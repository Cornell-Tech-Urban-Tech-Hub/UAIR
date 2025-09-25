from typing import Any, Dict, List
import pandas as pd
import os
import logging
import json
from omegaconf import OmegaConf

try:
    import ray  # noqa: F401
    from ray.data.llm import build_llm_processor, vLLMEngineProcessorConfig  # type: ignore
    _RAY_OK = True
except Exception:
    _RAY_OK = False

_VLLM_LOGS_SILENCED = False

def _maybe_silence_vllm_logs() -> None:
    global _VLLM_LOGS_SILENCED
    if _VLLM_LOGS_SILENCED:
        return
    try:
        if os.environ.get("RULE_TUPLES_SILENT"):
            os.environ.setdefault("VLLM_LOGGING_LEVEL", "ERROR")
            for name in ("vllm", "vllm.logger", "vllm.engine", "vllm.core", "vllm.worker"):
                lg = logging.getLogger(name)
                lg.setLevel(logging.ERROR)
                lg.propagate = False
        _VLLM_LOGS_SILENCED = True
    except Exception:
        pass


def _to_json_str(value: Any):
    """Serialize Python objects to JSON string for Arrow/Parquet friendliness.

    Returns None for None input; falls back to str(value) on failure.
    """
    try:
        if value is None:
            return None
        return json.dumps(value, ensure_ascii=False, default=str)
    except Exception:
        try:
            return str(value)
        except Exception:
            return None


def _serialize_arrow_unfriendly_in_row(row: Dict[str, Any], columns: List[str]) -> None:
    """In-place convert nested/dict/list columns to JSON strings in a row dict."""
    for col in columns:
        if col in row:
            val = row.get(col)
            if isinstance(val, (dict, list, tuple)):
                row[col] = _to_json_str(val)


def _filter_vllm_engine_kwargs(ek: Dict[str, Any]) -> Dict[str, Any]:
    """Drop engine kwargs unsupported by the installed vLLM version.

    We try to introspect vllm.AsyncEngineArgs for accepted fields. If that
    fails, conservatively drop known newer flags.
    """
    try:
        import vllm as _v
        accepted = None
        # Prefer dataclass fields (older vLLM uses dataclasses)
        try:
            fields = getattr(getattr(_v, "AsyncEngineArgs", None), "__dataclass_fields__", None)
            if isinstance(fields, dict) and fields:
                accepted = set(fields.keys())
        except Exception:
            accepted = None
        # Fallback to signature introspection
        if accepted is None:
            try:
                import inspect as _inspect
                sig = _inspect.signature(_v.AsyncEngineArgs.__init__)
                accepted = set(k for k in sig.parameters.keys() if k != "self")
            except Exception:
                accepted = None
        if accepted:
            filtered = {k: v for k, v in ek.items() if k in accepted}
            if len(filtered) != len(ek):
                try:
                    if not os.environ.get("RULE_TUPLES_SILENT"):
                        dropped = [k for k in ek.keys() if k not in accepted]
                        print(f"Filtering unsupported vLLM engine kwargs: {dropped}")
                except Exception:
                    pass
            return filtered
    except Exception:
        pass
    # Conservative fallback for unknown versions: drop newer flags
    ek = dict(ek)
    for k in ("use_v2_block_manager",):
        ek.pop(k, None)
    return ek


def run_classification_stage(df: pd.DataFrame, cfg):
    """Placeholder classification stage.

    For now, implement a trivial heuristic baseline so wiring works:
    - is_relevant = True if rule text contains any of {privacy, consent, data, information, share, disclose}
    Later, replace with LLM inference via Ray/vLLM using prompts in conf/prompt/classify.yaml.
    """
    # Streaming path: if a Ray Dataset is passed, use it end-to-end
    is_ray_ds = hasattr(df, "map_batches") and hasattr(df, "count") and _RAY_OK
    if not is_ray_ds:
        if df is None or len(df) == 0:
            return pd.DataFrame(columns=["name","rule_text","is_relevant"])  # minimal
        out = df.copy()
    def _heuristic(text: Any) -> bool:
        s = str(text or "").lower()
        if not s:
            return False
        keywords = ["privacy","consent","data","information","share","disclose","leak","dox"]
        return any(k in s for k in keywords)
    use_llm = bool(getattr(cfg.runtime, "use_llm_classify", False))
    if not use_llm:
        if is_ray_ds:
            def _heuristic_batch(pdf: pd.DataFrame) -> pd.DataFrame:
                pdf = pdf.copy()
                pdf["is_relevant"] = pdf["rule_text"].apply(_heuristic)
                pdf["classification_mode"] = "heuristic"
                return pdf
            return df.map_batches(_heuristic_batch, batch_format="pandas")
        out["is_relevant"] = out["rule_text"].apply(_heuristic)
        out["classification_mode"] = "heuristic"
        return out

    if not _RAY_OK:
        out["is_relevant"] = out["rule_text"].apply(_heuristic)
        out["classification_mode"] = "heuristic_fallback"
        try:
            print("Warning: Ray LLM not available; falling back to heuristic classification.")
        except Exception:
            pass
        return out

    # LLM classification via Ray vLLM
    system_prompt = str(getattr(cfg.prompt, "system_prompt", ""))
    prompt_template = str(getattr(cfg.prompt, "prompt_template", ""))

    def _format_prompt(rule_text: str) -> str:
        # Minimal templating: replace {{rule_text}}
        return prompt_template.replace("{{rule_text}}", str(rule_text or ""))

    # Constrain GPU mem via vLLM engine args: prefer provided config; otherwise set conservative defaults
    ek = dict(getattr(cfg.model, "engine_kwargs", {}))
    ek.setdefault("max_model_len", 4096)
    ek.setdefault("max_num_seqs", 16)
    ek.setdefault("gpu_memory_utilization", 0.85)
    # vLLM best-practice safe defaults (overridable via config)
    ek.setdefault("enable_prefix_caching", True)
    ek.setdefault("use_v2_block_manager", True)
    ek.setdefault("tokenizer_mode", "auto")
    ek.setdefault("trust_remote_code", True)
    ek.setdefault("dtype", "auto")
    ek.setdefault("kv_cache_dtype", "auto")
    ek = _filter_vllm_engine_kwargs(ek)
    engine_config = vLLMEngineProcessorConfig(
        model_source=str(getattr(cfg.model, "model_source")),
        runtime_env={
            "env_vars": {
                "VLLM_LOGGING_LEVEL": "ERROR",
            }
        },
        engine_kwargs=ek,
        concurrency=int(getattr(cfg.model, "concurrency", 1) or 1),
        batch_size=int(getattr(cfg.model, "batch_size", 16) or 16),
    )

    # Prefer stage-specific sampling params when present; convert nested DictConfig -> dict
    try:
        sp_src = getattr(cfg, "sampling_params_classify", getattr(cfg, "sampling_params", {}))
        sampling_params = OmegaConf.to_container(sp_src, resolve=True) if isinstance(sp_src, (dict,)) or hasattr(sp_src, "_get_node") else dict(sp_src)
    except Exception:
        sampling_params = dict(getattr(cfg, "sampling_params", {}))

    def _pre(row: Dict[str, Any]) -> Dict[str, Any]:
        _maybe_silence_vllm_logs()
        user = _format_prompt(row.get("rule_text"))
        from datetime import datetime as _dt
        return {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user},
            ],
            "sampling_params": sampling_params,
            "ts_start": _dt.now().timestamp(),
            **row,
        }

    def _post(row: Dict[str, Any]) -> Dict[str, Any]:
        from datetime import datetime as _dt
        text = str(row.get("generated_text") or "").strip().upper()
        is_rel = text.startswith("YES") or ("YES" in text and "NO" not in text)
        ts_end = _dt.now().timestamp()
        usage = row.get("usage") or row.get("token_counts") or None
        # Optionally serialize nested columns for Arrow/Parquet compatibility
        try:
            serialize_nested = bool(getattr(cfg.runtime, "serialize_nested_json", True))
        except Exception:
            serialize_nested = True
        if serialize_nested:
            _serialize_arrow_unfriendly_in_row(row, [
                "messages",
                "sampling_params",
                "usage",
                "token_counts",
            ])
        return {
            **row,
            "is_relevant": bool(is_rel),
            "llm_output": row.get("generated_text"),
            "classification_mode": "llm",
            "latency_s": (float(ts_end) - float(row.get("ts_start", ts_end))),
            "token_usage_prompt": ((usage or {}).get("prompt_tokens") or (usage or {}).get("input_tokens")),
            "token_usage_output": ((usage or {}).get("completion_tokens") or (usage or {}).get("output_tokens")),
            "token_usage_total": ((usage or {}).get("total_tokens")),
            # Signal a single-row progress marker for downstream logging
            "_progress_row": 1,
        }

    processor = build_llm_processor(engine_config, preprocess=_pre, postprocess=_post)
    if is_ray_ds:
        return processor(df)
    ds = ray.data.from_pandas(out)
    out_ds = processor(ds).materialize()
    out_df = out_ds.to_pandas()
    return out_df


