from typing import Any, Dict, List
import re
from bisect import bisect_right
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


def _build_relevant_regex() -> re.Pattern:
    phrases = [
        r"artificial\s+intelligence",
        r"machine\s+learning",
        r"neural\s+network",
        r"large\s+language\s+model",
        r"transformer",
        r"chatgpt|gpt-\d+|gpt-",
        r"openai|anthropic|claude|gemini|qwen",
        r"fine-?tuning|inference|prompt(ing)?|agent(s)?",
        # light domain cues mirroring keyword_extract guidance
        r"(?:city|cities)",
        r"urban",
        r"climate",
        r"earth",
        r"environment",
        r"transport",
    ]
    pattern = r"(" + r"|".join(phrases) + r")"
    return re.compile(pattern, flags=re.IGNORECASE)


def _generate_relevant_blocks(text: str, compiled_regex: re.Pattern, window_words: int = 100) -> List[str]:
    if not isinstance(text, str) or not text:
        return []
    token_matches = list(re.finditer(r"\S+", text))
    if not token_matches:
        return []
    token_starts = [m.start() for m in token_matches]
    intervals: List[List[int]] = []
    for m in compiled_regex.finditer(text):
        idx = max(0, min(len(token_starts) - 1, bisect_right(token_starts, m.start()) - 1))
        start_token = max(0, idx - window_words)
        end_token = min(len(token_matches) - 1, idx + window_words)
        start_char = token_matches[start_token].start()
        end_char = token_matches[end_token].end()
        intervals.append([start_char, end_char])
    if not intervals:
        return []
    intervals.sort(key=lambda x: x[0])
    merged: List[List[int]] = []
    for s, e in intervals:
        if not merged or s > merged[-1][1]:
            merged.append([s, e])
        else:
            merged[-1][1] = max(merged[-1][1], e)
    return [text[s:e] for s, e in merged]


def run_classification_stage(df: pd.DataFrame, cfg):
    """Article relevance classification stage.

    - Heuristic: detect AI-related keywords in text
    - LLM: borrow prompt shape from experiments/prompts/base.yaml (relevance_v1) when available,
      else fall back to UAIR classify prompt. Produces both `relevance_answer` and `is_relevant`.
    Uses `article_text` as the canonical input text column.
    """
    # Streaming path: if a Ray Dataset is passed, use it end-to-end
    is_ray_ds = hasattr(df, "map_batches") and hasattr(df, "count") and _RAY_OK
    if not is_ray_ds:
        if df is None or len(df) == 0:
            return pd.DataFrame(columns=["article_text","is_relevant"])  # minimal
        out = df.copy()
    def _heuristic(text: Any) -> bool:
        s = str(text or "").lower()
        if not s:
            return False
        # Simple AI/news relevance heuristic aligned to experiments.relevance_stage intent
        keywords = [
            "artificial intelligence"," ai ","machine learning","ml ","neural network","deep learning",
            "large language model","llm","chatgpt","gpt-","openai","anthropic","claude","gemini","qwen",
            "transformer model","fine-tuning","inference","prompting","agents","autonomous agent","model weights",
        ]
        return any(k in s for k in keywords)
    use_llm = bool(getattr(cfg.runtime, "use_llm_classify", False))
    # Prefilter gating: pre_gating (filter before LLM), post_gating (compute flag only), off
    try:
        prefilter_mode = str(getattr(cfg.runtime, "prefilter_mode", "pre_gating")).strip().lower()
    except Exception:
        prefilter_mode = "pre_gating"
    if not use_llm:
        if is_ray_ds:
            def _heuristic_batch(pdf: pd.DataFrame) -> pd.DataFrame:
                pdf = pdf.copy()
                pdf["is_relevant"] = pdf["article_text"].apply(_heuristic)
                pdf["classification_mode"] = "heuristic"
                return pdf
            return df.map_batches(_heuristic_batch, batch_format="pandas")
        out["is_relevant"] = out["article_text"].apply(_heuristic)
        out["classification_mode"] = "heuristic"
        out_df = out
        # Stage-scoped W&B logging (optional) even for heuristic path
        try:
            if bool(getattr(cfg.wandb, "enabled", False)):
                try:
                    import wandb as _wandb  # type: ignore
                except Exception:
                    _wandb = None  # type: ignore
                if _wandb is not None:
                    _created_run = False
                    try:
                        group = getattr(cfg.wandb, "group", None) or os.environ.get("WANDB_GROUP")
                    except Exception:
                        group = None
                    try:
                        proj = str(getattr(cfg.wandb, "project", "UAIR") or "UAIR")
                    except Exception:
                        proj = "UAIR"
                    try:
                        ent = getattr(cfg.wandb, "entity", None)
                        ent = str(ent) if (ent is not None and str(ent).strip() != "") else None
                    except Exception:
                        ent = None
                    try:
                        existing = getattr(_wandb, "run", None)
                        if existing is None:
                            run = _wandb.init(project=proj, entity=ent, job_type=str(getattr(cfg.runtime, "stage", "classify")), name=f"{getattr(cfg.experiment, 'name', 'UAIR')}-classify", group=group, config=OmegaConf.to_container(cfg, resolve=True))
                            _created_run = True
                        else:
                            run = existing
                    except Exception:
                        run = None  # type: ignore
                    try:
                        total = int(len(out_df))
                    except Exception:
                        total = 0
                    rel_count = 0
                    ratio = 0.0
                    try:
                        if total > 0 and "is_relevant" in out_df.columns:
                            rel_count = int(out_df["is_relevant"].astype(bool).sum())
                            ratio = float(rel_count) / float(total) if total > 0 else 0.0
                    except Exception:
                        pass
                    try:
                        _wandb.log({
                            "classify/rows": total,
                            "classify/relevant_count": rel_count,
                            "classify/relevant_ratio": ratio,
                            "classify/avg_latency_s": None,
                        })
                    except Exception:
                        pass
                    try:
                        if _created_run and not bool(getattr(cfg.wandb, "single_run", False)):
                            _wandb.finish()
                    except Exception:
                        pass
        except Exception:
            pass
        return out_df

    if not _RAY_OK:
        out["is_relevant"] = out["article_text"].apply(_heuristic)
        out["classification_mode"] = "heuristic_fallback"
        try:
            print("Warning: Ray LLM not available; falling back to heuristic classification.")
        except Exception:
            pass
        return out

    # LLM classification via Ray vLLM
    # Optional keyword-based buffering (±window words around matches)
    try:
        enable_kw_buf = bool(getattr(cfg.runtime, "keyword_buffering", True))
    except Exception:
        enable_kw_buf = True
    try:
        window_words = int(getattr(cfg.runtime, "keyword_window_words", 100) or 100)
    except Exception:
        window_words = 100
    kw_regex = _build_relevant_regex() if enable_kw_buf else None
    # Compute keyword prefilter flag for gating
    def _kw_flag(text: Any) -> bool:
        if kw_regex is None:
            return True
        try:
            return bool(kw_regex.search(str(text or "")))
        except Exception:
            return True
    # Prefer experiments-style prompts when available; otherwise fall back to UAIR prompt
    try:
        system_prompt = (
            getattr(getattr(cfg, "prompts", {}), "relevance_v1", {}).get("system")  # type: ignore
            if hasattr(cfg, "prompts") else None
        )
    except Exception:
        system_prompt = None
    if not system_prompt:
        system_prompt = str(getattr(cfg.prompt, "system_prompt", ""))

    try:
        user_template = (
            getattr(getattr(cfg, "prompts", {}), "relevance_v1", {}).get("user_template")  # type: ignore
            if hasattr(cfg, "prompts") else None
        )
    except Exception:
        user_template = None
    if not user_template:
        user_template = str(getattr(cfg.prompt, "prompt_template", ""))

    def _format_user(article_text: str, row: Dict[str, Any]) -> str:
        text_val = str((row.get("chunk_text") if row.get("chunk_text") else article_text) or "")
        # Provide basic ids to satisfy experiments-style templates when present
        try:
            art_id = row.get("article_id") or row.get("name") or None
        except Exception:
            art_id = None
        if not art_id:
            try:
                import hashlib as _hash
                art_id = _hash.sha1(text_val.encode("utf-8")).hexdigest()[:12]
            except Exception:
                art_id = "unknown"
        # If template uses '{chunk_text}' style, format with .format; else support UAIR '{{rule_text}}'
        if "{chunk_text}" in user_template or "{article_id}" in user_template:
            try:
                return user_template.format(article_id=art_id, chunk_id=0, num_chunks=1, chunk_text=text_val)
            except Exception:
                pass
        # Maintain UAIR template compatibility
        return user_template.replace("{{rule_text}}", text_val).replace("{{article_text}}", text_val)

    # Tokenizer-based trimming helpers
    def _get_tokenizer(model_source: str):
        try:
            from transformers import AutoTokenizer  # type: ignore
            return AutoTokenizer.from_pretrained(model_source, trust_remote_code=True, use_fast=True)
        except Exception:
            return None

    def _get_max_user_input_tokens(tokenizer, system_text: str) -> int:
        try:
            system_tokens = len(tokenizer.encode(system_text, add_special_tokens=False)) if tokenizer else 0
            max_model_len = int(getattr(cfg.model, "engine_kwargs", {}).get("max_model_len", 4096))
            try:
                sp = getattr(cfg, "sampling_params_classify", getattr(cfg, "sampling_params", {}))
                if hasattr(sp, "max_tokens"):
                    max_output = int(getattr(sp, "max_tokens"))
                else:
                    max_output = int((sp or {}).get("max_tokens", 8))
            except Exception:
                max_output = 8
            safety = 512
            return max(512, max_model_len - max_output - system_tokens - safety)
        except Exception:
            return 2048

    def _trim_text_for_prompt(text: str, tokenizer, system_text: str) -> str:
        # Tokenizer-aware trimming when available; otherwise conservative char-based fallback
        if tokenizer:
            try:
                ids = tokenizer.encode(text or "", add_special_tokens=False)
                max_user = _get_max_user_input_tokens(tokenizer, system_text)
                if len(ids) <= max_user:
                    return text
                ids = ids[:max_user]
                return tokenizer.decode(ids, skip_special_tokens=True)
            except Exception:
                pass
        # Fallback: approximate 4 chars per token budget
        try:
            max_user = _get_max_user_input_tokens(tokenizer, system_text)
        except Exception:
            max_user = 2048
        approx_chars_per_token = 4
        max_chars = int(max_user) * approx_chars_per_token
        try:
            return text if len(text or "") <= max_chars else str(text or "")[:max_chars]
        except Exception:
            return text

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
    # Borrow from experiments: short label budget; optionally allow rationales
    try:
        if bool(getattr(cfg.runtime, "log_rationales", False)):
            sampling_params.setdefault("max_tokens", 32)
        else:
            sampling_params.setdefault("max_tokens", 8)
    except Exception:
        sampling_params.setdefault("max_tokens", 8)

    # Driver-computed conservative char budget for user content (captured into Ray workers)
    try:
        mm_len = int(getattr(cfg.model, "engine_kwargs", {}).get("max_model_len", 4096))
    except Exception:
        mm_len = 4096
    try:
        max_out = int(sampling_params.get("max_tokens", 8) or 8)
    except Exception:
        max_out = 8
    _approx_user_char_budget = max(2048, (mm_len - max_out - 512) * 4)

    def _attach_chunk_text(row: Dict[str, Any]) -> Dict[str, Any]:
        if not enable_kw_buf or kw_regex is None:
            return row
        text_val = row.get("article_text")
        try:
            blocks = _generate_relevant_blocks(text_val, kw_regex, window_words)
        except Exception:
            blocks = []
        if blocks:
            row["chunk_text"] = "\n\n".join(blocks)
        else:
            row.setdefault("chunk_text", str(text_val or ""))
        return row

    def _pre(row: Dict[str, Any]) -> Dict[str, Any]:
        _maybe_silence_vllm_logs()
        if enable_kw_buf and kw_regex is not None:
            row = _attach_chunk_text(dict(row))
        user = _format_user(row.get("article_text"), row)
        try:
            if isinstance(user, str) and len(user) > int(_approx_user_char_budget):
                user = user[: int(_approx_user_char_budget)]
        except Exception:
            pass
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
            "relevance_answer": row.get("generated_text"),
            "is_relevant": bool(is_rel),
            "llm_output": row.get("generated_text"),
            "classification_mode": "llm_relevance",
            "latency_s": (float(ts_end) - float(row.get("ts_start", ts_end))),
            "token_usage_prompt": ((usage or {}).get("prompt_tokens") or (usage or {}).get("input_tokens")),
            "token_usage_output": ((usage or {}).get("completion_tokens") or (usage or {}).get("output_tokens")),
            "token_usage_total": ((usage or {}).get("total_tokens")),
            # Signal a single-row progress marker for downstream logging
            "_progress_row": 1,
        }

    # Pre-attach chunk_text in a separate map when streaming; apply prefilter gating and trimming
    if is_ray_ds:
        ds_in = df
        # Keyword flag + gating
        if prefilter_mode in ("pre_gating", "post_gating"):
            ds_in = ds_in.map(lambda r: {**r, "relevant_keyword": _kw_flag(r.get("article_text"))})
            if prefilter_mode == "pre_gating":
                ds_in = ds_in.filter(lambda r: bool(r.get("relevant_keyword", True)))
        # Attach chunk_text
        if enable_kw_buf and kw_regex is not None:
            ds_in = ds_in.map(_attach_chunk_text)
        # Trim chunk_text to fit model context budget
        tok = _get_tokenizer(str(getattr(cfg.model, "model_source", "")))
        sys_text = system_prompt
        def _trim_row(r: Dict[str, Any]) -> Dict[str, Any]:
            txt = r.get("chunk_text") or r.get("article_text")
            r["chunk_text"] = _trim_text_for_prompt(str(txt or ""), tok, sys_text)
            return r
        ds_in = ds_in.map(_trim_row)
        processor = build_llm_processor(engine_config, preprocess=_pre, postprocess=_post)
        return processor(ds_in)

    # Pandas path: prefilter, attach chunk_text, trim, then Ray Dataset
    out = out.copy()
    # Track keyword gating statistics before filtering
    _kw_orig_total = None
    _kw_marked_total = None
    _kw_kept_after_gate = None
    if prefilter_mode in ("pre_gating", "post_gating"):
        try:
            _kw_orig_total = int(len(out))
        except Exception:
            _kw_orig_total = None
        try:
            out["relevant_keyword"] = out["article_text"].apply(lambda t: _kw_flag(t))
        except Exception:
            out["relevant_keyword"] = True
        try:
            _kw_marked_total = int(out["relevant_keyword"].astype(bool).sum())
        except Exception:
            _kw_marked_total = None
        if prefilter_mode == "pre_gating":
            out = out[out["relevant_keyword"] == True]
            try:
                _kw_kept_after_gate = int(len(out))
            except Exception:
                _kw_kept_after_gate = None
    if enable_kw_buf and kw_regex is not None:
        try:
            out["chunk_text"] = out["article_text"].apply(lambda t: "\n\n".join(_generate_relevant_blocks(t, kw_regex, window_words)) or str(t or ""))
        except Exception:
            out["chunk_text"] = out["article_text"].astype(str)
    # Early W&B log of keyword gating (before vLLM) so metrics appear promptly
    try:
        if bool(getattr(cfg.wandb, "enabled", False)):
            try:
                import wandb as _wandb  # type: ignore
            except Exception:
                _wandb = None  # type: ignore
            if _wandb is not None:
                _created_run_early = False
                try:
                    group = getattr(cfg.wandb, "group", None) or os.environ.get("WANDB_GROUP")
                except Exception:
                    group = None
                try:
                    proj = str(getattr(cfg.wandb, "project", "UAIR") or "UAIR")
                except Exception:
                    proj = "UAIR"
                try:
                    ent = getattr(cfg.wandb, "entity", None)
                    ent = str(ent) if (ent is not None and str(ent).strip() != "") else None
                except Exception:
                    ent = None
                try:
                    existing = getattr(_wandb, "run", None)
                    if existing is None:
                        run = _wandb.init(project=proj, entity=ent, job_type=str(getattr(cfg.runtime, "stage", "classify")), name=f"{getattr(cfg.experiment, 'name', 'UAIR')}-classify", group=group, config=OmegaConf.to_container(cfg, resolve=True))
                        _created_run_early = True
                    else:
                        run = existing
                except Exception:
                    run = None  # type: ignore
                try:
                    payload_early = {}
                    if _kw_orig_total is not None and _kw_marked_total is not None and _kw_orig_total > 0:
                        payload_early["keyword/marked"] = int(_kw_marked_total)
                        payload_early["keyword/marked_ratio"] = float(_kw_marked_total) / float(_kw_orig_total)
                        payload_early["keyword/total_checked"] = int(_kw_orig_total)
                    if _kw_kept_after_gate is not None:
                        payload_early["keyword/pre_gated_rows"] = int(_kw_kept_after_gate)
                    if payload_early:
                        _wandb.log(payload_early)
                except Exception:
                    pass
                try:
                    if _created_run_early and not bool(getattr(cfg.wandb, "single_run", False)):
                        _wandb.finish()
                except Exception:
                    pass
    except Exception:
        pass
    tok = _get_tokenizer(str(getattr(cfg.model, "model_source", "")))
    try:
        sys_text = system_prompt
        out["chunk_text"] = out["chunk_text"].apply(lambda t: _trim_text_for_prompt(str(t or ""), tok, sys_text))
    except Exception:
        pass
    ds = ray.data.from_pandas(out)
    processor = build_llm_processor(engine_config, preprocess=_pre, postprocess=_post)
    out_ds = processor(ds).materialize()
    out_df = out_ds.to_pandas()
    # Stage-scoped W&B logging (optional)
    try:
        if bool(getattr(cfg.wandb, "enabled", False)):
            try:
                import wandb as _wandb  # type: ignore
            except Exception:
                _wandb = None  # type: ignore
            if _wandb is not None:
                _created_run = False
                try:
                    group = getattr(cfg.wandb, "group", None) or os.environ.get("WANDB_GROUP")
                except Exception:
                    group = None
                try:
                    proj = str(getattr(cfg.wandb, "project", "UAIR") or "UAIR")
                except Exception:
                    proj = "UAIR"
                try:
                    ent = getattr(cfg.wandb, "entity", None)
                    ent = str(ent) if (ent is not None and str(ent).strip() != "") else None
                except Exception:
                    ent = None
                try:
                    existing = getattr(_wandb, "run", None)
                    if existing is None:
                        run = _wandb.init(project=proj, entity=ent, job_type=str(getattr(cfg.runtime, "stage", "classify")), name=f"{getattr(cfg.experiment, 'name', 'UAIR')}-classify", group=group, config=OmegaConf.to_container(cfg, resolve=True))
                        _created_run = True
                    else:
                        run = existing
                except Exception:
                    run = None  # type: ignore
                try:
                    total = int(len(out_df))
                except Exception:
                    total = 0
                rel_count = 0
                ratio = 0.0
                avg_lat = None
                try:
                    if total > 0 and "is_relevant" in out_df.columns:
                        rel_count = int(out_df["is_relevant"].astype(bool).sum())
                        ratio = float(rel_count) / float(total) if total > 0 else 0.0
                except Exception:
                    pass
                try:
                    if "latency_s" in out_df.columns:
                        lat = [float(v) for v in out_df.get("latency_s", []).tolist() if isinstance(v, (int, float))]
                        avg_lat = (sum(lat) / len(lat)) if lat else None
                except Exception:
                    pass
                try:
                    payload = {
                        "classify/rows": total,
                        "classify/relevant_count": rel_count,
                        "classify/relevant_ratio": ratio,
                        "classify/avg_latency_s": avg_lat,
                    }
                    _wandb.log(payload)
                    # Log keyword gating under its own section
                    kw_payload = {}
                    try:
                        if _kw_orig_total is not None and _kw_marked_total is not None and _kw_orig_total > 0:
                            kw_payload["keyword/marked"] = int(_kw_marked_total)
                            kw_payload["keyword/marked_ratio"] = float(_kw_marked_total) / float(_kw_orig_total)
                            kw_payload["keyword/total_checked"] = int(_kw_orig_total)
                    except Exception:
                        pass
                    try:
                        if _kw_kept_after_gate is not None:
                            kw_payload["keyword/pre_gated_rows"] = int(_kw_kept_after_gate)
                    except Exception:
                        pass
                    if kw_payload:
                        _wandb.log(kw_payload)
                except Exception:
                    pass
                try:
                    if _created_run and not bool(getattr(cfg.wandb, "single_run", False)):
                        _wandb.finish()
                except Exception:
                    pass
    except Exception:
        pass
    return out_df


