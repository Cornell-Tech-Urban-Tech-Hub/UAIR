from typing import Any, Dict, Optional
import pandas as pd
import json
import os
import logging
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


def _serialize_arrow_unfriendly_in_row(row: Dict[str, Any], columns):
    """In-place convert nested/dict/list columns to JSON strings in a row dict."""
    for col in columns:
        if col in row:
            val = row.get(col)
            if isinstance(val, (dict, list, tuple)):
                row[col] = _to_json_str(val)


def _detect_num_gpus() -> int:
    """Detect the number of GPUs allocated to this job.
    
    Priority order:
    1. CUDA_VISIBLE_DEVICES environment variable (set by launcher)
    2. SLURM_GPUS_PER_NODE or SLURM_GPUS_ON_NODE
    3. torch.cuda.device_count() if CUDA is available
    4. Return 1 as safe fallback
    """
    # Priority 1: CUDA_VISIBLE_DEVICES (most reliable for actual allocation)
    try:
        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if cuda_visible and cuda_visible.strip():
            # Parse comma-separated GPU indices (e.g., "0,1,2,3" -> 4 GPUs)
            gpu_indices = [x.strip() for x in cuda_visible.split(",") if x.strip()]
            if gpu_indices:
                return len(gpu_indices)
    except Exception:
        pass
    
    # Priority 2: SLURM environment variables
    try:
        slurm_gpus = os.environ.get("SLURM_GPUS_PER_NODE") or os.environ.get("SLURM_GPUS_ON_NODE")
        if slurm_gpus:
            # Can be a number like "4" or format like "gpu:4"
            try:
                if ":" in slurm_gpus:
                    return int(slurm_gpus.split(":")[-1])
                return int(slurm_gpus)
            except Exception:
                pass
    except Exception:
        pass
    
    # Priority 3: Torch CUDA device count
    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            count = torch.cuda.device_count()
            if count > 0:
                return count
    except Exception:
        pass
    
    # Fallback: 1 GPU
    return 1


def _detect_gpu_type() -> str:
    """Detect the GPU type/model name.
    
    Returns a normalized GPU type string (e.g., 'rtx_a6000', 'rtx_a5000', 'unknown').
    """
    try:
        import torch  # type: ignore
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            # Get the name of the first GPU (assuming homogeneous GPUs)
            gpu_name = torch.cuda.get_device_name(0).lower()
            
            # Normalize common GPU names
            if "a6000" in gpu_name:
                return "rtx_a6000"
            elif "a5000" in gpu_name:
                return "rtx_a5000"
            elif "a100" in gpu_name:
                return "a100"
            elif "v100" in gpu_name:
                return "v100"
            elif "a40" in gpu_name:
                return "a40"
            elif "rtx" in gpu_name:
                # Generic RTX
                return "rtx_generic"
            
            return "unknown"
    except Exception:
        pass
    
    return "unknown"


def _apply_gpu_aware_batch_settings(engine_kwargs: Dict[str, Any], cfg) -> Dict[str, Any]:
    """Apply GPU-type-aware batch size and max_num_seqs if not explicitly set.
    
    GPU-specific defaults (can be overridden in config):
    - RTX A6000: batch_size=4, max_num_seqs=4
    - RTX A5000: batch_size=2, max_num_seqs=2
    - Others: Use config defaults or fallback values
    """
    # GPU-aware defaults mapping
    GPU_BATCH_SETTINGS = {
        "rtx_a6000": {"batch_size": 4, "max_num_seqs": 4},
        "rtx_a5000": {"batch_size": 2, "max_num_seqs": 2},
        "a100": {"batch_size": 8, "max_num_seqs": 8},
        "v100": {"batch_size": 4, "max_num_seqs": 4},
        "a40": {"batch_size": 4, "max_num_seqs": 4},
    }
    
    gpu_type = _detect_gpu_type()
    gpu_settings = GPU_BATCH_SETTINGS.get(gpu_type, {})
    
    # Apply max_num_seqs from GPU settings if not in engine_kwargs and GPU type is recognized
    if "max_num_seqs" not in engine_kwargs and gpu_settings:
        try:
            engine_kwargs["max_num_seqs"] = gpu_settings["max_num_seqs"]
            if not os.environ.get("RULE_TUPLES_SILENT"):
                print(f"Auto-set max_num_seqs={gpu_settings['max_num_seqs']} for {gpu_type}")
        except Exception:
            pass
    
    return gpu_settings


def _filter_vllm_engine_kwargs(ek: Dict[str, Any]) -> Dict[str, Any]:
    """Drop engine kwargs unsupported by the installed vLLM version.

    We try to introspect vllm.AsyncEngineArgs for accepted fields. If that
    fails, conservatively drop known newer flags.
    """
    try:
        import vllm as _v
        accepted = None
        try:
            fields = getattr(getattr(_v, "AsyncEngineArgs", None), "__dataclass_fields__", None)
            if isinstance(fields, dict) and fields:
                accepted = set(fields.keys())
        except Exception:
            accepted = None
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
    ek = dict(ek)
    for k in ("use_v2_block_manager",):
        ek.pop(k, None)
    return ek

def _extract_last_json(text: str) -> Optional[Dict[str, Any]]:
    if not isinstance(text, str) or not text.strip():
        return None
    s = text
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    import re
    snippets = re.findall(r"\{[\s\S]*\}", s)
    for snip in reversed(snippets or []):
        try:
            obj = json.loads(snip)
            if isinstance(obj, dict):
                return obj
        except Exception:
            continue
    return None


def run_decomposition_stage(df: pd.DataFrame, cfg):
    """Placeholder decomposition stage.

    Produces CI tuple columns with None defaults; later replaced by LLM extraction.
    Columns: ci_subject, ci_sender, ci_receiver, ci_information, ci_transmission_principle, ci_missing_elements (list)
    """
    base_cols = [
        "ci_subject",
        "ci_sender",
        "ci_receiver",
        "ci_information",
        "ci_transmission_principle",
        "ci_missing_elements",
    ]
    is_ray_ds = hasattr(df, "map_batches") and hasattr(df, "count") and _RAY_OK
    if not is_ray_ds:
        if df is None or len(df) == 0:
            return pd.DataFrame(columns=["article_text"] + base_cols)
        out = df.copy()
    use_llm = bool(getattr(cfg.runtime, "use_llm_decompose", False))
    # Whether to JSON-serialize nested columns for Arrow/Parquet friendliness
    try:
        serialize_nested = bool(getattr(cfg.runtime, "serialize_nested_json", True))
    except Exception:
        serialize_nested = True
    if not use_llm or not _RAY_OK:
        if is_ray_ds:
            def _fill_empty(pdf: pd.DataFrame) -> pd.DataFrame:
                pdf = pdf.copy()
                for c in base_cols:
                    pdf[c] = None
                pdf["ci_missing_elements"] = pdf.apply(lambda r: ["subject","sender","receiver","information","transmission_principle"], axis=1)
                if serialize_nested:
                    pdf["ci_missing_elements"] = pdf["ci_missing_elements"].map(_to_json_str)
                return pdf
            return df.map_batches(_fill_empty, batch_format="pandas")
        for c in base_cols:
            out[c] = None
        out["ci_missing_elements"] = out.apply(lambda r: ["subject","sender","receiver","information","transmission_principle"], axis=1)
        if serialize_nested:
            out["ci_missing_elements"] = out["ci_missing_elements"].map(_to_json_str)
        return out

    system_prompt = str(getattr(cfg.prompt_decompose, "system_prompt", ""))
    prompt_template = str(getattr(cfg.prompt_decompose, "prompt_template", ""))

    def _format_prompt(article_text: str) -> str:
        return (
            prompt_template
            .replace("{{rule_text}}", str(article_text or ""))
            .replace("{{article_text}}", str(article_text or ""))
        )

    # Constrain GPU mem via vLLM engine args: prefer provided config; otherwise set conservative defaults
    ek = dict(getattr(cfg.model, "engine_kwargs", {}))
    ek.setdefault("max_model_len", 4096)
    ek.setdefault("max_num_seqs", 4)  # Reduced from 16 for 2-GPU setup
    ek.setdefault("gpu_memory_utilization", 0.6)  # Reduced from 0.85 for 2-GPU setup
    # Auto-detect tensor_parallel_size from allocated GPUs if not explicitly set
    if "tensor_parallel_size" not in ek:
        try:
            num_gpus = _detect_num_gpus()
            ek["tensor_parallel_size"] = num_gpus
            if not os.environ.get("RULE_TUPLES_SILENT"):
                print(f"Auto-detected {num_gpus} GPU(s) for tensor parallelism")
        except Exception:
            ek.setdefault("tensor_parallel_size", 1)
    # Apply GPU-type-aware batch settings (max_num_seqs and batch_size)
    gpu_settings = _apply_gpu_aware_batch_settings(ek, cfg)
    # vLLM best-practice safe defaults (overridable via config)
    ek.setdefault("enable_prefix_caching", True)
    ek.setdefault("use_v2_block_manager", True)
    ek.setdefault("tokenizer_mode", "auto")
    ek.setdefault("trust_remote_code", True)
    ek.setdefault("dtype", "auto")
    ek.setdefault("kv_cache_dtype", "auto")
    ek = _filter_vllm_engine_kwargs(ek)
    # Determine batch_size: explicit config > GPU-aware default > fallback
    try:
        batch_size_cfg = getattr(cfg.model, "batch_size", None)
        if batch_size_cfg is not None:
            batch_size = int(batch_size_cfg)
        elif gpu_settings and "batch_size" in gpu_settings:
            batch_size = gpu_settings["batch_size"]
            if not os.environ.get("RULE_TUPLES_SILENT"):
                print(f"Auto-set batch_size={batch_size} for {_detect_gpu_type()}")
        else:
            batch_size = 16
    except Exception:
        batch_size = 16
    engine_config = vLLMEngineProcessorConfig(
        model_source=str(getattr(cfg.model, "model_source")),
        runtime_env={
            "env_vars": {
                "VLLM_LOGGING_LEVEL": "ERROR",
                # Propagate wandb config to Ray workers (uses in-process mode)
                "WANDB_DISABLE_SERVICE": str(os.environ.get("WANDB_DISABLE_SERVICE", "true")),
                "WANDB_SILENT": str(os.environ.get("WANDB_SILENT", "true")),
            }
        },
        engine_kwargs=ek,
        concurrency=int(getattr(cfg.model, "concurrency", 1) or 1),
        batch_size=int(batch_size),
    )
    # Prefer stage-specific sampling params when present; convert nested DictConfig -> dict
    try:
        sp_src = getattr(cfg, "sampling_params_decompose", getattr(cfg, "sampling_params", {}))
        sampling_params = OmegaConf.to_container(sp_src, resolve=True) if isinstance(sp_src, (dict,)) or hasattr(sp_src, "_get_node") else dict(sp_src)
    except Exception:
        sampling_params = dict(getattr(cfg, "sampling_params", {}))

    def _pre(row: Dict[str, Any]) -> Dict[str, Any]:
        _maybe_silence_vllm_logs()
        user = _format_prompt(row.get("article_text"))
        sp = dict(sampling_params)
        # Encourage short, structured JSON-only outputs
        sp.setdefault("max_tokens", 256)
        sp.setdefault("detokenize", False)
        # Optional guided decoding hook (future-proof; requires vLLM support)
        try:
            if bool(getattr(cfg.runtime, "guided_decoding_decompose", False)):
                schema = {
                    "type": "object",
                    "properties": {
                        "subject": {"type": ["string","null"]},
                        "sender": {"type": ["string","null"]},
                        "receiver": {"type": ["string","null"]},
                        "information": {"type": ["string","null"]},
                        "transmission_principle": {"type": ["string","null"]},
                        "missing": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["missing"],
                    "additionalProperties": True,
                }
                sp["guided_decoding"] = {"json": schema}
        except Exception:
            pass
        # Remove classification artifacts that might override our messages
        base = {k: v for k, v in row.items() if k not in {"messages", "sampling_params", "generated_text", "llm_output"}}
        base.update({
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user},
            ],
            "sampling_params": sp,
        })
        return base

    def _post(row: Dict[str, Any]) -> Dict[str, Any]:
        txt = row.get("generated_text")
        obj = _extract_last_json(txt if isinstance(txt, str) else "") or {}

        # Robust key normalization and synonyms mapping
        def _norm_key(k: Any) -> str:
            s = str(k).strip().lower()
            out = []
            for ch in s:
                out.append(ch if (ch.isalnum() or ch == "_") else "_")
            return "".join(out)

        norm_obj: Dict[str, Any] = {}
        try:
            for k, v in (obj.items() if isinstance(obj, dict) else []):
                norm_obj[_norm_key(k)] = v
        except Exception:
            pass

        def _first_key(keys):
            for k in keys:
                if k in norm_obj:
                    return norm_obj.get(k)
            return None

        subject = _first_key(["subject", "data_subject", "person", "individual"])
        sender = _first_key(["sender", "source", "discloser", "actor_sender"])
        receiver = _first_key(["receiver", "recipient", "audience", "actor_receiver", "target"])
        information = _first_key(["information", "info", "data", "content", "pii"])
        tp = _first_key(["transmission_principle", "transmissionprinciple", "principle", "condition", "constraint", "rule"])
        missing = _first_key(["missing", "missing_elements", "missing_fields"]) or []
        if not isinstance(missing, list):
            try:
                missing = list(missing) if missing is not None else []
            except Exception:
                missing = []

        # Optionally serialize nested columns for Arrow/Parquet compatibility
        if serialize_nested:
            _serialize_arrow_unfriendly_in_row(row, [
                "messages",
                "sampling_params",
                "usage",
                "token_counts",
            ])
            missing_out = _to_json_str(missing)
        else:
            missing_out = missing

        # Ensure ci_* scalar columns are Arrow-friendly (no lists/dicts in columns)
        def _norm_ci_value(v: Any) -> Any:
            if isinstance(v, (list, tuple)):
                return _to_json_str(v) if serialize_nested else ", ".join([str(x) for x in v if x is not None])
            if isinstance(v, dict):
                return _to_json_str(v) if serialize_nested else str(v)
            return v

        subject_out = _norm_ci_value(subject)
        sender_out = _norm_ci_value(sender)
        receiver_out = _norm_ci_value(receiver)
        information_out = _norm_ci_value(information)
        tp_out = _norm_ci_value(tp)

        return {
            **row,
            "ci_subject": subject_out,
            "ci_sender": sender_out,
            "ci_receiver": receiver_out,
            "ci_information": information_out,
            "ci_transmission_principle": tp_out,
            "ci_missing_elements": missing_out,
            "llm_output": row.get("generated_text"),
            "_progress_row": 1,
        }

    processor = build_llm_processor(engine_config, preprocess=_pre, postprocess=_post)
    if is_ray_ds:
        return processor(df)
    ds = ray.data.from_pandas(out)
    out_ds = processor(ds).materialize()
    out_df = out_ds.to_pandas()
    for c in base_cols:
        if c not in out_df.columns:
            out_df[c] = None
    # Stage-scoped logging is handled by the orchestrator
    return out_df


