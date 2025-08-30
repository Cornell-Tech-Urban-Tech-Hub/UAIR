from typing import Any, Dict


def build_engine_kwargs(
    *,
    max_model_len: int,
    gpu_memory_utilization: float,
    tensor_parallel_size: int,
    max_num_batched_tokens: int,
    max_num_seqs: int,
    kv_cache_dtype: str | None,
    enable_chunked_prefill: bool,
    enforce_eager: bool,
    tokenizer_pool_size: int | None = None,
) -> Dict[str, Any]:
    ekw: Dict[str, Any] = {
        "max_model_len": int(max_model_len),
        "gpu_memory_utilization": float(gpu_memory_utilization),
        "tensor_parallel_size": int(tensor_parallel_size),
        "disable_log_stats": True,
        "max_num_batched_tokens": int(max_num_batched_tokens),
        "max_num_seqs": int(max_num_seqs),
        "enforce_eager": bool(enforce_eager),
        "trust_remote_code": True,
    }
    if enable_chunked_prefill and gpu_memory_utilization <= 0.8:
        ekw["enable_chunked_prefill"] = True
    if kv_cache_dtype and kv_cache_dtype != "auto":
        ekw["kv_cache_dtype"] = kv_cache_dtype
    if tokenizer_pool_size is not None:
        ekw["tokenizer_pool_size"] = int(tokenizer_pool_size)
    return ekw


