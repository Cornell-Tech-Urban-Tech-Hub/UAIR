from typing import Any, Dict, List, Tuple
import os
import json
import pandas as pd
import sys
import logging
import time
import re
from bisect import bisect_right
try:
    from omegaconf import ListConfig as _OmegaListConfig  # type: ignore
except Exception:
    _OmegaListConfig = None  # type: ignore

try:
    import ray  # type: ignore
    _RAY_OK = True
except Exception:
    _RAY_OK = False

def _ensure_ids(pdf: pd.DataFrame) -> pd.DataFrame:
    if "article_id" not in pdf.columns:
        try:
            import hashlib as _h
            def _gen(r: pd.Series) -> str:
                src = r.get("article_path")
                if not isinstance(src, str) or src.strip() == "":
                    src = r.get("article_text") or r.get("chunk_text") or ""
                return _h.sha1(str(src).encode("utf-8")).hexdigest()
            pdf = pdf.copy()
            pdf["article_id"] = pdf.apply(_gen, axis=1)
        except Exception:
            pdf["article_id"] = None
    return pdf

def _get_logger() -> logging.Logger:
    logger = logging.getLogger("uair.topic")
    if not logger.handlers:
        level_name = str(os.environ.get("UAIR_TOPIC_LOG_LEVEL", "INFO")).upper()
        try:
            logger.setLevel(getattr(logging, level_name, logging.INFO))
        except Exception:
            logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s uair.topic - %(message)s"))
        logger.addHandler(handler)
        logger.propagate = False
    return logger

def _log_event(event: str, logger=None, **kwargs: Any) -> None:
    """Log a topic stage event.
    
    Args:
        event: Event name
        logger: Optional WandbLogger instance for W&B logging
        **kwargs: Event metadata
    """
    payload: Dict[str, Any] = {"topic": {"event": event}}
    if kwargs:
        try:
            payload["topic"].update(kwargs)
        except Exception:
            pass
    try:
        _get_logger().info(json.dumps(payload))
    except Exception:
        pass
    try:
        if not os.environ.get("RULE_TUPLES_SILENT"):
            print(json.dumps(payload, indent=2))
    except Exception:
        pass
    # Best-effort forward to W&B run when available
    if logger:
        try:
            # Optional gate via env to disable
            gate = str(os.environ.get("UAIR_TOPIC_WANDB_LOG_EVENTS", "1")).strip().lower() in ("1","true","yes","on")
            if gate:
                # Route categorical/string fields to summary, numeric to metrics
                try:
                    logger.set_summary("topic/event", str(event))
                except Exception:
                    pass
                numeric_metrics: Dict[str, Any] = {}
                for k, v in (kwargs or {}).items():
                    try:
                        if isinstance(v, (int, float)):
                            numeric_metrics[f"topic/{k}"] = v
                        elif isinstance(v, str):
                            logger.set_summary(f"topic/{k}", v)
                        else:
                            # Best-effort: store JSON-serializable values in summary
                            try:
                                logger.set_summary(f"topic/{k}", v)
                            except Exception:
                                pass
                    except Exception:
                        pass
                if numeric_metrics:
                    logger.log_metrics(numeric_metrics)
        except Exception:
            pass

def _select_text(pdf: pd.DataFrame, text_pref: str) -> pd.Series:
    if text_pref == "chunk_text" and "chunk_text" in pdf.columns:
        return pdf["chunk_text"].fillna("").astype(str)
    if text_pref == "article_text" and "article_text" in pdf.columns:
        return pdf["article_text"].fillna("").astype(str)
    # auto: prefer full article_text when available, otherwise fall back to chunk_text
    if "article_text" in pdf.columns:
        return pdf["article_text"].fillna("").astype(str)
    if "chunk_text" in pdf.columns:
        return pdf["chunk_text"].fillna("").astype(str)
    return pd.Series([""] * len(pdf))

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
        r"(?:city|cities)",
        r"urban",
        r"climate",
        r"earth",
        r"environment",
        r"transport",
    ]
    return re.compile(r"(" + r"|".join(phrases) + r")", flags=re.IGNORECASE)

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

def _resolve_torch_dtype(val: Any):
    try:
        import torch  # type: ignore
    except Exception:
        return None
    if val is None:
        return None
    # Already a torch dtype
    try:
        if val in (torch.float16, torch.bfloat16, torch.float32):  # type: ignore[attr-defined]
            return val
    except Exception:
        pass
    # Map common string aliases
    try:
        s = str(val).strip().lower()
    except Exception:
        return None
    mapping = {
        "float16": getattr(__import__('torch'), 'float16', None),
        "fp16": getattr(__import__('torch'), 'float16', None),
        "half": getattr(__import__('torch'), 'float16', None),
        "bfloat16": getattr(__import__('torch'), 'bfloat16', None),
        "bf16": getattr(__import__('torch'), 'bfloat16', None),
        "float32": getattr(__import__('torch'), 'float32', None),
        "fp32": getattr(__import__('torch'), 'float32', None),
    }
    return mapping.get(s)

def _trim_texts(texts: List[str], max_tokens: int) -> List[str]:
    # Lightweight char-based clamp as embedding models handle truncation internally.
    if max_tokens is None or max_tokens <= 0:
        return texts
    approx_chars = int(max_tokens) * 4
    return [t if len(t) <= approx_chars else t[:approx_chars] for t in texts]

def _embed_batch(model_name: str, texts: List[str], device: str, normalize: bool, trust_remote_code: bool, matryoshka_dim: Any, max_seq_length: Any, batch_size: Any, devices: Any = None, torch_dtype: Any = None) -> List[List[float]]:
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
        import torch.nn.functional as F  # type: ignore
    except Exception as e:
        raise RuntimeError(f"sentence-transformers required for embeddings: {e}")
    dev = None if device == "auto" else device
    # Optional dtype (fp16/bf16) via model_kwargs
    mk = None
    if torch_dtype is not None:
        try:
            mk = {"torch_dtype": torch_dtype}
        except Exception:
            mk = None
    model = SentenceTransformer(model_name, device=dev, trust_remote_code=bool(trust_remote_code), model_kwargs=mk)
    # Apply task instruction prefix for clustering per model card best-practice
    prefixed = [f"clustering: {t}" for t in texts]
    # Optional: adjust model max sequence length (if provided)
    try:
        if max_seq_length:
            try:
                model.max_seq_length = int(max_seq_length)
            except Exception:
                pass
    except Exception:
        pass
    # Encode to tensor for matryoshka post-processing
    try:
        bs = int(batch_size) if batch_size is not None else 64
    except Exception:
        bs = 64
    # Multi-GPU via sentence-transformers encode(device=["cuda:0", ...])
    if devices is not None:
        try:
            dev_list = list(devices) if isinstance(devices, (list, tuple)) else None
        except Exception:
            dev_list = None
    else:
        dev_list = None
    if dev_list and len(dev_list) > 0:
        try:
            import numpy as np  # type: ignore
        except Exception as _e:
            np = None  # type: ignore
        # encode across multiple devices (returns numpy array)
        emb_np = model.encode(prefixed, batch_size=bs, convert_to_tensor=False, normalize_embeddings=False, show_progress_bar=True, device=dev_list)
        # Ensure np array
        try:
            import numpy as _np  # type: ignore
            arr = _np.asarray(emb_np, dtype=_np.float32)
            # Layer norm
            try:
                mean = arr.mean(axis=1, keepdims=True)
                var = arr.var(axis=1, keepdims=True)
                arr = (arr - mean) / _np.sqrt(var + 1e-5)
            except Exception:
                pass
            # Matryoshka slice
            try:
                if matryoshka_dim:
                    arr = arr[:, :int(matryoshka_dim)]
            except Exception:
                pass
            # L2 normalize
            if normalize:
                try:
                    denom = _np.linalg.norm(arr, axis=1, keepdims=True)
                    denom[denom == 0] = 1.0
                    arr = arr / denom
                except Exception:
                    pass
            return arr.tolist()
        except Exception:
            # Fallback to single-device path below
            pass
    # Single device path (torch tensor post-processing)
    emb = model.encode(prefixed, batch_size=bs, convert_to_tensor=True, normalize_embeddings=False, show_progress_bar=True)
    try:
        emb = F.layer_norm(emb, normalized_shape=(emb.shape[1],))
    except Exception:
        pass
    try:
        if matryoshka_dim:
            emb = emb[:, :int(matryoshka_dim)]
    except Exception:
        pass
    if normalize:
        try:
            emb = F.normalize(emb, p=2, dim=1)
        except Exception:
            pass
    return emb.detach().cpu().tolist()

def _reduce_umap(emb: List[List[float]], cfg) -> Tuple[List[List[float]], Any]:
    import numpy as np  # type: ignore
    # Guard tiny datasets and clamp neighbors to valid range
    n_samples = int(len(emb) if emb is not None else 0)
    if n_samples <= 1:
        return emb, None
    red = getattr(cfg.topic.reduce, "n_components", 15)
    nnei = getattr(cfg.topic.reduce, "n_neighbors", 15)
    mind = getattr(cfg.topic.reduce, "min_dist", 0.1)
    metric = getattr(cfg.topic.reduce, "metric", "cosine")
    try:
        nnei_eff = max(2, min(int(nnei), n_samples - 1))
    except Exception:
        nnei_eff = max(2, n_samples - 1)
    arr = np.array(emb, dtype="float32")
    # Prefer cuML UMAP when enabled
    use_rapids = False
    try:
        use_rapids = bool(getattr(getattr(getattr(cfg, "topic", object()), "gpu", object()), "use_rapids", False))
    except Exception:
        use_rapids = False
    if use_rapids:
        try:
            from cuml.manifold import UMAP as cuUMAP  # type: ignore
            reducer = cuUMAP(n_components=int(red), n_neighbors=int(nnei_eff), min_dist=float(mind), metric=str(metric), random_state=int(getattr(cfg.topic, "seed", 777)))
            out = reducer.fit_transform(arr)
            return out.tolist(), reducer
        except Exception:
            pass
    # CPU fallback (umap-learn)
    try:
        import umap  # type: ignore
        reducer = umap.UMAP(n_components=int(red), n_neighbors=int(nnei_eff), min_dist=float(mind), metric=str(metric), random_state=int(getattr(cfg.topic, "seed", 777)))
        out = reducer.fit_transform(arr)
        return out.tolist(), reducer
    except Exception as e:
        raise RuntimeError(f"UMAP not available: {e}")

def _cluster_hdbscan(emb_red: List[List[float]], cfg) -> Tuple[List[int], List[float], Any]:
    import numpy as np  # type: ignore
    mcs = getattr(cfg.topic.hdbscan, "min_cluster_size", 30)
    ms = getattr(cfg.topic.hdbscan, "min_samples", None)
    metric_cfg = str(getattr(cfg.topic.hdbscan, "metric", "euclidean"))
    eps = getattr(cfg.topic.hdbscan, "cluster_selection_epsilon", 0.0)
    arr = np.array(emb_red, dtype="float32")
    # Prefer cuML HDBSCAN when enabled (GPU supports only euclidean and no callable/precomputed)
    use_rapids = False
    try:
        use_rapids = bool(getattr(getattr(getattr(cfg, "topic", object()), "gpu", object()), "use_rapids", False))
    except Exception:
        use_rapids = False
    if use_rapids:
        try:
            from cuml.cluster import HDBSCAN as cuHDBSCAN  # type: ignore
            metric = "euclidean"  # GPU limitation
            clusterer = cuHDBSCAN(min_cluster_size=int(mcs), min_samples=(None if ms in (None, "null") else int(ms)), metric=metric, cluster_selection_epsilon=float(eps), prediction_data=True)
            clusterer.fit(arr)
            labels = clusterer.labels_.tolist()
            # Soft membership (optional)
            probs_list = [1.0] * len(labels)
            try:
                from cuml.cluster.hdbscan import all_points_membership_vectors  # type: ignore
                mv = all_points_membership_vectors(clusterer)
                # Ensure host list
                try:
                    import cupy as cp  # type: ignore
                    probs_arr = mv.max(axis=1)
                    probs_list = cp.asnumpy(probs_arr).tolist()
                except Exception:
                    try:
                        probs_list = mv.max(axis=1).get().tolist()  # type: ignore[attr-defined]
                    except Exception:
                        probs_list = [1.0] * len(labels)
            except Exception:
                pass
            return labels, probs_list, clusterer
        except Exception:
            pass
    # CPU fallback
    try:
        import hdbscan  # type: ignore
        clusterer = hdbscan.HDBSCAN(min_cluster_size=int(mcs), min_samples=(None if ms in (None, "null") else int(ms)), metric=metric_cfg, cluster_selection_epsilon=float(eps))
        labels = clusterer.fit_predict(arr)
        probs = getattr(clusterer, "probabilities_", None)
        probs_list = probs.tolist() if probs is not None else [1.0] * len(labels)
        return labels.tolist(), probs_list, clusterer
    except Exception as e:
        raise RuntimeError(f"hdbscan not available: {e}")

def run_topic_stage(df, cfg, logger=None):
    # Load into pandas if Ray Dataset
    if hasattr(df, "to_pandas") and hasattr(df, "count") and _RAY_OK:
        pdf = df.to_pandas()
    else:
        pdf = df.copy() if isinstance(df, pd.DataFrame) else pd.DataFrame([])
    if pdf is None or len(pdf) == 0:
        _log_event("input_empty", logger=logger)
        return pd.DataFrame([])

    pdf = _ensure_ids(pdf)
    # Gate to relevant rows when present
    try:
        if "is_relevant" in pdf.columns:
            pdf = pdf[pdf["is_relevant"].astype(bool) == True]
    except Exception:
        pass
    if len(pdf) == 0:
        _log_event("no_rows_after_gate", logger=logger)
        return pd.DataFrame([])

    # Environment preflight logging (CUDA / RAPIDS visibility)
    try:
        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    except Exception:
        cuda_visible = None
    try:
        rapids_requested = bool(getattr(getattr(getattr(cfg, "topic", object()), "gpu", object()), "use_rapids", False))
    except Exception:
        rapids_requested = False
    try:
        torch_info: Dict[str, Any] = {}
        try:
            import torch  # type: ignore
            cuda_ok = bool(torch.cuda.is_available())
            dev_count = int(torch.cuda.device_count()) if cuda_ok else 0
            dev_names: List[str] = []
            if cuda_ok:
                for i in range(dev_count):
                    try:
                        dev_names.append(str(torch.cuda.get_device_name(i)))
                    except Exception:
                        dev_names.append("gpu")
            torch_info = {"cuda_available": cuda_ok, "device_count": dev_count, "device_names": dev_names}
        except Exception:
            torch_info = {"cuda_available": False}
        # RAPIDS availability
        has_cuml = False
        try:
            import cuml  # type: ignore
            has_cuml = True
        except Exception:
            has_cuml = False
        # Selected embed device from config
        try:
            sel_dev = str(getattr(cfg.topic.embed, "device", "auto") or "auto")
        except Exception:
            sel_dev = "auto"
        # Compute effective embed device for logging (auto-promote to CUDA unless disabled)
        eff_dev_pre = sel_dev
        try:
            disable_auto = str(os.environ.get("UAIR_TOPIC_DISABLE_AUTO_PROMOTE", "")).strip().lower() in ("1","true","yes","on")
            sel_lower = str(sel_dev).lower()
            if not disable_auto and sel_lower in ("auto", "cpu"):
                eff_dev_pre = "cuda" if torch_info.get("cuda_available") else "cpu"
        except Exception:
            pass
        _log_event(
            "preflight",
            cuda_visible_devices=cuda_visible,
            torch_cuda_available=torch_info.get("cuda_available"),
            torch_device_count=torch_info.get("device_count"),
            torch_device_names=torch_info.get("device_names"),
            selected_embed_device=sel_dev,
            effective_embed_device=eff_dev_pre,
            rapids_requested=rapids_requested,
            cuml_available=has_cuml,
        )
    except Exception:
        pass

    # Decide unit of clustering
    cluster_on = str(getattr(cfg.topic, "cluster_on", "article") or "article").strip().lower()
    text_pref = str(getattr(cfg.topic, "text_column", "auto") or "auto").strip().lower()
    max_tok = int(getattr(cfg.topic, "max_tokens_for_embed", 3072) or 3072)

    # If embeddings should use chunk_text, synthesize keyword-window buffers when missing
    if text_pref == "chunk_text" and "chunk_text" not in pdf.columns:
        try:
            enable_kw_buf = bool(getattr(getattr(cfg, "runtime", object()), "keyword_buffering", True))
        except Exception:
            enable_kw_buf = True
        try:
            window_words = int(getattr(getattr(cfg, "runtime", object()), "keyword_window_words", 100) or 100)
        except Exception:
            window_words = 100
        kw_regex = _build_relevant_regex() if enable_kw_buf else None

        def _mk_chunk_text(t: Any) -> str:
            s = "" if t is None else str(t)
            if not kw_regex:
                return s
            try:
                blocks = _generate_relevant_blocks(s, kw_regex, window_words)
            except Exception:
                blocks = []
            return ("\n\n".join(blocks)) if blocks else s

        try:
            base = pdf.get("article_text", pd.Series([""] * len(pdf))).fillna("").astype(str)
        except Exception:
            base = pd.Series([""] * len(pdf))
        try:
            pdf["chunk_text"] = base.apply(_mk_chunk_text)
        except Exception:
            # Fallback: copy base text when apply fails
            pdf["chunk_text"] = base

    if cluster_on == "chunk" and "chunk_id" in pdf.columns:
        df_units = pdf[["article_id", "chunk_id"]].copy()
        df_units["unit_id"] = df_units.apply(lambda r: f"{r['article_id']}__{r['chunk_id']}", axis=1)
        texts = _select_text(pdf, text_pref)
        units = pd.DataFrame({"unit_id": df_units["unit_id"], "text": texts})
    else:
        # collapse to article_id → first/concat text
        texts = _select_text(pdf, text_pref)
        tmp = pd.DataFrame({"article_id": pdf["article_id"], "text": texts})
        units = tmp.groupby("article_id", dropna=False)["text"].apply(lambda s: "\n\n".join([t for t in s.tolist() if isinstance(t, str) and t.strip()])).reset_index()
        units = units.rename(columns={"article_id": "unit_id"})

    # Optional sampling to bound memory footprint (prefer runtime.sample_n for consistency across stages)
    try:
        sample_cap = getattr(getattr(cfg, "runtime", object()), "sample_n", None)
    except Exception:
        sample_cap = None
    if sample_cap is None:
        try:
            sample_cap = getattr(getattr(cfg, "topic", object()), "sample_max_units", None)
        except Exception:
            sample_cap = None
    try:
        if sample_cap is not None:
            cap = int(sample_cap)
            if cap > 0 and len(units) > cap:
                try:
                    seed = int(getattr(getattr(cfg, "topic", object()), "seed", 777))
                except Exception:
                    seed = 777
                units = units.sample(n=cap, random_state=seed).reset_index(drop=True)
    except Exception:
        pass

    # Progress: units prepared
    _log_event("units_prepared", logger=logger, units=int(len(units)))

    texts_list = [str(t or "") for t in units["text"].tolist()]
    texts_list = _trim_texts(texts_list, max_tok)

    # Embeddings
    model_name = str(getattr(cfg.topic.embed, "model_source", "nomic-ai/nomic-embed-text-v1.5"))
    dev_cfg = getattr(cfg.topic.embed, "device", "auto")
    # Optional precision override (e.g., fp16, bfloat16) – pass through to SentenceTransformer via model_kwargs
    try:
        torch_dtype_cfg_raw = getattr(cfg.topic.embed, "torch_dtype", None)
    except Exception:
        torch_dtype_cfg_raw = None
    torch_dtype_cfg = _resolve_torch_dtype(torch_dtype_cfg_raw)
    devices_list = None
    # Accept OmegaConf ListConfig or native list/tuple
    try:
        if (_OmegaListConfig is not None and isinstance(dev_cfg, _OmegaListConfig)) or isinstance(dev_cfg, (list, tuple)):
            devices_list = [str(d) for d in list(dev_cfg)]
    except Exception:
        devices_list = None
    # Also accept a string that looks like a list, e.g. "[cuda:0,cuda:1]" or "['cuda:0','cuda:1']" or "cuda:0,cuda:1"
    if devices_list is None and isinstance(dev_cfg, str):
        s = dev_cfg.strip()
        try:
            if s.startswith("[") and s.endswith("]"):
                s = s[1:-1]
            s = s.replace("'", "").replace('"', "")
            if "," in s:
                parts = [p.strip() for p in s.split(",") if p.strip()]
                if parts:
                    devices_list = parts
        except Exception:
            devices_list = None
    device = (str(dev_cfg) if not ((_OmegaListConfig is not None and isinstance(dev_cfg, _OmegaListConfig)) or isinstance(dev_cfg, (list, tuple))) else "auto")
    # If we detected a devices list (from OmegaConf or string), avoid passing that as the model device
    if devices_list is not None:
        device = "auto"
    # Resolve effective single-device for logging/model load; multi-device handled in encode calls
    eff_device = device
    try:
        import torch  # type: ignore
        disable_auto = str(os.environ.get("UAIR_TOPIC_DISABLE_AUTO_PROMOTE", "")).strip().lower() in ("1","true","yes","on")
        if not disable_auto and device in ("auto", "cpu"):
            eff_device = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        pass
    normalize = bool(getattr(cfg.topic.embed, "normalize", True))
    trust_rc = bool(getattr(cfg.topic.embed, "trust_remote_code", True))
    mat_dim = getattr(cfg.topic.embed, "matryoshka_dim", None)
    max_len = getattr(cfg.topic.embed, "max_seq_length", None)
    try:
        bs_cfg = getattr(cfg.topic.embed, "batch_size", None)
    except Exception:
        bs_cfg = None
    # Embedding start log
    try:
        bs_log = int(getattr(cfg.topic.embed, "batch_size", 64) or 64)
    except Exception:
        bs_log = 64
    _log_event(
        "embedding_start",
        model=model_name,
        units=int(len(texts_list)),
        batch_size=bs_log,
        device=(devices_list if devices_list else eff_device),
        matryoshka_dim=(int(mat_dim) if isinstance(mat_dim, int) or (isinstance(mat_dim, str) and str(mat_dim).isdigit()) else None),
        normalize=bool(normalize),
    )
    # Also log an explicit W&B metric counter for progress tracking dashboarding
    if logger:
        try:
            logger.log_metrics({"topic/embed/total_units": int(len(texts_list))})
        except Exception:
            pass
    # Optional progress bar via env flag
    try:
        show_pb = str(os.environ.get("UAIR_TOPIC_SHOW_PB", "")).strip().lower() in ("1","true","yes","on")
    except Exception:
        show_pb = False
    # Timed embedding with incremental progress logs
    start_embed = time.perf_counter()
    try:
        if show_pb:
            from sentence_transformers import SentenceTransformer  # type: ignore
            # Optional dtype passed via model_kwargs
            mk = None
            if torch_dtype_cfg is not None:
                try:
                    mk = {"torch_dtype": torch_dtype_cfg}
                except Exception:
                    mk = None
            m = SentenceTransformer(model_name, device=eff_device, trust_remote_code=bool(trust_rc), model_kwargs=mk)
            import torch.nn.functional as F  # type: ignore
            # mirror _embed_batch with progress
            prefixed = [f"clustering: {t}" for t in texts_list]
            bs = int(bs_cfg) if bs_cfg is not None else 64
            total = len(prefixed)
            total_batches = (total + bs - 1) // bs
            try:
                log_every_env = os.environ.get("UAIR_TOPIC_LOG_EVERY", "")
                log_every_batches = int(log_every_env) if log_every_env else max(1, total_batches // 10)
            except Exception:
                log_every_batches = max(1, total_batches // 10)
            emb_list: List[Any] = []
            for i in range(0, total, bs):
                batch_idx = i // bs
                chunk = prefixed[i:i+bs]
                # If multi-device is requested, let encode distribute across devices
                if devices_list:
                    e_np = m.encode(chunk, batch_size=bs, show_progress_bar=True, convert_to_tensor=False, normalize_embeddings=False, device=devices_list)
                    try:
                        import numpy as _np  # type: ignore
                        arr = _np.asarray(e_np, dtype=_np.float32)
                        # layer norm
                        try:
                            mean = arr.mean(axis=1, keepdims=True)
                            var = arr.var(axis=1, keepdims=True)
                            arr = (arr - mean) / _np.sqrt(var + 1e-5)
                        except Exception:
                            pass
                        if mat_dim:
                            try:
                                arr = arr[:, :int(mat_dim)]
                            except Exception:
                                pass
                        if normalize:
                            try:
                                denom = _np.linalg.norm(arr, axis=1, keepdims=True)
                                denom[denom == 0] = 1.0
                                arr = arr / denom
                            except Exception:
                                pass
                        # Convert back to torch tensor slice for concatenation
                        import torch as _torch  # type: ignore
                        e = _torch.from_numpy(arr)
                    except Exception:
                        # fallback to single-device encode
                        e = m.encode(chunk, batch_size=bs, show_progress_bar=True, convert_to_tensor=True, normalize_embeddings=False)
                else:
                    e = m.encode(chunk, batch_size=bs, show_progress_bar=True, convert_to_tensor=True, normalize_embeddings=False)
                try:
                    e = F.layer_norm(e, normalized_shape=(e.shape[1],))
                except Exception:
                    pass
                if mat_dim:
                    try:
                        e = e[:, :int(mat_dim)]
                    except Exception:
                        pass
                if normalize:
                    try:
                        e = F.normalize(e, p=2, dim=1)
                    except Exception:
                        pass
                emb_list.append(e.detach().cpu())
                # progress logging
                done = min(i + bs, total)
                if (batch_idx % log_every_batches) == 0 or done == total:
                    elapsed = time.perf_counter() - start_embed
                    rate = (done / elapsed) if elapsed > 0 else None
                    eta = ((total - done) / rate) if (rate and rate > 0) else None
                    _log_event(
                        "embedding_progress",
                        done=int(done),
                        total=int(total),
                        pct=round(100.0 * done / total, 2) if total else 100.0,
                        elapsed_s=round(elapsed, 2),
                        rate_items_per_s=(round(rate, 2) if rate else None),
                        eta_s=(round(eta, 2) if eta else None),
                    )
                    if logger:
                        try:
                            logger.log_metrics({
                                "topic/embed/done_units": int(done),
                                "topic/embed/elapsed_s": round(elapsed, 2),
                                "topic/embed/rate_units_per_s": (round(rate, 2) if rate else None),
                                "topic/embed/pct": (round(100.0 * done / total, 2) if total else 100.0),
                            })
                        except Exception:
                            pass
            import torch  # type: ignore
            emb = torch.cat(emb_list, dim=0).tolist()
        else:
            emb = _embed_batch(model_name, texts_list, eff_device, normalize, trust_rc, mat_dim, max_len, bs_cfg, devices=devices_list, torch_dtype=torch_dtype_cfg)
    except Exception:
        emb = _embed_batch(model_name, texts_list, eff_device, normalize, trust_rc, mat_dim, max_len, bs_cfg, devices=devices_list, torch_dtype=torch_dtype_cfg)
    elapsed_embed = time.perf_counter() - start_embed
    _log_event("embedding_done", logger=logger, dim=(len(emb[0]) if emb else None), elapsed_s=round(elapsed_embed, 2))
    if logger:
        try:
            logger.log_metrics({"topic/embed/elapsed_s": round(elapsed_embed, 2)})
        except Exception:
            pass

    # Reduction
    method = str(getattr(cfg.topic.reduce, "method", "umap") or "umap")
    # Reduction start
    try:
        _log_event(
            "reduction_start",
            method=method,
            n_components=int(getattr(cfg.topic.reduce, "n_components", 15) or 15),
            n_neighbors=int(getattr(cfg.topic.reduce, "n_neighbors", 15) or 15),
            min_dist=float(getattr(cfg.topic.reduce, "min_dist", 0.1) or 0.1),
            metric=str(getattr(cfg.topic.reduce, "metric", "cosine") or "cosine"),
        )
    except Exception:
        pass
    n_units = len(texts_list)
    if n_units <= 1:
        emb_red, reducer = emb, None
    elif method == "umap":
        try:
            start_red = time.perf_counter()
            emb_red, reducer = _reduce_umap(emb, cfg)
            elapsed_red = time.perf_counter() - start_red
            backend_name = None
            try:
                backend_name = getattr(type(reducer), "__module__", None)
            except Exception:
                backend_name = None
            _log_event(
                "reduction_done",
                method=method,
                backend=backend_name,
                out_dim=(len(emb_red[0]) if emb_red else None),
                elapsed_s=round(elapsed_red, 2),
            )
            if logger:
                try:
                    logger.log_metrics({
                        "topic/reduction/elapsed_s": round(elapsed_red, 2),
                    })
                    try:
                        logger.set_config({"topic": {"reduction": {"method": str(method)}}})
                    except Exception:
                        logger.set_summary("topic/reduction/method", str(method))
                except Exception:
                    pass
        except Exception:
            # Fallback to PCA, then identity
            try:
                import numpy as np  # type: ignore
                from sklearn.decomposition import PCA  # type: ignore
                comps = int(getattr(cfg.topic.reduce, "n_components", 15) or 15)
                arr = np.array(emb, dtype="float32")
                start_red = time.perf_counter()
                emb_red = PCA(n_components=min(comps, max(1, n_units - 1)), random_state=int(getattr(cfg.topic, "seed", 777))).fit_transform(arr).tolist()
                elapsed_red = time.perf_counter() - start_red
                reducer = None
                _log_event("reduction_done", logger=logger, method="pca", backend="sklearn", out_dim=(len(emb_red[0]) if emb_red else None), elapsed_s=round(elapsed_red, 2))
            except Exception:
                emb_red, reducer = emb, None
                _log_event("reduction_done", logger=logger, method="identity", backend=None, out_dim=(len(emb_red[0]) if emb_red else None))
    elif method == "pca":
        try:
            import numpy as np  # type: ignore
            from sklearn.decomposition import PCA  # type: ignore
            comps = int(getattr(cfg.topic.reduce, "n_components", 15) or 15)
            arr = np.array(emb, dtype="float32")
            start_red = time.perf_counter()
            emb_red = PCA(n_components=min(comps, max(1, n_units - 1)), random_state=int(getattr(cfg.topic, "seed", 777))).fit_transform(arr).tolist()
            elapsed_red = time.perf_counter() - start_red
            reducer = None
            _log_event("reduction_done", method="pca", backend="sklearn", out_dim=(len(emb_red[0]) if emb_red else None), elapsed_s=round(elapsed_red, 2))
            if logger:
                try:
                    logger.log_metrics({
                        "topic/reduction/elapsed_s": round(elapsed_red, 2),
                    })
                    try:
                        logger.set_config({"topic": {"reduction": {"method": "pca"}}})
                    except Exception:
                        logger.set_summary("topic/reduction/method", "pca")
                except Exception:
                    pass
        except Exception:
            emb_red, reducer = emb, None
            _log_event("reduction_done", logger=logger, method="identity", backend=None, out_dim=(len(emb_red[0]) if emb_red else None))
            if logger:
                try:
                    try:
                        logger.set_config({"topic": {"reduction": {"method": "identity"}}})
                    except Exception:
                        logger.set_summary("topic/reduction/method", "identity")
                except Exception:
                    pass
    else:
        emb_red, reducer = emb, None
        _log_event("reduction_done", logger=logger, method="identity", backend=None, out_dim=(len(emb_red[0]) if emb_red else None))

    # Clustering
    # Log start with parameters
    try:
        _log_event(
            "clustering_start",
            min_cluster_size=int(getattr(cfg.topic.hdbscan, "min_cluster_size", 30) or 30),
            min_samples=(int(getattr(cfg.topic.hdbscan, "min_samples", 0)) if getattr(cfg.topic.hdbscan, "min_samples", None) not in (None, "null") else None),
            metric=str(getattr(cfg.topic.hdbscan, "metric", "euclidean") or "euclidean"),
            cluster_selection_epsilon=float(getattr(cfg.topic.hdbscan, "cluster_selection_epsilon", 0.0) or 0.0),
        )
    except Exception:
        pass
    try:
        start_clu = time.perf_counter()
        labels, probs, clusterer = _cluster_hdbscan(emb_red, cfg)
        elapsed_clu = time.perf_counter() - start_clu
    except Exception:
        # Fallback: single cluster assignment
        labels = [0 for _ in range(len(emb_red))]
        probs = [1.0 for _ in range(len(emb_red))]
        clusterer = None
        elapsed_clu = None
    try:
        import collections as _c
        ctr = dict(_c.Counter([int(x) for x in labels])) if labels else {}
        total_pts = int(len(labels))
        noise = int(ctr.get(-1, 0))
        non_noise = total_pts - noise
        num_clusters = max(0, len([k for k in ctr.keys() if int(k) != -1]))
        noise_ratio = (noise / total_pts) if total_pts > 0 else None
        _log_event(
            "clustering_done",
            clusters=ctr,
            num_clusters=int(num_clusters),
            noise=int(noise),
            total=int(total_pts),
            noise_ratio=(round(noise_ratio, 4) if noise_ratio is not None else None),
            elapsed_s=(round(elapsed_clu, 2) if elapsed_clu is not None else None),
        )
        if logger:
            try:
                logger.log_metrics({
                    "topic/clustering/num_clusters": int(num_clusters),
                    "topic/clustering/noise": int(noise),
                    "topic/clustering/total": int(total_pts),
                    "topic/clustering/noise_ratio": (round(noise_ratio, 4) if noise_ratio is not None else None),
                    "topic/clustering/elapsed_s": (round(elapsed_clu, 2) if elapsed_clu is not None else None),
                })
            except Exception:
                pass
    except Exception:
        pass
    # Prepare 2D coordinates for plotting (use first two reduced dims, else PCA to 2D)
    try:
        import numpy as _np  # type: ignore
        arr_red = _np.array(emb_red, dtype="float32") if emb_red is not None else None
        if arr_red is not None and arr_red.shape[1] >= 2:
            plot_xy = arr_red[:, :2]
        else:
            arr0 = _np.array(emb, dtype="float32") if emb is not None else None
            if arr0 is not None and arr0.shape[1] >= 2:
                try:
                    from sklearn.decomposition import PCA as _PCA  # type: ignore
                    plot_xy = _PCA(n_components=2, random_state=int(getattr(cfg.topic, "seed", 777))).fit_transform(arr0)
                except Exception:
                    plot_xy = arr0[:, :2]
            else:
                plot_xy = None
    except Exception:
        plot_xy = None

    out = units.copy()
    out["topic_id"] = labels
    out["topic_prob"] = probs
    try:
        if plot_xy is not None:
            out["plot_x"] = [float(x) for x in (plot_xy[:, 0].tolist())]
            out["plot_y"] = [float(y) for y in (plot_xy[:, 1].tolist())]
    except Exception:
        pass

    # Summaries (top terms via TF-IDF) with RAPIDS/cuML when available; fallback to scikit-learn
    try:
        # Common config
        min_df_v = 1 if len(texts_list) < 50 else 2
        try:
            stop_words_cfg = getattr(getattr(cfg, "topic", object()), "tfidf_stop_words", "english")
        except Exception:
            stop_words_cfg = "english"
        try:
            max_df_cfg = getattr(getattr(cfg, "topic", object()), "tfidf_max_df", 0.95)
        except Exception:
            max_df_cfg = 0.95
        try:
            ngram_max = int(getattr(getattr(cfg, "topic", object()), "tfidf_ngram_max", 3))
        except Exception:
            ngram_max = 3
        _log_event(
            "tfidf_start",
            max_features=20000,
            min_df=min_df_v,
            max_df=float(max_df_cfg),
            stop_words=str(stop_words_cfg),
            ngram_max=int(ngram_max),
            docs=int(len(texts_list)),
        )
        tfidf_start = time.perf_counter()
        used_gpu = False
        terms = None
        X = None
        # Try cuML TF-IDF when RAPIDS enabled
        try:
            use_rapids = bool(getattr(getattr(getattr(cfg, "topic", object()), "gpu", object()), "use_rapids", False))
        except Exception:
            use_rapids = False
        if use_rapids:
            try:
                import cudf as _cudf  # type: ignore
                import cupy as _cp  # type: ignore
                from cuml.feature_extraction.text import TfidfVectorizer as _cuTfidf  # type: ignore
                # Sanitize inputs for cuDF: ensure strings, no None
                try:
                    texts_gpu = ["" if (t is None) else str(t) for t in texts_list]
                except Exception:
                    texts_gpu = texts_list
                # Use cuDF Series input so cuML can apply .str ops internally
                gser = _cudf.Series(texts_gpu)
                # Map stop_words to supported form
                _sw = stop_words_cfg
                try:
                    if isinstance(_sw, str):
                        s = _sw.strip().lower()
                        if s in ("english", "en"):
                            try:
                                from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as _EN  # type: ignore
                                _sw = list(_EN)
                            except Exception:
                                _sw = None
                except Exception:
                    _sw = None
                cu_vect = _cuTfidf(
                    max_features=20000,
                    ngram_range=(1, int(max(1, ngram_max))),
                    min_df=min_df_v,
                    max_df=max_df_cfg,
                    stop_words=_sw,
                    lowercase=True,
                )
                Xg = cu_vect.fit_transform(gser)
                try:
                    terms_gpu = cu_vect.get_feature_names_out()
                except Exception:
                    terms_gpu = cu_vect.get_feature_names()
                # Safely convert feature names to host list without iterating the cuDF Series directly
                try:
                    import cudf as __cudf  # type: ignore
                    if isinstance(terms_gpu, __cudf.Series):
                        try:
                            terms = terms_gpu.to_pandas().astype(str).tolist()
                        except Exception:
                            terms = terms_gpu.to_arrow().to_pylist()
                    else:
                        # Already list-like or numpy/cupy host array
                        try:
                            terms = list(terms_gpu)
                        except Exception:
                            terms = [str(t) for t in terms_gpu]
                except Exception:
                    try:
                        terms = list(terms_gpu)
                    except Exception:
                        terms = []
                # Per-article keywords
                try:
                    try:
                        doc_k = int(getattr(getattr(cfg, "topic", object()), "doc_terms_k", 10))
                    except Exception:
                        doc_k = 10
                    article_kw: List[List[str]] = []
                    X_csr = Xg.tocsr()
                    total_docs = int(X_csr.shape[0])
                    log_every_docs = max(1, total_docs // 10)
                    doc_kw_start = time.perf_counter()
                    for i in range(total_docs):
                        row = X_csr.getrow(i)
                        if row.nnz == 0:
                            article_kw.append([])
                        else:
                            idxs = row.indices
                            vals = row.data
                            # bring small row arrays to host for top-k
                            try:
                                import cupy as __cp  # type: ignore
                                vals_np = __cp.asnumpy(vals)
                                idxs_np = __cp.asnumpy(idxs)
                            except Exception:
                                vals_np = vals
                                idxs_np = idxs
                            k = int(min(doc_k, len(vals_np))) if len(vals_np) > 0 else 0
                            if k > 0:
                                order = vals_np.argsort()[-k:][::-1]
                                top_idx = [int(idxs_np[j]) for j in order]
                                kws = [str(terms[j]) for j in top_idx]
                                article_kw.append(kws)
                            else:
                                article_kw.append([])
                        if (i % log_every_docs) == 0 or (i + 1) == total_docs:
                            elapsed = time.perf_counter() - doc_kw_start
                            _log_event(
                                "tfidf_doc_keywords_progress",
                                done=int(i + 1),
                                total=int(total_docs),
                                pct=round(100.0 * (i + 1) / total_docs, 2) if total_docs else 100.0,
                                elapsed_s=round(elapsed, 2),
                            )
                    try:
                        out["article_keywords"] = article_kw
                    except Exception:
                        pass
                except Exception:
                    pass
                # Cluster-level top terms on GPU
                top_terms: Dict[int, Any] = {}
                for lab in sorted(set(labels)):
                    if lab == -1:
                        continue
                    idx = [i for i, y in enumerate(labels) if y == lab]
                    if not idx:
                        continue
                    try:
                        sub = Xg[idx]
                        arr = sub.sum(axis=0)
                        try:
                            import cupy as __cp  # type: ignore
                            arr_dense = __cp.asnumpy(arr.toarray()).ravel()
                        except Exception:
                            arr_dense = arr.A.ravel()
                        top_idx = arr_dense.argsort()[-10:][::-1]
                        top_terms[int(lab)] = [str(terms[j]) for j in top_idx]
                    except Exception:
                        top_terms[int(lab)] = None
                out["topic_top_terms"] = out["topic_id"].apply(lambda l: top_terms.get(int(l)) if int(l) in top_terms else None)
                used_gpu = True
                tfidf_elapsed = time.perf_counter() - tfidf_start
                _log_event("tfidf_done", logger=logger, elapsed_s=round(tfidf_elapsed, 2))
                if logger:
                    try:
                        logger.log_metrics({
                            "topic/tfidf/elapsed_s": round(tfidf_elapsed, 2),
                            "topic/tfidf/vocab_size": int(len(terms) if terms is not None else 0),
                        })
                        # Backend is categorical; prefer config/summary over scalar metric
                        try:
                            logger.set_config({"topic": {"tfidf": {"backend": "cuml"}}})
                        except Exception:
                            logger.set_summary("topic/tfidf/backend", "cuml")
                    except Exception:
                        pass
            except Exception as _e:
                used_gpu = False
                try:
                    _log_event("tfidf_gpu_skip", logger=logger, reason=str(_e)[:200])
                except Exception:
                    pass
        if not used_gpu:
            # CPU fallback (scikit-learn)
            from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
            import numpy as np  # type: ignore
            vect = TfidfVectorizer(max_features=20000, ngram_range=(1,int(max(1, ngram_max))), min_df=min_df_v, max_df=max_df_cfg, stop_words=stop_words_cfg)
            X = vect.fit_transform(texts_list)
            terms = np.array(vect.get_feature_names_out())
            top_terms: Dict[int, Any] = {}
            try:
                try:
                    doc_k = int(getattr(getattr(cfg, "topic", object()), "doc_terms_k", 10))
                except Exception:
                    doc_k = 10
                article_kw: List[List[str]] = []
                X_csr = X.tocsr()
                total_docs = X_csr.shape[0]
                log_every_docs = max(1, total_docs // 10)
                doc_kw_start = time.perf_counter()
                for i in range(total_docs):
                    row = X_csr.getrow(i)
                    if row.nnz == 0:
                        article_kw.append([])
                    else:
                        idxs = row.indices
                        vals = row.data
                        k = int(min(doc_k, len(vals))) if len(vals) > 0 else 0
                        if k > 0:
                            order = vals.argsort()[-k:][::-1]
                            top_idx = [int(idxs[j]) for j in order]
                            kws = [str(terms[j]) for j in top_idx]
                            article_kw.append(kws)
                        else:
                            article_kw.append([])
                    if (i % log_every_docs) == 0 or (i + 1) == total_docs:
                        elapsed = time.perf_counter() - doc_kw_start
                        _log_event(
                            "tfidf_doc_keywords_progress",
                            done=int(i + 1),
                            total=int(total_docs),
                            pct=round(100.0 * (i + 1) / total_docs, 2) if total_docs else 100.0,
                            elapsed_s=round(elapsed, 2),
                        )
                try:
                    out["article_keywords"] = article_kw
                except Exception:
                    pass
            except Exception:
                pass
            for lab in sorted(set(labels)):
                if lab == -1:
                    continue
                idx = [i for i, y in enumerate(labels) if y == lab]
                if not idx:
                    continue
                try:
                    import scipy.sparse as sp  # type: ignore
                    cls_vec = X[idx].sum(axis=0)
                    arr = np.asarray(cls_vec).ravel()
                except Exception:
                    sub = X[idx].mean(axis=0)
                    arr = np.asarray(sub).ravel()
                try:
                    top_idx = arr.argsort()[-10:][::-1]
                    top_terms[int(lab)] = terms[top_idx].tolist()
                except Exception:
                    top_terms[int(lab)] = None
            out["topic_top_terms"] = out["topic_id"].apply(lambda l: top_terms.get(int(l)) if int(l) in top_terms else None)
            tfidf_elapsed = time.perf_counter() - tfidf_start
            _log_event("tfidf_done", elapsed_s=round(tfidf_elapsed, 2))
            if logger:
                try:
                    logger.log_metrics({
                        "topic/tfidf/elapsed_s": round(tfidf_elapsed, 2),
                        "topic/tfidf/vocab_size": int(len(terms) if terms is not None else 0),
                    })
                    try:
                        logger.set_config({"topic": {"tfidf": {"backend": "sklearn"}}})
                    except Exception:
                        logger.set_summary("topic/tfidf/backend", "sklearn")
                except Exception:
                    pass
    except Exception:
        pass

    # Enrich outputs with basic metadata for downstream usability
    try:
        if cluster_on == "chunk" and "chunk_id" in pdf.columns:
            meta = pdf[["article_id", "chunk_id", "article_path", "country", "year"]].copy()
            meta["unit_id"] = meta.apply(lambda r: f"{r['article_id']}__{r['chunk_id']}", axis=1)
            meta = meta.drop_duplicates(subset=["unit_id"])
            out = out.merge(meta[["unit_id", "article_id", "article_path", "country", "year"]], on="unit_id", how="left")
        else:
            if "article_id" in pdf.columns:
                meta = pdf[["article_id", "article_path", "country", "year"]].drop_duplicates(subset=["article_id"])
                out = out.merge(meta, left_on="unit_id", right_on="article_id", how="left")
            else:
                out["article_id"] = out["unit_id"]
    except Exception:
        try:
            if "article_id" not in out.columns:
                out["article_id"] = out["unit_id"].apply(lambda s: str(s).split("__", 1)[0] if isinstance(s, str) else None)
        except Exception:
            pass

    return out


