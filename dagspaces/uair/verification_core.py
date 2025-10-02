import re
import os
import json
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import torch  # type: ignore
    from torch.nn import functional as F  # type: ignore
except Exception:  # pragma: no cover - torch is required at runtime
    torch = None  # type: ignore
    F = None  # type: ignore

try:
    from transformers import (  # type: ignore
        AutoTokenizer,
        AutoModel,
        AutoModelForSequenceClassification,
    )
except Exception:  # pragma: no cover - transformers is required at runtime
    AutoTokenizer = None  # type: ignore
    AutoModel = None  # type: ignore
    AutoModelForSequenceClassification = None  # type: ignore


# ------------------------------
# Global caches (initialized via init_verification)
# ------------------------------
_EMBED_TOKENIZER = None
_EMBED_MODEL = None
_NLI_TOKENIZER = None
_NLI_MODEL = None
_DEVICE: Optional[str] = None
_CLAIM_MAP: Dict[str, str] = {}
_DEBUG: bool = False


@dataclass
class VerificationConfig:
    method: str = "combo"  # one of: off|embed|nli|combo|combo_judge
    top_k: int = 3
    sim_threshold: float = 0.55
    entail_threshold: float = 0.85
    contra_max: float = 0.05
    embed_model_name: str = "intfloat/multilingual-e5-base"
    nli_model_name: str = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
    device: Optional[str] = None


_CONFIG = VerificationConfig()


def _ensure_imports():
    if torch is None or AutoTokenizer is None or AutoModel is None:
        raise RuntimeError("verification requires torch and transformers to be installed")


def _detect_device(explicit: Optional[str]) -> str:
    if explicit:
        return explicit
    try:
        import torch as _t  # type: ignore
        return "cuda" if _t.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def parse_thresholds_string(s: str) -> Tuple[float, float, float]:
    """Parse a thresholds string like 'sim=0.55,ent=0.85,contra=0.05'."""
    sim = 0.55
    ent = 0.85
    contra = 0.05
    try:
        parts = [p.strip() for p in str(s).split(",") if p.strip()]
        for p in parts:
            if "=" in p:
                k, v = p.split("=", 1)
                k = k.strip().lower()
                v = float(v.strip())
                if k == "sim":
                    sim = v
                elif k == "ent":
                    ent = v
                elif k == "contra":
                    contra = v
    except Exception:
        pass
    return sim, ent, contra


def _build_label_to_text_map(taxonomy: Dict[str, List[str]]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    idx = 1
    # Preserve insertion order of taxonomy JSON
    for _category, subcats in taxonomy.items():
        for sub in subcats:
            mapping[str(idx)] = str(sub)
            idx += 1
    return mapping


def _mean_pool(last_hidden_state: "torch.Tensor", attention_mask: "torch.Tensor") -> "torch.Tensor":
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    masked = last_hidden_state * mask
    summed = masked.sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def _encode_embeddings(texts: List[str], is_query: bool) -> np.ndarray:
    assert _EMBED_MODEL is not None and _EMBED_TOKENIZER is not None and _DEVICE is not None
    # E5-style prefix improves performance for E5 models; harmless for others
    prefix = "query: " if is_query else "passage: "
    proc_texts = [(prefix + t) for t in texts]
    with torch.inference_mode():
        inputs = _EMBED_TOKENIZER(proc_texts, padding=True, truncation=True, return_tensors="pt")
        inputs = {k: v.to(_DEVICE) for k, v in inputs.items()}
        outputs = _EMBED_MODEL(**inputs)
        if hasattr(outputs, "last_hidden_state"):
            emb = _mean_pool(outputs.last_hidden_state, inputs["attention_mask"])  # type: ignore[attr-defined]
        else:
            # Fallback: some models expose different output keys
            last = getattr(outputs, "last_hidden_state", None)
            if last is None:
                raise RuntimeError("Unexpected embedding model outputs; last_hidden_state not found")
            emb = _mean_pool(last, inputs["attention_mask"])  # type: ignore[index]
        emb = F.normalize(emb, p=2, dim=1)
        return emb.detach().cpu().numpy()


def _cosine_sim_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a @ b.T)


def _split_sentences(text: str) -> List[str]:
    if not isinstance(text, str) or not text.strip():
        return []
    # Simple rule-based splitter; avoids heavy deps
    pieces = re.split(r"(?<=[\.!?])\s+|[\n\r]+", text)
    sentences = [p.strip() for p in pieces if isinstance(p, str) and p.strip()]
    # Filter very short fragments
    sentences = [s for s in sentences if len(s) >= 8]
    return sentences[:256]


def _nli_probs(premises: List[str], hypotheses: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    assert _NLI_MODEL is not None and _NLI_TOKENIZER is not None and _DEVICE is not None
    with torch.inference_mode():
        enc = _NLI_TOKENIZER(premises, hypotheses, padding=True, truncation=True, return_tensors="pt")
        enc = {k: v.to(_DEVICE) for k, v in enc.items()}
        outputs = _NLI_MODEL(**enc)
        logits = outputs.logits  # type: ignore[attr-defined]
        probs = F.softmax(logits, dim=-1)
        # Try to map to entailment/neutral/contradiction indices
        id2label = getattr(_NLI_MODEL.config, "id2label", {})  # type: ignore[attr-defined]
        label2id: Dict[str, int] = {str(v).lower(): int(k) for k, v in getattr(_NLI_MODEL.config, "label2id", {}).items()}  # type: ignore[attr-defined]
        def idx(name: str) -> int:
            if name.lower() in label2id:
                return label2id[name.lower()]
            # fallback: search id2label
            for i, lab in id2label.items():
                if str(lab).lower() == name.lower():
                    return int(i)
            # common defaults
            mapping = {"entailment": 2, "neutral": 1, "contradiction": 0}
            return mapping[name.lower()]
        ent_idx = idx("entailment")
        neu_idx = idx("neutral")
        con_idx = idx("contradiction")
        ent = probs[:, ent_idx].detach().cpu().numpy()
        neu = probs[:, neu_idx].detach().cpu().numpy()
        con = probs[:, con_idx].detach().cpu().numpy()
        return ent, neu, con


def init_verification(
    taxonomy: Dict[str, List[str]],
    method: str = "combo",
    top_k: int = 3,
    embed_model_name: str = "intfloat/multilingual-e5-base",
    nli_model_name: str = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
    thresholds: Optional[Dict[str, float]] = None,
    device: Optional[str] = None,
    debug: Optional[bool] = None,
) -> None:
    """Initialize global models and configuration for verification.

    Call once per Ray worker process.
    """
    _ensure_imports()
    global _EMBED_TOKENIZER, _EMBED_MODEL, _NLI_TOKENIZER, _NLI_MODEL, _CONFIG, _DEVICE, _CLAIM_MAP, _DEBUG
    _CLAIM_MAP = _build_label_to_text_map(taxonomy)
    _DEVICE = _detect_device(device)
    # Resolve debug flag from argument or env var UAIR_VERIFY_DEBUG
    try:
        _DEBUG = bool(debug) if debug is not None else (str(os.environ.get("UAIR_VERIFY_DEBUG", "")).strip().lower() in {"1","true","yes","on"})
    except Exception:
        _DEBUG = False
    _CONFIG = VerificationConfig(
        method=str(method),
        top_k=int(top_k),
        sim_threshold=float(thresholds.get("sim", 0.55) if thresholds else 0.55),
        entail_threshold=float(thresholds.get("ent", 0.85) if thresholds else 0.85),
        contra_max=float(thresholds.get("contra", 0.05) if thresholds else 0.05),
        embed_model_name=embed_model_name,
        nli_model_name=nli_model_name,
        device=_DEVICE,
    )
    # Embedding model
    if _EMBED_MODEL is None:
        _EMBED_TOKENIZER = AutoTokenizer.from_pretrained(_CONFIG.embed_model_name)
        _EMBED_MODEL = AutoModel.from_pretrained(_CONFIG.embed_model_name)
        _EMBED_MODEL.to(_DEVICE)
        _EMBED_MODEL.eval()
    # NLI model (only if needed)
    if _CONFIG.method in {"nli", "combo", "combo_judge"} and _NLI_MODEL is None:
        _NLI_TOKENIZER = AutoTokenizer.from_pretrained(_CONFIG.nli_model_name)
        _NLI_MODEL = AutoModelForSequenceClassification.from_pretrained(_CONFIG.nli_model_name)
        _NLI_MODEL.to(_DEVICE)
        _NLI_MODEL.eval()


def _verify_one(chunk_text: str, chunk_label: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "ver_sim_max": None,
        "ver_top_sent": None,
        "ver_evidence_topk": None,
        "ver_nli_ent_max": None,
        "ver_nli_label_max": None,
        "ver_nli_evidence": None,
        "ver_verified_chunk": False,
    }
    if _DEBUG:
        out.update({
            "ver_dbg_method": getattr(_CONFIG, "method", None),
            "ver_dbg_device": _DEVICE,
            "ver_dbg_has_embed": bool(_EMBED_MODEL is not None),
            "ver_dbg_has_nli": bool(_NLI_MODEL is not None),
            "ver_dbg_reason": None,
            "ver_dbg_num_sentences": None,
            "ver_dbg_label_text": None,
            "ver_dbg_embed_error": None,
            "ver_dbg_nli_skipped": None,
            "ver_dbg_nli_skip_reason": None,
            "ver_dbg_nli_error": None,
        })
    # Check if verification is properly initialized
    if _EMBED_MODEL is None or _DEVICE is None:
        if _DEBUG:
            out["ver_dbg_reason"] = "not_initialized"
        return out
    if not isinstance(chunk_label, str) or (chunk_label.strip().lower() == "none"):
        if _DEBUG:
            out["ver_dbg_reason"] = "no_label"
        return out
    label_text = _CLAIM_MAP.get(str(chunk_label))
    if not isinstance(label_text, str) or not label_text:
        if _DEBUG:
            out["ver_dbg_reason"] = "label_text_missing"
        return out
    sentences = _split_sentences(str(chunk_text or ""))
    if not sentences:
        sentences = [str(chunk_text or "").strip()]
    if _DEBUG:
        out["ver_dbg_num_sentences"] = len(sentences)
        out["ver_dbg_label_text"] = label_text
    try:
        query_emb = _encode_embeddings([label_text], is_query=True)
        sent_emb = _encode_embeddings(sentences, is_query=False)
        sims = _cosine_sim_matrix(query_emb, sent_emb)[0]
        order = np.argsort(-sims)
        k = int(max(1, _CONFIG.top_k))
        top_idx = order[:k]
        top_sents = [sentences[i] for i in top_idx]
        top_sims = [float(sims[i]) for i in top_idx]
        out["ver_sim_max"] = float(max(top_sims) if len(top_sims) > 0 else float(sims.max()))
        if len(top_sents) > 0:
            out["ver_top_sent"] = top_sents[0]
            out["ver_evidence_topk"] = top_sents
    except Exception as e:
        # Embedding errors → leave defaults
        if _DEBUG:
            out["ver_dbg_embed_error"] = str(e)
    # Pure embedding method
    if _CONFIG.method == "embed":
        out["ver_verified_chunk"] = bool((out["ver_sim_max"] or 0.0) >= _CONFIG.sim_threshold)
        if _DEBUG:
            out["ver_dbg_nli_skipped"] = True
            out["ver_dbg_nli_skip_reason"] = "method=embed"
        return out
    # NLI or combo
    try:
        if _NLI_MODEL is not None and out.get("ver_evidence_topk"):
            premises = list(out["ver_evidence_topk"])  # type: ignore[arg-type]
            hypotheses = [f"This article is about {label_text}." for _ in premises]
            ent, neu, con = _nli_probs(premises, hypotheses)
            best = int(np.argmax(ent)) if ent.size > 0 else 0
            ent_max = float(ent[best]) if ent.size > 0 else None
            con_best = float(con[best]) if con.size > 0 else None
            out["ver_nli_ent_max"] = ent_max
            out["ver_nli_label_max"] = ("entailment" if ent_max is not None and ent_max >= _CONFIG.entail_threshold else "neutral")
            out["ver_nli_evidence"] = premises[best] if premises else None
            out["ver_verified_chunk"] = bool(
                (ent_max is not None and ent_max >= _CONFIG.entail_threshold) and (con_best is not None and con_best <= _CONFIG.contra_max)
            )
        else:
            # If NLI not available, fall back to embedding threshold
            if _DEBUG:
                out["ver_dbg_nli_skipped"] = True
                out["ver_dbg_nli_skip_reason"] = ("nli_model_none" if _NLI_MODEL is None else "no_evidence_topk")
            out["ver_verified_chunk"] = bool((out["ver_sim_max"] or 0.0) >= _CONFIG.sim_threshold)
    except Exception as e:
        out["ver_verified_chunk"] = bool((out["ver_sim_max"] or 0.0) >= _CONFIG.sim_threshold)
        if _DEBUG:
            out["ver_dbg_nli_error"] = str(e)
    return out


def verify_batch_pandas(df: pd.DataFrame) -> pd.DataFrame:
    """Ray Data map_batches-compatible function (batch_format='pandas').

    Requires init_verification() to have been called on the worker.
    """
    # Ensure expected columns exist; create defaults if missing
    if "chunk_text" not in df.columns:
        df["chunk_text"] = ""
    if "chunk_label" not in df.columns:
        df["chunk_label"] = "None"
    
    # Early return with defaults if verification not initialized
    if _EMBED_MODEL is None or _DEVICE is None:
        df["ver_verified_chunk"] = False
        df["ver_sim_max"] = None
        df["ver_nli_ent_max"] = None
        df["ver_nli_evidence"] = None
        if _DEBUG:
            try:
                df["ver_dbg_reason"] = "not_initialized"
                df["ver_dbg_method"] = getattr(_CONFIG, "method", None)
                df["ver_dbg_device"] = _DEVICE
                df["ver_dbg_has_embed"] = bool(_EMBED_MODEL is not None)
                df["ver_dbg_has_nli"] = bool(_NLI_MODEL is not None)
            except Exception:
                pass
        return df
    
    # Work on a copy to avoid index-alignment issues during assignment
    df = df.copy()
    results: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        try:
            res = _verify_one(str(row.get("chunk_text", "")), str(row.get("chunk_label", "None")))
        except Exception:
            # Use defaults on error
            res = {
                "ver_sim_max": None,
                "ver_top_sent": None,
                "ver_evidence_topk": None,
                "ver_nli_ent_max": None,
                "ver_nli_label_max": None,
                "ver_nli_evidence": None,
                "ver_verified_chunk": False,
            }
        results.append(res)
    
    try:
        res_df = pd.DataFrame(results)
        # Assign by position to avoid misalignment with arbitrary indices
        for col in res_df.columns:
            df[col] = res_df[col].to_numpy()
    except Exception:
        # Add default columns if DataFrame creation fails
        df["ver_verified_chunk"] = False
        df["ver_sim_max"] = None
        df["ver_nli_ent_max"] = None
        df["ver_nli_evidence"] = None
    
    return df



