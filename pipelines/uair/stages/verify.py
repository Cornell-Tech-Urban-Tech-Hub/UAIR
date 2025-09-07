from typing import Any, Dict, List
import os
import json
import pandas as pd

try:
    import ray  # type: ignore
    _RAY_OK = True
except Exception:
    _RAY_OK = False

try:
    import yaml  # type: ignore
except Exception:
    yaml = None  # type: ignore


def run_verification_stage(df, cfg):
    """
    Wraps experiments.verification over Ray Datasets or pandas.

    Expects taxonomy JSON at cfg.taxonomy_json and chunk rows with
    at least: article_id, chunk_text, chunk_label (string index or "None").
    """
    try:
        from experiments.verification import (
            init_verification,
            verify_batch_pandas,
            parse_thresholds_string,
        )
    except Exception as e:
        raise RuntimeError(f"verification components unavailable: {e}")

    # Resolve taxonomy (YAML or JSON)
    taxonomy_path = str(getattr(cfg, "taxonomy_json", "") or "")
    taxonomy = {}
    try:
        if taxonomy_path and os.path.exists(taxonomy_path):
            if taxonomy_path.endswith((".yaml", ".yml")) and yaml is not None:
                with open(taxonomy_path, "r") as f:
                    data = yaml.safe_load(f)
            else:
                with open(taxonomy_path, "r") as f:
                    data = json.load(f)
            taxonomy = data.get("taxonomy", data) if isinstance(data, dict) else {}
    except Exception:
        taxonomy = {}

    # Verify configuration
    try:
        method = str(getattr(cfg.verify, "method", "combo"))
    except Exception:
        method = "combo"
    try:
        top_k = int(getattr(cfg.verify, "top_k", 3) or 3)
    except Exception:
        top_k = 3
    try:
        thr_s = str(getattr(cfg.verify, "thresholds", "sim=0.55,ent=0.85,contra=0.05"))
    except Exception:
        thr_s = "sim=0.55,ent=0.85,contra=0.05"
    sim_thr, ent_thr, con_thr = parse_thresholds_string(thr_s)
    thresholds = {"sim": sim_thr, "ent": ent_thr, "contra": con_thr}
    try:
        device = getattr(cfg.verify, "device", None)
    except Exception:
        device = None

    is_ray_ds = hasattr(df, "map_batches") and hasattr(df, "count") and _RAY_OK
    if is_ray_ds:
        ds_in = df

        # Build name->index mapping consistent with taxonomy stage ordering
        def _norm(s: str) -> str:
            try:
                import re as _re
                return _re.sub(r"\s+", " ", str(s or "").strip().lower())
            except Exception:
                return str(s or "").strip().lower()
        name_to_idx: Dict[str, int] = {}
        try:
            if isinstance(taxonomy, dict):
                idx = 1
                for _cat, subs in taxonomy.items():
                    if isinstance(subs, list):
                        for s in subs:
                            if isinstance(s, str) and s.strip():
                                name_to_idx[_norm(s)] = idx
                                idx += 1
        except Exception:
            name_to_idx = {}

        def _coerce_label(row: Dict[str, Any]) -> Dict[str, Any]:
            r = dict(row)
            # Ensure chunk_text exists
            if not r.get("chunk_text"):
                r["chunk_text"] = str(r.get("article_text") or "")
            lab = r.get("chunk_label")
            lab_name = r.get("chunk_label_name")
            lab_str = str(lab) if lab is not None else "None"
            # If numeric-like, keep
            is_numeric = False
            try:
                int(lab_str)
                is_numeric = True
            except Exception:
                is_numeric = False
            if not is_numeric:
                # Try to map name to index
                nm = str(lab_name if lab_name not in (None, "None", "") else lab_str)
                idx_val = name_to_idx.get(_norm(nm))
                if isinstance(idx_val, int):
                    r["chunk_label"] = str(idx_val)
                else:
                    r["chunk_label"] = "None"
            else:
                r["chunk_label"] = lab_str
            return r

        try:
            ds_in = ds_in.map(_coerce_label)
        except Exception:
            pass
        # Keep only rows with a usable chunk_label
        def _has_label(r: Dict[str, Any]) -> bool:
            v = r.get("chunk_label")
            if v is None:
                return False
            s = str(v).strip()
            if s == "" or s.lower() == "none":
                return False
            try:
                int(s)
                return True
            except Exception:
                return False
        try:
            ds_in = ds_in.filter(_has_label)
        except Exception:
            pass

        # Initialize per-worker via map_batches constructor, or do a lightweight pre-map init
        def _init_fn() -> None:
            init_verification(
                taxonomy=taxonomy,
                method=method,
                top_k=top_k,
                thresholds=thresholds,
                device=device,
            )

        # Wrapper ensures initialization per worker before verifying batches
        def _verify_with_init(pdf: pd.DataFrame) -> pd.DataFrame:
            try:
                init_verification(
                    taxonomy=taxonomy,
                    method=method,
                    top_k=top_k,
                    thresholds=thresholds,
                    device=device,
                )
            except Exception:
                pass
            return verify_batch_pandas(pdf)

        try:
            ds_in = ds_in.map_batches(lambda x: x, batch_format="pandas", fn_constructor=_init_fn)  # type: ignore[arg-type]
        except Exception:
            pass
        ds_out = ds_in.map_batches(_verify_with_init, batch_format="pandas")
        # Perform doc-level aggregation similar to experiments/verification_stage.py
        def _reduce_verify(pdf: pd.DataFrame) -> pd.DataFrame:
            if pdf is None or len(pdf) == 0:
                return pd.DataFrame([])
            try:
                verified_chunk = bool(pdf.get("ver_verified_chunk", pd.Series([], dtype=bool)).any())
            except Exception:
                verified_chunk = False
            best_ent = None
            best_evi = None
            max_sim = None
            max_sim_evi = None
            if "ver_nli_ent_max" in pdf.columns:
                try:
                    idx = int(pdf["ver_nli_ent_max"].astype(float).fillna(-1).idxmax())
                    best_ent = float(pdf.loc[idx, "ver_nli_ent_max"]) if pd.notna(pdf.loc[idx, "ver_nli_ent_max"]) else None
                    best_evi = pdf.loc[idx, "ver_nli_evidence"] if "ver_nli_evidence" in pdf.columns else None
                except Exception:
                    pass
            if "ver_sim_max" in pdf.columns:
                try:
                    idx2 = int(pdf["ver_sim_max"].astype(float).fillna(-1).idxmax())
                    max_sim = float(pdf.loc[idx2, "ver_sim_max"]) if pd.notna(pdf.loc[idx2, "ver_sim_max"]) else None
                    max_sim_evi = pdf.loc[idx2, "ver_top_sent"] if "ver_top_sent" in pdf.columns else None
                except Exception:
                    pass
            out = {
                "article_id": pdf.get("article_id", pd.Series([None])).iloc[0],
                "verified_doc": bool(verified_chunk),
                "ver_doc_best_ent": best_ent,
                "ver_doc_best_evidence": best_evi,
                "ver_doc_max_sim": max_sim,
                "ver_doc_max_sim_evidence": max_sim_evi,
            }
            return pd.DataFrame([out])

        try:
            ver_docs_ds = ds_out.groupby("article_id").map_groups(_reduce_verify, batch_format="pandas")
        except Exception:
            ver_docs_ds = None

        # Emit side outputs when runtime.output_dir is set
        try:
            out_dir = str(getattr(cfg.runtime, "output_dir", "") or "")
        except Exception:
            out_dir = ""
        if out_dir:
            try:
                import os as _os
                _os.makedirs(out_dir, exist_ok=True)
            except Exception:
                pass
            try:
                ds_out.write_parquet(os.path.join(out_dir, "chunks_verification"))
            except Exception:
                pass
            try:
                if ver_docs_ds is not None:
                    ver_docs_ds.write_parquet(os.path.join(out_dir, "docs_verification"))
            except Exception:
                pass
        return ds_out

    # Pandas fallback
    pdf = df.copy() if df is not None else pd.DataFrame([])
    init_verification(
        taxonomy=taxonomy,
        method=method,
        top_k=top_k,
        thresholds=thresholds,
        device=device,
    )
    if len(pdf) == 0:
        return pdf
    # Coerce labels by name when needed
    try:
        def _norm2(s: str) -> str:
            try:
                import re as _re
                return _re.sub(r"\s+", " ", str(s or "").strip().lower())
            except Exception:
                return str(s or "").strip().lower()
        name_to_idx2: Dict[str, int] = {}
        if isinstance(taxonomy, dict):
            idx = 1
            for _cat, subs in taxonomy.items():
                if isinstance(subs, list):
                    for s in subs:
                        if isinstance(s, str) and s.strip():
                            name_to_idx2[_norm2(s)] = idx
                            idx += 1
        def _fix_row(r: pd.Series) -> pd.Series:
            lab = r.get("chunk_label")
            lab_name = r.get("chunk_label_name")
            lab_str = str(lab) if lab is not None else "None"
            # numeric?
            numeric = False
            try:
                int(lab_str)
                numeric = True
            except Exception:
                numeric = False
            if not numeric:
                nm = str(lab_name if lab_name not in (None, "None", "") else lab_str)
                idx_v = name_to_idx2.get(_norm2(nm))
                r["chunk_label"] = (str(idx_v) if isinstance(idx_v, int) else "None")
            else:
                r["chunk_label"] = lab_str
            if not r.get("chunk_text"):
                r["chunk_text"] = str(r.get("article_text") or "")
            return r
        if len(name_to_idx2):
            pdf = pdf.apply(_fix_row, axis=1)
    except Exception:
        pass
    # Filter to rows with usable chunk_label
    try:
        def _keep_row(r: pd.Series) -> bool:
            v = r.get("chunk_label")
            if v is None:
                return False
            s = str(v).strip()
            if s == "" or s.lower() == "none":
                return False
            try:
                int(s)
                return True
            except Exception:
                return False
        if len(pdf):
            pdf = pdf[pdf.apply(_keep_row, axis=1)]
    except Exception:
        pass

    try:
        out = verify_batch_pandas(pdf)
    except Exception:
        out = pdf
    # Pandas doc-level aggregation
    try:
        def _reduce_verify_pdf(df_in: pd.DataFrame) -> pd.DataFrame:
            if df_in.empty:
                return pd.DataFrame([])
            verified_chunk = bool(df_in.get("ver_verified_chunk", pd.Series([], dtype=bool)).any())
            best_ent = None
            best_evi = None
            max_sim = None
            max_sim_evi = None
            if "ver_nli_ent_max" in df_in.columns:
                try:
                    idx = int(df_in["ver_nli_ent_max"].astype(float).fillna(-1).idxmax())
                    best_ent = float(df_in.loc[idx, "ver_nli_ent_max"]) if pd.notna(df_in.loc[idx, "ver_nli_ent_max"]) else None
                    best_evi = df_in.loc[idx, "ver_nli_evidence"] if "ver_nli_evidence" in df_in.columns else None
                except Exception:
                    pass
            if "ver_sim_max" in df_in.columns:
                try:
                    idx2 = int(df_in["ver_sim_max"].astype(float).fillna(-1).idxmax())
                    max_sim = float(df_in.loc[idx2, "ver_sim_max"]) if pd.notna(df_in.loc[idx2, "ver_sim_max"]) else None
                    max_sim_evi = df_in.loc[idx2, "ver_top_sent"] if "ver_top_sent" in df_in.columns else None
                except Exception:
                    pass
            out_row = {
                "article_id": df_in.get("article_id", pd.Series([None])).iloc[0],
                "verified_doc": bool(verified_chunk),
                "ver_doc_best_ent": best_ent,
                "ver_doc_best_evidence": best_evi,
                "ver_doc_max_sim": max_sim,
                "ver_doc_max_sim_evidence": max_sim_evi,
            }
            return pd.DataFrame([out_row])

        ver_docs_pdf = out.groupby("article_id", dropna=False).apply(_reduce_verify_pdf).reset_index(drop=True)
    except Exception:
        ver_docs_pdf = pd.DataFrame([])

    # Persist side outputs if requested
    try:
        out_dir = str(getattr(cfg.runtime, "output_dir", "") or "")
    except Exception:
        out_dir = ""
    if out_dir:
        try:
            os.makedirs(out_dir, exist_ok=True)
        except Exception:
            pass
        try:
            out.to_parquet(os.path.join(out_dir, "chunks_verification.parquet"), index=False)
        except Exception:
            pass
        try:
            if len(ver_docs_pdf):
                ver_docs_pdf.to_parquet(os.path.join(out_dir, "docs_verification.parquet"), index=False)
        except Exception:
            pass
    return out


