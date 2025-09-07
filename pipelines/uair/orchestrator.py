from __future__ import annotations

from typing import Any, Dict, List, Optional
import os
import json
import pandas as pd
from datetime import datetime
from uuid import uuid4
import platform
import socket
import sys
import subprocess
import shlex
import threading
import time as _time
try:
    import ray  # type: ignore
    _RAY_OK_E = True
except Exception:
    _RAY_OK_E = False

# Lightweight Ray actor to aggregate streaming usage stats without driver blocking
if _RAY_OK_E:
    try:
        @ray.remote
        class _UsageAggregator:
            def __init__(self):
                # Stage-scoped and overall cumulative counters
                self._totals: Dict[str, Dict[str, int]] = {
                    "classify": {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                    "decompose": {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                    "taxonomy": {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                    "overall": {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                }

            def _coerce_int(self, v: Any) -> int:
                try:
                    if v is None:
                        return 0
                    if isinstance(v, bool):
                        return int(v)
                    if isinstance(v, (int,)):
                        return int(v)
                    if isinstance(v, float):
                        return int(v)
                    return int(float(v))
                except Exception:
                    return 0

            def record(self, stage: str, calls: Any = 0, prompt_tokens: Any = 0, completion_tokens: Any = 0, total_tokens: Any = 0) -> None:
                st = str(stage or "overall").strip().lower()
                if st not in self._totals:
                    st = "overall"
                c = self._coerce_int(calls)
                p = self._coerce_int(prompt_tokens)
                q = self._coerce_int(completion_tokens)
                t = self._coerce_int(total_tokens)
                self._totals[st]["calls"] += c
                self._totals[st]["prompt_tokens"] += p
                self._totals[st]["completion_tokens"] += q
                self._totals[st]["total_tokens"] += t
                # Mirror into overall
                if st != "overall":
                    self._totals["overall"]["calls"] += c
                    self._totals["overall"]["prompt_tokens"] += p
                    self._totals["overall"]["completion_tokens"] += q
                    # If total_tokens isn't provided, derive when possible
                    if t == 0 and (p or q):
                        t = p + q
                    self._totals["overall"]["total_tokens"] += t

            def snapshot(self) -> Dict[str, Dict[str, int]]:
                # Return a cheap copy
                return {
                    k: dict(v) for k, v in self._totals.items()
                }
    except Exception:
        pass


def _load_parquet_dataset(parquet_path: str, columns: Dict[str, str], debug: bool, sample_n: Optional[int]) -> pd.DataFrame:
    if not isinstance(parquet_path, str) or not parquet_path:
        raise ValueError("data.parquet_path is required")
    if not os.path.isabs(parquet_path):
        # Resolve relative to repo root if possible
        parquet_path = os.path.abspath(parquet_path)
    df = pd.read_parquet(parquet_path)
    # Build a rename map for article dataset columns only
    col_map = {
        columns.get("article_text", "article_text"): "article_text",
        columns.get("article_path", "article_path"): "article_path",
        columns.get("country", "country"): "country",
        columns.get("year", "year"): "year",
        columns.get("article_id", "article_id"): "article_id",
    }
    present = {src: dst for src, dst in col_map.items() if src in df.columns}
    if present:
        df = df.rename(columns=present)
    # Ensure the canonical article text column exists
    if "article_text" not in df.columns:
        raise RuntimeError("Parquet missing required text column (article_text); configure data.columns.article_text to map your source text column")

    # Normalize a minimal working set; coerce a few known text-ish columns when present
    def _safe_str(x: Any) -> str:
        try:
            return "" if x is None or (isinstance(x, float) and pd.isna(x)) else str(x).strip()
        except Exception:
            return str(x) if x is not None else ""
    # Ensure article metadata columns exist
    for c in ("article_path", "country", "year", "article_id"):
        if c not in df.columns:
            df[c] = None
        else:
            try:
                df[c] = df[c].apply(_safe_str)
            except Exception:
                pass

    # Drop legacy rules dataset columns if present
    legacy_cols = [
        "name", "public_description", "subscribers", "rule_text",
        "rule_index", "total_rules_count",
    ]
    try:
        drop_now = [c for c in legacy_cols if c in df.columns]
        if drop_now:
            df = df.drop(columns=drop_now)
    except Exception:
        pass

    if debug and isinstance(sample_n, int) and sample_n > 0:
        try:
            n = min(int(sample_n), int(len(df)))
        except Exception:
            n = int(sample_n)
        try:
            seed = int(os.environ.get("UAIR_SAMPLE_SEED", "777"))
        except Exception:
            seed = 777
        try:
            df = df.sample(n=n, random_state=seed).reset_index(drop=True)
        except Exception:
            df = df.head(n)
    return df


def _project_root() -> str:
    try:
        here = os.path.dirname(__file__)
        return os.path.abspath(os.path.join(here, "..", ".."))
    except Exception:
        return os.getcwd()


def _default_output_dir(stage_name: str) -> str:
    try:
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    except Exception:
        ts = "now"
    try:
        label = str(stage_name or "stage").strip().lower()
    except Exception:
        label = "stage"
    base = os.path.join(_project_root(), "outputs", f"{ts}_{label}")
    return base


def run_experiment(cfg) -> None:
    """Simplified experiment entry point: pandas IO, stage routing, and SLURM pipeline launcher.

    Notes:
    - W&B runs are initialized in stage implementations; orchestrator only forwards a shared group id.
    - Hydra/Submitit should be used for SLURM sweeps and child stage jobs.
    """
    stage = str(getattr(cfg.runtime, "stage", "classify")).strip().lower()
    debug = bool(getattr(cfg.runtime, "debug", True))
    sample_n = getattr(cfg.runtime, "sample_n", 100)

    parquet_path = str(getattr(cfg.data, "parquet_path"))
    columns = dict(getattr(cfg.data, "columns", {}))

    # Always use pandas DataFrame for control-plane simplicity
    df = _load_parquet_dataset(parquet_path, columns, debug=debug, sample_n=sample_n)

    if not bool(os.environ.get("RULE_TUPLES_SILENT", "0")):
        print(json.dumps({
            "loaded_rows": int(len(df)),
            "stage": stage,
            "columns": list(df.columns)[:12],
            "streaming": False,
        }, indent=2))

    # If running the pipeline coordinator, ensure a non-null W&B group is set before init
    if stage == "pipeline":
        try:
            _existing_group = os.environ.get("WANDB_GROUP")
        except Exception:
            _existing_group = None
        try:
            if not _existing_group:
                _pre_group = getattr(cfg.wandb, "group", None)
                if not _pre_group:
                    _pre_group = uuid4().hex
                os.environ["WANDB_GROUP"] = str(_pre_group)
        except Exception:
            try:
                os.environ["WANDB_GROUP"] = uuid4().hex
            except Exception:
                pass

    # Optional W&B init (orchestrator-level) for consolidated logging
    use_wandb = False
    try:
        use_wandb = bool(getattr(cfg.wandb, "enabled", False))
    except Exception:
        use_wandb = False
    wb = None
    _wb_group = None
    if use_wandb:
        try:
            import wandb as _wandb  # type: ignore
            wb = _wandb
            try:
                _wb_group = getattr(cfg.wandb, "group", None) or os.environ.get("WANDB_GROUP")
            except Exception:
                _wb_group = None
            run_name = f"{getattr(cfg.experiment, 'name', 'UAIR')}-{stage}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            try:
                if _wb_group:
                    run_name = f"{_wb_group}-{run_name}"
            except Exception:
                pass
            # Build informative run config capturing model/runtime
            try:
                model_source = str(getattr(cfg.model, "model_source", ""))
            except Exception:
                model_source = ""
            try:
                engine_kwargs = dict(getattr(cfg.model, "engine_kwargs", {}))
            except Exception:
                engine_kwargs = {}
            try:
                batch_size = int(getattr(cfg.model, "batch_size", 16) or 16)
            except Exception:
                batch_size = 16
            try:
                concurrency = int(getattr(cfg.model, "concurrency", 1) or 1)
            except Exception:
                concurrency = 1
            run_config = {
                "stage": stage,
                "parquet_path": parquet_path,
                "data_columns_map": columns,
                "sample_n": sample_n,
                "output_csv": str(getattr(cfg.runtime, "output_csv", "") or ""),
                "use_llm_classify": bool(getattr(cfg.runtime, "use_llm_classify", False)),
                "use_llm_decompose": bool(getattr(cfg.runtime, "use_llm_decompose", False)),
                "prefilter_mode": str(getattr(cfg.runtime, "prefilter_mode", "pre_gating")),
                # Model/runtime
                "model_source": model_source,
                "engine_kwargs": engine_kwargs,
                "batch_size": batch_size,
                "concurrency": concurrency,
                "sampling_params": dict(getattr(cfg, "sampling_params", {})),
                # System metadata
                "python_version": sys.version.split()[0],
                "os": platform.platform(),
                "hostname": socket.gethostname(),
            }
            try:
                proj = str(getattr(cfg.wandb, "project", "UAIR") or "UAIR")
            except Exception:
                proj = "UAIR"
            try:
                ent = getattr(cfg.wandb, "entity", None)
                ent = str(ent) if (ent is not None and str(ent).strip() != "") else None
            except Exception:
                ent = None
            wb.init(project=proj, entity=ent, group=_wb_group, job_type=stage, name=run_name, config=run_config)
            try:
                wb.log({"dataset/loaded_rows": int(len(df))})
            except Exception:
                pass
        except Exception:
            wb = None

    def _wb_log(data: Dict[str, Any]) -> None:
        if wb is not None:
            try:
                wb.log(data)
            except Exception:
                pass

    def _wb_log_table(df_in: pd.DataFrame, key: str, prefer_cols: List[str], max_rows: int = 1000) -> None:
        if wb is None:
            return
        # If a stage closed the global W&B run, re-init a lightweight run
        try:
            if getattr(wb, "run", None) is None:
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
                    name = f"{getattr(cfg.experiment, 'name', 'UAIR')}-{stage}-tables-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
                except Exception:
                    name = None
                try:
                    wb.init(project=proj, entity=ent, group=_wb_group, job_type=stage, name=name)
                except Exception:
                    pass
        except Exception:
            pass
        try:
            df_local = df_in
            cols = [c for c in prefer_cols if c in df_local.columns]
            if not cols:
                cols = list(df_local.columns)[:12]
            try:
                table = wb.Table(columns=cols, log_mode="MUTABLE")
            except Exception:
                table = wb.Table(columns=cols)
            n = 0
            for _, r in df_local.reset_index(drop=True).iterrows():
                if n >= int(max_rows):
                    break
                row_vals: List[str] = []
                for c in cols:
                    v = r.get(c)
                    try:
                        import json as _json
                        if isinstance(v, (dict, list, tuple, set)):
                            if isinstance(v, set):
                                v = list(v)
                            v = _json.dumps(v, ensure_ascii=False)
                    except Exception:
                        pass
                    try:
                        row_vals.append(str(v) if v is not None else "")
                    except Exception:
                        row_vals.append("")
                table.add_data(*row_vals)
                n += 1
            wb.log({key: table, f"{key}/rows": n})
        except Exception:
            pass

    if stage == "classify":
        from .stages.classify import run_classification_stage
        out_df = run_classification_stage(df, cfg)
        out_path = str(getattr(cfg.runtime, "output_csv", "") or "")
        # Best-effort sanitization for Parquet: convert nested objects to JSON strings
        def _sanitize_for_parquet_pdf(pdf: pd.DataFrame) -> pd.DataFrame:
            pdf = pdf.copy()
            def _ser(v: Any) -> Any:
                try:
                    import numpy as _np  # type: ignore
                    if isinstance(v, _np.generic):
                        try:
                            return v.item()
                        except Exception:
                            pass
                    if isinstance(v, _np.ndarray):
                        try:
                            import json as _json
                            return _json.dumps(v.tolist(), ensure_ascii=False)
                        except Exception:
                            return str(v)
                except Exception:
                    pass
                if isinstance(v, (dict, list, tuple, set)):
                    try:
                        import json as _json
                        if isinstance(v, set):
                            v = list(v)
                        return _json.dumps(v, ensure_ascii=False)
                    except Exception:
                        return str(v)
                return v
            for col in list(pdf.columns):
                try:
                    if pdf[col].dtype == object:
                        s = pdf[col].apply(_ser)
                        try:
                            s = s.apply(lambda x: "" if x is None else x)
                        except Exception:
                            pass
                        pdf[col] = s
                except Exception:
                    pass
            return pdf
        if out_path:
            try:
                out_df.to_csv(out_path, index=False)
                if not bool(os.environ.get("RULE_TUPLES_SILENT", "0")):
                    print(json.dumps({"classified_rows": int(len(out_df)), "output_csv": out_path}, indent=2))
            except Exception:
                if not bool(os.environ.get("RULE_TUPLES_SILENT", "0")):
                    print(json.dumps({"classified_rows": int(len(out_df)), "output_csv": None}, indent=2))
        else:
            if not bool(os.environ.get("RULE_TUPLES_SILENT", "0")):
                print(json.dumps({"classified_rows": int(len(out_df))}, indent=2))
        # Write Parquet to runtime.output_dir if provided
        out_dir = str(getattr(cfg.runtime, "output_dir", "") or _default_output_dir("classify"))
        if out_dir:
            try:
                os.makedirs(out_dir, exist_ok=True)
            except Exception:
                pass
            try:
                # All results
                pq_all = os.path.join(out_dir, "classify_all.parquet")
                _sanitize_for_parquet_pdf(out_df).to_parquet(pq_all, index=False)
                # Relevant-only results (when available)
                pq_rel = None
                try:
                    if "is_relevant" in out_df.columns:
                        rel_df = out_df[out_df["is_relevant"].astype(bool) == True]
                        pq_rel = os.path.join(out_dir, "classify_relevant.parquet")
                        _sanitize_for_parquet_pdf(rel_df).to_parquet(pq_rel, index=False)
                except Exception:
                    pq_rel = None
                if not bool(os.environ.get("RULE_TUPLES_SILENT", "0")):
                    print(json.dumps({
                        "classified_rows": int(len(out_df)),
                        "output_parquet_all": pq_all,
                        "output_parquet_relevant": pq_rel,
                    }, indent=2))
            except Exception as e:
                if not bool(os.environ.get("RULE_TUPLES_SILENT", "0")):
                    print(json.dumps({"error": "classify_parquet_write_failed", "detail": str(e)}, indent=2))
        # Orchestrator-level W&B fallback: aggregate usage tokens and throughput if available
        try:
            import numpy as _np  # type: ignore
        except Exception:
            _np = None  # type: ignore
        try:
            def _safe_sum(col: str) -> int:
                try:
                    if col in out_df.columns:
                        s = pd.to_numeric(out_df[col], errors="coerce").fillna(0)
                        return int(s.astype("int64").sum())
                except Exception:
                    pass
                return 0
            p_tok = _safe_sum("token_usage_prompt")
            q_tok = _safe_sum("token_usage_output")
            t_tok = _safe_sum("token_usage_total")
            # Derive avg tokens/sec using output tokens and per-row latency if present
            tok_per_s = None
            try:
                if "latency_s" in out_df.columns:
                    lat = pd.to_numeric(out_df["latency_s"], errors="coerce").fillna(0.0)
                    out = pd.to_numeric(out_df.get("token_usage_output", 0), errors="coerce").fillna(0.0)
                    total_lat = float(lat.sum())
                    total_out = float(out.sum())
                    if total_lat > 0.0 and total_out > 0.0:
                        tok_per_s = total_out / total_lat
            except Exception:
                tok_per_s = None
            payload: Dict[str, Any] = {
                "usage/classify/prompt_tokens": int(p_tok),
                "usage/classify/completion_tokens": int(q_tok if q_tok > 0 else max(0, t_tok - p_tok)),
                "usage/classify/total_tokens": int(t_tok if t_tok > 0 else p_tok + q_tok),
            }
            if tok_per_s is not None:
                payload["classify/output_tok_per_s"] = float(tok_per_s)
            # If classify stage used keyword gating, try to surface its counters
            try:
                if "relevant_keyword" in out_df.columns:
                    kw_marked = int(out_df["relevant_keyword"].astype(bool).sum())
                else:
                    kw_marked = None
            except Exception:
                kw_marked = None
            if kw_marked is not None:
                payload["keyword/marked"] = kw_marked
            _wb_log(payload)
        except Exception:
            pass
        # Log final results table to W&B (sampled)
        try:
            try:
                max_rows = int(getattr(cfg.wandb, "table_sample_rows", 1000) or 1000)
            except Exception:
                max_rows = 1000
            _wb_log_table(
                out_df,
                key="tables/classify_results",
                prefer_cols=[
                    "article_id","article_path","country","year",
                    "is_relevant","relevance_answer","classification_mode",
                    "relevant_keyword","latency_s","token_usage_prompt","token_usage_output","token_usage_total",
                ],
                max_rows=max_rows,
            )
            # Also log only the rows marked relevant by classify
            try:
                if "is_relevant" in out_df.columns:
                    rel_df = out_df[out_df["is_relevant"].astype(bool) == True]
                    if len(rel_df) > 0:
                        _wb_log_table(
                            rel_df,
                            key="tables/classify_relevant_results",
                            prefer_cols=[
                                "article_id","article_path","country","year",
                                "is_relevant","relevance_answer","classification_mode",
                                "relevant_keyword","latency_s","token_usage_prompt","token_usage_output","token_usage_total",
                            ],
                            max_rows=max_rows,
                        )
            except Exception:
                pass
        except Exception:
            pass
        return

    if stage == "decompose":
        from .stages.decompose import run_decomposition_stage
        out_df = run_decomposition_stage(df, cfg)
        out_path = str(getattr(cfg.runtime, "output_csv", "") or "")
        if out_path:
            try:
                out_df.to_csv(out_path, index=False)
                if not bool(os.environ.get("RULE_TUPLES_SILENT", "0")):
                    print(json.dumps({"decomposed_rows": int(len(out_df)), "output_csv": out_path}, indent=2))
            except Exception:
                if not bool(os.environ.get("RULE_TUPLES_SILENT", "0")):
                    print(json.dumps({"decomposed_rows": int(len(out_df)), "output_csv": None}, indent=2))
        else:
            if not bool(os.environ.get("RULE_TUPLES_SILENT", "0")):
                print(json.dumps({"decomposed_rows": int(len(out_df))}, indent=2))
        return

    if stage == "taxonomy":
        from .stages.taxonomy import run_taxonomy_stage
        out_df = run_taxonomy_stage(df, cfg)
        out_path = str(getattr(cfg.runtime, "output_csv", "") or "")
        # Doc-level aggregation (non-streaming)
        try:
            def _docs_from_chunks_pdf(chunks_df: pd.DataFrame) -> pd.DataFrame:
                rows: List[Dict[str, Any]] = []
                if chunks_df is None or len(chunks_df) == 0:
                    return pd.DataFrame([])
                for aid, grp in chunks_df.groupby("article_id", dropna=False):
                    labels_num = [str(x) for x in grp.get("chunk_label", pd.Series([], dtype=str)).tolist()]
                    labels_name = [str(x) for x in grp.get("chunk_label_name", pd.Series([], dtype=str)).tolist()]
                    # Majority vote on names (excluding 'None')
                    name_counts: Dict[str, int] = {}
                    for nm in labels_name:
                        if str(nm).strip().lower() == "none":
                            continue
                        name_counts[nm] = name_counts.get(nm, 0) + 1
                    if name_counts:
                        final_name = max(name_counts.items(), key=lambda it: it[1])[0]
                    else:
                        final_name = "None"
                    # Majority vote on numbers (excluding 'None')
                    num_counts: Dict[str, int] = {}
                    for lb in labels_num:
                        if str(lb).strip().lower() == "none":
                            continue
                        num_counts[lb] = num_counts.get(lb, 0) + 1
                    if num_counts:
                        def _num_sort_key(item):
                            label, count = item
                            try:
                                numeric_val = int(label)
                            except Exception:
                                numeric_val = -1
                            return (count, numeric_val)
                        final_num = max(num_counts.items(), key=_num_sort_key)[0]
                    else:
                        final_num = "None"
                    paths = [x for x in grp.get("article_path", pd.Series([], dtype=str)).tolist() if isinstance(x, str) and x]
                    rows.append({
                        "article_id": aid,
                        "article_path": (paths[0] if paths else None),
                        "predicted_category_number": final_num,
                        "predicted_category_name": final_name,
                        "num_chunks": int(grp.get("num_chunks", pd.Series([len(labels_num)])).max()) if "num_chunks" in grp.columns else len(labels_num),
                        "chunk_labels": ",".join(labels_num),
                        "chunk_label_names": ",".join(labels_name),
                    })
                return pd.DataFrame(rows)
            docs_df = _docs_from_chunks_pdf(out_df)
        except Exception:
            docs_df = pd.DataFrame([])
        # Robust W&B metric for none_fraction at chunk level (normalize names/numbers)
        try:
            def _none_mask(sr: pd.Series) -> pd.Series:
                try:
                    s = sr.astype(str).str.strip().str.strip("\"'").str.lower()
                    return s.isin(["none", ""])
                except Exception:
                    try:
                        return sr.astype(str).str.lower().eq("none")
                    except Exception:
                        return pd.Series([False] * len(sr))
            total_tx = int(len(out_df))
            if total_tx > 0 and use_wandb:
                if "chunk_label_name" in out_df.columns:
                    none_ct = int(_none_mask(out_df["chunk_label_name"]).sum())
                elif "chunk_label" in out_df.columns:
                    none_ct = int(_none_mask(out_df["chunk_label"]).sum())
                else:
                    none_ct = 0
                _wb_log({"taxonomy/none_fraction": (float(none_ct) / float(total_tx))})
        except Exception:
            pass

        if out_path:
            try:
                out_df.to_csv(out_path, index=False)
                # Write docs alongside chunk CSV with suffix
                try:
                    root, ext = os.path.splitext(out_path)
                    docs_path = f"{root}_docs{ext or '.csv'}"
                    if len(docs_df):
                        docs_df.to_csv(docs_path, index=False)
                except Exception:
                    pass
                if not bool(os.environ.get("RULE_TUPLES_SILENT", "0")):
                    print(json.dumps({"taxonomy_rows": int(len(out_df)), "output_csv": out_path}, indent=2))
            except Exception:
                if not bool(os.environ.get("RULE_TUPLES_SILENT", "0")):
                    print(json.dumps({"taxonomy_rows": int(len(out_df)), "output_csv": None}, indent=2))
        else:
            if not bool(os.environ.get("RULE_TUPLES_SILENT", "0")):
                print(json.dumps({"taxonomy_rows": int(len(out_df))}, indent=2))
        # Also write Parquet to runtime.output_dir if provided
        out_dir = str(getattr(cfg.runtime, "output_dir", "") or _default_output_dir("taxonomy"))
        if out_dir:
            try:
                os.makedirs(out_dir, exist_ok=True)
            except Exception:
                pass
            try:
                pq_path = os.path.join(out_dir, "results.parquet")
                # Reuse sanitizer from classify branch
                def _sanitize_for_parquet_pdf(pdf: pd.DataFrame) -> pd.DataFrame:
                    pdf = pdf.copy()
                    def _ser(v: Any) -> Any:
                        try:
                            import numpy as _np  # type: ignore
                            if isinstance(v, _np.generic):
                                try:
                                    return v.item()
                                except Exception:
                                    pass
                            if isinstance(v, _np.ndarray):
                                try:
                                    import json as _json
                                    return _json.dumps(v.tolist(), ensure_ascii=False)
                                except Exception:
                                    return str(v)
                        except Exception:
                            pass
                        if isinstance(v, (dict, list, tuple, set)):
                            try:
                                import json as _json
                                if isinstance(v, set):
                                    v = list(v)
                                return _json.dumps(v, ensure_ascii=False)
                            except Exception:
                                return str(v)
                        return v
                    for col in list(pdf.columns):
                        try:
                            if pdf[col].dtype == object:
                                s = pdf[col].apply(_ser)
                                try:
                                    s = s.apply(lambda x: "" if x is None else x)
                                except Exception:
                                    pass
                                pdf[col] = s
                        except Exception:
                            pass
                    return pdf
                _sanitize_for_parquet_pdf(out_df).to_parquet(pq_path, index=False)
                if not bool(os.environ.get("RULE_TUPLES_SILENT", "0")):
                    print(json.dumps({"taxonomy_rows": int(len(out_df)), "output_parquet": pq_path}, indent=2))
            except Exception as e:
                if not bool(os.environ.get("RULE_TUPLES_SILENT", "0")):
                    print(json.dumps({"error": "taxonomy_parquet_write_failed", "detail": str(e)}, indent=2))
        # Log final results tables to W&B (chunks and aggregated docs)
        try:
            try:
                max_rows = int(getattr(cfg.wandb, "table_sample_rows", 1000) or 1000)
            except Exception:
                max_rows = 1000
            _wb_log_table(
                out_df,
                key="tables/taxonomy_chunks",
                prefer_cols=[
                    "article_id","article_path","chunk_id","num_chunks",
                    "chunk_label","chunk_label_name","answer","relevant_keyword",
                ],
                max_rows=max_rows,
            )
            if len(docs_df):
                _wb_log_table(
                    docs_df,
                    key="tables/taxonomy_docs",
                    prefer_cols=[
                        "article_id","article_path","predicted_category_number","predicted_category_name","num_chunks",
                    ],
                    max_rows=max_rows,
                )
        except Exception:
            pass
        return

    if stage == "verification":
        from .stages.verify import run_verification_stage
        out_df = run_verification_stage(df, cfg)
        out_path = str(getattr(cfg.runtime, "output_csv", "") or "")
        if out_path:
            try:
                out_df.to_csv(out_path, index=False)
                if not bool(os.environ.get("RULE_TUPLES_SILENT", "0")):
                    print(json.dumps({"verification_rows": int(len(out_df)), "output_csv": out_path}, indent=2))
            except Exception:
                if not bool(os.environ.get("RULE_TUPLES_SILENT", "0")):
                    print(json.dumps({"verification_rows": int(len(out_df)), "output_csv": None}, indent=2))
        else:
            if not bool(os.environ.get("RULE_TUPLES_SILENT", "0")):
                print(json.dumps({"verification_rows": int(len(out_df))}, indent=2))
        # Also write Parquet to runtime.output_dir if provided
        out_dir = str(getattr(cfg.runtime, "output_dir", "") or _default_output_dir("verification"))
        if out_dir:
            try:
                os.makedirs(out_dir, exist_ok=True)
            except Exception:
                pass
            try:
                pq_path = os.path.join(out_dir, "verification.parquet")
                # Reuse sanitizer from classify branch
                def _sanitize_for_parquet_pdf(pdf: pd.DataFrame) -> pd.DataFrame:
                    pdf = pdf.copy()
                    def _ser(v: Any) -> Any:
                        try:
                            import numpy as _np  # type: ignore
                            if isinstance(v, _np.generic):
                                try:
                                    return v.item()
                                except Exception:
                                    pass
                            if isinstance(v, _np.ndarray):
                                try:
                                    import json as _json
                                    return _json.dumps(v.tolist(), ensure_ascii=False)
                                except Exception:
                                    return str(v)
                        except Exception:
                            pass
                        if isinstance(v, (dict, list, tuple, set)):
                            try:
                                import json as _json
                                if isinstance(v, set):
                                    v = list(v)
                                return _json.dumps(v, ensure_ascii=False)
                            except Exception:
                                return str(v)
                        return v
                    for col in list(pdf.columns):
                        try:
                            if pdf[col].dtype == object:
                                s = pdf[col].apply(_ser)
                                try:
                                    s = s.apply(lambda x: "" if x is None else x)
                                except Exception:
                                    pass
                                pdf[col] = s
                        except Exception:
                            pass
                    return pdf
                _sanitize_for_parquet_pdf(out_df).to_parquet(pq_path, index=False)
                if not bool(os.environ.get("RULE_TUPLES_SILENT", "0")):
                    print(json.dumps({"verification_rows": int(len(out_df)), "output_parquet": pq_path}, indent=2))
            except Exception as e:
                if not bool(os.environ.get("RULE_TUPLES_SILENT", "0")):
                    print(json.dumps({"error": "verification_parquet_write_failed", "detail": str(e)}, indent=2))
        # W&B table for verification results (sampled)
        try:
            try:
                max_rows = int(getattr(cfg.wandb, "table_sample_rows", 1000) or 1000)
            except Exception:
                max_rows = 1000
            _wb_log_table(
                out_df,
                key="tables/verification_chunks",
                prefer_cols=[
                    "article_id","article_path","chunk_id","num_chunks",
                    "chunk_label","ver_verified_chunk","ver_sim_max","ver_nli_ent_max",
                ],
                max_rows=max_rows,
            )
        except Exception:
            pass
        return

    if stage == "pipeline":
        # Run each stage via child Hydra/Submitit jobs by default; optionally run locally when pipeline_local=true.
        def _format_override(k: str, v: Any) -> str:
            if isinstance(v, bool):
                vv = "true" if v else "false"
            else:
                vv = str(v)
            return f"{k}={vv}"

        def _run_stage_subprocess(stage_name: str, overrides: Dict[str, Any], group_id: str) -> None:
            args: List[str] = [sys.executable, "-m", "pipelines.uair.cli"]
            try:
                child_launcher = str(getattr(cfg.runtime, "child_launcher", "") or "")
            except Exception:
                child_launcher = ""
            run_local = bool(getattr(cfg.runtime, "pipeline_local", False))
            if (child_launcher and (not run_local)):
                args.append("-m")
                args.append(f"hydra/launcher={child_launcher}")
                try:
                    wckey = getattr(cfg.runtime, "wckey", None)
                except Exception:
                    wckey = None
                if not wckey:
                    try:
                        wckey = os.environ.get("UAIR_WCKEY")
                    except Exception:
                        wckey = None
                if wckey and str(wckey).strip() != "":
                    args.append(_format_override("hydra.launcher.additional_parameters.wckey", str(wckey)))
            args.append(_format_override("runtime.stage", stage_name))
            # Pass wandb.group explicitly so Hydra picks it up even without env
            overrides = {**overrides, "wandb.group": group_id}
            for k, v in overrides.items():
                args.append(_format_override(k, v))
            if not bool(os.environ.get("RULE_TUPLES_SILENT", "0")):
                try:
                    printable = " ".join(shlex.quote(a) for a in args)
                    print(json.dumps({"launch": stage_name, "cmd": printable, "wandb_group": group_id}, indent=2))
                except Exception:
                    pass
            if (child_launcher and (not run_local)):
                env = os.environ.copy()
                env["WANDB_GROUP"] = group_id
                subprocess.run(args, check=True, env=env)
            else:
                # Local in-process run using Hydra app entry
                from .cli import main as _main
                # Reconstruct Hydra-style args for in-process call
                hydra_args = []
                for k, v in overrides.items():
                    if isinstance(v, bool):
                        vv = "true" if v else "false"
                    else:
                        vv = str(v)
                    hydra_args.append(f"{k}={vv}")
                hydra_args.append(f"runtime.stage={stage_name}")
                # Set group in env for child
                os.environ["WANDB_GROUP"] = group_id
                # Call Hydra main with overrides
                _main.callback(hydra_args) if hasattr(_main, 'callback') else _main()

        # Prepare output dirs
        base_out = str(getattr(cfg.runtime, "output_dir", "") or _default_output_dir("pipeline"))
        try:
            os.makedirs(base_out, exist_ok=True)
        except Exception:
            pass
        classify_dir = os.path.join(base_out, "classify")
        taxonomy_dir = os.path.join(base_out, "taxonomy")
        verify_dir = os.path.join(base_out, "verification")
        try:
            os.makedirs(classify_dir, exist_ok=True)
            os.makedirs(taxonomy_dir, exist_ok=True)
            os.makedirs(verify_dir, exist_ok=True)
        except Exception:
            pass

        # Group id shared by child stage runs
        try:
            group_id = str(getattr(cfg.wandb, "group", None) or os.environ.get("WANDB_GROUP") or uuid4().hex)
        except Exception:
            group_id = uuid4().hex

        # Stage 1: classify
        cls_overrides: Dict[str, Any] = {
            "runtime.output_dir": classify_dir,
            "data.parquet_path": str(getattr(cfg.data, "parquet_path")),
        }
        # Propagate debug and sample_n so child classify behavior matches the parent
        try:
            cls_overrides["runtime.debug"] = bool(getattr(cfg.runtime, "debug", False))
        except Exception:
            pass
        try:
            _sn = getattr(cfg.runtime, "sample_n", None)
            if _sn is not None:
                cls_overrides["runtime.sample_n"] = int(_sn)
        except Exception:
            pass
        try:
            if hasattr(cfg.runtime, "use_llm_classify"):
                cls_overrides["runtime.use_llm_classify"] = bool(getattr(cfg.runtime, "use_llm_classify"))
        except Exception:
            pass
        # Log classify stage start and completion to W&B if available
        try:
            cls_t0 = _time.time()
            _wb_log({
                "pipeline/classify_started_at": float(cls_t0),
            })
        except Exception:
            pass
        try:
            _run_stage_subprocess("classify", cls_overrides, group_id)
            try:
                cls_dt = float(_time.time() - cls_t0)
            except Exception:
                cls_dt = None  # type: ignore
            _wb_log({
                "pipeline/classify_done": 1,
                "pipeline/classify_failed": 0,
                **({"pipeline/classify_duration_s": cls_dt} if cls_dt is not None else {}),
            })
        except Exception:
            try:
                cls_dt = float(_time.time() - cls_t0)
            except Exception:
                cls_dt = None  # type: ignore
            _wb_log({
                "pipeline/classify_done": 0,
                "pipeline/classify_failed": 1,
                **({"pipeline/classify_duration_s": cls_dt} if cls_dt is not None else {}),
            })
            raise

        # Stage 2: taxonomy consumes classify_dir
        # Prefer relevant-only parquet if present; otherwise all-results parquet; fallback to directory
        try:
            rel_pq = os.path.join(classify_dir, "classify_relevant.parquet")
            all_pq = os.path.join(classify_dir, "classify_all.parquet")
            if os.path.exists(rel_pq):
                tax_input_path = rel_pq
            elif os.path.exists(all_pq):
                tax_input_path = all_pq
            else:
                tax_input_path = classify_dir
        except Exception:
            tax_input_path = classify_dir
        tax_overrides: Dict[str, Any] = {
            "runtime.output_dir": taxonomy_dir,
            "data.parquet_path": tax_input_path,
        }
        # Log taxonomy stage start and completion to W&B if available
        try:
            tax_t0 = _time.time()
            _wb_log({
                "pipeline/taxonomy_started_at": float(tax_t0),
            })
        except Exception:
            pass
        try:
            _run_stage_subprocess("taxonomy", tax_overrides, group_id)
            try:
                tax_dt = float(_time.time() - tax_t0)
            except Exception:
                tax_dt = None  # type: ignore
            _wb_log({
                "pipeline/taxonomy_done": 1,
                "pipeline/taxonomy_failed": 0,
                **({"pipeline/taxonomy_duration_s": tax_dt} if tax_dt is not None else {}),
            })
        except Exception:
            try:
                tax_dt = float(_time.time() - tax_t0)
            except Exception:
                tax_dt = None  # type: ignore
            _wb_log({
                "pipeline/taxonomy_done": 0,
                "pipeline/taxonomy_failed": 1,
                **({"pipeline/taxonomy_duration_s": tax_dt} if tax_dt is not None else {}),
            })
            raise

        # Stage 3: verification consumes taxonomy_dir/results.parquet when present
        try:
            tax_res = os.path.join(taxonomy_dir, "results.parquet")
            if os.path.exists(tax_res):
                ver_input_path = tax_res
            else:
                ver_input_path = taxonomy_dir
        except Exception:
            ver_input_path = taxonomy_dir
        ver_overrides: Dict[str, Any] = {
            "runtime.output_dir": verify_dir,
            "data.parquet_path": ver_input_path,
        }
        # Log verification stage start and completion to W&B if available
        try:
            ver_t0 = _time.time()
            _wb_log({
                "pipeline/verification_started_at": float(ver_t0),
            })
        except Exception:
            pass
        try:
            _run_stage_subprocess("verification", ver_overrides, group_id)
            try:
                ver_dt = float(_time.time() - ver_t0)
            except Exception:
                ver_dt = None  # type: ignore
            _wb_log({
                "pipeline/verification_done": 1,
                "pipeline/verification_failed": 0,
                **({"pipeline/verification_duration_s": ver_dt} if ver_dt is not None else {}),
            })
        except Exception:
            try:
                ver_dt = float(_time.time() - ver_t0)
            except Exception:
                ver_dt = None  # type: ignore
            _wb_log({
                "pipeline/verification_done": 0,
                "pipeline/verification_failed": 1,
                **({"pipeline/verification_duration_s": ver_dt} if ver_dt is not None else {}),
            })
            raise

        if not bool(os.environ.get("RULE_TUPLES_SILENT", "0")):
            print(json.dumps({
                "pipeline": {
                    "output_dir": base_out,
                    "classify_dir": classify_dir,
                    "taxonomy_dir": taxonomy_dir,
                    "verification_dir": verify_dir,
                    "wandb_group": group_id,
                }
            }, indent=2))
        return

    raise ValueError(f"Unknown runtime.stage: {stage}")


