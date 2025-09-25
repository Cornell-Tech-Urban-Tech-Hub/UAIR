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
from .logging_util import (
    is_enabled as _wb_enabled,
    start_run as _wb_start,
    finish_run as _wb_finish,
    log_metrics as _wb_log,
    log_table as _wb_log_table,
)
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
                    "classify": {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "kw_marked": 0, "kw_checked": 0},
                    "decompose": {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "kw_marked": 0, "kw_checked": 0},
                    "taxonomy": {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "kw_marked": 0, "kw_checked": 0},
                    "overall": {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "kw_marked": 0, "kw_checked": 0},
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

            def record(self, stage: str, calls: Any = 0, prompt_tokens: Any = 0, completion_tokens: Any = 0, total_tokens: Any = 0, kw_marked: Any = 0, kw_checked: Any = 0) -> None:
                st = str(stage or "overall").strip().lower()
                if st not in self._totals:
                    st = "overall"
                c = self._coerce_int(calls)
                p = self._coerce_int(prompt_tokens)
                q = self._coerce_int(completion_tokens)
                t = self._coerce_int(total_tokens)
                km = self._coerce_int(kw_marked)
                kc = self._coerce_int(kw_checked)
                self._totals[st]["calls"] += c
                self._totals[st]["prompt_tokens"] += p
                self._totals[st]["completion_tokens"] += q
                self._totals[st]["total_tokens"] += t
                self._totals[st]["kw_marked"] += km
                self._totals[st]["kw_checked"] += kc
                # Mirror into overall
                if st != "overall":
                    self._totals["overall"]["calls"] += c
                    self._totals["overall"]["prompt_tokens"] += p
                    self._totals["overall"]["completion_tokens"] += q
                    # If total_tokens isn't provided, derive when possible
                    if t == 0 and (p or q):
                        t = p + q
                    self._totals["overall"]["total_tokens"] += t
                    self._totals["overall"]["kw_marked"] += km
                    self._totals["overall"]["kw_checked"] += kc

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
    # If dataset is empty, allow it to pass through (downstream will no-op)
    try:
        if len(df) == 0:
            return df
    except Exception:
        pass
    # Ensure a usable text column exists: prefer article_text; allow chunk_text for downstream stages
    if "article_text" not in df.columns and "chunk_text" not in df.columns:
        raise RuntimeError("Parquet missing required text column (article_text) or chunk_text; configure data.columns.article_text or provide chunk_text")

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

    # Choose IO mode: Ray Dataset streaming or pandas DataFrame (only for non-pipeline stages)
    use_streaming = False
    ds = None
    # Ensure df is defined for all stages (incl. topic) to avoid NameError in fallbacks/prints
    df = None  # type: ignore[assignment]
    try:
        _stream_flag = bool(getattr(cfg.runtime, "streaming_io", False))
        _stream_allowed = stage in ("classify", "taxonomy", "verification")
        use_streaming = (_stream_flag and _stream_allowed and _RAY_OK_E)
    except Exception:
        use_streaming = False
    if use_streaming:
        try:
            import ray  # type: ignore
            if not ray.is_initialized():
                # Use a stable namespace so detached actors are discoverable across sessions
                try:
                    _ns = os.environ.get("RAY_NAMESPACE") or os.environ.get("WANDB_GROUP") or "uair"
                except Exception:
                    _ns = "uair"
                try:
                    ray.init(log_to_driver=True, namespace=str(_ns))
                except Exception:
                    # Fallback without namespace if version doesn't support it
                    ray.init(log_to_driver=True)
            try:
                ctx = ray.data.DataContext.get_current()
                ctx.enable_progress_bars = False
                # Encourage smaller blocks to improve time-to-first-output
                ctx.target_min_block_size = 1 * 1024 * 1024
                ctx.target_max_block_size = 64 * 1024 * 1024
            except Exception:
                pass
            ds = ray.data.read_parquet(parquet_path)
            # Ensure canonical columns exist; add renamed copies when needed
            col_map = {
                columns.get("article_text", "article_text"): "article_text",
                columns.get("article_path", "article_path"): "article_path",
                columns.get("country", "country"): "country",
                columns.get("year", "year"): "year",
                columns.get("article_id", "article_id"): "article_id",
            }
            def _ensure_canon(row: Dict[str, Any]) -> Dict[str, Any]:
                out = dict(row)
                for src, dst in col_map.items():
                    try:
                        if dst not in out and src in row:
                            out[dst] = row.get(src)
                    except Exception:
                        pass
                return out
            try:
                ds = ds.map(_ensure_canon)
            except Exception:
                pass
            if debug and isinstance(sample_n, int) and sample_n:
                try:
                    ds = ds.limit(max(1, int(sample_n)))
                except Exception:
                    pass
        except Exception:
            ds = None
            use_streaming = False
    if (not use_streaming) and (stage in ("classify", "taxonomy", "verification")):
        # Pandas control-plane path for non-pipeline stages only
        df = _load_parquet_dataset(parquet_path, columns, debug=debug, sample_n=sample_n)

    if not bool(os.environ.get("RULE_TUPLES_SILENT", "0")):
        try:
            loaded = (int(len(df)) if not use_streaming else None)
        except Exception:
            loaded = None
        try:
            cols = (list(df.columns)[:12] if not use_streaming else None)
        except Exception:
            cols = None
        print(json.dumps({
            "loaded_rows": loaded,
            "stage": stage,
            "columns": cols,
            "streaming": bool(use_streaming),
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

    # W&B enable flag
    use_wandb = _wb_enabled(cfg)

    if stage == "classify":
        from .stages.classify import run_classification_stage
        # Stage-scoped W&B run (Option B)
        try:
            run_cfg = {
                "stage": "classify",
                "parquet_path": parquet_path,
                "data_columns_map": columns,
                "sample_n": sample_n,
                "output_csv": str(getattr(cfg.runtime, "output_csv", "") or ""),
                "use_llm_classify": bool(getattr(cfg.runtime, "use_llm_classify", False)),
                "prefilter_mode": str(getattr(cfg.runtime, "prefilter_mode", "pre_gating")),
                "model_source": str(getattr(cfg.model, "model_source", "")),
                "engine_kwargs": dict(getattr(cfg.model, "engine_kwargs", {})),
                "batch_size": int(getattr(cfg.model, "batch_size", 16) or 16),
                "concurrency": int(getattr(cfg.model, "concurrency", 1) or 1),
                "sampling_params": dict(getattr(cfg, "sampling_params", {})),
                "python_version": sys.version.split()[0],
                "os": platform.platform(),
                "hostname": socket.gethostname(),
            }
        except Exception:
            run_cfg = {"stage": "classify"}
        if use_wandb:
            _wb_start(cfg, "classify", run_config=run_cfg)
        # Support streaming IO when enabled
        in_obj = None
        try:
            in_obj = ds if ("ds" in locals() and ds is not None) else df
        except Exception:
            in_obj = df
        # Create a named usage aggregator actor only when streaming is enabled
        named_usage_actor = None
        if use_streaming:
            try:
                if use_wandb and _RAY_OK_E:
                    import ray  # type: ignore
                    # Ensure Ray is initialized with a stable namespace when using detached actors
                    try:
                        if not ray.is_initialized():
                            _ns = os.environ.get("RAY_NAMESPACE") or os.environ.get("WANDB_GROUP") or "uair"
                            try:
                                ray.init(log_to_driver=True, namespace=str(_ns))
                            except Exception:
                                ray.init(log_to_driver=True)
                    except Exception:
                        pass
                    try:
                        named_usage_actor = ray.get_actor("uair_usage_agg")
                    except Exception:
                        try:
                            named_usage_actor = _UsageAggregator.options(name="uair_usage_agg", lifetime="detached").remote()  # type: ignore
                        except Exception:
                            named_usage_actor = None
            except Exception:
                named_usage_actor = None
        out_any = run_classification_stage(in_obj, cfg)
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
        # If streaming dataset output, write Parquet via Ray Data
        is_ds_out = False
        try:
            import ray  # type: ignore
            is_ds_out = hasattr(out_any, "write_parquet") and hasattr(out_any, "count")
        except Exception:
            is_ds_out = False
        if is_ds_out:
            # Accumulate usage via actor during streaming writes (no background thread)
            usage_actor = None
            try:
                if use_wandb and _RAY_OK_E:
                    usage_actor = _UsageAggregator.remote()  # type: ignore
            except Exception:
                usage_actor = None
            # Attach accumulator map to stream per-batch token usage into the actor
            out_ds = out_any
            if usage_actor is not None:
                def _accumulate_usage(pdf: pd.DataFrame) -> pd.DataFrame:
                    try:
                        s_prompt = pd.to_numeric(pdf.get("token_usage_prompt", 0), errors="coerce").fillna(0).astype("int64").sum()
                    except Exception:
                        s_prompt = 0
                    try:
                        s_output = pd.to_numeric(pdf.get("token_usage_output", 0), errors="coerce").fillna(0).astype("int64").sum()
                    except Exception:
                        s_output = 0
                    try:
                        s_total = pd.to_numeric(pdf.get("token_usage_total", 0), errors="coerce").fillna(0).astype("int64").sum()
                    except Exception:
                        s_total = 0
                    # Keyword counters
                    try:
                        s_kw_checked = int(len(pdf))
                    except Exception:
                        s_kw_checked = 0
                    try:
                        if "relevant_keyword" in pdf.columns:
                            s_kw_marked = int(pdf["relevant_keyword"].astype(bool).sum())
                        else:
                            s_kw_marked = 0
                    except Exception:
                        s_kw_marked = 0
                    try:
                        usage_actor.record.remote("classify", calls=len(pdf), prompt_tokens=int(s_prompt), completion_tokens=int(s_output), total_tokens=int(s_total), kw_marked=int(s_kw_marked), kw_checked=int(s_kw_checked))  # type: ignore
                    except Exception:
                        pass
                    return pdf
                try:
                    out_ds = out_ds.map_batches(_accumulate_usage, batch_format="pandas", batch_size=256)
                except Exception:
                    pass
            out_dir = str(getattr(cfg.runtime, "output_dir", "") or _default_output_dir("classify"))
            if out_dir:
                try:
                    os.makedirs(out_dir, exist_ok=True)
                except Exception:
                    pass
                try:
                    out_ds = out_ds.map_batches(_sanitize_for_parquet_pdf, batch_format="pandas", batch_size=256)
                    pq_all = os.path.join(out_dir, "classify_all.parquet")
                    out_ds.write_parquet(pq_all)
                    # Relevant-only
                    pq_rel = None
                    try:
                        rel_ds = out_ds.filter(lambda r: bool(r.get("is_relevant", False)))
                        pq_rel = os.path.join(out_dir, "classify_relevant.parquet")
                        rel_ds.write_parquet(pq_rel)
                    except Exception:
                        pq_rel = None
                    if not bool(os.environ.get("RULE_TUPLES_SILENT", "0")):
                        print(json.dumps({
                            "classified_rows": None,
                            "output_parquet_all": pq_all,
                            "output_parquet_relevant": pq_rel,
                        }, indent=2))
                except Exception as e:
                    if not bool(os.environ.get("RULE_TUPLES_SILENT", "0")):
                        print(json.dumps({"error": "classify_parquet_write_failed", "detail": str(e)}, indent=2))
            # Final usage snapshot logging (once)
            try:
                if usage_actor is not None and use_wandb:
                    import ray as __ray  # type: ignore
                    snap = __ray.get(usage_actor.snapshot.remote())  # type: ignore
                    cls = (snap.get("classify") or {}) if isinstance(snap, dict) else {}
                    payload2 = {
                        "usage/classify/calls": int(cls.get("calls", 0) or 0),
                        "usage/classify/prompt_tokens": int(cls.get("prompt_tokens", 0) or 0),
                        "usage/classify/completion_tokens": int(cls.get("completion_tokens", 0) or 0),
                        "usage/classify/total_tokens": int(cls.get("total_tokens", 0) or 0),
                        "keyword/total_checked": int(cls.get("kw_checked", 0) or 0),
                        "keyword/marked": int(cls.get("kw_marked", 0) or 0),
                    }
                    try:
                        kc = float(payload2.get("keyword/total_checked") or 0)
                        km = float(payload2.get("keyword/marked") or 0)
                        if kc > 0:
                            payload2["keyword/marked_ratio"] = (km / kc)
                    except Exception:
                        pass
                    _wb_log(cfg, payload2)  # type: ignore[arg-type]
            except Exception:
                pass
            if use_wandb:
                _wb_finish(cfg)
            return

        # Pandas output path
        out_df = out_any
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
            _wb_log(cfg, payload)  # type: ignore[arg-type]
        except Exception:
            pass
        # Log final results table to W&B (sampled)
        try:
            try:
                max_rows = int(getattr(cfg.wandb, "table_sample_rows", 1000) or 1000)
            except Exception:
                max_rows = 1000
            _wb_log_table(cfg, out_df,
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
                        _wb_log_table(cfg, rel_df,
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
        if use_wandb:
            _wb_finish(cfg)
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
        # Stage-scoped W&B run (Option B)
        try:
            run_cfg = {
                "stage": "taxonomy",
                "parquet_path": parquet_path,
                "data_columns_map": columns,
                "sample_n": sample_n,
                "output_csv": str(getattr(cfg.runtime, "output_csv", "") or ""),
                "prefilter_mode": str(getattr(cfg.runtime, "prefilter_mode", "pre_gating")),
                "model_source": str(getattr(cfg.model, "model_source", "")),
                "engine_kwargs": dict(getattr(cfg.model, "engine_kwargs", {})),
                "batch_size": int(getattr(cfg.model, "batch_size", 16) or 16),
                "concurrency": int(getattr(cfg.model, "concurrency", 1) or 1),
                "sampling_params": dict(getattr(cfg, "sampling_params", {})),
                "python_version": sys.version.split()[0],
                "os": platform.platform(),
                "hostname": socket.gethostname(),
            }
        except Exception:
            run_cfg = {"stage": "taxonomy"}
        if use_wandb:
            _wb_start(cfg, "taxonomy", run_config=run_cfg)
        in_obj = None
        try:
            in_obj = ds if ("ds" in locals() and ds is not None) else df
        except Exception:
            in_obj = df
        out_any = run_taxonomy_stage(in_obj, cfg)
        # If streaming dataset output, write Parquet via Ray Data
        is_ds_out = False
        try:
            import ray  # type: ignore
            is_ds_out = hasattr(out_any, "write_parquet") and hasattr(out_any, "count")
        except Exception:
            is_ds_out = hasattr(out_any, "write_parquet") and hasattr(out_any, "count")
        if is_ds_out:
            out_ds = out_any
            # Write chunks-level results to Parquet directory
            out_dir = str(getattr(cfg.runtime, "output_dir", "") or _default_output_dir("taxonomy"))
            if out_dir:
                try:
                    os.makedirs(out_dir, exist_ok=True)
                except Exception:
                    pass
                # Best-effort sanitization for Arrow/Parquet
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
                try:
                    out_ds = out_ds.map_batches(_sanitize_for_parquet_pdf, batch_format="pandas", batch_size=256)
                except Exception:
                    pass
                try:
                    pq_path = os.path.join(out_dir, "results.parquet")
                    out_ds.write_parquet(pq_path)
                    if not bool(os.environ.get("RULE_TUPLES_SILENT", "0")):
                        print(json.dumps({"taxonomy_rows": None, "output_parquet": pq_path}, indent=2))
                except Exception as e:
                    if not bool(os.environ.get("RULE_TUPLES_SILENT", "0")):
                        print(json.dumps({"error": "taxonomy_parquet_write_failed", "detail": str(e)}, indent=2))
            # Sampled W&B tables are skipped for streaming path to avoid driver collection
            return
        # Pandas output path
        out_df = out_any
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
                _wb_log(cfg, {"taxonomy/none_fraction": (float(none_ct) / float(total_tx))})  # type: ignore[arg-type]
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
            _wb_log_table(cfg, out_df,
                key="tables/taxonomy_chunks",
                prefer_cols=[
                    "article_id","article_path","chunk_id","num_chunks",
                    "chunk_label","chunk_label_name","answer","relevant_keyword",
                ],
                max_rows=max_rows,
            )
            if len(docs_df):
                _wb_log_table(cfg, docs_df,
                    key="tables/taxonomy_docs",
                    prefer_cols=[
                        "article_id","article_path","predicted_category_number","predicted_category_name","num_chunks",
                    ],
                    max_rows=max_rows,
                )
        except Exception:
            pass
        if use_wandb:
            _wb_finish(cfg)
        return

    if stage == "topic":
        from .stages.topic import run_topic_stage
        try:
            run_cfg = {"stage": "topic", "parquet_path": parquet_path}
        except Exception:
            run_cfg = {"stage": "topic"}
        if use_wandb:
            _wb_start(cfg, "topic", run_config=run_cfg)
        in_obj = None
        try:
            in_obj = ds if ("ds" in locals() and ds is not None) else df
        except Exception:
            in_obj = df
        # If input looks like a directory or classify output dir, try to read classify_relevant.parquet
        try:
            p = parquet_path
            if os.path.isdir(p):
                # Expect classify outputs at the root of the provided directory
                cand_rel = os.path.join(p, "classify_relevant.parquet")
                cand_all = os.path.join(p, "classify_all.parquet")
                if os.path.exists(cand_rel):
                    in_obj = pd.read_parquet(cand_rel)
                elif os.path.exists(cand_all):
                    in_obj = pd.read_parquet(cand_all)
            elif os.path.isfile(p):
                # Direct parquet file input
                in_obj = pd.read_parquet(p)
        except Exception:
            pass
        out_any = run_topic_stage(in_obj, cfg)
        out_df = out_any if isinstance(out_any, pd.DataFrame) else None
        out_path = str(getattr(cfg.runtime, "output_csv", "") or "")
        if out_path:
            try:
                out_df.to_csv(out_path, index=False)
                if not bool(os.environ.get("RULE_TUPLES_SILENT", "0")):
                    print(json.dumps({"topic_rows": int(len(out_df)), "output_csv": out_path}, indent=2))
            except Exception:
                if not bool(os.environ.get("RULE_TUPLES_SILENT", "0")):
                    print(json.dumps({"topic_rows": int(len(out_df)), "output_csv": None}, indent=2))
        else:
            if not bool(os.environ.get("RULE_TUPLES_SILENT", "0")):
                print(json.dumps({"topic_rows": int(len(out_df))}, indent=2))
        out_dir = str(getattr(cfg.runtime, "output_dir", "") or _default_output_dir("topic"))
        if out_dir:
            try:
                os.makedirs(out_dir, exist_ok=True)
            except Exception:
                pass
            try:
                pq_path = os.path.join(out_dir, "docs_topics.parquet")
                out_df.to_parquet(pq_path, index=False)
                if not bool(os.environ.get("RULE_TUPLES_SILENT", "0")):
                    print(json.dumps({"topic_rows": int(len(out_df)), "output_parquet": pq_path}, indent=2))
            except Exception as e:
                if not bool(os.environ.get("RULE_TUPLES_SILENT", "0")):
                    print(json.dumps({"error": "topic_parquet_write_failed", "detail": str(e)}, indent=2))
        if use_wandb:
            try:
                _wb_log_table(cfg, out_df, key="tables/topic_docs", prefer_cols=[
                    "unit_id","article_id","article_path","topic_id","topic_prob","topic_top_terms","article_keywords","plot_x","plot_y"
                ], max_rows=int(getattr(cfg.wandb, "table_sample_rows", 1000) or 1000))
            except Exception:
                pass
            # Cluster-level summary table (one row per topic_id) with top terms
            try:
                if out_df is not None and "topic_id" in out_df.columns:
                    import pandas as _pd  # type: ignore
                    def _first_nonnull(s):
                        try:
                            for v in s:
                                if v is not None:
                                    return v
                        except Exception:
                            pass
                        return None
                    grp = out_df.groupby("topic_id", dropna=False)
                    rows = []
                    for tid, g in grp:
                        try:
                            ex = g.iloc[0]
                        except Exception:
                            ex = None
                        rows.append({
                            "topic_id": int(tid) if tid is not None else -1,
                            "topic_size": int(len(g)),
                            "topic_top_terms": _first_nonnull(g.get("topic_top_terms", [])),
                            "example_article_id": (ex.get("article_id") if ex is not None else None),
                            "example_article_path": (ex.get("article_path") if ex is not None else None),
                        })
                    clust_df = _pd.DataFrame(rows)
                    _wb_log_table(cfg, clust_df, key="tables/topic_clusters", prefer_cols=[
                        "topic_id","topic_size","topic_top_terms","example_article_id","example_article_path"
                    ], max_rows=1000)
            except Exception:
                pass
            # Plot and log a 2D cluster scatter when coordinates are present
            try:
                if out_df is not None and ("plot_x" in out_df.columns and "plot_y" in out_df.columns):
                    from .stages.topic_plot import log_cluster_scatter_to_wandb, log_cluster_scatter_plotly_to_wandb  # type: ignore
                    try:
                        log_cluster_scatter_to_wandb(out_df, cfg, title="topic_cluster_map")
                    except Exception:
                        pass
                    try:
                        log_cluster_scatter_plotly_to_wandb(out_df, cfg, title="topic_cluster_map")
                    except Exception:
                        pass
            except Exception:
                pass
            # Log basic topic metrics
            try:
                n_units = int(len(out_df)) if out_df is not None else 0
                noise_ct = 0
                num_clusters = 0
                try:
                    vc = out_df["topic_id"].value_counts(dropna=False)
                    noise_ct = int(vc.get(-1, 0)) if hasattr(vc, "get") else (int(vc[-1]) if (-1 in getattr(vc, 'index', [])) else 0)
                    try:
                        uniq = out_df["topic_id"].astype(int).unique().tolist()
                        num_clusters = int(len([x for x in uniq if int(x) != -1]))
                    except Exception:
                        num_clusters = 0
                except Exception:
                    pass
                payload = {
                    "topic/units": n_units,
                    "topic/noise_count": noise_ct,
                    "topic/noise_fraction": (float(noise_ct) / float(n_units)) if n_units else 0.0,
                    "topic/num_clusters": num_clusters,
                }
                _wb_log(cfg, payload)  # type: ignore[arg-type]
            except Exception:
                pass
            _wb_finish(cfg)
        return

    if stage == "verification":
        from .stages.verify import run_verification_stage
        # Stage-scoped W&B run (Option B)
        try:
            run_cfg = {
                "stage": "verification",
                "parquet_path": parquet_path,
                "data_columns_map": columns,
                "sample_n": sample_n,
                "output_csv": str(getattr(cfg.runtime, "output_csv", "") or ""),
                "model_source": str(getattr(cfg.model, "model_source", "")),
                "engine_kwargs": dict(getattr(cfg.model, "engine_kwargs", {})),
                "batch_size": int(getattr(cfg.model, "batch_size", 16) or 16),
                "concurrency": int(getattr(cfg.model, "concurrency", 1) or 1),
                "sampling_params": dict(getattr(cfg, "sampling_params", {})),
                "python_version": sys.version.split()[0],
                "os": platform.platform(),
                "hostname": socket.gethostname(),
            }
        except Exception:
            run_cfg = {"stage": "verification"}
        if use_wandb:
            _wb_start(cfg, "verification", run_config=run_cfg)
        in_obj = None
        try:
            in_obj = ds if ("ds" in locals() and ds is not None) else df
        except Exception:
            in_obj = df
        out_any = run_verification_stage(in_obj, cfg)
        # If streaming dataset output, write Parquet via Ray Data
        is_ds_out = False
        try:
            import ray  # type: ignore
            is_ds_out = hasattr(out_any, "write_parquet") and hasattr(out_any, "count")
        except Exception:
            is_ds_out = hasattr(out_any, "write_parquet") and hasattr(out_any, "count")
        if is_ds_out:
            out_ds = out_any
            out_dir = str(getattr(cfg.runtime, "output_dir", "") or _default_output_dir("verification"))
            if out_dir:
                try:
                    os.makedirs(out_dir, exist_ok=True)
                except Exception:
                    pass
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
                try:
                    out_ds = out_ds.map_batches(_sanitize_for_parquet_pdf, batch_format="pandas", batch_size=256)
                except Exception:
                    pass
                try:
                    pq_path = os.path.join(out_dir, "verification.parquet")
                    out_ds.write_parquet(pq_path)
                    if not bool(os.environ.get("RULE_TUPLES_SILENT", "0")):
                        print(json.dumps({"verification_rows": None, "output_parquet": pq_path}, indent=2))
                except Exception as e:
                    if not bool(os.environ.get("RULE_TUPLES_SILENT", "0")):
                        print(json.dumps({"error": "verification_parquet_write_failed", "detail": str(e)}, indent=2))
            # W&B table logging is skipped for streaming to avoid driver collection
            return
        # Pandas output path
        out_df = out_any
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
            base_cols = [
                "article_id","article_path","chunk_id","num_chunks",
                "chunk_label","ver_verified_chunk","ver_sim_max","ver_top_sent","ver_nli_ent_max",
            ]
            _wb_log_table(cfg, out_df, key="tables/verification_chunks", prefer_cols=base_cols, max_rows=max_rows)
            # Optional debug columns
            try:
                dbg = str(os.environ.get("UAIR_VERIFY_DEBUG", "")).strip().lower() in ("1","true","yes","on")
            except Exception:
                dbg = False
            if dbg:
                debug_cols = base_cols + [
                    "chunk_label_name","ver_nli_label_max","ver_top_sent","ver_evidence_topk",
                ]
                _wb_log_table(cfg, out_df, key="tables/verification_debug_chunks", prefer_cols=debug_cols, max_rows=max_rows)
        except Exception:
            pass
        if use_wandb:
            _wb_finish(cfg)
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
            # Prefer mapping runtime.child_launchers[stage]; fallback to runtime.child_launcher
            # Resolve per-stage launcher from DictConfig or dict; fallback to runtime.child_launcher
            try:
                cl_map = getattr(cfg.runtime, "child_launchers", None)
            except Exception:
                cl_map = None
            stage_launcher = None
            if cl_map is not None:
                try:
                    # OmegaConf DictConfig supports attribute access
                    stage_launcher = getattr(cl_map, stage_name)
                except Exception:
                    stage_launcher = None
                if stage_launcher is None:
                    try:
                        stage_launcher = cl_map.get(stage_name)  # type: ignore[attr-defined]
                    except Exception:
                        stage_launcher = None
            try:
                child_launcher = str(stage_launcher or getattr(cfg.runtime, "child_launcher", "") or "")
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
                    # Mask any sensitive args when printing
                    def _mask_arg(a: str) -> str:
                        if a.startswith("wandb.api_key="):
                            return "wandb.api_key=***"
                        return a
                    printable = " ".join(shlex.quote(_mask_arg(a)) for a in args)
                    print(json.dumps({"launch": stage_name, "cmd": printable, "wandb_group": group_id}, indent=2))
                except Exception:
                    pass
            if (child_launcher and (not run_local)):
                env = os.environ.copy()
                env["WANDB_GROUP"] = group_id
                # Ensure W&B is not disabled in child env and API key is propagated
                try:
                    if str(getattr(cfg.wandb, "enabled", True)).lower() in ("true", "1", "yes"):
                        env.pop("WANDB_DISABLED", None)
                        env.setdefault("WANDB_MODE", "online")
                except Exception:
                    pass
                # Propagate common W&B auth/env if present in parent
                try:
                    for _k in ("WANDB_API_KEY", "WANDB_BASE_URL", "WANDB_ENTITY", "WANDB_PROJECT"):
                        if os.environ.get(_k) and not env.get(_k):
                            env[_k] = os.environ.get(_k)  # type: ignore
                except Exception:
                    pass
                # no WANDB_BASE_URL propagation
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

        # Start a W&B pipeline monitor run (coordinator)
        if use_wandb:
            try:
                run_cfg = {
                    "stage": "pipeline",
                    "parquet_path": str(getattr(cfg.data, "parquet_path")),
                    "output_dir": base_out,
                    "classify_dir": classify_dir,
                    "taxonomy_dir": taxonomy_dir,
                    "verification_dir": verify_dir,
                    "pipeline_topic": bool(getattr(cfg.runtime, "pipeline_topic", False)),
                    "streaming_io": bool(getattr(cfg.runtime, "streaming_io", False)),
                    "child_launcher": str(getattr(cfg.runtime, "child_launcher", "")),
                    "python_version": sys.version.split()[0],
                    "os": platform.platform(),
                    "hostname": socket.gethostname(),
                    "wandb_group": group_id,
                }
            except Exception:
                run_cfg = {"stage": "pipeline", "wandb_group": group_id}
            try:
                _wb_start(cfg, "pipeline", run_config=run_cfg)
            except Exception:
                pass

        # Stage 1: classify
        cls_overrides: Dict[str, Any] = {
            "runtime.output_dir": classify_dir,
            "data.parquet_path": str(getattr(cfg.data, "parquet_path")),
        }
        # Do NOT pass API keys via args; rely on env or prior login
        # Propagate debug and sample_n so child classify behavior matches the parent
        try:
            cls_overrides["runtime.debug"] = bool(getattr(cfg.runtime, "debug", False))
        except Exception:
            pass
        # Propagate streaming IO preference to child classify
        try:
            cls_overrides["runtime.streaming_io"] = bool(getattr(cfg.runtime, "streaming_io", False))
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
        # Ensure W&B is explicitly enabled for child classify
        try:
            cls_overrides["wandb.enabled"] = True
        except Exception:
            pass
        # Log pipeline stage timings to console only (no W&B in pipeline coordinator)
        cls_t0 = _time.time()
        try:
            _run_stage_subprocess("classify", cls_overrides, group_id)
            cls_dt = float(_time.time() - cls_t0)
        except Exception:
            cls_dt = float(_time.time() - cls_t0)
            raise
        # Log classify stage duration to W&B
        if use_wandb:
            try:
                _wb_log(cfg, {"pipeline/classify_seconds": float(cls_dt)})  # type: ignore[arg-type]
            except Exception:
                pass

        # Optional Stage 2: topic consumes classify_dir when pipeline_topic=true; otherwise taxonomy
        pipeline_topic = bool(getattr(cfg.runtime, "pipeline_topic", False))
        if pipeline_topic:
            try:
                rel_pq = os.path.join(classify_dir, "classify_relevant.parquet")
                all_pq = os.path.join(classify_dir, "classify_all.parquet")
                if os.path.exists(rel_pq):
                    topic_input_path = rel_pq
                elif os.path.exists(all_pq):
                    topic_input_path = all_pq
                else:
                    topic_input_path = classify_dir
            except Exception:
                topic_input_path = classify_dir
            topic_overrides: Dict[str, Any] = {
                "runtime.output_dir": os.path.join(base_out, "topic"),
                "data.parquet_path": topic_input_path,
                # Ensure SentenceTransformer runs on CPU to avoid GPU contention
                "topic.embed.device": "cpu",
            }
            try:
                topic_overrides["wandb.enabled"] = True
            except Exception:
                pass
            t_t0 = _time.time()
            try:
                _run_stage_subprocess("topic", topic_overrides, group_id)
                _ = float(_time.time() - t_t0)
            except Exception:
                _ = float(_time.time() - t_t0)
                raise
            # Log topic stage duration and mode
            if use_wandb:
                try:
                    _wb_log(cfg, {"pipeline/topic_seconds": float(_), "pipeline/topic_mode": True})  # type: ignore[arg-type]
                except Exception:
                    pass
            if not bool(getattr(cfg.runtime, "pipeline_topic", False)):
                pass
            # Skip taxonomy and verification when topic pipeline is enabled
            if not bool(os.environ.get("RULE_TUPLES_SILENT", "0")):
                print(json.dumps({"pipeline": {"output_dir": base_out, "topic_mode": True}}, indent=2))
            # Finish pipeline W&B run early in topic mode
            if use_wandb:
                try:
                    _wb_finish(cfg)
                except Exception:
                    pass
            return

        # Stage 2: taxonomy consumes classify_dir
        # Prefer relevant-only parquet if present; also handle historical misspelling; otherwise fall back
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

        # Best-effort: create a sanitized copy for taxonomy by dropping heavy/nested columns
        # This avoids accidental collisions (e.g., stringified 'messages') and reduces memory.
        sanitized_path = None
        try:
            import pandas as _pd  # local alias to avoid confusion with global pd
            if os.path.exists(tax_input_path):
                try:
                    pdf_in = _pd.read_parquet(tax_input_path)
                    # Columns to drop if present
                    drop_cols = [
                        "messages", "sampling_params", "usage", "token_counts",
                        "generated_tokens", "embeddings", "generated_text",
                        "llm_output", "relevance_answer", "classification_mode",
                        "latency_s", "token_usage_prompt", "token_usage_output",
                        "token_usage_total",
                    ]
                    cols_present = [c for c in drop_cols if c in list(pdf_in.columns)]
                    if cols_present:
                        pdf_in = pdf_in.drop(columns=cols_present)
                    # Write sanitized parquet alongside taxonomy outputs
                    sanitized_path = os.path.join(taxonomy_dir, "taxonomy_input_sanitized.parquet")
                    try:
                        pdf_in.to_parquet(sanitized_path, index=False)
                    except Exception:
                        # Attempt object sanitization similar to classify branch
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
                        pdf_in = pdf_in.applymap(_ser)
                        pdf_in.to_parquet(sanitized_path, index=False)
                except Exception:
                    sanitized_path = None
        except Exception:
            sanitized_path = None

        tax_overrides: Dict[str, Any] = {
            "runtime.output_dir": taxonomy_dir,
            "data.parquet_path": (sanitized_path or tax_input_path),
        }
        # Do NOT pass API keys via args; rely on env or prior login
        # Propagate streaming IO preference to child taxonomy
        try:
            tax_overrides["runtime.streaming_io"] = bool(getattr(cfg.runtime, "streaming_io", False))
        except Exception:
            pass
        # Ensure W&B is explicitly enabled for child taxonomy
        try:
            tax_overrides["wandb.enabled"] = True
        except Exception:
            pass
        tax_t0 = _time.time()
        try:
            _run_stage_subprocess("taxonomy", tax_overrides, group_id)
            tax_dt = float(_time.time() - tax_t0)
        except Exception:
            tax_dt = float(_time.time() - tax_t0)
            raise
        # Log taxonomy stage duration
        if use_wandb:
            try:
                _wb_log(cfg, {"pipeline/taxonomy_seconds": float(tax_dt)})  # type: ignore[arg-type]
            except Exception:
                pass

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
        # If the taxonomy results path is a directory (Ray-style Parquet dataset), prefer streaming IO
        try:
            if os.path.isdir(ver_input_path):
                ver_overrides["runtime.streaming_io"] = True
        except Exception:
            pass
        # Do NOT pass API keys via args; rely on env or prior login
        # Propagate streaming IO preference to child verification
        try:
            ver_overrides["runtime.streaming_io"] = bool(getattr(cfg.runtime, "streaming_io", False))
        except Exception:
            pass
        # Ensure W&B is explicitly enabled for child verification
        try:
            ver_overrides["wandb.enabled"] = True
        except Exception:
            pass
        ver_t0 = _time.time()
        try:
            _run_stage_subprocess("verification", ver_overrides, group_id)
            ver_dt = float(_time.time() - ver_t0)
        except Exception:
            ver_dt = float(_time.time() - ver_t0)
            raise
        # Log verification stage duration
        if use_wandb:
            try:
                _wb_log(cfg, {"pipeline/verification_seconds": float(ver_dt)})  # type: ignore[arg-type]
            except Exception:
                pass

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
        # Log final pipeline outputs and finish W&B run
        if use_wandb:
            try:
                _wb_log(cfg, {
                    "pipeline/output_dir": str(base_out),
                    "pipeline/classify_dir": str(classify_dir),
                    "pipeline/taxonomy_dir": str(taxonomy_dir),
                    "pipeline/verification_dir": str(verify_dir),
                    "pipeline/topic_mode": False,
                })  # type: ignore[arg-type]
            except Exception:
                pass
            try:
                _wb_finish(cfg)
            except Exception:
                pass
        return

    raise ValueError(f"Unknown runtime.stage: {stage}")


