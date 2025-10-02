from __future__ import annotations

from typing import Any, Dict, List, Optional
import os
import json
import pandas as pd
from datetime import datetime
import platform
import socket
import sys
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
    # Ensure required columns exist; coerce types sanely
    required = [
        columns.get("name", "name"),
        columns.get("public_description", "public_description"),
        columns.get("subscribers", "subscribers"),
        columns.get("rule_text", "rule_text"),
        columns.get("rule_index", "rule_index"),
        columns.get("total_rules_count", "total_rules_count"),
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Parquet missing expected columns: {missing}")
    # Normalize a minimal working set
    def _safe_str(x: Any) -> str:
        try:
            return "" if x is None or (isinstance(x, float) and pd.isna(x)) else str(x).strip()
        except Exception:
            return str(x) if x is not None else ""
    df[columns.get("name", "name")] = df[columns.get("name", "name")].apply(_safe_str)
    df[columns.get("public_description", "public_description")] = df[columns.get("public_description", "public_description")].apply(_safe_str)
    # Keep subscribers as string for now; downstream can coerce
    df[columns.get("subscribers", "subscribers")] = df[columns.get("subscribers", "subscribers")].apply(_safe_str)
    # Subset and rename to canonical names used in stages
    df = df.rename(columns={
        columns.get("name", "name"): "name",
        columns.get("public_description", "public_description"): "public_description",
        columns.get("subscribers", "subscribers"): "subscribers",
        columns.get("rule_text", "rule_text"): "rule_text",
        columns.get("rule_index", "rule_index"): "rule_index",
        columns.get("total_rules_count", "total_rules_count"): "total_rules_count",
    })
    if debug and isinstance(sample_n, int) and sample_n > 0:
        df = df.head(sample_n)
    return df


def run_experiment(cfg) -> None:
    """Entry point for the experiment.

    Current skeleton:
    - Loads flattened rules parquet into a pandas DataFrame
    - Routes by stage: classify | decompose | pipeline (pipeline = classify then decompose subset)
    - LLM wiring to be added in subsequent steps
    """
    stage = str(getattr(cfg.runtime, "stage", "classify")).strip().lower()
    debug = bool(getattr(cfg.runtime, "debug", True))
    sample_n = getattr(cfg.runtime, "sample_n", 100)

    parquet_path = str(getattr(cfg.data, "parquet_path"))
    columns = dict(getattr(cfg.data, "columns", {}))

    # Optional Ray streaming IO
    use_streaming = bool(getattr(cfg.runtime, "streaming_io", False)) and _RAY_OK_E
    ds = None
    if use_streaming:
        try:
            # Ensure Ray Data streams small blocks to encourage early writes
            try:
                ctx = ray.data.DataContext.get_current()
                # Smaller target blocks promote streaming and reduce resident memory
                # Use conservative sizes to keep memory bounded
                ctx.target_min_block_size = 1 * 1024 * 1024
                ctx.target_max_block_size = 64 * 1024 * 1024
                ctx.execution_options.verbose_progress = False
            except Exception:
                pass
            ds = ray.data.read_parquet(parquet_path)
            # Select and rename columns to canonical names used in stages
            col_map = {
                columns.get("name", "name"): "name",
                columns.get("public_description", "public_description"): "public_description",
                columns.get("subscribers", "subscribers"): "subscribers",
                columns.get("rule_text", "rule_text"): "rule_text",
                columns.get("rule_index", "rule_index"): "rule_index",
                columns.get("total_rules_count", "total_rules_count"): "total_rules_count",
            }
            # Intersect with existing columns to avoid select errors
            try:
                schema_names = list(ds.schema().names)
            except Exception:
                schema_names = []
            keep_keys = [k for k in col_map.keys() if k in schema_names]
            if keep_keys:
                ds = ds.select_columns(keep_keys)
            def _prep(pdf: pd.DataFrame) -> pd.DataFrame:
                def _safe_str(x: Any) -> str:
                    try:
                        return "" if x is None or (isinstance(x, float) and pd.isna(x)) else str(x).strip()
                    except Exception:
                        return str(x) if x is not None else ""
                pdf = pdf.rename(columns=col_map)
                # Ensure canonical columns exist
                for c in [
                    "name","public_description","subscribers","rule_text","rule_index","total_rules_count"
                ]:
                    if c not in pdf.columns:
                        pdf[c] = None
                pdf["name"] = pdf["name"].apply(_safe_str)
                pdf["public_description"] = pdf["public_description"].apply(_safe_str)
                pdf["subscribers"] = pdf["subscribers"].apply(_safe_str)
                return pdf
            ds = ds.map_batches(_prep, batch_format="pandas", batch_size=256)
            if debug and isinstance(sample_n, int) and sample_n > 0:
                ds = ds.limit(sample_n)
        except Exception:
            use_streaming = False
            ds = None
    if not use_streaming:
        df = _load_parquet_dataset(parquet_path, columns, debug=debug, sample_n=sample_n)

    # Optional W&B init
    use_wandb = False
    try:
        use_wandb = bool(getattr(cfg.wandb, "enabled", False))
    except Exception:
        use_wandb = False
    wb = None
    if use_wandb:
        try:
            import wandb as _wandb
            wb = _wandb
            run_name = f"{getattr(cfg.experiment, 'name', 'UAIR')}-{stage}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            # Build rich experiment config
            try:
                model_source = str(getattr(cfg.model, "model_source", ""))
            except Exception:
                model_source = ""
            try:
                engine_kwargs = dict(getattr(cfg.model, "engine_kwargs", {}))
            except Exception:
                engine_kwargs = {}
            try:
                batch_size = int(getattr(cfg.model, "batch_size", getattr(cfg, "model_runtime", {}).get("batch_size", 16)))
            except Exception:
                batch_size = 16
            try:
                concurrency = int(getattr(cfg.model, "concurrency", getattr(cfg, "model_runtime", {}).get("concurrency", 1)))
            except Exception:
                concurrency = 1
            classify_sp = dict(getattr(cfg, "sampling_params", {}))
            decomp_sp = dict(getattr(cfg, "sampling_params", {}))
            # Prompts
            classify_sys = str(getattr(cfg.prompt, "system_prompt", ""))
            classify_tpl = str(getattr(cfg.prompt, "prompt_template", ""))
            try:
                decomp_sys = str(getattr(cfg.prompt_decompose, "system_prompt", ""))
                decomp_tpl = str(getattr(cfg.prompt_decompose, "prompt_template", ""))
            except Exception:
                decomp_sys = ""
                decomp_tpl = ""
            # System info
            try:
                ray_ver = ray.__version__ if _RAY_OK_E else None
                cluster_gpu = int((ray.cluster_resources() or {}).get("GPU", 0)) if _RAY_OK_E else None
                avail_gpu = int((ray.available_resources() or {}).get("GPU", 0)) if _RAY_OK_E else None
            except Exception:
                ray_ver = None
                cluster_gpu = None
                avail_gpu = None
            run_config = {
                "stage": stage,
                "parquet_path": parquet_path,
                "data_columns_map": columns,
                "sample_n": sample_n,
                "output_csv": str(getattr(cfg.runtime, "output_csv", "") or ""),
                "use_llm_classify": bool(getattr(cfg.runtime, "use_llm_classify", False)),
                "use_llm_decompose": bool(getattr(cfg.runtime, "use_llm_decompose", False)),
                "guided_decoding_decompose": bool(getattr(cfg.runtime, "guided_decoding_decompose", False)),
                "max_errored_blocks": int(getattr(cfg.runtime, "max_errored_blocks", 0) or 0),
                # Model/runtime
                "model_source": model_source,
                "engine_kwargs": engine_kwargs,
                "batch_size": batch_size,
                "concurrency": concurrency,
                "sampling_params": dict(getattr(cfg, "sampling_params", {})),
                # Prompts
                "classify_system_prompt": classify_sys,
                "classify_prompt_template": classify_tpl,
                "decompose_system_prompt": decomp_sys,
                "decompose_prompt_template": decomp_tpl,
                # System metadata
                "ray_version": ray_ver,
                "python_version": sys.version.split()[0],
                "os": platform.platform(),
                "hostname": socket.gethostname(),
                "cluster_gpu_count": cluster_gpu,
                "available_gpu_count": avail_gpu,
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
            wb.init(project=proj, entity=ent, job_type=stage, name=run_name, config=run_config)
            try:
                wb.log({"dataset/loaded_rows": int(len(df))})
            except Exception:
                pass
            try:
                wb.log({"status/started": 1})
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

    def _wb_log_table(df: pd.DataFrame, key: str, max_rows: int = 1000) -> None:
        if wb is None:
            return
        try:
            cols_pref = [
                "name", "public_description", "subscribers", "rule_index", "rule_text",
                "is_relevant", "llm_output",
                "ci_subject", "ci_sender", "ci_receiver", "ci_information", "ci_transmission_principle", "ci_missing_elements",
            ]
            cols = [c for c in cols_pref if c in df.columns]
            if not cols:
                # Fallback: take first N columns to ensure something is logged
                cols = list(df.columns)[:12]
            table = wb.Table(columns=cols, log_mode="MUTABLE")
            def _to_str(v: Any) -> str:
                try:
                    import json as _json
                    if isinstance(v, (dict, list, tuple)):
                        return _json.dumps(v, ensure_ascii=False)
                except Exception:
                    pass
                try:
                    return str(v) if v is not None else ""
                except Exception:
                    return ""
            # Fixed-seed random sampling when exceeding max_rows
            try:
                total_rows = int(len(df))
            except Exception:
                total_rows = None
            sample_n = int(max_rows)
            seed = 777
            try:
                env_seed = os.environ.get("UAIR_WB_TABLE_SEED") or os.environ.get("UAIR_TABLE_SAMPLE_SEED")
                if env_seed is not None:
                    seed = int(env_seed)
            except Exception:
                seed = 777
            try:
                if total_rows is not None and total_rows > sample_n:
                    df_iter = df.sample(n=sample_n, random_state=seed).reset_index(drop=True)
                else:
                    df_iter = df.reset_index(drop=True)
            except Exception:
                df_iter = df.reset_index(drop=True).head(sample_n)
            n = 0
            for _, r in df_iter.iterrows():
                table.add_data(*[_to_str(r.get(c)) for c in cols])
                n += 1
            wb.log({key: table, f"{key}/rows": n, f"{key}/total_rows": (total_rows if total_rows is not None else n)})
        except Exception:
            pass

    # Streaming W&B usage logging: create an optional Ray actor and background logger
    usage_agg_actor_name: Optional[str] = None
    _usage_logger_stop: Optional[threading.Event] = None
    _usage_logger_thread: Optional[threading.Thread] = None

    def _make_usage_accumulator(stage_label: str):
        """Return a Ray map_batches function that aggregates token usage and records into the actor.

        The returned function is executed on Ray workers; it must look up the named actor itself.
        """
        if not (use_streaming and _RAY_OK_E):
            return None
        name = usage_agg_actor_name
        if not name:
            return None

        def _accumulate(pdf: pd.DataFrame) -> pd.DataFrame:
            try:
                calls = int(len(pdf))
                def _sum_col(col: str) -> int:
                    try:
                        if col in pdf.columns:
                            s = pd.to_numeric(pdf[col], errors="coerce").fillna(0)
                            return int(s.astype("int64").sum())
                    except Exception:
                        pass
                    return 0
                prompt = _sum_col("token_usage_prompt")
                completion = _sum_col("token_usage_output")
                total = _sum_col("token_usage_total")
                # Emit lightweight progress every batch
                try:
                    nrows = int(_sum_col("_progress_row") or len(pdf))
                    _wb_log({f"progress/{stage_label}/rows_processed": nrows})
                except Exception:
                    pass
                # Fallback to nested usage dicts when explicit columns are absent
                if (prompt == 0 or completion == 0 or total == 0) and ("usage" in pdf.columns or "token_counts" in pdf.columns):
                    try:
                        def _safe_get(d: Any, *keys: str) -> int:
                            for k in keys:
                                try:
                                    if isinstance(d, dict) and k in d and d.get(k) is not None:
                                        v = d.get(k)
                                        return int(v) if isinstance(v, (int,)) else int(float(v))
                                except Exception:
                                    continue
                            return 0
                        p = 0; q = 0; t = 0
                        if "usage" in pdf.columns:
                            for u in pdf["usage"].tolist():
                                p += _safe_get(u, "prompt_tokens", "input_tokens")
                                q += _safe_get(u, "completion_tokens", "output_tokens")
                                tot = _safe_get(u, "total_tokens")
                                if tot == 0:
                                    tot = _safe_get(u, "prompt_tokens", "input_tokens") + _safe_get(u, "completion_tokens", "output_tokens")
                                t += tot
                        if "token_counts" in pdf.columns:
                            for u in pdf["token_counts"].tolist():
                                p += _safe_get(u, "prompt_tokens", "input_tokens")
                                q += _safe_get(u, "completion_tokens", "output_tokens")
                                tot = _safe_get(u, "total_tokens")
                                if tot == 0:
                                    tot = _safe_get(u, "prompt_tokens", "input_tokens") + _safe_get(u, "completion_tokens", "output_tokens")
                                t += tot
                        # prefer explicit columns when present
                        if prompt == 0:
                            prompt = p
                        if completion == 0:
                            completion = q
                        if total == 0:
                            total = t if (t > 0) else (p + q)
                    except Exception:
                        pass
                try:
                    ray.get_actor(name).record.remote(stage_label, calls=calls, prompt_tokens=prompt, completion_tokens=completion, total_tokens=total)
                except Exception:
                    pass
            except Exception:
                pass
            return pdf

        return _accumulate

    def _start_usage_logger_if_needed() -> None:
        nonlocal usage_agg_actor_name, _usage_logger_stop, _usage_logger_thread
        if not (use_streaming and _RAY_OK_E):
            return
        if usage_agg_actor_name is None:
            try:
                usage_agg_actor_name = f"usage_agg_{os.getpid()}_{int(_time.time())}"
                _ = _UsageAggregator.options(name=usage_agg_actor_name).remote()
            except Exception:
                usage_agg_actor_name = None
                return
        if wb is None:
            return
        _usage_logger_stop = threading.Event()

        def _log_loop():
            while _usage_logger_stop is not None and not _usage_logger_stop.is_set():
                try:
                    snap = ray.get(ray.get_actor(usage_agg_actor_name).snapshot.remote())
                    if isinstance(snap, dict):
                        data = {}
                        for scope in ("classify", "decompose", "overall"):
                            d = snap.get(scope) or {}
                            try:
                                data[f"usage/{scope}/calls"] = int(d.get("calls", 0) or 0)
                                data[f"usage/{scope}/prompt_tokens"] = int(d.get("prompt_tokens", 0) or 0)
                                data[f"usage/{scope}/completion_tokens"] = int(d.get("completion_tokens", 0) or 0)
                                data[f"usage/{scope}/total_tokens"] = int(d.get("total_tokens", 0) or 0)
                            except Exception:
                                continue
                        if data:
                            _wb_log(data)
                except Exception:
                    pass
                if _usage_logger_stop is not None:
                    _usage_logger_stop.wait(2.0)

        _usage_logger_thread = threading.Thread(target=_log_loop, name="wb_usage_logger", daemon=True)
        _usage_logger_thread.start()

    def _stop_usage_logger(final_log: bool = True) -> None:
        nonlocal _usage_logger_stop, _usage_logger_thread
        try:
            if _usage_logger_stop is not None:
                _usage_logger_stop.set()
            if _usage_logger_thread is not None:
                _usage_logger_thread.join(timeout=3.0)
            if final_log and usage_agg_actor_name is not None:
                try:
                    snap = ray.get(ray.get_actor(usage_agg_actor_name).snapshot.remote())
                    if isinstance(snap, dict):
                        data = {}
                        for scope in ("classify", "decompose", "overall"):
                            d = snap.get(scope) or {}
                            data[f"usage/{scope}/calls"] = int(d.get("calls", 0) or 0)
                            data[f"usage/{scope}/prompt_tokens"] = int(d.get("prompt_tokens", 0) or 0)
                            data[f"usage/{scope}/completion_tokens"] = int(d.get("completion_tokens", 0) or 0)
                            data[f"usage/{scope}/total_tokens"] = int(d.get("total_tokens", 0) or 0)
                        if data:
                            _wb_log(data)
                except Exception:
                    pass
        except Exception:
            pass

    # Periodic progress logger based on parquet shards written
    _progress_logger_stop: Optional[threading.Event] = None
    _progress_logger_thread: Optional[threading.Thread] = None

    def _start_progress_logger_if_needed(out_dir: str) -> None:
        nonlocal _progress_logger_stop, _progress_logger_thread
        if wb is None or not out_dir:
            return
        if _progress_logger_stop is not None:
            return
        _progress_logger_stop = threading.Event()

        def _loop():
            last_ct = -1
            while _progress_logger_stop is not None and not _progress_logger_stop.is_set():
                try:
                    # Count parquet files (best-effort); avoid heavy ops
                    ct = 0
                    try:
                        for nm in os.listdir(out_dir):
                            if nm.endswith(".parquet") or nm.endswith(".pq") or nm.startswith("part-"):
                                ct += 1
                    except Exception:
                        ct = 0
                    if ct != last_ct:
                        _wb_log({"progress/parquet_shards": ct})
                        last_ct = ct
                except Exception:
                    pass
                if _progress_logger_stop is not None:
                    _progress_logger_stop.wait(5.0)

        _progress_logger_thread = threading.Thread(target=_loop, name="wb_progress_logger", daemon=True)
        _progress_logger_thread.start()

    def _stop_progress_logger() -> None:
        nonlocal _progress_logger_stop, _progress_logger_thread
        try:
            if _progress_logger_stop is not None:
                _progress_logger_stop.set()
            if _progress_logger_thread is not None:
                _progress_logger_thread.join(timeout=2.0)
        except Exception:
            pass

    def _drop_heavy_columns(ds_in):
        cols = ["messages", "sampling_params", "generated_text", "usage", "token_counts", "prompt", "prompt_token_ids", "params"]
        try:
            return ds_in.drop_columns(cols)
        except Exception:
            def _drop_batch(pdf: pd.DataFrame) -> pd.DataFrame:
                try:
                    return pdf.drop(columns=[c for c in cols if c in pdf.columns])
                except Exception:
                    return pdf
            return ds_in.map_batches(_drop_batch, batch_format="pandas", batch_size=256)

    def _sanitize_for_parquet(pdf: pd.DataFrame) -> pd.DataFrame:
        pdf = pdf.copy()
        # Convert problematic object columns to JSON strings
        def _ser(v: Any) -> Any:
            try:
                import numpy as _np
                if isinstance(v, _np.generic):
                    try:
                        return v.item()
                    except Exception:
                        pass
                if isinstance(v, _np.ndarray):
                    try:
                        return json.dumps(v.tolist(), ensure_ascii=False)
                    except Exception:
                        return str(v)
            except Exception:
                pass
            if isinstance(v, (dict, list, tuple, set)):
                try:
                    if isinstance(v, set):
                        v = list(v)
                    return json.dumps(v, ensure_ascii=False)
                except Exception:
                    return str(v)
            return v
        for col in list(pdf.columns):
            if col in ("messages", "sampling_params", "generated_text", "usage", "token_counts"):
                try:
                    pdf.drop(columns=[col], inplace=True)
                except Exception:
                    pass
                continue
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

    if not bool(os.environ.get("RULE_TUPLES_SILENT", "0")):
        if use_streaming and ds is not None:
            print(json.dumps({
                "loaded_rows": None,
                "stage": stage,
                "columns": ["name","public_description","subscribers","rule_index","rule_text"],
                "streaming": True,
            }, indent=2))
        else:
            print(json.dumps({
                "loaded_rows": int(len(df)),
                "stage": stage,
                "columns": list(df.columns)[:12],
                "streaming": False,
            }, indent=2))

    if stage == "classify":
        from .stages.classify import run_classification_stage
        if use_streaming and ds is not None:
            _start_usage_logger_if_needed()
            out_ds = run_classification_stage(ds, cfg)
            acc = _make_usage_accumulator("classify")
            if acc is not None:
                try:
                    # Place after LLM processor to avoid extra memory growth and stream counts with outputs
                    out_ds = out_ds.map_batches(acc, batch_format="pandas", batch_size=256)
                except Exception:
                    pass
            # Drop heavy/nested columns before writing
            try:
                out_ds = _drop_heavy_columns(out_ds)
            except Exception:
                pass
            out_dir = str(getattr(cfg.runtime, "output_dir", "") or "")
            if out_dir:
                try:
                    _start_progress_logger_if_needed(out_dir)
                    # Stream to Parquet with moderate file sizes and safe schema
                    out_ds = out_ds.map_batches(_sanitize_for_parquet, batch_format="pandas", batch_size=256)
                    out_ds.write_parquet(out_dir, min_rows_per_file=5000)
                    if not bool(os.environ.get("RULE_TUPLES_SILENT", "0")):
                        print(json.dumps({"classified_rows": None, "output_dir": out_dir}, indent=2))
                except Exception as e:
                    if not bool(os.environ.get("RULE_TUPLES_SILENT", "0")):
                        print(json.dumps({"error": "classify_streaming_write_failed", "detail": str(e)}, indent=2))
                    raise
            _stop_progress_logger()
            _stop_usage_logger(final_log=True)
            return
        out_df = run_classification_stage(df, cfg)
        out_path = str(getattr(cfg.runtime, "output_csv", "") or "")
        try:
            if "is_relevant" in out_df.columns:
                rel_count = int(out_df["is_relevant"].astype(bool).sum())
                total = int(len(out_df))
                ratio = float(rel_count) / float(total) if total > 0 else 0.0
                avg_lat = None
                try:
                    lat = [float(v) for v in out_df.get("latency_s", []).tolist() if isinstance(v, (int, float))]
                    avg_lat = (sum(lat) / len(lat)) if lat else None
                except Exception:
                    pass
                _wb_log({
                    "classify/rows": total,
                    "classify/relevant_count": rel_count,
                    "classify/relevant_ratio": ratio,
                    "classify/avg_latency_s": avg_lat,
                })
        except Exception:
            pass
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
        _wb_log_table(out_df, key="inspection/classify")
        return

    if stage == "decompose":
        from .stages.decompose import run_decomposition_stage
        if use_streaming and ds is not None:
            _start_usage_logger_if_needed()
            out_ds = run_decomposition_stage(ds, cfg)
            acc = _make_usage_accumulator("decompose")
            if acc is not None:
                try:
                    out_ds = out_ds.map_batches(acc, batch_format="pandas", batch_size=256)
                except Exception:
                    pass
            try:
                out_ds = _drop_heavy_columns(out_ds)
            except Exception:
                pass
            out_dir = str(getattr(cfg.runtime, "output_dir", "") or "")
            if out_dir:
                try:
                    _start_progress_logger_if_needed(out_dir)
                    out_ds = out_ds.map_batches(_sanitize_for_parquet, batch_format="pandas", batch_size=256)
                    out_ds.write_parquet(out_dir, min_rows_per_file=5000)
                    if not bool(os.environ.get("RULE_TUPLES_SILENT", "0")):
                        print(json.dumps({"decomposed_rows": None, "output_dir": out_dir}, indent=2))
                except Exception as e:
                    if not bool(os.environ.get("RULE_TUPLES_SILENT", "0")):
                        print(json.dumps({"error": "decompose_streaming_write_failed", "detail": str(e)}, indent=2))
                    raise
            _stop_progress_logger()
            _stop_usage_logger(final_log=True)
            return
        out_df = run_decomposition_stage(df, cfg)
        out_path = str(getattr(cfg.runtime, "output_csv", "") or "")
        try:
            total = int(len(out_df))
            have_any = int((out_df[[
                "ci_subject","ci_sender","ci_receiver","ci_information","ci_transmission_principle"
            ]].notna().any(axis=1)).sum()) if total > 0 else 0
            _wb_log({
                "decompose/rows": total,
                "decompose/any_tuple_present": have_any,
            })
        except Exception:
            pass
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
        _wb_log_table(out_df, key="inspection/decompose")
        return

    if stage == "pipeline":
        from .stages.classify import run_classification_stage
        from .stages.decompose import run_decomposition_stage
        if use_streaming and ds is not None:
            _start_usage_logger_if_needed()
            cls_ds = run_classification_stage(ds, cfg)
            acc_c = _make_usage_accumulator("classify")
            if acc_c is not None:
                try:
                    cls_ds = cls_ds.map_batches(acc_c, batch_format="pandas", batch_size=256)
                except Exception:
                    pass
            rel_ds = cls_ds.filter(lambda r: bool(r.get("is_relevant", False)))
            dec_ds = run_decomposition_stage(rel_ds, cfg)
            acc_d = _make_usage_accumulator("decompose")
            if acc_d is not None:
                try:
                    dec_ds = dec_ds.map_batches(acc_d, batch_format="pandas", batch_size=256)
                except Exception:
                    pass
            try:
                dec_ds = _drop_heavy_columns(dec_ds)
            except Exception:
                pass
            out_dir = str(getattr(cfg.runtime, "output_dir", "") or "")
            if out_dir:
                try:
                    _start_progress_logger_if_needed(out_dir)
                    dec_ds = dec_ds.map_batches(_sanitize_for_parquet, batch_format="pandas", batch_size=256)
                    dec_ds.write_parquet(out_dir, min_rows_per_file=5000)
                    if not bool(os.environ.get("RULE_TUPLES_SILENT", "0")):
                        print(json.dumps({
                            "pipeline_rows_in": None,
                            "pipeline_relevant": None,
                            "pipeline_decomposed": None,
                            "output_dir": out_dir,
                        }, indent=2))
                except Exception as e:
                    if not bool(os.environ.get("RULE_TUPLES_SILENT", "0")):
                        print(json.dumps({"error": "pipeline_streaming_write_failed", "detail": str(e)}, indent=2))
                    raise
            _stop_progress_logger()
            _stop_usage_logger(final_log=True)
            return
        cls_df = run_classification_stage(df, cfg)
        # Filter to relevant
        rel_df = cls_df[cls_df.get("is_relevant", False) == True] if "is_relevant" in cls_df.columns else cls_df
        # Log classification metrics in pipeline stage as well
        try:
            if "is_relevant" in cls_df.columns:
                rel_count_p = int(cls_df["is_relevant"].astype(bool).sum())
                total_p = int(len(cls_df))
                ratio_p = float(rel_count_p) / float(total_p) if total_p > 0 else 0.0
                avg_lat_p = None
                try:
                    lat_p = [float(v) for v in cls_df.get("latency_s", []).tolist() if isinstance(v, (int, float))]
                    avg_lat_p = (sum(lat_p) / len(lat_p)) if lat_p else None
                except Exception:
                    pass
                _wb_log({
                    "classify/rows": total_p,
                    "classify/relevant_count": rel_count_p,
                    "classify/relevant_ratio": ratio_p,
                    "classify/avg_latency_s": avg_lat_p,
                })
                _wb_log_table(cls_df, key="inspection/classify")
        except Exception:
            pass
        dec_df = run_decomposition_stage(rel_df, cfg)
        out_path = str(getattr(cfg.runtime, "output_csv", "") or "")
        try:
            total = int(len(df))
            rel = int(len(rel_df))
            dec = int(len(dec_df))
            ratio = float(rel) / float(total) if total > 0 else 0.0
            _wb_log({
                "pipeline/rows_in": total,
                "pipeline/relevant": rel,
                "pipeline/decomposed": dec,
                "pipeline/relevant_ratio": ratio,
            })
        except Exception:
            pass
        if out_path:
            try:
                dec_df.to_csv(out_path, index=False)
                if not bool(os.environ.get("RULE_TUPLES_SILENT", "0")):
                    print(json.dumps({
                        "pipeline_rows_in": int(len(df)),
                        "pipeline_relevant": int(len(rel_df)),
                        "pipeline_decomposed": int(len(dec_df)),
                        "output_csv": out_path,
                    }, indent=2))
            except Exception:
                if not bool(os.environ.get("RULE_TUPLES_SILENT", "0")):
                    print(json.dumps({
                        "pipeline_rows_in": int(len(df)),
                        "pipeline_relevant": int(len(rel_df)),
                        "pipeline_decomposed": int(len(dec_df)),
                        "output_csv": None,
                    }, indent=2))
        else:
            if not bool(os.environ.get("RULE_TUPLES_SILENT", "0")):
                print(json.dumps({
                    "pipeline_rows_in": int(len(df)),
                    "pipeline_relevant": int(len(rel_df)),
                    "pipeline_decomposed": int(len(dec_df)),
                }, indent=2))
        _wb_log_table(dec_df, key="inspection/pipeline")
        return

    raise ValueError(f"Unknown runtime.stage: {stage}")


