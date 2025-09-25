from __future__ import annotations

from typing import Any, Dict, List, Optional
import os
import sys
import datetime as _dt

_WB = None  # Lazy-imported wandb module or None


def _wandb_available() -> bool:
    global _WB
    if _WB is not None:
        return True
    try:
        import wandb as _wandb  # type: ignore
        _WB = _wandb
        return True
    except Exception:
        _WB = None
        return False


def is_enabled(cfg) -> bool:
    try:
        return bool(getattr(cfg.wandb, "enabled", False))
    except Exception:
        return False


def _get_group(cfg) -> Optional[str]:
    try:
        grp = getattr(cfg.wandb, "group", None)
        if grp and str(grp).strip() != "":
            return str(grp)
    except Exception:
        pass
    try:
        env_g = os.environ.get("WANDB_GROUP")
        if env_g and str(env_g).strip() != "":
            return env_g
    except Exception:
        pass
    return None


def start_run(cfg, stage: str, run_config: Optional[Dict[str, Any]] = None, name_suffix: Optional[str] = None) -> None:
    if not is_enabled(cfg):
        return
    if not _wandb_available():
        return
    proj = "UAIR"
    try:
        proj = str(getattr(cfg.wandb, "project", "UAIR") or "UAIR")
    except Exception:
        pass
    try:
        ent = getattr(cfg.wandb, "entity", None)
        ent = str(ent) if (ent is not None and str(ent).strip() != "") else None
    except Exception:
        ent = None
    group = _get_group(cfg)
    try:
        exp_name = str(getattr(cfg.experiment, "name", "UAIR") or "UAIR")
    except Exception:
        exp_name = "UAIR"
    ts = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    base = f"{exp_name}-{stage}-{ts}"
    if group:
        base = f"{group}-{base}"
    if isinstance(name_suffix, str) and name_suffix:
        base = f"{base}-{name_suffix}"
    try:
        _WB.init(project=proj, entity=ent, group=group, job_type=str(stage), name=base, config=(run_config or {}))
    except Exception:
        pass


def finish_run(cfg) -> None:
    if not is_enabled(cfg):
        return
    if not _wandb_available():
        return
    try:
        if getattr(_WB, "run", None) is not None:
            _WB.finish()
    except Exception:
        pass


def log_metrics(cfg, data: Dict[str, Any]) -> None:
    if not is_enabled(cfg):
        return
    if not _wandb_available():
        return
    try:
        if data:
            _WB.log(data)
    except Exception:
        pass


def _to_str(v: Any) -> str:
    try:
        import json as _json
        if isinstance(v, (dict, list, tuple, set)):
            if isinstance(v, set):
                v = list(v)
            return _json.dumps(v, ensure_ascii=False)
    except Exception:
        pass
    try:
        return str(v) if v is not None else ""
    except Exception:
        return ""


def log_table(cfg, df, key: str, prefer_cols: Optional[List[str]] = None, max_rows: int = 1000) -> None:
    if not is_enabled(cfg):
        return
    if not _wandb_available():
        return
    if df is None:
        return
    try:
        import pandas as _pd  # type: ignore
    except Exception:
        _pd = None  # type: ignore
    try:
        cols = [c for c in (prefer_cols or []) if c in df.columns]
        if not cols:
            cols = list(df.columns)[:12]
        try:
            table = _WB.Table(columns=cols)
        except Exception:
            table = _WB.Table(columns=cols)
        # Determine sampling
        try:
            total_rows = int(len(df))
        except Exception:
            total_rows = None
        sample_n = int(max_rows)
        # Fixed seed selection
        seed = 777
        try:
            # Prefer cfg.wandb.table_sample_seed when available
            seed = int(getattr(getattr(cfg, "wandb", object()), "table_sample_seed", 777))
        except Exception:
            try:
                env_seed = os.environ.get("UAIR_WB_TABLE_SEED") or os.environ.get("UAIR_TABLE_SAMPLE_SEED")
                if env_seed is not None:
                    seed = int(env_seed)
            except Exception:
                seed = 777
        try:
            if total_rows is not None and total_rows > sample_n and _pd is not None:
                df_iter = df.sample(n=sample_n, random_state=seed).reset_index(drop=True)
            else:
                df_iter = df.reset_index(drop=True)
        except Exception:
            # Fallback to first N if sampling fails
            try:
                df_iter = df.reset_index(drop=True).head(sample_n)
            except Exception:
                df_iter = df
        n = 0
        for _, r in df_iter.iterrows():
            row_vals: List[str] = []
            for c in cols:
                row_vals.append(_to_str(r.get(c)))
            table.add_data(*row_vals)
            n += 1
        _WB.log({key: table, f"{key}/rows": n, f"{key}/total_rows": (total_rows if total_rows is not None else n)})
    except Exception:
        pass


