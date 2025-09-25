from __future__ import annotations

from typing import Optional
import os

def log_cluster_scatter_to_wandb(df, cfg, title: str = "topic_cluster_map") -> Optional[object]:
    try:
        import wandb  # type: ignore
    except Exception:
        return None
    try:
        if not getattr(getattr(cfg, "wandb", object()), "enabled", False):
            return None
    except Exception:
        return None
    if df is None or len(df) == 0:
        return None
    # Guard: skip heavy wandb.Table scatter for large datasets
    try:
        # Config chain: cfg.topic.plot.table_scatter_max_rows or env UAIR_TOPIC_TABLE_MAX_ROWS
        try:
            max_rows_cfg = getattr(getattr(getattr(cfg, "topic", object()), "plot", object()), "table_scatter_max_rows", None)
        except Exception:
            max_rows_cfg = None
        max_rows_env = os.environ.get("UAIR_TOPIC_TABLE_MAX_ROWS")
        table_limit = int(max_rows_env) if (max_rows_env and max_rows_env.isdigit()) else int(max_rows_cfg) if isinstance(max_rows_cfg, int) else 20000
    except Exception:
        table_limit = 20000
    try:
        if len(df) > table_limit:
            # Downsample with fixed seed instead of skipping
            try:
                # Prefer cfg.wandb.table_sample_seed; fallback to topic.seed; else env; else default
                try:
                    seed = int(getattr(getattr(cfg, "wandb", object()), "table_sample_seed", 777))
                except Exception:
                    try:
                        seed = int(getattr(getattr(cfg, "topic", object()), "seed", 777))
                    except Exception:
                        seed = 777
                seed_env = os.environ.get("UAIR_WB_TABLE_SEED") or os.environ.get("UAIR_TABLE_SAMPLE_SEED")
                if seed_env is not None:
                    seed = int(seed_env)
            except Exception:
                seed = 777
            try:
                df = df.sample(n=int(table_limit), random_state=seed).reset_index(drop=True)
            except Exception:
                df = df.reset_index(drop=True).head(int(table_limit))
    except Exception:
        pass
    # Ensure required columns
    if not ("plot_x" in df.columns and "plot_y" in df.columns and "topic_id" in df.columns):
        return None
    try:
        # Color by topic_id; show hover with article_id and path when present
        data = []
        cols = ["plot_x", "plot_y", "topic_id"]
        extra = []
        for c in ("article_id", "article_path"):
            if c in df.columns:
                extra.append(c)
        cols = cols + extra
        table = wandb.Table(columns=cols)
        for _, r in df.iterrows():
            row = [float(r.get("plot_x", 0.0)), float(r.get("plot_y", 0.0)), int(r.get("topic_id", -1))]
            for c in extra:
                row.append(str(r.get(c)) if r.get(c) is not None else "")
            table.add_data(*row)
        plt = wandb.plot.scatter(table, x="plot_x", y="plot_y", title=title, label="topic_id")
        wandb.log({"plots/" + title: plt})
        return plt
    except Exception:
        return None

def log_cluster_scatter_plotly_to_wandb(df, cfg, title: str = "topic_cluster_map") -> Optional[object]:
    try:
        import wandb  # type: ignore
        import plotly.express as px  # type: ignore
    except Exception:
        return None
    try:
        if not getattr(getattr(cfg, "wandb", object()), "enabled", False):
            return None
    except Exception:
        return None
    if df is None or len(df) == 0:
        return None
    if not ("plot_x" in df.columns and "plot_y" in df.columns and "topic_id" in df.columns):
        return None
    try:
        import plotly.graph_objects as go  # type: ignore
    except Exception:
        return None
    try:
        n = int(len(df))
    except Exception:
        n = 0

    # Read plotting configuration with safe fallbacks
    def _get_plot_cfg_int(path_keys, default_val):
        try:
            node = getattr(cfg, path_keys[0])
            for k in path_keys[1:]:
                node = getattr(node, k)
            if isinstance(node, int):
                return int(node)
        except Exception:
            pass
        return int(default_val)

    def _get_env_int(name, default_val):
        try:
            v = os.environ.get(name)
            if v is None:
                return int(default_val)
            return int(v)
        except Exception:
            return int(default_val)

    def _get_plot_cfg_str(path_keys, default_val):
        try:
            node = getattr(cfg, path_keys[0])
            for k in path_keys[1:]:
                node = getattr(node, k)
            return str(node)
        except Exception:
            return str(default_val)

    try:
        seed = _get_plot_cfg_int(["topic", "seed"], 777)
    except Exception:
        seed = 777
    # thresholds and params
    max_points_cfg = _get_plot_cfg_int(["topic", "plot", "max_points"], 120000)
    max_points = _get_env_int("UAIR_TOPIC_PLOT_MAX_POINTS", max_points_cfg)
    heat_thr_cfg = _get_plot_cfg_int(["topic", "plot", "heatmap_threshold"], 250000)
    heat_threshold = _get_env_int("UAIR_TOPIC_PLOT_HEATMAP_THRESHOLD", heat_thr_cfg)
    bins_cfg = _get_plot_cfg_int(["topic", "plot", "heatmap_bins"], 200)
    heat_bins = _get_env_int("UAIR_TOPIC_HEATMAP_BINS", bins_cfg)
    msize_cfg = _get_plot_cfg_int(["topic", "plot", "marker_size"], 3)
    marker_size = _get_env_int("UAIR_TOPIC_PLOT_MARKER_SIZE", msize_cfg)
    try:
        op_env = os.environ.get("UAIR_TOPIC_PLOT_MARKER_OPACITY")
        if op_env is not None:
            marker_opacity = float(op_env)
        else:
            marker_opacity = float(getattr(getattr(getattr(cfg, "topic", object()), "plot", object()), "marker_opacity", 0.6))
    except Exception:
        marker_opacity = 0.6
    method_env = os.environ.get("UAIR_TOPIC_PLOT_METHOD", "auto").strip().lower()
    method_cfg = _get_plot_cfg_str(["topic", "plot", "method"], "auto").strip().lower()
    method = method_env or method_cfg or "auto"

    # Prepare working frame with required columns only
    try:
        base_cols = ["plot_x", "plot_y", "topic_id"]
        opt_cols = []
        for c in ("article_id", "article_path", "article_keywords", "topic_top_terms"):
            if c in df.columns:
                opt_cols.append(c)
        df2 = df[base_cols + opt_cols].copy()
    except Exception:
        df2 = df[["plot_x", "plot_y", "topic_id"]].copy()
    # Drop rows with missing coordinates to avoid plotly rendering issues
    try:
        df2 = df2.dropna(subset=["plot_x", "plot_y"]).reset_index(drop=True)
    except Exception:
        pass

    # Decide visualization mode
    use_heatmap = (method == "heatmap") or (method == "auto" and n > heat_threshold)

    try:
        if use_heatmap:
            # Use server-side aggregation to create a compact 2D histogram (heatmap)
            fig = go.Figure()
            fig.add_trace(
                go.Histogram2d(
                    x=df2["plot_x"],
                    y=df2["plot_y"],
                    nbinsx=int(heat_bins),
                    nbinsy=int(heat_bins),
                    colorscale="YlGnBu",
                    colorbar=dict(title="count"),
                )
            )
            fig.update_layout(
                title=title,
                showlegend=False,
                hovermode="closest",
                margin=dict(l=40, r=10, t=40, b=40),
            )
            wandb.log({"plots/" + title + "_plotly": fig})
            return fig

        # Otherwise, scattergl with optional downsampling
        df_plot = df2
        try:
            if n > max_points and max_points > 0:
                # Sample per-topic to preserve cluster structure
                try:
                    import pandas as _pd  # type: ignore
                except Exception:
                    _pd = None  # type: ignore
                try:
                    unique_topics = df2["topic_id"].nunique(dropna=False)
                except Exception:
                    unique_topics = 1
                per_cap = max(1, int(max_points // max(1, int(unique_topics))))
                try:
                    df_plot = (
                        df2.groupby("topic_id", dropna=False, group_keys=False)
                        .apply(lambda g: g.sample(n=min(len(g), per_cap), random_state=int(seed)))
                        .reset_index(drop=True)
                    )
                except Exception:
                    df_plot = df2.sample(n=int(max_points), random_state=int(seed))
        except Exception:
            pass

        # Prepare hover fields and labels
        def _norm_terms(val):
            try:
                if isinstance(val, (list, tuple)):
                    return [str(x) for x in val if x is not None]
                if isinstance(val, str):
                    import ast as _ast  # type: ignore
                    try:
                        parsed = _ast.literal_eval(val)
                        if isinstance(parsed, (list, tuple)):
                            return [str(x) for x in parsed if x is not None]
                    except Exception:
                        return [val]
            except Exception:
                pass
            return []
        try:
            if "topic_top_terms" in df_plot.columns:
                df_plot["topic_label"] = df_plot["topic_top_terms"].apply(lambda v: ", ".join(_norm_terms(v)[:3]) if _norm_terms(v) else (f"topic {int(v)}" if "topic_id" in df_plot.columns else "topic"))
            else:
                df_plot["topic_label"] = df_plot["topic_id"].apply(lambda x: f"topic {int(x)}" if x is not None else "topic")
        except Exception:
            try:
                df_plot["topic_label"] = df_plot.get("topic_id", "").apply(lambda x: f"topic {x}")
            except Exception:
                df_plot["topic_label"] = "topic"
        # Normalize keywords string for hover
        try:
            if "article_keywords" in df_plot.columns:
                def _kw_to_str(v):
                    try:
                        if isinstance(v, (list, tuple)):
                            return ", ".join([str(x) for x in v if x is not None][:10])
                        if isinstance(v, str):
                            import ast as _ast  # type: ignore
                            parsed = None
                            try:
                                parsed = _ast.literal_eval(v)
                            except Exception:
                                parsed = None
                            if isinstance(parsed, (list, tuple)):
                                return ", ".join([str(x) for x in parsed if x is not None][:10])
                            return v
                    except Exception:
                        return str(v) if v is not None else ""
                df_plot["article_keywords_str"] = df_plot["article_keywords"].apply(_kw_to_str)
        except Exception:
            pass

        # Build a single-trace Scattergl with continuous color by topic_id to avoid huge legends
        try:
            color_vals = df_plot["topic_id"].astype(float)
        except Exception:
            color_vals = None
        # Customdata for hover: [topic_id, topic_label, article_id, article_path, article_keywords]
        try:
            import numpy as _np  # type: ignore
        except Exception:
            _np = None  # type: ignore
        try:
            cd_cols = []
            for c in ("topic_id", "topic_label", "article_id", "article_path", "article_keywords_str"):
                if c in df_plot.columns:
                    cd_cols.append(c)
                elif c == "article_keywords_str" and "article_keywords" in df_plot.columns:
                    cd_cols.append("article_keywords")
            custom = df_plot[cd_cols].to_numpy() if cd_cols else None
        except Exception:
            custom = None
        hovertemplate = "<b>%{customdata[1]}</b> (id=%{customdata[0]})<br>article: %{customdata[2]}<br>path: %{customdata[3]}<br>keywords: %{customdata[4]}<br>x=%{x:.3f}, y=%{y:.3f}<extra></extra>" if custom is not None and len(cd_cols) >= 5 else "x=%{x:.3f}, y=%{y:.3f}<br>topic=%{marker.color}<extra></extra>"
        scatter = go.Scattergl(
            x=df_plot["plot_x"],
            y=df_plot["plot_y"],
            mode="markers",
            customdata=custom,
            hovertemplate=hovertemplate,
            marker=dict(
                size=int(marker_size),
                opacity=float(marker_opacity),
                color=color_vals,
                colorscale="Turbo",
                line=dict(width=0),
                showscale=True,
            ),
        )
        fig = go.Figure(data=[scatter])
        fig.update_layout(
            title=title,
            showlegend=False,
            margin=dict(l=40, r=10, t=40, b=40),
        )
        # Prefer explicit W&B Plotly datatype for best compatibility; fallback to raw fig, then HTML
        logged = False
        try:
            from wandb.data_types import Plotly as _WbPlotly  # type: ignore
            wandb.log({"plots/" + title + "_plotly": _WbPlotly(fig)})
            logged = True
        except Exception:
            try:
                wandb.log({"plots/" + title + "_plotly": fig})
                logged = True
            except Exception:
                pass
        if not logged:
            try:
                html = fig.to_html(include_plotlyjs="cdn", full_html=False)
                wandb.log({"plots/" + title + "_html": wandb.Html(html)})
            except Exception:
                pass
        return fig
    except Exception:
        return None

# (Static image fallback removed)


