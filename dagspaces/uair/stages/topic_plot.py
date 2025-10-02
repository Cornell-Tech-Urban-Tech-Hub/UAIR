from __future__ import annotations

from typing import Optional
import os


def log_cluster_scatter_to_wandb(df, logger=None, title: str = "topic_cluster_map") -> Optional[object]:
    """Log cluster scatter plot to wandb using wandb.Table (legacy).
    
    Args:
        df: DataFrame with plot data
        logger: Optional WandbLogger instance
        title: Plot title
    """
    if not logger:
        return None
    try:
        import wandb  # type: ignore
    except Exception:
        return None
    if df is None or len(df) == 0:
        return None
    # Guard: skip heavy wandb.Table scatter for large datasets
    try:
        max_rows_env = os.environ.get("UAIR_TOPIC_TABLE_MAX_ROWS")
        table_limit = int(max_rows_env) if (max_rows_env and max_rows_env.isdigit()) else 20000
    except Exception:
        table_limit = 20000
    try:
        if len(df) > table_limit:
            # Downsample with fixed seed instead of skipping
            try:
                seed_env = os.environ.get("UAIR_WB_TABLE_SEED") or os.environ.get("UAIR_TABLE_SAMPLE_SEED")
                seed = int(seed_env) if seed_env else 777
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
        logger.log_plot("plots/" + title, plt)
        return plt
    except Exception:
        return None

def log_cluster_scatter_plotly_to_wandb(df, logger=None, title: str = "topic_cluster_map") -> Optional[object]:
    """Log cluster scatter plot to wandb.
    
    Args:
        df: DataFrame with plot data
        logger: Optional WandbLogger instance
        title: Plot title
    """
    if not logger:
        return None
    try:
        import wandb  # type: ignore
        import plotly.express as px  # type: ignore
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

    # Read plotting configuration from environment variables (with fallback defaults)
    def _get_env_int(name, default_val):
        try:
            v = os.environ.get(name)
            if v is None:
                return int(default_val)
            return int(v)
        except Exception:
            return int(default_val)

    # Configuration defaults (can be overridden via environment variables)
    seed = 777
    max_points = _get_env_int("UAIR_TOPIC_PLOT_MAX_POINTS", 120000)
    heat_threshold = _get_env_int("UAIR_TOPIC_PLOT_HEATMAP_THRESHOLD", 250000)
    heat_bins = _get_env_int("UAIR_TOPIC_HEATMAP_BINS", 200)
    marker_size = _get_env_int("UAIR_TOPIC_PLOT_MARKER_SIZE", 3)
    try:
        op_env = os.environ.get("UAIR_TOPIC_PLOT_MARKER_OPACITY")
        marker_opacity = float(op_env) if op_env is not None else 0.6
    except Exception:
        marker_opacity = 0.6
    method = os.environ.get("UAIR_TOPIC_PLOT_METHOD", "auto").strip().lower()

    # Default large-plot optimizations: enable when n > 10k
    optimize_threshold = _get_env_int("UAIR_TOPIC_PLOT_OPTIMIZE_THRESHOLD", 10000)
    optimize_cap = _get_env_int("UAIR_TOPIC_PLOT_MAX_POINTS_LARGE", 10000)
    optimize_large = n > int(optimize_threshold)
    target_max_points = int(min(max_points, optimize_cap)) if optimize_large else int(max_points)

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
            logger.log_plot("plots/" + title + "_plotly", fig)
            return fig

        # Otherwise, scattergl with optional downsampling
        df_plot = df2
        try:
            mp = int(target_max_points)
            if n > mp and mp > 0:
                # Sample per-topic to preserve cluster structure
                try:
                    import pandas as _pd  # type: ignore
                except Exception:
                    _pd = None  # type: ignore
                try:
                    unique_topics = df2["topic_id"].nunique(dropna=False)
                except Exception:
                    unique_topics = 1
                per_cap = max(1, int(mp // max(1, int(unique_topics))))
                try:
                    df_plot = (
                        df2.groupby("topic_id", dropna=False, group_keys=False)
                        .apply(lambda g: g.sample(n=min(len(g), per_cap), random_state=int(seed)))
                        .reset_index(drop=True)
                    )
                except Exception:
                    df_plot = df2.sample(n=int(mp), random_state=int(seed))
        except Exception:
            pass

        # Prepare hover fields and labels (rich hover disabled for large plots)
        enable_rich_hover = not optimize_large
        if enable_rich_hover:
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
        custom = None
        hovertemplate = None
        if enable_rich_hover:
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
            hovertemplate = "<b>%{customdata[1]}</b> (id=%{customdata[0]})<br>article: %{customdata[2]}<br>path: %{customdata[3]}<br>keywords: %{customdata[4]}<br>x=%{x:.3f}, y=%{y:.3f}<extra></extra>" if custom is not None and 'cd_cols' in locals() and len(cd_cols) >= 5 else "x=%{x:.3f}, y=%{y:.3f}<br>topic=%{marker.color}<extra></extra>"

        scatter = go.Scattergl(
            x=df_plot["plot_x"],
            y=df_plot["plot_y"],
            mode="markers",
            customdata=custom,
            hovertemplate=(hovertemplate if enable_rich_hover else None),
            hoverinfo=(None if enable_rich_hover else "skip"),
            marker=dict(
                size=(int(marker_size) if not optimize_large else max(1, min(int(marker_size), 2))),
                opacity=(float(marker_opacity) if not optimize_large else 0.5),
                color=color_vals,
                colorscale="Turbo",
                line=dict(width=0),
                showscale=(False if optimize_large else True),
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
            logger.log_plot("plots/" + title + "_plotly", _WbPlotly(fig))
            logged = True
        except Exception:
            try:
                logger.log_plot("plots/" + title + "_plotly", fig)
                logged = True
            except Exception:
                pass
        if not logged:
            try:
                import wandb  # type: ignore
                html = fig.to_html(include_plotlyjs="cdn", full_html=False)
                logger.log_plot("plots/" + title + "_html", wandb.Html(html))
            except Exception:
                pass
        return fig
    except Exception:
        return None

# (Static image fallback removed)


