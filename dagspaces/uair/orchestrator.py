from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Mapping, Optional

import pandas as pd
from omegaconf import DictConfig, OmegaConf
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.core.hydra_config import HydraConfig

from .config_schema import (
    PipelineGraphSpec,
    PipelineNodeSpec,
    load_pipeline_graph,
    resolve_output_root,
)
from .stages.classify import run_classification_stage
from .stages.decompose import run_decomposition_stage
from .stages.taxonomy import run_taxonomy_stage
from .stages.topic import run_topic_stage
from .stages.verify import run_verification_stage
from .wandb_logger import WandbLogger

try:
    import ray  # type: ignore

    _RAY_AVAILABLE = True
except Exception:  # pragma: no cover - Ray optional dependency
    ray = None  # type: ignore
    _RAY_AVAILABLE = False

try:
    import submitit  # type: ignore
    _SUBMITIT_AVAILABLE = True
except Exception:
    submitit = None  # type: ignore
    _SUBMITIT_AVAILABLE = False


@dataclass
class StageExecutionContext:
    cfg: DictConfig
    node: PipelineNodeSpec
    inputs: Dict[str, str]
    output_paths: Dict[str, str]
    output_dir: str
    output_root: str
    logger: Optional['WandbLogger'] = None


@dataclass
class StageResult:
    outputs: Dict[str, str]
    metadata: Dict[str, Any] = field(default_factory=dict)


class StageRunner:
    stage_name: str

    def run(self, context: StageExecutionContext) -> StageResult:
        raise NotImplementedError


class ArtifactRegistry:
    def __init__(self) -> None:
        self._artifacts: Dict[str, str] = {}

    def register_source(self, name: str, path: str) -> None:
        self._artifacts[name] = path

    def register_outputs(self, node_key: str, outputs: Mapping[str, str]) -> None:
        for out_name, out_path in outputs.items():
            self._artifacts[f"{node_key}.{out_name}"] = out_path

    def resolve(self, ref: str) -> str:
        if ref in self._artifacts:
            return self._artifacts[ref]
        candidate = os.path.abspath(os.path.expanduser(ref))
        if os.path.exists(candidate) or os.path.isabs(ref):
            return candidate
        raise KeyError(f"Unknown artifact reference '{ref}'")

    def resolve_output_path(self, path: str, output_root: str, node_key: str) -> str:
        if not path:
            raise ValueError(f"Node '{node_key}' output path is empty")
        resolved = path
        if not os.path.isabs(resolved):
            resolved = os.path.join(output_root, resolved)
        return os.path.abspath(resolved)


def clone_config(cfg: DictConfig) -> DictConfig:
    return OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))  # type: ignore[return-value]


def merge_overrides(base_cfg: DictConfig, overrides: Optional[Mapping[str, Any]]) -> DictConfig:
    if not overrides:
        return base_cfg
    return OmegaConf.merge(base_cfg, OmegaConf.create(overrides))  # type: ignore[return-value]


def ensure_section(cfg: DictConfig, section: str) -> None:
    if OmegaConf.select(cfg, section) is None:
        OmegaConf.update(cfg, section, {}, merge=True)


def common_parent(paths: Iterable[str]) -> Optional[str]:
    try:
        parents = [os.path.dirname(p) for p in paths]
        if not parents:
            return None
        return os.path.commonpath(parents)
    except Exception:
        return None


def prepare_node_config(base_cfg: DictConfig, node: PipelineNodeSpec, output_dir: str) -> DictConfig:
    cfg_copy = clone_config(base_cfg)
    cfg_copy = merge_overrides(cfg_copy, node.overrides)
    ensure_section(cfg_copy, "runtime")
    OmegaConf.update(cfg_copy, "runtime.stage", node.stage, merge=True)
    OmegaConf.update(cfg_copy, "runtime.output_dir", output_dir, merge=True)
    OmegaConf.update(cfg_copy, "runtime.output_csv", None, merge=True)
    return cfg_copy


def _load_parquet_dataset(parquet_path: str, columns: Mapping[str, str], debug: bool, sample_n: Optional[int]) -> pd.DataFrame:
    if not isinstance(parquet_path, str) or parquet_path.strip() == "":
        raise ValueError("data.parquet_path is required")
    if not os.path.isabs(parquet_path):
        parquet_path = os.path.abspath(parquet_path)
    df = pd.read_parquet(parquet_path)
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
    if "article_text" not in df.columns and "chunk_text" not in df.columns:
        raise RuntimeError("Parquet missing required text column (article_text) or chunk_text")

    def _safe_str(x: Any) -> str:
        if x is None:
            return ""
        try:
            return "" if (isinstance(x, float) and pd.isna(x)) else str(x).strip()
        except Exception:
            return str(x) if x is not None else ""

    for column in ("article_path", "country", "year", "article_id"):
        if column not in df.columns:
            df[column] = None
        else:
            try:
                df[column] = df[column].apply(_safe_str)
            except Exception:
                pass

    legacy_cols = [
        "name",
        "public_description",
        "subscribers",
        "rule_text",
        "rule_index",
        "total_rules_count",
    ]
    drop_now = [col for col in legacy_cols if col in df.columns]
    if drop_now:
        df = df.drop(columns=drop_now)

    if debug and isinstance(sample_n, int) and sample_n > 0:
        try:
            n = min(int(sample_n), int(len(df)))
        except Exception:
            n = int(sample_n)
        try:
            seed_env = os.environ.get("UAIR_SAMPLE_SEED", "777")
            seed = int(seed_env) if seed_env is not None else 777
        except Exception:
            seed = 777
        try:
            df = df.sample(n=n, random_state=seed).reset_index(drop=True)
        except Exception:
            df = df.head(n)
    return df


def _prepare_streaming_dataset(dataset_path: str, columns: Mapping[str, str], cfg: DictConfig, stage: str) -> tuple[Optional[Any], bool]:
    if not _RAY_AVAILABLE:
        return None, False
    streaming_allowed = stage in {"classify", "taxonomy", "verification"}
    if not streaming_allowed:
        return None, False
    try:
        if not os.path.isabs(dataset_path):
            dataset_path = os.path.abspath(dataset_path)
        if not ray.is_initialized():
            namespace = os.environ.get("RAY_NAMESPACE") or os.environ.get("WANDB_GROUP") or "uair"
            try:
                ray.init(log_to_driver=True, namespace=str(namespace))
            except Exception:
                ray.init(log_to_driver=True)
        ds = ray.data.read_parquet(dataset_path)
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
                if dst not in out and src in row:
                    out[dst] = row.get(src)
            return out

        try:
            ds = ds.map(_ensure_canon)
        except Exception:
            pass
        debug = bool(getattr(cfg.runtime, "debug", False))
        sample_n = getattr(cfg.runtime, "sample_n", None)
        if debug and isinstance(sample_n, int) and sample_n > 0:
            try:
                ds = ds.limit(max(1, int(sample_n)))
            except Exception:
                pass
        return ds, True
    except Exception:
        return None, False


def prepare_stage_input(cfg: DictConfig, dataset_path: str, stage: str) -> tuple[Optional[pd.DataFrame], Optional[Any], bool]:
    debug = bool(getattr(cfg.runtime, "debug", False))
    sample_n = getattr(cfg.runtime, "sample_n", None)
    columns = dict(getattr(cfg.data, "columns", {})) if getattr(cfg, "data", None) else {}
    streaming_enabled = bool(getattr(cfg.runtime, "streaming_io", False))
    ds = None
    use_streaming = False
    if streaming_enabled:
        ds, use_streaming = _prepare_streaming_dataset(dataset_path, columns, cfg, stage)
    df = None
    if not use_streaming:
        df = _load_parquet_dataset(dataset_path, columns, debug=debug, sample_n=sample_n)
    return df, ds, use_streaming


def _collect_outputs(context: StageExecutionContext, optional: Mapping[str, bool]) -> Dict[str, str]:
    resolved: Dict[str, str] = {}
    for key, path in context.output_paths.items():
        if os.path.exists(path):
            resolved[key] = path
        else:
            if optional.get(key, False):
                continue
            raise FileNotFoundError(
                f"Expected output '{key}' for node '{context.node.key}' at '{path}' not found"
            )
    return resolved


class ClassificationRunner(StageRunner):
    stage_name = "classify"

    def run(self, context: StageExecutionContext) -> StageResult:
        dataset_path = context.inputs.get("dataset")
        if not dataset_path:
            raise ValueError(f"Node '{context.node.key}' requires 'dataset' input")
        cfg = context.cfg
        OmegaConf.update(cfg, "data.parquet_path", dataset_path, merge=True)
        df, ds, use_streaming = prepare_stage_input(cfg, dataset_path, self.stage_name)
        in_obj = ds if use_streaming and ds is not None else df
        out = run_classification_stage(in_obj, cfg)
        
        # Convert Ray Dataset to pandas if needed
        if hasattr(out, "to_pandas"):
            out = out.to_pandas()
        
        # Calculate row count
        row_count = None
        if isinstance(out, pd.DataFrame):
            row_count = len(out)
        elif isinstance(out, pd.Series):
            row_count = len(out)
        elif hasattr(out, "__len__"):
            try:
                row_count = len(out)
            except Exception:
                pass
        
        # Save outputs to disk
        if isinstance(out, pd.DataFrame):
            # Save "all" output (all classified articles)
            if "all" in context.output_paths:
                out.to_parquet(context.output_paths["all"], index=False)
            
            # Save "relevant" output (only relevant articles)
            if "relevant" in context.output_paths and "is_relevant" in out.columns:
                relevant_df = out[out["is_relevant"] == True].copy()
                relevant_df.to_parquet(context.output_paths["relevant"], index=False)
        
        # Log results table to wandb (in inspect_results panel group)
        if isinstance(out, pd.DataFrame) and context.logger:
            try:
                prefer_cols = ["article_id", "is_relevant", "article_text", "article_path", "country", "year"]
                context.logger.log_table(out, "classify/results", prefer_cols=prefer_cols, panel_group="inspect_results")
            except Exception as e:
                print(f"Warning: Failed to log classify results table to wandb: {e}", flush=True)
        
        metadata: Dict[str, Any] = {
            "rows": row_count,
            "streaming": bool(use_streaming),
        }
        outputs = _collect_outputs(
            context,
            {name: spec.optional for name, spec in context.node.outputs.items()},
        )
        return StageResult(outputs=outputs, metadata=metadata)


class TaxonomyRunner(StageRunner):
    stage_name = "taxonomy"

    def run(self, context: StageExecutionContext) -> StageResult:
        dataset_path = context.inputs.get("dataset")
        if not dataset_path:
            raise ValueError(f"Node '{context.node.key}' requires 'dataset' input")
        cfg = context.cfg
        OmegaConf.update(cfg, "data.parquet_path", dataset_path, merge=True)
        df, ds, use_streaming = prepare_stage_input(cfg, dataset_path, self.stage_name)
        in_obj = ds if use_streaming and ds is not None else df
        out = run_taxonomy_stage(in_obj, cfg)
        
        # Convert to pandas if needed
        if hasattr(out, "to_pandas"):
            out = out.to_pandas()
        
        # Save outputs to disk
        if isinstance(out, pd.DataFrame):
            for output_name, output_path in context.output_paths.items():
                out.to_parquet(output_path, index=False)
        
        # Log results table to wandb (in inspect_results panel group)
        if isinstance(out, pd.DataFrame) and context.logger:
            try:
                prefer_cols = ["article_id", "chunk_id", "taxonomy_json", "chunk_text", "article_path"]
                context.logger.log_table(out, "taxonomy/results", prefer_cols=prefer_cols, panel_group="inspect_results")
            except Exception as e:
                print(f"Warning: Failed to log taxonomy results table to wandb: {e}", flush=True)
        
        metadata: Dict[str, Any] = {
            "rows": len(out) if isinstance(out, pd.DataFrame) else None,
            "streaming": bool(use_streaming),
        }
        outputs = _collect_outputs(
            context,
            {name: spec.optional for name, spec in context.node.outputs.items()},
        )
        return StageResult(outputs=outputs, metadata=metadata)


class DecomposeRunner(StageRunner):
    stage_name = "decompose"

    def run(self, context: StageExecutionContext) -> StageResult:
        dataset_path = context.inputs.get("dataset")
        if not dataset_path:
            raise ValueError(f"Node '{context.node.key}' requires 'dataset' input")
        cfg = context.cfg
        OmegaConf.update(cfg, "data.parquet_path", dataset_path, merge=True)
        df, ds, use_streaming = prepare_stage_input(cfg, dataset_path, self.stage_name)
        in_obj = ds if use_streaming and ds is not None else df
        out = run_decomposition_stage(in_obj, cfg)
        
        # Convert to pandas if needed
        if hasattr(out, "to_pandas"):
            out = out.to_pandas()
        
        # Save outputs to disk
        if isinstance(out, pd.DataFrame):
            for output_name, output_path in context.output_paths.items():
                out.to_parquet(output_path, index=False)
        
        # Log results table to wandb (in inspect_results panel group)
        if isinstance(out, pd.DataFrame) and context.logger:
            try:
                prefer_cols = ["article_id", "chunk_id", "chunk_text", "article_path"]
                context.logger.log_table(out, "decompose/results", prefer_cols=prefer_cols, panel_group="inspect_results")
            except Exception as e:
                print(f"Warning: Failed to log decompose results table to wandb: {e}", flush=True)
        
        metadata: Dict[str, Any] = {
            "rows": len(out) if isinstance(out, pd.DataFrame) else None,
            "streaming": bool(use_streaming),
        }
        outputs = _collect_outputs(
            context,
            {name: spec.optional for name, spec in context.node.outputs.items()},
        )
        return StageResult(outputs=outputs, metadata=metadata)


class TopicRunner(StageRunner):
    stage_name = "topic"

    @staticmethod
    def _resolve_topic_path(path: str) -> str:
        if os.path.isdir(path):
            candidates = [
                os.path.join(path, "classify_relevant.parquet"),
                os.path.join(path, "classify_all.parquet"),
                os.path.join(path, "results.parquet"),
            ]
            for cand in candidates:
                if os.path.exists(cand):
                    return cand
        return path

    def run(self, context: StageExecutionContext) -> StageResult:
        dataset_path = context.inputs.get("dataset")
        if not dataset_path:
            raise ValueError(f"Node '{context.node.key}' requires 'dataset' input")
        resolved_path = self._resolve_topic_path(dataset_path)
        cfg = context.cfg
        OmegaConf.update(cfg, "data.parquet_path", resolved_path, merge=True)
        df, ds, use_streaming = prepare_stage_input(cfg, resolved_path, self.stage_name)
        in_obj = ds if use_streaming and ds is not None else df
        out = run_topic_stage(in_obj, cfg, logger=context.logger)
        
        # Convert to pandas if needed
        if hasattr(out, "to_pandas"):
            out = out.to_pandas()
        
        # Save outputs to disk
        if isinstance(out, pd.DataFrame):
            for output_name, output_path in context.output_paths.items():
                out.to_parquet(output_path, index=False)
        
        # Log results table and plots to wandb
        if isinstance(out, pd.DataFrame) and context.logger:
            try:
                # Log results table (in inspect_results panel group)
                prefer_cols = ["unit_id", "topic_id", "topic_prob", "topic_top_terms", "article_keywords", "plot_x", "plot_y"]
                context.logger.log_table(out, "topic/results", prefer_cols=prefer_cols, panel_group="inspect_results")
            except Exception as e:
                print(f"Warning: Failed to log topic results table to wandb: {e}", flush=True)
            
            try:
                # Log plotly visualization
                from .stages.topic_plot import log_cluster_scatter_plotly_to_wandb
                log_cluster_scatter_plotly_to_wandb(out, context.logger, title="topic_cluster_map")
            except Exception as e:
                print(f"Warning: Failed to log topic plot to wandb: {e}", flush=True)
        
        metadata: Dict[str, Any] = {
            "rows": len(out) if isinstance(out, pd.DataFrame) else None,
            "streaming": bool(use_streaming),
        }
        outputs = _collect_outputs(
            context,
            {name: spec.optional for name, spec in context.node.outputs.items()},
        )
        return StageResult(outputs=outputs, metadata=metadata)


class VerificationRunner(StageRunner):
    stage_name = "verification"

    def run(self, context: StageExecutionContext) -> StageResult:
        dataset_path = context.inputs.get("dataset")
        if not dataset_path:
            raise ValueError(f"Node '{context.node.key}' requires 'dataset' input")
        cfg = context.cfg
        OmegaConf.update(cfg, "data.parquet_path", dataset_path, merge=True)
        df, ds, use_streaming = prepare_stage_input(cfg, dataset_path, self.stage_name)
        in_obj = ds if use_streaming and ds is not None else df
        out = run_verification_stage(in_obj, cfg)
        
        # Convert to pandas if needed
        if hasattr(out, "to_pandas"):
            out = out.to_pandas()
        
        # Save outputs to disk
        if isinstance(out, pd.DataFrame):
            for output_name, output_path in context.output_paths.items():
                out.to_parquet(output_path, index=False)
        
        # Log results table to wandb (in inspect_results panel group)
        if isinstance(out, pd.DataFrame) and context.logger:
            try:
                prefer_cols = ["article_id", "chunk_id", "chunk_text", "verification_result", "article_path"]
                context.logger.log_table(out, "verification/results", prefer_cols=prefer_cols, panel_group="inspect_results")
            except Exception as e:
                print(f"Warning: Failed to log verification results table to wandb: {e}", flush=True)
        
        metadata: Dict[str, Any] = {
            "rows": len(out) if isinstance(out, pd.DataFrame) else None,
            "streaming": bool(use_streaming),
        }
        outputs = _collect_outputs(
            context,
            {name: spec.optional for name, spec in context.node.outputs.items()},
        )
        return StageResult(outputs=outputs, metadata=metadata)


_STAGE_REGISTRY: Dict[str, StageRunner] = {
    "classify": ClassificationRunner(),
    "decompose": DecomposeRunner(),
    "taxonomy": TaxonomyRunner(),
    "topic": TopicRunner(),
    "verification": VerificationRunner(),
}


def _ensure_output_dirs(paths: Iterable[str]) -> None:
    for path in paths:
        os.makedirs(os.path.dirname(path), exist_ok=True)


def _node_optional_outputs(node: PipelineNodeSpec) -> Dict[str, bool]:
    return {name: spec.optional for name, spec in node.outputs.items()}


def _node_output_paths(node: PipelineNodeSpec, registry: ArtifactRegistry, output_root: str) -> Dict[str, str]:
    resolved: Dict[str, str] = {}
    for out_name, spec in node.outputs.items():
        resolved[out_name] = registry.resolve_output_path(spec.path, output_root, node.key)
    _ensure_output_dirs(resolved.values())
    return resolved


def _node_inputs(node: PipelineNodeSpec, registry: ArtifactRegistry) -> Dict[str, str]:
    resolved: Dict[str, str] = {}
    for alias, ref in node.inputs.items():
        resolved[alias] = registry.resolve(ref)
    return resolved


def _print_status(payload: Dict[str, Any]) -> None:
    try:
        print(json.dumps(payload, indent=2))
    except Exception:
        pass


def _load_launcher_config(cfg: DictConfig, launcher_name: str) -> Optional[DictConfig]:
    """Load a launcher configuration from Hydra config."""
    try:
        # Find the config path - use the location of this file as reference
        config_path = os.path.join(os.path.dirname(__file__), "conf")
        
        if not os.path.exists(config_path):
            # Try to get from hydra runtime
            hydra_cfg = getattr(cfg, "hydra", None)
            if hydra_cfg:
                runtime_cfg = getattr(hydra_cfg, "runtime", None)
                if runtime_cfg:
                    sources = getattr(runtime_cfg, "config_sources", [])
                    for source in sources:
                        if hasattr(source, "provider") and source.provider == "main":
                            config_path = source.path
                            break
        
        if not config_path or not os.path.exists(config_path):
            raise ValueError(f"Could not find config directory")
            
        launcher_file = os.path.join(config_path, "hydra", "launcher", f"{launcher_name}.yaml")
        if not os.path.exists(launcher_file):
            raise ValueError(f"Launcher config file not found: {launcher_file}")
        
        # Load the launcher config
        launcher_cfg = OmegaConf.load(launcher_file)
        # Resolve interpolations with the main config as context
        launcher_cfg = OmegaConf.merge({"runtime": cfg.get("runtime", {})}, launcher_cfg)
        return launcher_cfg
    except Exception as e:
        raise ValueError(f"Failed to load launcher config '{launcher_name}': {e}") from e


def _create_submitit_executor(launcher_cfg: DictConfig, job_name: str, log_folder: str) -> Any:
    """Create a submitit executor from launcher configuration."""
    if not _SUBMITIT_AVAILABLE or submitit is None:
        raise RuntimeError("submitit is not available but is required for SLURM job submission")
    
    executor = submitit.AutoExecutor(folder=log_folder)
    
    # Map launcher config to submitit parameters
    executor.update_parameters(
        timeout_min=int(launcher_cfg.get("timeout_min", 120)),
        slurm_partition=str(launcher_cfg.get("partition", "pierson")),
        slurm_mem=f"{int(launcher_cfg.get('mem_gb', 8))}GB",
        slurm_cpus_per_task=int(launcher_cfg.get("cpus_per_task", 2)),
        slurm_gpus_per_node=int(launcher_cfg.get("gpus_per_node", 0)),
        slurm_nodes=int(launcher_cfg.get("nodes", 1)),
        slurm_tasks_per_node=int(launcher_cfg.get("tasks_per_node", 1)),
        slurm_array_parallelism=int(launcher_cfg.get("array_parallelism", 1)),
        name=job_name,
        slurm_additional_parameters=launcher_cfg.get("additional_parameters", {}),
        slurm_setup=launcher_cfg.get("setup", []),
    )
    
    return executor


def execute_stage_job(context_data: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a single stage - designed to be submitted as a SLURM job."""
    # Reconstruct context from serialized data
    from omegaconf import OmegaConf
    cfg = OmegaConf.create(context_data["cfg"])
    node_dict = context_data["node"]
    
    # Reconstruct PipelineNodeSpec
    from .config_schema import PipelineNodeSpec, OutputSpec
    outputs = {}
    for out_key, out_val in node_dict.get("outputs", {}).items():
        outputs[out_key] = OutputSpec.from_config(out_key, out_val)
    
    node = PipelineNodeSpec(
        key=node_dict["key"],
        stage=node_dict["stage"],
        depends_on=node_dict.get("depends_on", []),
        inputs=node_dict.get("inputs", {}),
        outputs=outputs,
        overrides=node_dict.get("overrides", {}),
        launcher=node_dict.get("launcher"),
        parallel_group=node_dict.get("parallel_group"),
        max_attempts=node_dict.get("max_attempts", 1),
        retry_backoff_s=node_dict.get("retry_backoff_s", 0.0),
        wandb_suffix=node_dict.get("wandb_suffix"),
    )
    
    context = StageExecutionContext(
        cfg=cfg,
        node=node,
        inputs=context_data["inputs"],
        output_paths=context_data["output_paths"],
        output_dir=context_data["output_dir"],
        output_root=context_data["output_root"],
    )
    
    # Get the stage runner
    stage_registry = dict(_STAGE_REGISTRY)
    runner = stage_registry.get(node.stage)
    if runner is None:
        raise ValueError(f"No runner registered for stage '{node.stage}' (node '{node.key}')")
    
    # Execute stage with wandb logging context
    wandb_run_id = node.wandb_suffix or node.key
    run_config = {
        "node": node.key,
        "stage": node.stage,
        "inputs": list(context.inputs.keys()),
        "outputs": list(context.output_paths.keys()),
    }
    
    with WandbLogger(cfg, stage=node.stage, run_id=wandb_run_id, run_config=run_config) as logger:
        try:
            # Update context with logger
            context.logger = logger
            
            # Execute the stage
            _print_status({"node": node.key, "stage": node.stage, "status": "running", "inputs": context.inputs})
            stage_start = time.time()
            
            result = runner.run(context)
            
            # Log completion metrics
            duration_s = time.time() - stage_start
            try:
                logger.set_summary(f"{node.stage}/status", "completed")
            except Exception:
                pass
            logger.log_metrics({
                f"{node.stage}/duration_s": duration_s,
                f"{node.stage}/rows_processed": result.metadata.get("rows", 0),
            })
            
            return {
                "outputs": result.outputs,
                "metadata": result.metadata,
            }
        except Exception as e:
            # Log failure
            try:
                logger.set_summary(f"{node.stage}/status", "failed")
                logger.set_summary(f"{node.stage}/error", str(e))
            except Exception:
                pass
            raise


def run_experiment(cfg: DictConfig) -> None:
    # Execute entire pipeline with wandb logging context
    with WandbLogger(cfg, stage="orchestrator", run_id="monitor", run_config={"type": "pipeline"}) as logger:
        try:
            # Get the parent/monitor group ID to pass to child jobs
            # This ensures all stages in one pipeline run are grouped together
            parent_group = logger.wb_config.group if logger.wb_config else None
            if parent_group:
                # Set in environment so child jobs can inherit it
                os.environ["WANDB_GROUP"] = parent_group
                print(f"[orchestrator] Setting WANDB_GROUP={parent_group} for child stages", flush=True)
            
            graph_spec: PipelineGraphSpec = load_pipeline_graph(cfg)
            output_root = resolve_output_root(graph_spec, cfg)
            os.makedirs(output_root, exist_ok=True)
            registry = ArtifactRegistry()
            for source_key, source in graph_spec.sources.items():
                path = source.path
                if not os.path.isabs(path):
                    path = os.path.abspath(os.path.expanduser(path))
                registry.register_source(source_key, path)
            manifest: Dict[str, Any] = {
                "output_root": output_root,
                "nodes": {},
            }
            stage_registry = dict(_STAGE_REGISTRY)
            ordered_nodes = graph_spec.topological_order()
            pipeline_start = time.time()
            
            # Log pipeline structure to wandb: numeric to charts; structure to config
            logger.log_metrics({
                "orchestrator/total_nodes": len(ordered_nodes),
            })
            try:
                logger.set_config({
                    "orchestrator": {
                        "node_order": ordered_nodes,
                        "total_nodes": len(ordered_nodes),
                    }
                })
            except Exception:
                pass
            
            for node_key in ordered_nodes:
                node = graph_spec.nodes[node_key]
                runner = stage_registry.get(node.stage)
                if runner is None:
                    raise ValueError(f"No runner registered for stage '{node.stage}' (node '{node.key}')")
                inputs = _node_inputs(node, registry)
                output_paths = _node_output_paths(node, registry, output_root)
                output_dir = common_parent(output_paths.values())
                if not output_dir:
                    output_dir = os.path.join(output_root, node.key)
                os.makedirs(output_dir, exist_ok=True)
                node_cfg = prepare_node_config(cfg, node, output_dir)
                context = StageExecutionContext(
                    cfg=node_cfg,
                    node=node,
                    inputs=inputs,
                    output_paths=output_paths,
                    output_dir=output_dir,
                    output_root=output_root,
                )
                
                node_start = time.time()
                
                # Check if this node should be launched as a separate SLURM job
                if node.launcher:
                    _print_status({"node": node.key, "stage": node.stage, "status": "submitting", "launcher": node.launcher, "inputs": inputs})
                    try:
                        launcher_cfg = _load_launcher_config(cfg, node.launcher)
                    except ValueError as e:
                        raise ValueError(f"Could not load launcher config '{node.launcher}' for node '{node.key}': {e}") from e
                    
                    # Create submitit executor - store logs in the Hydra multirun directory
                    # Structure: multirun/YYYY-MM-DD/HH-MM-SS/0/.slurm_jobs/STAGE_NAME/
                    log_folder = None
                    try:
                        # Priority 1: Use HydraConfig to get runtime output directory
                        hydra_cfg = HydraConfig.get()
                        if hydra_cfg and hydra_cfg.runtime and hydra_cfg.runtime.output_dir:
                            hydra_output_dir = hydra_cfg.runtime.output_dir
                            log_folder = os.path.join(hydra_output_dir, ".slurm_jobs", node.key)
                            _print_status({"debug": "using_hydra_output_dir", "log_folder": log_folder})
                    except Exception as e:
                        _print_status({"debug": "hydra_config_error", "error": str(e)})
                    
                    # Priority 2: Fall back to output_root
                    if not log_folder:
                        log_folder = os.path.join(output_root, ".slurm_jobs", node.key)
                        _print_status({"debug": "using_output_root_fallback", "log_folder": log_folder, "output_root": output_root})
                    
                    log_folder = os.path.abspath(log_folder)
                    os.makedirs(log_folder, exist_ok=True)
                    job_name = f"UAIR-{node.key}"
                    executor = _create_submitit_executor(launcher_cfg, job_name, log_folder)
                    
                    # Ensure child job uses parent's W&B group for proper grouping
                    # Submitit doesn't auto-inherit env vars, so we need to explicitly set them
                    if parent_group:
                        # Method 1: Set environment variable on executor
                        # This ensures it's available in the SLURM job's environment
                        try:
                            # Get current setup commands and prepend WANDB_GROUP export
                            current_setup = list(launcher_cfg.get("setup", []))
                            # Insert explicit WANDB_GROUP export at the beginning (after shebang/source commands)
                            # Find insertion point (after source commands)
                            insert_idx = 0
                            for i, cmd in enumerate(current_setup):
                                if "source" in cmd or "export HYDRA_FULL_ERROR" in cmd:
                                    insert_idx = i + 1
                            # Insert WANDB_GROUP export
                            wandb_group_export = f"export WANDB_GROUP={parent_group}"
                            if wandb_group_export not in current_setup:
                                current_setup.insert(insert_idx, wandb_group_export)
                                executor.update_parameters(slurm_setup=current_setup)
                                _print_status({"debug": "injected_wandb_group", "group": parent_group, "node": node.key})
                        except Exception as e:
                            _print_status({"debug": "failed_to_inject_wandb_group", "error": str(e)})
                    
                    # Prepare serializable context data
                    context_data = {
                        "cfg": OmegaConf.to_container(node_cfg, resolve=True),
                        "node": {
                            "key": node.key,
                            "stage": node.stage,
                            "depends_on": node.depends_on,
                            "inputs": node.inputs,
                            "outputs": {k: {"path": v.path, "type": v.type, "optional": v.optional} for k, v in node.outputs.items()},
                            "overrides": node.overrides,
                            "launcher": node.launcher,
                            "parallel_group": node.parallel_group,
                            "max_attempts": node.max_attempts,
                            "retry_backoff_s": node.retry_backoff_s,
                            "wandb_suffix": node.wandb_suffix,
                        },
                        "inputs": inputs,
                        "output_paths": output_paths,
                        "output_dir": output_dir,
                        "output_root": output_root,
                    }
                    
                    # Submit the job
                    job = executor.submit(execute_stage_job, context_data)
                    _print_status({"node": node.key, "stage": node.stage, "status": "submitted", "job_id": job.job_id})
                    
                    # Wait for the job to complete
                    try:
                        job_result = job.result()  # This blocks until the job completes
                        result = StageResult(
                            outputs=job_result["outputs"],
                            metadata=job_result["metadata"],
                        )
                    except Exception as exc:
                        _print_status({"node": node.key, "stage": node.stage, "status": "failed", "job_id": job.job_id, "error": str(exc)})
                        raise
                else:
                    # Run locally in the current process
                    _print_status({"node": node.key, "stage": node.stage, "status": "running", "inputs": inputs})
                    try:
                        result = runner.run(context)
                    except Exception as exc:
                        _print_status({"node": node.key, "stage": node.stage, "status": "failed", "error": str(exc)})
                        raise
                
                registry.register_outputs(node.key, result.outputs)
                duration = time.time() - node_start
                manifest["nodes"][node.key] = {
                    "stage": node.stage,
                    "inputs": inputs,
                    "outputs": result.outputs,
                    "metadata": result.metadata,
                    "duration_s": duration,
                }
                _print_status({
                    "node": node.key,
                    "stage": node.stage,
                    "status": "completed",
                    "duration_s": round(duration, 3),
                    "outputs": result.outputs,
                })
            
            manifest_path = os.path.join(output_root, "pipeline_manifest.json")
            try:
                with open(manifest_path, "w", encoding="utf-8") as fh:
                    json.dump(manifest, fh, indent=2)
            except Exception:
                pass
            total_duration = time.time() - pipeline_start
            
            # Log final pipeline metrics to wandb
            try:
                logger.set_summary("orchestrator/status", "completed")
            except Exception:
                pass
            logger.log_metrics({
                "orchestrator/total_duration_s": round(total_duration, 3),
                "orchestrator/nodes_completed": len(manifest["nodes"]),
            })
            
            _print_status({
                "pipeline": {
                    "output_root": output_root,
                    "nodes": ordered_nodes,
                    "duration_s": round(total_duration, 3),
                    "manifest": manifest_path,
                }
            })
        except Exception as e:
            try:
                logger.set_summary("orchestrator/status", "failed")
                logger.set_summary("orchestrator/error", str(e))
            except Exception:
                pass
            raise
