from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional
import os

from omegaconf import DictConfig, OmegaConf


def _to_dict(obj: Any) -> Dict[str, Any]:
    if isinstance(obj, DictConfig):
        return OmegaConf.to_container(obj, resolve=True)  # type: ignore[return-value]
    if isinstance(obj, dict):
        return dict(obj)
    raise TypeError(f"Unsupported mapping type: {type(obj)!r}")


def _infer_artifact_type(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext in {".parquet"}:
        return "parquet"
    if ext in {".csv"}:
        return "csv"
    if ext in {".json"}:
        return "json"
    if ext in {".ndjson"}:
        return "ndjson"
    if ext in {".txt"}:
        return "text"
    if ext in {".yaml", ".yml"}:
        return "yaml"
    if ext in {".log"}:
        return "log"
    if ext in {".pb", ".bin"}:
        return "binary"
    return "dir" if (not ext or ext == "") else ext.lstrip(".")


@dataclass
class ArtifactSpec:
    key: str
    type: str
    path: str
    optional: bool = False

    @classmethod
    def from_config(cls, key: str, value: Any) -> "ArtifactSpec":
        if isinstance(value, str):
            artifact_path = value
            art_type = _infer_artifact_type(artifact_path)
            return cls(key=key, type=art_type, path=artifact_path, optional=False)
        data = _to_dict(value)
        if "path" not in data:
            raise ValueError(f"Artifact '{key}' is missing required 'path'")
        path = str(data["path"])
        art_type = str(data.get("type") or _infer_artifact_type(path))
        optional = bool(data.get("optional", False))
        return cls(key=key, type=art_type, path=path, optional=optional)


@dataclass
class SourceSpec(ArtifactSpec):
    pass


@dataclass
class OutputSpec(ArtifactSpec):
    pass


@dataclass
class PipelineNodeSpec:
    key: str
    stage: str
    depends_on: List[str] = field(default_factory=list)
    inputs: Dict[str, str] = field(default_factory=dict)
    outputs: Dict[str, OutputSpec] = field(default_factory=dict)
    overrides: Dict[str, Any] = field(default_factory=dict)
    launcher: Optional[str] = None
    parallel_group: Optional[str] = None
    max_attempts: int = 1
    retry_backoff_s: float = 0.0
    wandb_suffix: Optional[str] = None

    @classmethod
    def from_config(cls, key: str, value: Any) -> "PipelineNodeSpec":
        data = _to_dict(value)
        stage = str(data.get("stage")) if data.get("stage") is not None else None
        if not stage:
            raise ValueError(f"Pipeline node '{key}' must define a 'stage'")
        depends_on_val = data.get("depends_on", [])
        if isinstance(depends_on_val, str):
            depends_on = [depends_on_val]
        else:
            depends_on = [str(dep) for dep in depends_on_val]
        inputs_val = data.get("inputs", {})
        inputs = {str(k): str(v) for k, v in _to_dict(inputs_val).items()} if inputs_val else {}
        outputs_val = data.get("outputs", {})
        outputs: Dict[str, OutputSpec] = {}
        for out_key, out_value in _to_dict(outputs_val).items():
            outputs[str(out_key)] = OutputSpec.from_config(str(out_key), out_value)
        overrides_val = data.get("overrides", {})
        overrides = _to_dict(overrides_val) if overrides_val else {}
        launcher = data.get("launcher")
        parallel_group = data.get("parallel_group")
        max_attempts = int(data.get("max_attempts", 1) or 1)
        retry_backoff_s = float(data.get("retry_backoff_s", 0.0) or 0.0)
        wandb_suffix = data.get("wandb_suffix")
        return cls(
            key=key,
            stage=str(stage),
            depends_on=depends_on,
            inputs=inputs,
            outputs=outputs,
            overrides=overrides,
            launcher=str(launcher) if launcher else None,
            parallel_group=str(parallel_group) if parallel_group else None,
            max_attempts=max_attempts,
            retry_backoff_s=retry_backoff_s,
            wandb_suffix=str(wandb_suffix) if wandb_suffix else None,
        )


@dataclass
class PipelineGraphSpec:
    sources: Dict[str, SourceSpec] = field(default_factory=dict)
    nodes: Dict[str, PipelineNodeSpec] = field(default_factory=dict)
    output_root: Optional[str] = None
    allow_partial: bool = False

    def topological_order(self) -> List[str]:
        indegree: Dict[str, int] = {node_id: 0 for node_id in self.nodes.keys()}
        for node in self.nodes.values():
            for dep in node.depends_on:
                if dep not in self.nodes:
                    raise ValueError(f"Node '{node.key}' depends on unknown node '{dep}'")
                indegree[node.key] += 1
        ready = [node_id for node_id, degree in indegree.items() if degree == 0]
        ordered: List[str] = []
        while ready:
            current = ready.pop(0)
            ordered.append(current)
            for node in self.nodes.values():
                if current in node.depends_on:
                    indegree[node.key] -= 1
                    if indegree[node.key] == 0:
                        ready.append(node.key)
        if len(ordered) != len(self.nodes):
            missing = set(self.nodes.keys()) - set(ordered)
            raise ValueError(f"Cycle detected in pipeline graph; unresolved nodes: {sorted(missing)}")
        return ordered


def load_pipeline_graph(cfg: DictConfig) -> PipelineGraphSpec:
    if "pipeline" not in cfg:
        raise ValueError("Configuration is missing required 'pipeline' section")
    pipeline_section = cfg.pipeline
    sources_cfg = getattr(pipeline_section, "sources", {})
    sources: Dict[str, SourceSpec] = {}
    for src_key, src_val in _to_dict(sources_cfg).items():
        sources[src_key] = SourceSpec.from_config(src_key, src_val)
    graph_cfg = getattr(pipeline_section, "graph", None)
    if graph_cfg is None:
        raise ValueError("'pipeline.graph' must be defined in the configuration")
    nodes_cfg = getattr(graph_cfg, "nodes", None)
    if nodes_cfg is None:
        raise ValueError("'pipeline.graph.nodes' must be defined in the configuration")
    nodes: Dict[str, PipelineNodeSpec] = {}
    for node_key, node_val in _to_dict(nodes_cfg).items():
        nodes[node_key] = PipelineNodeSpec.from_config(node_key, node_val)
    output_root = getattr(pipeline_section, "output_root", None)
    allow_partial = bool(getattr(pipeline_section, "allow_partial", False))
    return PipelineGraphSpec(
        sources=sources,
        nodes=nodes,
        output_root=str(output_root) if output_root else None,
        allow_partial=allow_partial,
    )


def resolve_output_root(graph_spec: PipelineGraphSpec, cfg: DictConfig) -> str:
    root = graph_spec.output_root
    if root:
        return os.path.abspath(os.path.expanduser(root))
    # Fallback to runtime.output_root or hydra.run.dir if provided
    runtime_root = getattr(getattr(cfg, "runtime", object()), "output_root", None)
    if runtime_root:
        return os.path.abspath(os.path.expanduser(str(runtime_root)))
    hydra_cfg = getattr(cfg, "hydra", None)
    try:
        hydra_run_dir = getattr(getattr(hydra_cfg, "run", object()), "dir", None)
    except Exception:
        hydra_run_dir = None
    if hydra_run_dir:
        return os.path.abspath(os.path.expanduser(str(hydra_run_dir)))
    return os.path.abspath(os.getcwd())


def iter_topologically(nodes: Dict[str, PipelineNodeSpec]) -> Iterable[PipelineNodeSpec]:
    graph = PipelineGraphSpec(nodes=nodes)
    for node_id in graph.topological_order():
        yield nodes[node_id]
