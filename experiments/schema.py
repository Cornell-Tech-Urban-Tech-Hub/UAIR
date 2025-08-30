from dataclasses import dataclass, field
from typing import Optional, Any, List

from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf


@dataclass
class ModelRuntime:
    max_model_len_tokens: int = 8192
    gpu_memory_utilization: float = 0.65
    tensor_parallel_size: int = 2
    batch_size: int = 16
    concurrency: int = 1
    tokenizer_pool_size: Optional[int] = 4


@dataclass
class RelevanceCfg:
    model_source: str = ""
    tensor_parallel_size: int = 2
    batch_size: int = 16
    concurrency: int = 1
    max_model_len_tokens: int = 8192
    max_output_tokens: int = 8
    safety_margin_tokens: int = 2048
    chunk_overlap_tokens: int = 512
    gpu_memory_utilization: float = 0.65
    prefilter_mode: str = "pre_gating"
    disable_keyword_prefilter: bool = False
    disable_chunking: bool = False
    kv_cache_dtype: str = "auto"
    max_num_batched_tokens: int = 3072
    max_num_seqs: int = 4
    conservative_vllm: bool = False
    log_rationales: bool = False
    tokenizer_pool_size: Optional[int] = None


@dataclass
class TaxonomyCfg:
    model_source: str = ""
    tensor_parallel_size: int = 2
    batch_size: int = 16
    concurrency: int = 1
    max_model_len_tokens: int = 8192
    max_output_tokens: int = 8
    gpu_memory_utilization: float = 0.65
    kv_cache_dtype: str = "auto"
    max_num_batched_tokens: int = 3072
    max_num_seqs: int = 4
    tokenizer_pool_size: Optional[int] = None


@dataclass
class VerifyCfg:
    method: str = "combo"
    top_k: int = 3
    thresholds: str = "sim=0.55,ent=0.85,contra=0.05"
    device: str = "cpu"
    output: str = "verification"


@dataclass
class RuntimeCfg:
    num_cpus: int = 6
    num_gpus: int = 2


@dataclass
class WandbCfg:
    mode: str = "disabled"
    project: str = "sensing-ai-risks"
    prefix: str = ""
    suffix: str = ""


@dataclass
class DebugCfg:
    enable: bool = False
    limit: int = 10


@dataclass
class PipelineCfg:
    in_process: bool = False
    dry_run: bool = False


@dataclass
class AppConfig:
    model_runtime: ModelRuntime = field(default_factory=ModelRuntime)
    relevance: RelevanceCfg = field(default_factory=RelevanceCfg)
    taxonomy: TaxonomyCfg = field(default_factory=TaxonomyCfg)
    verify: VerifyCfg = field(default_factory=VerifyCfg)
    resources: RuntimeCfg = field(default_factory=RuntimeCfg)
    wandb: WandbCfg = field(default_factory=WandbCfg)
    debug: DebugCfg = field(default_factory=DebugCfg)
    pipeline: PipelineCfg = field(default_factory=PipelineCfg)
    seed: int = 777
    output_dir: str = ""
    data: Any = None


cs = ConfigStore.instance()
cs.store(name="schema_config", node=AppConfig)


def validate_and_resolve(cfg: DictConfig) -> DictConfig:
    # Compose onto the schema to validate types and fill defaults
    schema = OmegaConf.structured(AppConfig)
    merged = OmegaConf.merge(schema, cfg)
    return merged
