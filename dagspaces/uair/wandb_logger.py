"""Centralized W&B logging for UAIR pipeline.

This module provides a unified interface for W&B logging across all pipeline stages,
handling distributed execution (Ray, SLURM) and run lifecycle management.

Key features:
- Single source of truth for W&B configuration
- Proper context management for runs
- Online mode by default for real-time syncing (configurable via WANDB_MODE)
- Service daemon completely disabled using wandb.Settings() API
- Ray-aware: automatically skips initialization in Ray workers to avoid socket conflicts
- Thread-safe logging
- Graceful degradation when W&B is unavailable

Configuration (wandb 0.22.0+ best practices):
- Service daemon disabled via WANDB_DISABLE_SERVICE environment variable
- In-process mode is used instead of service daemon for SLURM/Ray compatibility
- Parameters like 'mode' and 'dir' passed directly to wandb.init()
- Settings() object used only for specific options (disable_git, disable_job_creation, etc.)
- WANDB_DIR (optional): specify writable directory for wandb files (defaults to SLURM_SUBMIT_DIR or CWD)
- WANDB_MODE: set to "offline" for deferred syncing, "online" (default) for real-time

Best Practices:
- Use WandbLogger context manager for run lifecycle
- All logging goes through log_metrics/log_table/log_artifact methods
- Never call wandb.init() or wandb.finish() directly outside this module
- Settings object properly configures wandb for distributed environments

References:
- wandb.Settings docs: https://docs.wandb.ai/ref/python/settings
- Distributed training guide: https://docs.wandb.ai/guides/track/advanced/distributed-training
"""

from __future__ import annotations

import os
import sys
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
import json
import platform
import socket
import traceback
import getpass
import shutil

# Configure wandb for distributed environments (SLURM, Ray)
# MUST be set before importing wandb
# Set wandb temp directory to a shared location accessible across SLURM nodes
# This prevents socket file issues when jobs run on different nodes (e.g., lisbeth vs klara)
if "TMPDIR" not in os.environ:
    shared_tmp = "/share/pierson/matt/tmp/wandb"
    os.makedirs(shared_tmp, exist_ok=True)
    os.environ["TMPDIR"] = shared_tmp

# Apply defaults from repo-level wandb/settings before importing wandb
def _apply_wandb_settings_defaults() -> None:
    try:
        # Allow explicit override via environment
        settings_path = os.environ.get("WANDB_SETTINGS_PATH")
        if not settings_path:
            # Default to repo root: <repo>/wandb/settings
            base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
            settings_path = os.path.join(base_dir, "wandb", "settings")
        if not (isinstance(settings_path, str) and os.path.exists(settings_path)):
            return
        import configparser  # Local import to avoid global dependency
        cp = configparser.ConfigParser()
        try:
            cp.read(settings_path)
        except Exception:
            return
        sect = "default" if cp.has_section("default") else (cp.sections()[0] if cp.sections() else None)
        if not sect:
            return
        sec = cp[sect]
        # Extract values; only set env vars if not already present to honor explicit overrides
        try:
            entity = sec.get("entity", fallback=None)
        except Exception:
            entity = None
        try:
            project = sec.get("project", fallback=None)
        except Exception:
            project = None
        try:
            base_url = sec.get("base_url", fallback=None)
        except Exception:
            base_url = None
        if entity and not os.environ.get("WANDB_ENTITY"):
            os.environ["WANDB_ENTITY"] = str(entity)
        if project and not os.environ.get("WANDB_PROJECT"):
            os.environ["WANDB_PROJECT"] = str(project)
        if base_url and not os.environ.get("WANDB_BASE_URL"):
            os.environ["WANDB_BASE_URL"] = str(base_url)
    except Exception:
        # Silent best-effort; never fail pipeline due to settings parsing
        pass

_apply_wandb_settings_defaults()

# Import wandb after environment is configured
import wandb as wandb_module

# In-process mode: do not require legacy service; service is disabled via env

# Import base classes after wandb is configured
from omegaconf import DictConfig, OmegaConf

@dataclass
class WandbConfig:
    """W&B configuration extracted from Hydra config."""
    
    enabled: bool = False
    project: str = "UAIR"
    entity: Optional[str] = None
    group: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    table_sample_rows: int = 1000
    table_sample_seed: int = 777
    
    @classmethod
    def from_hydra_config(cls, cfg) -> "WandbConfig":
        """Extract W&B config from Hydra config."""
        try:
            wandb_cfg = getattr(cfg, "wandb", None)
            # Pick up environment defaults (possibly injected from wandb/settings)
            env_entity = os.environ.get("WANDB_ENTITY")
            env_project = os.environ.get("WANDB_PROJECT")
            if wandb_cfg is None:
                return cls(
                    enabled=False,
                    project=(env_project or "UAIR"),
                    entity=(env_entity if env_entity and env_entity.strip() else None),
                    group=_get_group_from_config(cfg),
                    tags=[],
                    table_sample_rows=1000,
                    table_sample_seed=777,
                )
            
            # Resolve project with fallback to environment, then default
            proj_attr = getattr(wandb_cfg, "project", None)
            if proj_attr is None or str(proj_attr).strip() == "":
                project = env_project or "UAIR"
            else:
                project = str(proj_attr or "UAIR")
            # Resolve entity with fallback to environment
            entity_cfg = _get_optional_str(wandb_cfg, "entity")
            entity = entity_cfg or (env_entity if env_entity and env_entity.strip() else None)
            return cls(
                enabled=bool(getattr(wandb_cfg, "enabled", False)),
                project=project,
                entity=entity,
                group=_get_group_from_config(cfg),
                tags=_get_list(wandb_cfg, "tags"),
                table_sample_rows=int(getattr(wandb_cfg, "table_sample_rows", 1000)),
                table_sample_seed=int(getattr(wandb_cfg, "table_sample_seed", 777)),
            )
        except Exception as e:
            print(f"[wandb] Warning: Failed to parse config: {e}", file=sys.stderr)
            return cls()


def _get_optional_str(obj, attr: str) -> Optional[str]:
    """Get optional string attribute, returning None if empty."""
    try:
        val = getattr(obj, attr, None)
        if val is not None and str(val).strip():
            return str(val)
    except Exception:
        pass
    return None


def _get_list(obj, attr: str) -> List[str]:
    """Get list attribute safely."""
    try:
        val = getattr(obj, attr, None)
        if val is None:
            return []
        if isinstance(val, (list, tuple)):
            return [str(x) for x in val]
        return [str(val)]
    except Exception:
        return []


def _get_group_from_config(cfg) -> Optional[str]:
    """Extract group from config with fallback to environment variables.
    
    Priority order:
    1. cfg.wandb.group
    2. WANDB_GROUP env var
    3. SUBMITIT_JOB_ID (parent SLURM job)
    4. SLURM_JOB_ID (current SLURM job)
    """
    # Priority 1: Explicit config
    try:
        grp = getattr(cfg.wandb, "group", None)
        if grp and str(grp).strip():
            return str(grp)
    except Exception:
        pass
    
    # Priority 2: Environment variable
    env_group = os.environ.get("WANDB_GROUP")
    if env_group and env_group.strip():
        return env_group
    
    # Priority 3: Submitit parent job ID
    submitit_job_id = os.environ.get("SUBMITIT_JOB_ID")
    if submitit_job_id and submitit_job_id.strip():
        return f"slurm-{submitit_job_id}"
    
    # Priority 4: Current SLURM job ID
    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    if slurm_job_id and slurm_job_id.strip():
        return f"slurm-{slurm_job_id}"
    
    return None


def _detect_num_gpus() -> int:
    """Detect the number of GPUs allocated to this job.
    
    Priority order:
    1. CUDA_VISIBLE_DEVICES environment variable (set by launcher)
    2. SLURM_GPUS_PER_NODE or SLURM_GPUS_ON_NODE
    3. torch.cuda.device_count() if CUDA is available
    4. Return 0 if no GPUs detected
    """
    # Priority 1: CUDA_VISIBLE_DEVICES (most reliable for actual allocation)
    try:
        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if cuda_visible and cuda_visible.strip():
            # Parse comma-separated GPU indices (e.g., "0,1,2,3" -> 4 GPUs)
            gpu_indices = [x.strip() for x in cuda_visible.split(",") if x.strip()]
            if gpu_indices:
                return len(gpu_indices)
    except Exception:
        pass
    
    # Priority 2: SLURM environment variables
    try:
        slurm_gpus = os.environ.get("SLURM_GPUS_PER_NODE") or os.environ.get("SLURM_GPUS_ON_NODE")
        if slurm_gpus:
            # Can be a number like "4" or format like "gpu:4"
            try:
                if ":" in slurm_gpus:
                    return int(slurm_gpus.split(":")[-1])
                return int(slurm_gpus)
            except Exception:
                pass
    except Exception:
        pass
    
    # Priority 3: Torch CUDA device count
    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            count = torch.cuda.device_count()
            if count > 0:
                return count
    except Exception:
        pass
    
    return 0


def _detect_gpu_type() -> Optional[str]:
    """Detect the GPU type/model name.
    
    Returns a normalized GPU type string (e.g., 'NVIDIA RTX A6000', 'NVIDIA RTX A5000').
    """
    try:
        import torch  # type: ignore
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            # Get the name of the first GPU (assuming homogeneous GPUs)
            gpu_name = torch.cuda.get_device_name(0)
            return gpu_name
    except Exception:
        pass
    
    return None


def _parse_cpus_on_node(val: str) -> int:
    """Parse SLURM_CPUS_ON_NODE format (e.g., '32', '16(x2)', '2,2')."""
    if not isinstance(val, str):
        return -1
    try:
        v = val.strip()
        if "(x" in v and v.endswith(")"):
            import re as _re
            m = _re.match(r"^(\d+)\(x(\d+)\)$", v)
            if m:
                a = int(m.group(1))
                b = int(m.group(2))
                return max(1, a * b)
        if "," in v:
            parts = [p for p in v.split(",") if p.strip()]
            acc = 0
            for p in parts:
                try:
                    acc += int(p)
                except Exception:
                    return -1
            return max(1, acc)
        return max(1, int(v))
    except Exception:
        return -1


def _detect_num_cpus() -> Optional[int]:
    """Detect the number of CPUs allocated to this job.
    
    Priority order:
    1. SLURM_CPUS_PER_TASK or SLURM_CPUS_ON_NODE (job allocation)
    2. os.cpu_count() (system total)
    """
    try:
        cpt = os.environ.get("SLURM_CPUS_PER_TASK")
        if cpt is not None and str(cpt).strip() != "":
            return int(cpt)
        
        con = os.environ.get("SLURM_CPUS_ON_NODE")
        if con is not None and str(con).strip() != "":
            cpus = _parse_cpus_on_node(con)
            if cpus > 0:
                return cpus
    except Exception:
        pass
    
    # Fallback to system CPU count
    try:
        return os.cpu_count()
    except Exception:
        pass
    
    return None


def _read_int_file(path: str) -> int:
    """Read an integer from a file, handling 'max' keyword."""
    try:
        with open(path, "r") as f:
            s = f.read().strip()
        if s.lower() == "max":
            return -1
        return int(s)
    except Exception:
        return -1


def _detect_cgroup_mem_limit_bytes() -> int:
    """Return cgroup memory limit in bytes when available; otherwise -1.
    
    Supports cgroup v2 (memory.max) and v1 (memory.limit_in_bytes).
    """
    # cgroup v2
    v2 = "/sys/fs/cgroup/memory.max"
    lim = _read_int_file(v2)
    if lim > 0:
        # Filter out unrealistically large values (no limit)
        try:
            if lim > (1 << 56):
                return -1
        except Exception:
            pass
        return lim
    
    # cgroup v1
    v1 = "/sys/fs/cgroup/memory/memory.limit_in_bytes"
    lim = _read_int_file(v1)
    if lim > 0:
        try:
            if lim > (1 << 56):
                return -1
        except Exception:
            pass
        return lim
    
    return -1


def _detect_slurm_job_mem_bytes() -> int:
    """Infer SLURM job memory allocation in bytes from env vars.
    
    Prefers SLURM_MEM_PER_NODE; otherwise uses SLURM_MEM_PER_CPU * SLURM_CPUS_ON_NODE.
    Values are MB according to SLURM docs.
    """
    try:
        mem_per_node_mb = os.environ.get("SLURM_MEM_PER_NODE")
        if mem_per_node_mb:
            mb = int(mem_per_node_mb)
            if mb > 0:
                return mb * 1024 * 1024
    except Exception:
        pass
    
    try:
        mem_per_cpu_mb = os.environ.get("SLURM_MEM_PER_CPU")
        cpus_on_node = os.environ.get("SLURM_CPUS_ON_NODE")
        if mem_per_cpu_mb and cpus_on_node:
            mb = int(mem_per_cpu_mb)
            cpus = _parse_cpus_on_node(cpus_on_node)
            if mb > 0 and cpus > 0:
                return mb * cpus * 1024 * 1024
    except Exception:
        pass
    
    return -1


def _detect_memory_gb() -> Optional[float]:
    """Detect available memory in GB.
    
    Priority order:
    1. cgroup memory limit (container/SLURM cgroup)
    2. SLURM env-based memory inference
    3. System total memory (psutil/os.sysconf)
    """
    # cgroup limit first
    cg = _detect_cgroup_mem_limit_bytes()
    if cg > 0:
        return cg / (1024 ** 3)
    
    # SLURM-derived
    sj = _detect_slurm_job_mem_bytes()
    if sj > 0:
        return sj / (1024 ** 3)
    
    # Fallback to system total
    try:
        import psutil  # type: ignore
        tot = int(getattr(psutil.virtual_memory(), "total", 0))
        if tot > 0:
            return tot / (1024 ** 3)
    except Exception:
        pass
    
    try:
        tot = int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
        if tot > 0:
            return tot / (1024 ** 3)
    except Exception:
        pass
    
    return None


def _collect_gpu_details() -> List[Dict[str, Any]]:
    """Collect detailed information for each GPU."""
    gpus = []
    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                try:
                    props = torch.cuda.get_device_properties(i)
                    gpu_info = {
                        "index": i,
                        "name": torch.cuda.get_device_name(i),
                        "total_memory_gb": props.total_memory / (1024 ** 3),
                        "major": props.major,
                        "minor": props.minor,
                        "multi_processor_count": props.multi_processor_count,
                    }
                    gpus.append(gpu_info)
                except Exception:
                    pass
    except Exception:
        pass
    
    return gpus


def collect_compute_metadata(cfg=None) -> Dict[str, Any]:
    """Collect comprehensive compute metadata for wandb logging.
    
    Captures system configuration including:
    - CPU count and architecture
    - GPU count, type, and memory
    - RAM allocation
    - SLURM job parameters
    - Python environment
    - Model configuration (if available in cfg)
    
    Args:
        cfg: Optional Hydra config to extract model/runtime parameters
    
    Returns:
        Dictionary with compute metadata suitable for wandb.config
    """
    metadata: Dict[str, Any] = {}
    
    # System info
    try:
        metadata["system"] = {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "hostname": socket.gethostname(),
            "user": getpass.getuser(),
        }
    except Exception:
        pass
    
    # CPU info
    num_cpus = _detect_num_cpus()
    if num_cpus is not None:
        metadata["compute.cpu_count"] = num_cpus
    
    try:
        metadata["compute.cpu_architecture"] = platform.machine()
    except Exception:
        pass
    
    # GPU info
    num_gpus = _detect_num_gpus()
    metadata["compute.gpu_count"] = num_gpus
    
    if num_gpus > 0:
        gpu_type = _detect_gpu_type()
        if gpu_type:
            metadata["compute.gpu_type"] = gpu_type
        
        # Detailed GPU info
        gpu_details = _collect_gpu_details()
        if gpu_details:
            metadata["compute.gpus"] = gpu_details
    
    # Memory info
    mem_gb = _detect_memory_gb()
    if mem_gb is not None:
        metadata["compute.memory_gb"] = round(mem_gb, 2)
    
    # SLURM job info
    slurm_info = {}
    for key in [
        "SLURM_JOB_ID",
        "SLURM_JOB_NAME",
        "SLURM_NODELIST",
        "SLURM_JOB_NUM_NODES",
        "SLURM_CPUS_PER_TASK",
        "SLURM_CPUS_ON_NODE",
        "SLURM_MEM_PER_NODE",
        "SLURM_MEM_PER_CPU",
        "SLURM_GPUS_PER_NODE",
        "SLURM_TASKS_PER_NODE",
        "SLURM_PARTITION",
        "SLURM_SUBMIT_DIR",
        "SUBMITIT_JOB_ID",
    ]:
        val = os.environ.get(key)
        if val:
            slurm_info[key.lower()] = val
    
    if slurm_info:
        metadata["slurm"] = slurm_info
    
    # Extract model and runtime config if provided
    if cfg is not None:
        try:
            # Model config
            model_cfg = getattr(cfg, "model", None)
            if model_cfg:
                model_info = {}
                
                # Model source
                model_source = getattr(model_cfg, "model_source", None)
                if model_source:
                    model_info["model_source"] = str(model_source)
                
                # Engine kwargs (vLLM parameters)
                engine_kwargs = getattr(model_cfg, "engine_kwargs", None)
                if engine_kwargs:
                    ek = {}
                    for key in [
                        "max_model_len",
                        "max_num_seqs",
                        "max_num_batched_tokens",
                        "gpu_memory_utilization",
                        "tensor_parallel_size",
                        "enable_chunked_prefill",
                        "enable_prefix_caching",
                        "dtype",
                        "kv_cache_dtype",
                    ]:
                        try:
                            val = getattr(engine_kwargs, key, None)
                            if val is not None:
                                ek[key] = val
                        except Exception:
                            pass
                    if ek:
                        model_info["engine_kwargs"] = ek
                
                # Batch size and concurrency
                batch_size = getattr(model_cfg, "batch_size", None)
                if batch_size is not None:
                    model_info["batch_size"] = int(batch_size)
                
                concurrency = getattr(model_cfg, "concurrency", None)
                if concurrency is not None:
                    model_info["concurrency"] = int(concurrency)
                
                if model_info:
                    metadata["model"] = model_info
        except Exception:
            pass
        
        try:
            # Runtime config
            runtime_cfg = getattr(cfg, "runtime", None)
            if runtime_cfg:
                runtime_info = {}
                for key in [
                    "debug",
                    "sample_n",
                    "job_memory_gb",
                    "rows_per_block",
                    "streaming_io",
                    "use_llm_classify",
                    "use_llm_decompose",
                    "prefilter_mode",
                    "keyword_buffering",
                ]:
                    try:
                        val = getattr(runtime_cfg, key, None)
                        if val is not None:
                            runtime_info[key] = val
                    except Exception:
                        pass
                if runtime_info:
                    metadata["runtime"] = runtime_info
        except Exception:
            pass
    
    return metadata


class WandbLogger:
    """Thread-safe centralized W&B logger.
    
    Usage:
        # As context manager (recommended)
        with WandbLogger(cfg, stage="classify", run_id="classify-001") as logger:
            logger.log_metrics({"accuracy": 0.95})
            logger.log_table(df, "results")
        
        # Manual lifecycle
        logger = WandbLogger(cfg, stage="classify", run_id="classify-001")
        logger.start()
        try:
            logger.log_metrics({"accuracy": 0.95})
        finally:
            logger.finish()
    """
    
    _lock = threading.Lock()
    _wandb = None
    _wandb_available = None
    
    def __init__(
        self,
        cfg,
        stage: str,
        run_id: Optional[str] = None,
        run_config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize wandb logger.
        
        Args:
            cfg: Hydra config object
            stage: Stage name (e.g., "classify", "topic", "orchestrator")
            run_id: Optional run identifier/suffix
            run_config: Optional run configuration dict to log
        """
        self.cfg = cfg
        self.stage = stage
        self.run_id = run_id
        self.run_config = run_config or {}
        self.wb_config = WandbConfig.from_hydra_config(cfg)
        self._run = None
        
        # Use the globally imported wandb module configured with legacy service
        if WandbLogger._wandb is None:
            with WandbLogger._lock:
                if WandbLogger._wandb is None:
                    WandbLogger._wandb = wandb_module
                    WandbLogger._wandb_available = True
    
    @property
    def enabled(self) -> bool:
        """Check if wandb logging is enabled."""
        return self.wb_config.enabled and WandbLogger._wandb_available
    
    @property
    def wandb(self):
        """Get wandb module (or None if not available)."""
        return WandbLogger._wandb
    
    def _get_run_name(self) -> str:
        """Generate run name."""
        try:
            exp_name = str(getattr(self.cfg.experiment, "name", "UAIR") or "UAIR")
        except Exception:
            exp_name = "UAIR"
        
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        
        # Build name: [group-]experiment-stage-timestamp[-run_id]
        parts = []
        if self.wb_config.group:
            parts.append(self.wb_config.group)
        parts.extend([exp_name, self.stage, timestamp])
        if self.run_id:
            parts.append(self.run_id)
        
        return "-".join(parts)
    
    def _get_mode(self) -> str:
        """Determine wandb mode.
        
        Returns:
            Mode string: "online", "offline", or "disabled"
            
        Priority:
            1. WANDB_MODE environment variable
            2. Default to "online" for real-time syncing
            
        Note: We use online mode by default even for workers because:
        - WANDB__REQUIRE_LEGACY_SERVICE=TRUE prevents daemon/socket issues
        - Real-time syncing provides better visibility
        - Legacy service mode is more stable in distributed contexts (SLURM, Ray)
        """
        # Check for explicit mode override
        mode_env = os.environ.get("WANDB_MODE")
        if mode_env:
            mode_lower = mode_env.lower().strip()
            if mode_lower in ("online", "offline", "disabled"):
                return mode_lower
        
        # Default to online mode for real-time syncing
        return "online"
    
    def _debug_env_snapshot(self, wandb_dir: str) -> Dict[str, Any]:
        """Collect a safe environment snapshot for debugging wandb init issues."""
        snapshot: Dict[str, Any] = {}
        try:
            snapshot.update({
                "user": getpass.getuser(),
                "pid": os.getpid(),
                "hostname": socket.gethostname(),
                "python": platform.python_version(),
                "platform": {
                    "system": platform.system(),
                    "release": platform.release(),
                    "machine": platform.machine(),
                },
                "cwd": os.getcwd(),
                "wandb": {
                    "version": getattr(wandb_module, "__version__", None),
                    "module_file": getattr(wandb_module, "__file__", None),
                },
                "env": {
                    "WANDB_MODE": os.environ.get("WANDB_MODE"),
                    "WANDB_DISABLE_SERVICE": os.environ.get("WANDB_DISABLE_SERVICE"),
                    "WANDB_SERVICE_PRESENT": bool(os.environ.get("WANDB_SERVICE")),
                    "WANDB_BASE_URL": os.environ.get("WANDB_BASE_URL"),
                    "WANDB_DIR": os.environ.get("WANDB_DIR"),
                    "TMPDIR": os.environ.get("TMPDIR"),
                    "SLURM_JOB_ID": os.environ.get("SLURM_JOB_ID"),
                    "SLURM_SUBMIT_DIR": os.environ.get("SLURM_SUBMIT_DIR"),
                    "SUBMITIT_JOB_ID": os.environ.get("SUBMITIT_JOB_ID"),
                    # Do NOT log API keys/secrets
                    "WANDB_API_KEY_SET": bool(os.environ.get("WANDB_API_KEY")),
                },
            })
        except Exception:
            pass
        # Check directories
        def _dir_info(path: Optional[str]) -> Dict[str, Any]:
            info: Dict[str, Any] = {"path": path}
            try:
                if not path:
                    return info
                info["exists"] = os.path.exists(path)
                info["isdir"] = os.path.isdir(path)
                info["writable"] = os.access(path if os.path.isdir(path) else os.path.dirname(path), os.W_OK)
                try:
                    usage = shutil.disk_usage(path if os.path.isdir(path) else os.path.dirname(path))
                    info["disk_free_gb"] = round(usage.free / (1024 ** 3), 2)
                except Exception:
                    pass
            except Exception:
                pass
            return info
        try:
            snapshot["paths"] = {
                "wandb_dir": _dir_info(wandb_dir),
                "tmpdir": _dir_info(os.environ.get("TMPDIR")),
            }
        except Exception:
            pass
        return snapshot
    
    def _is_ray_worker(self) -> bool:
        """Check if running inside a Ray worker.
        
        Ray workers should not initialize wandb to avoid socket conflicts.
        Only the main process or rank 0 worker should init wandb.
        """
        try:
            import ray
            # Check if ray is initialized and we're in a worker
            if ray.is_initialized():
                # Check if we're in a worker context (not the driver)
                try:
                    worker = ray._private.worker.global_worker
                    # If we're a worker (not driver), skip wandb init
                    return worker.mode == ray.WORKER_MODE
                except Exception:
                    pass
        except ImportError:
            pass
        return False
    
    def start(self) -> None:
        """Start wandb run."""
        if not self.enabled:
            return
        
        # Skip wandb initialization in Ray workers to avoid socket conflicts
        if self._is_ray_worker():
            print(f"[wandb] Skipping initialization in Ray worker for {self.stage}", flush=True)
            return
        
        if self._run is not None:
            print(f"[wandb] Warning: Run already started for {self.stage}", file=sys.stderr)
            return
        
        # Ensure we do not inherit/target a parent service socket across nodes
        try:
            for k in ("WANDB_SERVICE", "WANDB__SERVICE", "WANDB_SERVICE_SOCKET", "WANDB_SERVICE_TRANSPORT"):
                if k in os.environ:
                    os.environ.pop(k, None)
            # Reinforce in-process mode
            os.environ.setdefault("WANDB_DISABLE_SERVICE", "true")
        except Exception:
            pass

        mode = self._get_mode()
        run_name = self._get_run_name()
        
        # Determine wandb directory (must be writable in SLURM environments)
        wandb_dir = os.environ.get("WANDB_DIR")
        if not wandb_dir:
            # Default to a writable location
            # Use SLURM_SUBMIT_DIR if available (job submission directory), otherwise CWD
            wandb_dir = os.environ.get("SLURM_SUBMIT_DIR", os.getcwd())
        
        # Emit detailed debug snapshot once per run start
        try:
            dbg = self._debug_env_snapshot(wandb_dir)
            print(f"[wandb] Debug init context: {json.dumps(dbg, ensure_ascii=False)}", flush=True)
        except Exception:
            pass
        print(f"[wandb] Starting run: {run_name} (mode={mode}, dir={wandb_dir})", flush=True)
        
        try:
            # Initialize run with legacy service (configured at module import)
            self._run = self.wandb.init(
                project=self.wb_config.project,
                entity=self.wb_config.entity,
                group=self.wb_config.group,
                job_type=self.stage,
                name=run_name,
                config=self.run_config,
                mode=mode,
                dir=wandb_dir,
                tags=self.wb_config.tags,
            )
            
            # Collect and log compute metadata
            try:
                compute_metadata = collect_compute_metadata(self.cfg)
                if compute_metadata:
                    self.set_config(compute_metadata, allow_val_change=True)
                    print(f"[wandb] ✓ Logged compute metadata: {compute_metadata.get('compute.cpu_count', 'N/A')} CPUs, {compute_metadata.get('compute.gpu_count', 0)} GPUs", flush=True)
            except Exception as e:
                print(f"[wandb] Warning: Failed to collect compute metadata: {e}", file=sys.stderr, flush=True)
            
            if mode == "offline":
                print(f"[wandb] ✓ Run started: {run_name} (OFFLINE - will sync on finish)", flush=True)
            elif mode == "online":
                print(f"[wandb] ✓ Run started: {run_name} (ONLINE - real-time syncing)", flush=True)
            else:
                print(f"[wandb] ✓ Run started: {run_name} (mode={mode})", flush=True)
            
        except Exception as e:
            # Print rich error context including traceback for easier diagnosis
            tb = traceback.format_exc()
            print(f"[wandb] ✗ Failed to start run: {e}", file=sys.stderr, flush=True)
            try:
                print(f"[wandb] Traceback:\n{tb}", file=sys.stderr, flush=True)
            except Exception:
                pass
            print(f"[wandb] Logging will be disabled for {self.stage}", file=sys.stderr, flush=True)
            self._run = None
    
    def finish(self) -> None:
        """Finish wandb run and sync if offline."""
        if not self.enabled or self._run is None:
            return
        
        try:
            run_name = getattr(self._run, "name", "unknown")
            run_mode = getattr(getattr(self._run, "settings", None), "mode", "unknown")
            
            print(f"[wandb] Finishing run: {run_name} (mode={run_mode})", flush=True)
            
            self.wandb.finish()
            
            if run_mode == "offline":
                print(f"[wandb] ✓ Offline run '{run_name}' synced to cloud", flush=True)
            else:
                print(f"[wandb] ✓ Online run '{run_name}' completed", flush=True)
            
            self._run = None
            
        except Exception as e:
            print(f"[wandb] ✗ Failed to finish run: {e}", file=sys.stderr, flush=True)
            self._run = None
    
    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None, commit: bool = True) -> None:
        """Log metrics to wandb.
        
        Args:
            metrics: Dictionary of metric name -> value
            step: Optional step number
            commit: Whether to increment step counter
        """
        if not self.enabled or self._run is None:
            return
        
        try:
            if metrics:
                self.wandb.log(metrics, step=step, commit=commit)
        except Exception as e:
            print(f"[wandb] Warning: Failed to log metrics: {e}", file=sys.stderr)
    
    def log_table(
        self,
        df,
        key: str,
        prefer_cols: Optional[List[str]] = None,
        max_rows: Optional[int] = None,
        panel_group: Optional[str] = None,
    ) -> None:
        """Log pandas DataFrame as wandb table with random sampling when needed.
        
        Args:
            df: Pandas DataFrame
            key: Table name/key (e.g., "classify/results")
            prefer_cols: Optional list of preferred columns to include
            max_rows: Max rows to sample (default from config)
            panel_group: Optional panel group name (e.g., "inspect_results")
                        Will prefix the key as "panel_group/key"
        """
        if not self.enabled or self._run is None or df is None:
            return
        
        try:
            import pandas as pd
            
            # Apply panel group prefix if specified
            table_key = f"{panel_group}/{key}" if panel_group else key
            
            # Select columns
            cols = [c for c in (prefer_cols or []) if c in df.columns]
            if not cols:
                cols = list(df.columns)[:12]
            
            # Sample if needed (always use random sampling, not head())
            max_rows = max_rows or self.wb_config.table_sample_rows
            total_rows = len(df)
            
            if total_rows > max_rows:
                df_sample = df.sample(
                    n=max_rows,
                    random_state=self.wb_config.table_sample_seed
                ).reset_index(drop=True)
                sampled = True
            else:
                df_sample = df.reset_index(drop=True)
                sampled = False
            
            # Create table
            table = self.wandb.Table(dataframe=df_sample[cols])
            
            # Log table with metadata
            log_data = {
                table_key: table,
                f"{table_key}/rows": len(df_sample),
                f"{table_key}/total_rows": total_rows,
            }
            
            # Add sampling metadata if applicable
            if sampled:
                log_data[f"{table_key}/sampled"] = True
                log_data[f"{table_key}/sample_seed"] = self.wb_config.table_sample_seed
            
            self.wandb.log(log_data)
            
            # Print sampling info
            if sampled:
                print(f"[wandb] ✓ Logged table '{table_key}': {len(df_sample):,} rows (randomly sampled from {total_rows:,})", flush=True)
            else:
                print(f"[wandb] ✓ Logged table '{table_key}': {total_rows:,} rows", flush=True)
            
        except Exception as e:
            print(f"[wandb] Warning: Failed to log table '{key}': {e}", file=sys.stderr)
    
    def log_artifact(self, artifact_path: str, name: str, type: str = "dataset") -> None:
        """Log artifact to wandb.
        
        Args:
            artifact_path: Path to artifact file/directory
            name: Artifact name
            type: Artifact type (e.g., "dataset", "model")
        """
        if not self.enabled or self._run is None:
            return
        
        try:
            artifact = self.wandb.Artifact(name=name, type=type)
            artifact.add_file(artifact_path)
            self.wandb.log_artifact(artifact)
        except Exception as e:
            print(f"[wandb] Warning: Failed to log artifact '{name}': {e}", file=sys.stderr)
    
    def log_plot(self, key: str, figure) -> None:
        """Log matplotlib/plotly figure or wandb plot object.
        
        Args:
            key: Plot name/key
            figure: Matplotlib figure, Plotly figure, or wandb plot object (Image, Plotly, Html, etc.)
        """
        if not self.enabled or self._run is None:
            return
        
        try:
            # Check if it's already a wandb data type (Plotly, Image, Html, etc.)
            if hasattr(figure, '__class__') and hasattr(figure.__class__, '__module__'):
                module = figure.__class__.__module__
                if module and 'wandb' in module:
                    # Already a wandb data type, log directly
                    self.wandb.log({key: figure})
                    return
            
            # Raw plotly figure (has to_html method)
            if hasattr(figure, 'to_html'):
                self.wandb.log({key: figure})
            # Matplotlib figure (has savefig method)
            elif hasattr(figure, 'savefig'):
                self.wandb.log({key: self.wandb.Image(figure)})
            else:
                # Log directly and let wandb handle it
                self.wandb.log({key: figure})
        except Exception as e:
            print(f"[wandb] Warning: Failed to log plot '{key}': {e}", file=sys.stderr)
    
    def set_summary(self, key: str, value: Any) -> None:
        """Set a run-level summary field (useful for categorical/string values)."""
        if not self.enabled or self._run is None:
            return
        try:
            self._run.summary[key] = value
        except Exception as e:
            print(f"[wandb] Warning: Failed to set summary '{key}': {e}", file=sys.stderr)
    
    def set_config(self, data: Dict[str, Any], allow_val_change: bool = True) -> None:
        """Update run config for stable, non-time-series metadata (e.g., backend name)."""
        if not self.enabled or self._run is None or not data:
            return
        try:
            self._run.config.update(dict(data), allow_val_change=allow_val_change)
        except Exception as e:
            print(f"[wandb] Warning: Failed to update config: {e}", file=sys.stderr)
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.finish()
        return False

