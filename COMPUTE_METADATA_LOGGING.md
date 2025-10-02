# Compute Metadata Logging for W&B

## Overview

The UAIR pipeline now automatically collects and logs comprehensive compute metadata to every W&B run, including the monitor (orchestrator) and all child stage runs. This provides full visibility into the hardware and system configuration used for each experiment.

## What's Logged

### System Information
- **Platform**: OS version and distribution
- **Python version**: Runtime Python version
- **Hostname**: Machine hostname
- **User**: Username running the job

### CPU Information
- **CPU count**: Number of CPUs allocated to the job (SLURM-aware)
- **CPU architecture**: Processor architecture (x86_64, arm64, etc.)

### GPU Information
- **GPU count**: Number of GPUs allocated (detected from CUDA_VISIBLE_DEVICES, SLURM, or torch)
- **GPU type**: GPU model name (e.g., "NVIDIA RTX A6000")
- **GPU details**: Per-GPU information including:
  - Index
  - Name
  - Total memory (GB)
  - Compute capability (major/minor)
  - Multi-processor count

### Memory Information
- **Total memory (GB)**: Available RAM (SLURM/cgroup-aware)
- Detection priority:
  1. cgroup memory limit (container/SLURM cgroup)
  2. SLURM job memory allocation
  3. System total memory

### SLURM Job Information (when applicable)
- Job ID, job name, nodelist
- Number of nodes
- CPUs per task / CPUs on node
- Memory per node / Memory per CPU
- GPUs per node
- Tasks per node
- Partition
- Submit directory
- Submitit job ID (for nested jobs)

### Model Configuration (from Hydra config)
- **Model source**: Path to model weights
- **Engine kwargs**: vLLM parameters including:
  - `max_model_len`
  - `max_num_seqs`
  - `max_num_batched_tokens`
  - `gpu_memory_utilization`
  - `tensor_parallel_size`
  - `enable_chunked_prefill`
  - `enable_prefix_caching`
  - `dtype`, `kv_cache_dtype`
- **Batch size**: Model batch size
- **Concurrency**: Number of concurrent requests

### Runtime Configuration (from Hydra config)
- Debug mode
- Sample size (`sample_n`)
- Job memory allocation
- Rows per block
- Streaming I/O settings
- LLM usage flags
- Prefilter mode
- Keyword buffering settings

## Implementation

### Core Function: `collect_compute_metadata(cfg=None)`

Located in `pipelines/uair/wandb_logger.py`, this function:

1. Detects compute resources using multiple detection strategies
2. Parses SLURM environment variables
3. Extracts model and runtime configuration from Hydra config
4. Returns a dictionary suitable for `wandb.config`

```python
from pipelines.uair.wandb_logger import collect_compute_metadata

# Collect metadata (with optional Hydra config)
metadata = collect_compute_metadata(cfg)

# Returns structure like:
{
    "system": {
        "platform": "Linux-6.8.0-83-generic-x86_64-with-glibc2.35",
        "python_version": "3.10.12",
        "hostname": "g2-node01",
        "user": "matt"
    },
    "compute.cpu_count": 8,
    "compute.cpu_architecture": "x86_64",
    "compute.gpu_count": 4,
    "compute.gpu_type": "NVIDIA RTX A6000",
    "compute.gpus": [
        {
            "index": 0,
            "name": "NVIDIA RTX A6000",
            "total_memory_gb": 48.0,
            "major": 8,
            "minor": 6,
            "multi_processor_count": 84
        },
        # ... more GPUs
    ],
    "compute.memory_gb": 128.5,
    "slurm": {
        "slurm_job_id": "12345",
        "slurm_cpus_per_task": "8",
        "slurm_gpus_per_node": "4",
        # ... more SLURM vars
    },
    "model": {
        "model_source": "/share/pierson/matt/zoo/models/Qwen3-30B-A3B-Instruct-2507",
        "engine_kwargs": {
            "max_model_len": 4096,
            "max_num_seqs": 4,
            "gpu_memory_utilization": 0.7,
            "tensor_parallel_size": 4
        },
        "batch_size": 4,
        "concurrency": 1
    },
    "runtime": {
        "debug": true,
        "sample_n": 1000,
        "job_memory_gb": 64,
        "streaming_io": false,
        "use_llm_classify": true,
        # ... more runtime settings
    }
}
```

### Automatic Integration

The `WandbLogger` class automatically calls `collect_compute_metadata()` when starting a run:

```python
# In WandbLogger.start()
compute_metadata = collect_compute_metadata(self.cfg)
if compute_metadata:
    self.set_config(compute_metadata, allow_val_change=True)
```

This happens for:
- **Monitor run**: The orchestrator's main run
- **All stage runs**: classify, taxonomy, decompose, topic, verification

## Usage

### No Action Required

The compute metadata logging is **automatically enabled** for all W&B runs. Simply use the existing `WandbLogger` as before:

```python
with WandbLogger(cfg, stage="classify", run_id="classify-001") as logger:
    # Compute metadata is automatically logged on start
    logger.log_metrics({"accuracy": 0.95})
    logger.log_table(df, "results")
```

### Viewing in W&B UI

1. Navigate to your W&B run
2. Go to the **Overview** tab
3. Expand the **Config** section
4. You'll see all compute metadata organized by category:
   - `system.*`
   - `compute.*`
   - `slurm.*`
   - `model.*`
   - `runtime.*`

### Querying Runs by Compute Config

You can filter and compare runs based on compute metadata:

```python
import wandb

api = wandb.Api()
runs = api.runs(
    "UAIR",
    filters={
        "config.compute.gpu_count": {"$gte": 4},
        "config.compute.gpu_type": "NVIDIA RTX A6000"
    }
)
```

## Benefits

1. **Reproducibility**: Full record of hardware used for each experiment
2. **Resource Planning**: Understand which configs work best on which hardware
3. **Debugging**: Identify hardware-related performance issues
4. **Auditing**: Track resource usage across experiments
5. **Cost Analysis**: Correlate hardware configuration with performance and cost

## Detection Strategies

### GPU Detection
1. **CUDA_VISIBLE_DEVICES** (most reliable for actual allocation)
2. SLURM environment variables (SLURM_GPUS_PER_NODE)
3. torch.cuda.device_count() (fallback)

### CPU Detection
1. **SLURM_CPUS_PER_TASK** (job allocation)
2. SLURM_CPUS_ON_NODE (with parsing for formats like "16(x2)")
3. os.cpu_count() (system total)

### Memory Detection
1. **cgroup limits** (container/SLURM cgroup - most accurate)
2. SLURM memory variables (SLURM_MEM_PER_NODE, SLURM_MEM_PER_CPU)
3. System total memory (psutil or os.sysconf)

This priority order ensures accurate detection in various environments (local, SLURM, containerized).

## Dependencies

The metadata collection gracefully handles missing dependencies:

- **torch**: Required for GPU detection (falls back to SLURM/CUDA_VISIBLE_DEVICES if unavailable)
- **psutil**: Optional for memory detection (has fallbacks)
- All detection functions use try/except to avoid failures

## Example Output

Console output when starting a run:

```
[wandb] Starting run: UAIR-classify-2025-10-02_14-30-00 (mode=online, dir=/share/pierson/matt/UAIR)
[wandb] ✓ Logged compute metadata: 8 CPUs, 4 GPUs
[wandb] ✓ Run started: UAIR-classify-2025-10-02_14-30-00 (ONLINE - real-time syncing)
```

## Files Modified

- `pipelines/uair/wandb_logger.py`:
  - Added `collect_compute_metadata()` function
  - Added GPU/CPU/memory detection helper functions
  - Modified `WandbLogger.start()` to automatically log compute metadata

## Testing

To verify the metadata is logged correctly:

1. Run any UAIR pipeline stage
2. Check the console output for "✓ Logged compute metadata"
3. View the W&B run's Config tab
4. Confirm all expected fields are present

Example test command:

```bash
cd /share/pierson/matt/UAIR
source .venv/bin/activate

# Run a simple test (will log to W&B)
python -m pipelines.uair.cli \
    runtime.debug=true \
    runtime.sample_n=10 \
    +pipeline=topic_modeling_of_relevant_classifications
```

Check the W&B run config for compute metadata fields.

## References

- W&B Config Docs: https://docs.wandb.ai/ref/python/run#config
- W&B Distributed Training: https://docs.wandb.ai/guides/track/advanced/distributed-training
- SLURM Environment Variables: https://slurm.schedmd.com/sbatch.html#SECTION_INPUT-ENVIRONMENT-VARIABLES


