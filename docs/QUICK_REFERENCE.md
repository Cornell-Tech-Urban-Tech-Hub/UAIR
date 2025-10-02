# UAIR Quick Reference

Fast lookup for common commands, patterns, and configurations.

---

## Command Line Interface

### Basic Execution

```bash
# Run with default config
python -m dagspaces.uair.cli

# Run specific pipeline
python -m dagspaces.uair.cli pipeline=taxonomy_full

# Override data path
python -m dagspaces.uair.cli data.parquet_path=/path/to/data.parquet

# Debug mode with sampling
python -m dagspaces.uair.cli runtime.debug=true runtime.sample_n=100

# Multiple overrides
python -m dagspaces.uair.cli \
  pipeline=my_pipeline \
  runtime.debug=true \
  model.batch_size=8 \
  data.parquet_path=/data/articles.parquet
```

### Configuration Groups

```bash
# Change data source
python -m dagspaces.uair.cli data=flattened_rules

# Change model
python -m dagspaces.uair.cli model=vllm_qwen3-30b

# Change launcher (execution environment)
python -m dagspaces.uair.cli hydra/launcher=g2_slurm_gpu_4x

# Change pipeline
python -m dagspaces.uair.cli pipeline=topic_modeling_of_relevant_classifications
```

### Inspection Commands

```bash
# Print resolved configuration
python -m dagspaces.uair.cli --cfg job

# Print help
python -m dagspaces.uair.cli --help

# List available config groups
python -m dagspaces.uair.cli --hydra-help
```

---

## Configuration Snippets

### Minimal Pipeline

```yaml
# conf/pipeline/minimal.yaml
# @package _global_
pipeline:
  sources:
    articles:
      path: ${data.parquet_path}
      type: parquet
  
  graph:
    nodes:
      my_stage:
        stage: topic
        depends_on: []
        inputs:
          dataset: articles
        outputs:
          results: outputs/results.parquet
```

### Node with Overrides

```yaml
my_stage:
  stage: classify
  depends_on: []
  inputs:
    dataset: articles
  outputs:
    results: outputs/classify/results.parquet
  overrides:
    runtime.sample_n: 500
    model.batch_size: 16
    model.engine_kwargs.max_model_len: 4096
  launcher: g2_slurm_gpu_4x
  wandb_suffix: my_stage
```

### Sequential Dependencies

```yaml
nodes:
  stage1:
    stage: classify
    depends_on: []
    inputs:
      dataset: articles
    outputs:
      results: outputs/stage1/results.parquet
  
  stage2:
    stage: taxonomy
    depends_on: [stage1]
    inputs:
      dataset: stage1.results  # Reference previous output
    outputs:
      results: outputs/stage2/results.parquet
```

### Parallel Stages

```yaml
nodes:
  stage1:
    depends_on: []
    # ...
  
  # These two run in parallel after stage1
  stage2a:
    depends_on: [stage1]
    inputs:
      dataset: stage1.results
    # ...
  
  stage2b:
    depends_on: [stage1]
    inputs:
      dataset: stage1.results
    # ...
```

---

## Common Overrides

### Runtime

```yaml
runtime.debug: true                 # Enable debug mode
runtime.sample_n: 1000             # Process only N rows
runtime.output_root: ./outputs     # Output directory
runtime.streaming_io: true         # Use Ray Data streaming
runtime.rows_per_block: 4000       # Ray Data block size
runtime.job_memory_gb: 64          # Memory allocation
runtime.use_llm_classify: true     # Enable LLM classification
runtime.prefilter_mode: pre_gating # Keyword filtering: pre_gating|post_gating|off
```

### Model

```yaml
model.model_source: /path/to/model
model.batch_size: 16
model.concurrency: 1
model.engine_kwargs.max_model_len: 8192
model.engine_kwargs.max_num_seqs: 8
model.engine_kwargs.gpu_memory_utilization: 0.85
model.engine_kwargs.tensor_parallel_size: 2
```

### Sampling Parameters

```yaml
sampling_params.temperature: 0.0
sampling_params.top_p: 1.0
sampling_params.max_tokens: 256
sampling_params_classify.max_tokens: 4
sampling_params_taxonomy.max_tokens: 16
```

---

## Stage Reference

### Built-in Stages

| Stage | Purpose | Key Outputs |
|-------|---------|-------------|
| `classify` | Relevance classification | `is_relevant`, `relevance_answer` |
| `taxonomy` | Category assignment | `chunk_label`, `chunk_label_name`, `answer` |
| `decompose` | Information extraction | `ci_*` fields (subject, sender, etc.) |
| `topic` | Topic modeling | `topic_id`, `topic_prob`, `topic_top_terms` |
| `verification` | Claim verification | `verification_result`, `verify_*` scores |

### Stage-Specific Config

**Topic**:
```yaml
topic.cluster_on: article           # article|chunk
topic.embed.device: cuda            # cuda|cpu|auto
topic.embed.batch_size: 64
topic.embed.matryoshka_dim: 256
topic.reduce.method: umap           # umap|pca|none
topic.reduce.n_components: 15
topic.hdbscan.min_cluster_size: 10
```

**Verification**:
```yaml
verify.method: combo                # off|embed|nli|combo|combo_judge
verify.top_k: 3
verify.thresholds: "sim=0.55,ent=0.85,contra=0.05"
verify.device: cuda
```

---

## SLURM Launchers

| Launcher | GPUs | CPUs | Mem | Use Case |
|----------|------|------|-----|----------|
| `null` | 0 | local | local | Local testing |
| `g2_slurm_cpu` | 0 | 2 | 16GB | Light CPU |
| `g2_slurm_cpu_beefy` | 0 | 8 | 64GB | Heavy CPU |
| `g2_slurm_gpu_1x` | 1 | 8 | 32GB | Single GPU |
| `g2_slurm_pierson` | 2 | 8 | 32GB | 2-GPU inference |
| `g2_slurm_gpu_4x` | 4 | 8 | 32GB | 4-GPU inference |
| `slurm_monitor` | 0 | 2 | 8GB | Orchestrator |

---

## File Locations

### Configuration

```
conf/
├── config.yaml              # Base config
├── data/                    # Data sources
├── model/                   # Model configs
├── prompt/                  # Prompt templates
├── taxonomy/                # Domain taxonomies
├── pipeline/                # Pipeline definitions
└── hydra/launcher/          # SLURM configs
```

### Code

```
dagspaces/uair/
├── cli.py                   # Entry point
├── orchestrator.py          # Pipeline execution
├── config_schema.py         # Config schemas
├── wandb_logger.py          # W&B integration
└── stages/                  # Stage implementations
    ├── classify.py
    ├── taxonomy.py
    ├── topic.py
    └── ...
```

### Outputs

```
outputs/
├── YYYY-MM-DD/
│   └── HH-MM-SS/
│       ├── pipeline_manifest.json
│       ├── outputs/
│       │   ├── classify/
│       │   ├── taxonomy/
│       │   └── topic/
│       └── .slurm_jobs/     # SLURM logs
```

---

## Environment Variables

### W&B

```bash
export WANDB_PROJECT=UAIR
export WANDB_ENTITY=my_team
export WANDB_GROUP=experiment_1
export WANDB_MODE=offline    # Disable W&B
```

### Runtime

```bash
export DATA_ROOT=/data
export OUTPUT_ROOT=/outputs
export TAXONOMY_JSON=/path/to/taxonomy.yaml
export HYDRA_FULL_ERROR=1    # Detailed errors
```

### GPU

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
export VLLM_LOGGING_LEVEL=WARNING
export TOKENIZERS_PARALLELISM=false
```

---

## Common Patterns

### Debug Small Sample

```yaml
runtime:
  debug: true
  sample_n: 10
```

```bash
python -m dagspaces.uair.cli runtime.debug=true runtime.sample_n=10
```

### Local Testing (No SLURM)

```yaml
defaults:
  - override /hydra/launcher: null

nodes:
  my_stage:
    launcher: null  # Run locally
```

### Filter Before Expensive Processing

```yaml
nodes:
  quick_filter:
    overrides:
      runtime.use_llm_classify: false  # Fast keyword filter
  
  expensive_llm:
    depends_on: [quick_filter]
    inputs:
      dataset: quick_filter.relevant   # Only filtered data
    overrides:
      runtime.use_llm_classify: true   # Slow but accurate
```

### Multi-Output

```yaml
classify:
  outputs:
    all: outputs/classify/all.parquet
    relevant: outputs/classify/relevant.parquet

taxonomy:
  inputs:
    dataset: classify.relevant  # Use only relevant
```

---

## Troubleshooting

### GPU OOM

```yaml
# Reduce memory usage
model.engine_kwargs.max_model_len: 2048        # Smaller context
model.engine_kwargs.gpu_memory_utilization: 0.6  # More conservative
model.engine_kwargs.max_num_seqs: 2            # Fewer parallel seqs
model.batch_size: 2                            # Smaller batches
```

### Ray Object Store Full

```bash
# Increase object store memory
export RAY_DEFAULT_OBJECT_STORE_MEMORY_PROPORTION=0.5
```

Or in config:
```yaml
runtime:
  object_store_proportion: 0.5
  rows_per_block: 1000  # Smaller blocks
```

### SLURM Job Failed

```bash
# Check SLURM logs
cat outputs/YYYY-MM-DD/HH-MM-SS/.slurm_jobs/stage_name/*.err
cat outputs/YYYY-MM-DD/HH-MM-SS/.slurm_jobs/stage_name/*.out

# Check job status
squeue -u $USER
sacct -j JOB_ID
```

### Configuration Not Found

```bash
# Verify config exists
ls conf/pipeline/my_pipeline.yaml

# Check search paths
python -m dagspaces.uair.cli --hydra-help
```

---

## Python API Usage

### Load Results

```python
import pandas as pd

# Load stage outputs
results = pd.read_parquet('outputs/.../stage_name/results.parquet')

# Load manifest
import json
with open('outputs/.../pipeline_manifest.json') as f:
    manifest = json.load(f)

print(manifest['nodes'])  # See all stage outputs
```

### Access Configuration

```python
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import os

# Initialize Hydra
config_dir = os.path.join(os.getcwd(), "dagspaces/uair/conf")
initialize_config_dir(config_dir=config_dir, version_base="1.3")

# Compose config
cfg = compose(
    config_name="config",
    overrides=["pipeline=my_pipeline", "runtime.debug=true"]
)

# Access values
print(OmegaConf.to_yaml(cfg))
```

---

## Links to Full Guides

- [User Guide](USER_GUIDE.md) - Complete introduction and concepts
- [Custom Stages](CUSTOM_STAGES_GUIDE.md) - Build custom stages
- [Configuration](CONFIGURATION_GUIDE.md) - Config recipes and patterns
- [SLURM Guide](SLURM_GUIDE.md) - Distributed execution
- [Examples](EXAMPLES.md) - End-to-end pipeline examples

---

*Last updated: 2025-10-02*

