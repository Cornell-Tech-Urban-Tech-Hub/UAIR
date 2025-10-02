# UAIR Configuration Guide

Hydra configuration patterns for building flexible, scalable pipelines.

---

## Table of Contents

1. [Configuration Fundamentals](#configuration-fundamentals)
2. [Pipeline Recipes](#pipeline-recipes)
3. [Per-Node Configuration](#per-node-configuration)
4. [Model Configuration](#model-configuration)
5. [SLURM Launcher Configuration](#slurm-launcher-configuration)
6. [Environment-Specific Configs](#environment-specific-configs)
7. [Advanced Patterns](#advanced-patterns)

---

## Configuration Fundamentals

### Config File Organization

```
conf/
├── config.yaml              # Base configuration
├── data/                    # Data source configs
│   ├── inputs.yaml
│   └── custom_dataset.yaml
├── model/                   # Model configs
│   ├── vllm_qwen3-30b.yaml
│   └── my_custom_model.yaml
├── prompt/                  # Prompt templates
│   ├── classify.yaml
│   ├── taxonomy.yaml
│   └── custom_prompt.yaml
├── taxonomy/                # Domain taxonomies
│   ├── weitz.yaml
│   └── custom_taxonomy.yaml
├── pipeline/                # Complete pipeline definitions
│   ├── taxonomy_full.yaml
│   └── my_pipeline.yaml
└── hydra/launcher/          # Execution environment configs
    ├── g2_slurm_cpu.yaml
    └── local.yaml
```

### Composition Order Matters

The `defaults` list determines how configs are merged:

```yaml
defaults:
  - _self_              # This file's values
  - data: inputs        # Load data/inputs.yaml
  - model: my_model     # Load model/my_model.yaml
  - prompt: classify    # Load prompt/classify.yaml

# Values defined here will override defaults above (because _self_ is first)
runtime:
  debug: true
```

**Note**: Later items override earlier items. The `_self_` directive controls when the current file's values apply.

### Variable Interpolation Patterns

```yaml
# 1. Reference other config values
runtime:
  output_root: /outputs
  sample_n: 1000

pipeline:
  output_root: ${runtime.output_root}  # → /outputs

# 2. Environment variables with defaults
data:
  parquet_path: ${oc.env:DATA_PATH,/default/path/data.parquet}

# 3. Nested references
model:
  model_source: /models/llama

sampling_params:
  model_ref: ${model.model_source}  # → /models/llama

# 4. Arithmetic (OmegaConf 2.1+)
runtime:
  base_memory: 32
  adjusted_memory: ${runtime.base_memory * 0.9}  # → 28.8
```

---

## Pipeline Recipes

### Recipe 1: Simple Linear Pipeline

**Use Case**: Process articles through sequential stages (classify → taxonomy → verify)

```yaml
# conf/pipeline/simple_linear.yaml
# @package _global_
defaults:
  - override /hydra/launcher: null  # Run locally

runtime:
  debug: false
  sample_n: null  # Process all data
  output_root: ./outputs/linear_pipeline

pipeline:
  output_root: ${runtime.output_root}
  allow_partial: false
  
  sources:
    articles:
      path: ${data.parquet_path}
      type: parquet
  
  graph:
    nodes:
      classify:
        stage: classify
        depends_on: []
        inputs:
          dataset: articles
        outputs:
          all: outputs/classify/all.parquet
          relevant: outputs/classify/relevant.parquet
        launcher: g2_slurm_pierson
        wandb_suffix: classify
      
      taxonomy:
        stage: taxonomy
        depends_on: [classify]
        inputs:
          dataset: classify.relevant  # Use only relevant articles
        outputs:
          results: outputs/taxonomy/results.parquet
        launcher: g2_slurm_pierson
        wandb_suffix: taxonomy
      
      verify:
        stage: verification
        depends_on: [taxonomy]
        inputs:
          dataset: taxonomy.results
        outputs:
          verified: outputs/verify/verified.parquet
        launcher: g2_slurm_cpu
        wandb_suffix: verify
```

**Execution**:
```bash
python -m pipelines.uair.cli \
  pipeline=simple_linear \
  data.parquet_path=/data/articles.parquet
```

### Recipe 2: Parallel Processing

**Use Case**: Run multiple independent analyses on the same dataset

```yaml
# conf/pipeline/parallel_analysis.yaml
# @package _global_
defaults:
  - override /hydra/launcher: slurm_monitor

runtime:
  output_root: ./outputs/parallel_analysis

pipeline:
  output_root: ${runtime.output_root}
  
  sources:
    articles:
      path: ${data.parquet_path}
      type: parquet
  
  graph:
    nodes:
      # Stage 1: Classify relevance
      classify:
        stage: classify
        depends_on: []
        inputs:
          dataset: articles
        outputs:
          relevant: outputs/classify/relevant.parquet
        launcher: g2_slurm_pierson
      
      # Stage 2a: Topic modeling (parallel with sentiment)
      topic:
        stage: topic
        depends_on: [classify]
        inputs:
          dataset: classify.relevant
        outputs:
          docs: outputs/topic/docs_topics.parquet
        launcher: g2_slurm_cpu_beefy
        wandb_suffix: topic
      
      # Stage 2b: Sentiment analysis (parallel with topic)
      sentiment:
        stage: sentiment
        depends_on: [classify]
        inputs:
          dataset: classify.relevant
        outputs:
          results: outputs/sentiment/results.parquet
        launcher: g2_slurm_cpu
        wandb_suffix: sentiment
      
      # Stage 3: Combine results (waits for both topic and sentiment)
      combine:
        stage: combine
        depends_on: [topic, sentiment]
        inputs:
          topic_data: topic.docs
          sentiment_data: sentiment.results
        outputs:
          combined: outputs/combined/results.parquet
```

### Recipe 3: Filter-Then-Process

**Use Case**: Apply expensive processing only to high-quality subset

```yaml
# conf/pipeline/filter_then_process.yaml
# @package _global_
defaults:
  - override /hydra/launcher: slurm_monitor

runtime:
  output_root: ./outputs/filtered_processing

pipeline:
  output_root: ${runtime.output_root}
  
  sources:
    articles:
      path: ${data.parquet_path}
      type: parquet
  
  graph:
    nodes:
      # Quick heuristic filter (CPU, fast)
      quick_filter:
        stage: classify
        depends_on: []
        inputs:
          dataset: articles
        outputs:
          relevant: outputs/quick_filter/relevant.parquet
        overrides:
          runtime.use_llm_classify: false  # Use keyword-based filtering
          runtime.prefilter_mode: pre_gating
        launcher: g2_slurm_cpu
      
      # Expensive LLM classification (GPU, slow) - only on filtered data
      llm_classify:
        stage: taxonomy
        depends_on: [quick_filter]
        inputs:
          dataset: quick_filter.relevant
        outputs:
          results: outputs/llm_classify/results.parquet
        overrides:
          runtime.use_llm_classify: true
        launcher: g2_slurm_gpu_4x
      
      # Verification (CPU, moderate) - only on classified data
      verify:
        stage: verification
        depends_on: [llm_classify]
        inputs:
          dataset: llm_classify.results
        outputs:
          verified: outputs/verify/verified.parquet
        launcher: g2_slurm_cpu_beefy
```

### Recipe 4: Multi-Output Fan-Out

**Use Case**: One stage produces multiple outputs consumed by different stages

```yaml
# conf/pipeline/fanout.yaml
# @package _global_
pipeline:
  sources:
    articles:
      path: ${data.parquet_path}
      type: parquet
  
  graph:
    nodes:
      # Classify produces two outputs
      classify:
        stage: classify
        depends_on: []
        inputs:
          dataset: articles
        outputs:
          all: outputs/classify/all.parquet
          relevant: outputs/classify/relevant.parquet
      
      # Path 1: Process all articles (including irrelevant)
      analyze_all:
        stage: topic
        depends_on: [classify]
        inputs:
          dataset: classify.all  # Use ALL articles
        outputs:
          topics: outputs/analyze_all/topics.parquet
      
      # Path 2: Deep analysis on relevant only
      deep_analysis:
        stage: taxonomy
        depends_on: [classify]
        inputs:
          dataset: classify.relevant  # Use only relevant
        outputs:
          taxonomy: outputs/deep_analysis/taxonomy.parquet
```

### Recipe 5: Iterative Refinement

**Use Case**: Run stage multiple times with different parameters

```yaml
# conf/pipeline/iterative.yaml
# @package _global_
pipeline:
  sources:
    articles:
      path: ${data.parquet_path}
      type: parquet
  
  graph:
    nodes:
      # Round 1: Coarse-grained classification
      classify_coarse:
        stage: classify
        depends_on: []
        inputs:
          dataset: articles
        outputs:
          results: outputs/classify_coarse/results.parquet
        overrides:
          model.engine_kwargs.temperature: 0.3  # Higher temp for exploration
      
      # Round 2: Fine-grained classification on relevant subset
      classify_fine:
        stage: taxonomy
        depends_on: [classify_coarse]
        inputs:
          dataset: classify_coarse.results
        outputs:
          results: outputs/classify_fine/results.parquet
        overrides:
          model.engine_kwargs.temperature: 0.0  # Deterministic for precision
          sampling_params_taxonomy.max_tokens: 64  # Allow longer responses
```

---

## Per-Node Configuration

### Overriding Runtime Parameters

Each node can override global runtime settings:

```yaml
nodes:
  my_stage:
    stage: classify
    overrides:
      # Runtime overrides
      runtime.debug: true
      runtime.sample_n: 500
      runtime.streaming_io: true
      runtime.rows_per_block: 2000
      runtime.prefilter_mode: pre_gating
      runtime.keyword_buffering: true
```

### Overriding Model Parameters

```yaml
nodes:
  my_llm_stage:
    stage: taxonomy
    overrides:
      # Model config overrides
      model.engine_kwargs.max_model_len: 4096
      model.engine_kwargs.max_num_seqs: 4
      model.engine_kwargs.gpu_memory_utilization: 0.7
      model.engine_kwargs.tensor_parallel_size: 2
      model.batch_size: 16
      model.concurrency: 2
```

### Overriding Sampling Parameters

```yaml
nodes:
  my_llm_stage:
    stage: taxonomy
    overrides:
      # Sampling parameter overrides
      sampling_params.temperature: 0.1
      sampling_params.top_p: 0.95
      sampling_params.max_tokens: 256
      sampling_params_taxonomy.guided_decoding.choice: ["Cat1", "Cat2", "Cat3"]
```

### Stage-Specific Configuration

Some stages have unique configuration sections:

```yaml
nodes:
  topic_stage:
    stage: topic
    overrides:
      # Topic modeling specific
      topic.cluster_on: article  # or 'chunk'
      topic.max_tokens_for_embed: 8192
      topic.embed.model_source: "nomic-ai/nomic-embed-text-v1.5"
      topic.embed.batch_size: 64
      topic.embed.device: cuda
      topic.embed.matryoshka_dim: 256
      topic.reduce.method: umap
      topic.reduce.n_components: 15
      topic.hdbscan.min_cluster_size: 10
```

---

## Model Configuration

### Creating a Custom Model Config

Create `conf/model/my_custom_model.yaml`:

```yaml
model_source: /path/to/my/model

engine_kwargs:
  # Context window
  max_model_len: 8192
  
  # Throughput settings
  max_num_batched_tokens: 2048
  max_num_seqs: 8
  
  # Memory management
  gpu_memory_utilization: 0.85
  
  # Parallelism
  tensor_parallel_size: 2  # Split across 2 GPUs
  pipeline_parallel_size: 1
  
  # Performance optimizations
  enable_chunked_prefill: true
  enable_prefix_caching: true
  use_v2_block_manager: true
  
  # Model loading
  tokenizer_mode: auto
  trust_remote_code: true
  dtype: auto
  kv_cache_dtype: auto
  
  # Stability
  disable_log_stats: true
  enforce_eager: false  # Enable CUDA graphs for speed

# Ray Data batch size
batch_size: 16

# Concurrent vLLM instances (usually 1)
concurrency: 1
```

Use it:
```bash
python -m pipelines.uair.cli model=my_custom_model
```

### GPU-Specific Model Configs

**For 1x RTX A6000 (48GB)**:
```yaml
engine_kwargs:
  max_model_len: 16384
  max_num_seqs: 4
  gpu_memory_utilization: 0.9
  tensor_parallel_size: 1
batch_size: 4
```

**For 2x RTX A5000 (24GB each)**:
```yaml
engine_kwargs:
  max_model_len: 4096
  max_num_seqs: 2
  gpu_memory_utilization: 0.7
  tensor_parallel_size: 2
batch_size: 2
```

**For 4x A100 (40GB each)**:
```yaml
engine_kwargs:
  max_model_len: 32768
  max_num_seqs: 16
  gpu_memory_utilization: 0.9
  tensor_parallel_size: 4
batch_size: 32
```

### Sampling Parameters by Task

**Classification (short output)**:
```yaml
sampling_params_classify:
  temperature: 0.0      # Deterministic
  max_tokens: 4
  detokenize: false
  guided_decoding:
    choice: ["YES", "NO"]
```

**Taxonomy (controlled vocabulary)**:
```yaml
sampling_params_taxonomy:
  temperature: 0.0
  max_tokens: 16
  detokenize: false
  guided_decoding:
    choice: ${taxonomy_categories}  # Constrain to valid categories
```

**Extraction (structured JSON)**:
```yaml
sampling_params_decompose:
  temperature: 0.1
  max_tokens: 1024
  detokenize: false
  guided_decoding:
    json: ${json_schema}  # JSON schema for structured output
```

**Generation (creative text)**:
```yaml
sampling_params_generate:
  temperature: 0.7
  top_p: 0.9
  top_k: 50
  max_tokens: 512
  detokenize: true
```

---

## SLURM Launcher Configuration

### Available Launchers

| Launcher | GPUs | CPUs | Memory | Use Case |
|----------|------|------|--------|----------|
| `null` | 0 | Local | Local | Local testing |
| `g2_slurm_cpu` | 0 | 2 | 16GB | Lightweight CPU tasks |
| `g2_slurm_cpu_beefy` | 0 | 8 | 64GB | Heavy CPU tasks (embeddings, clustering) |
| `g2_slurm_gpu_1x` | 1 | 8 | 32GB | Single-GPU LLM inference |
| `g2_slurm_pierson` | 2 | 8 | 32GB | Two-GPU model parallelism |
| `g2_slurm_gpu_4x` | 4 | 8 | 32GB | Four-GPU model parallelism |
| `slurm_monitor` | 0 | 2 | 8GB | Pipeline orchestrator (parent job) |

### Creating Custom Launcher

Create `conf/hydra/launcher/my_launcher.yaml`:

```yaml
_target_: hydra_plugins.hydra_submitit_launcher.submitit_launcher.SlurmLauncher
submitit_folder: ${hydra.sweep.dir}/.submitit/%j
timeout_min: 1440  # 24 hours
nodes: 1
tasks_per_node: 1
cpus_per_task: 16
mem_gb: 128
gpus_per_node: 2
partition: my_partition
array_parallelism: 1
name: UAIR-custom

additional_parameters:
  gres: gpu:2
  wckey: ""

setup:
  - export HYDRA_FULL_ERROR=1
  - source ~/.bashrc
  - source /path/to/venv/bin/activate
  - unset WANDB_DISABLED
  - export TOKENIZERS_PARALLELISM=false
  - export CUDA_VISIBLE_DEVICES=0,1
```

### Per-Node Launcher Selection

```yaml
nodes:
  lightweight_task:
    stage: sentiment
    launcher: g2_slurm_cpu  # CPU-only, fast
  
  heavy_embeddings:
    stage: topic
    launcher: g2_slurm_cpu_beefy  # 8 CPUs, 64GB RAM
  
  llm_inference:
    stage: taxonomy
    launcher: g2_slurm_gpu_4x  # 4 GPUs, tensor parallelism
  
  local_test:
    stage: verify
    launcher: null  # Run locally, no SLURM
```

---

## Environment-Specific Configs

### Development Environment

Create `conf/env/dev.yaml`:

```yaml
# @package _global_
defaults:
  - override /hydra/launcher: null

runtime:
  debug: true
  sample_n: 50  # Small sample for fast iteration
  output_root: ./outputs/dev

wandb:
  enabled: false  # Don't log to W&B during dev

model:
  engine_kwargs:
    max_model_len: 2048  # Smaller context for speed
    gpu_memory_utilization: 0.5  # Leave room for debugging
```

Use: `python -m pipelines.uair.cli env=dev`

### Staging Environment

Create `conf/env/staging.yaml`:

```yaml
# @package _global_
defaults:
  - override /hydra/launcher: slurm_monitor

runtime:
  debug: true
  sample_n: 1000  # Moderate sample
  output_root: /staging/outputs

wandb:
  enabled: true
  project: UAIR-staging
  group: staging-${now:%Y%m%d}

model:
  engine_kwargs:
    max_model_len: 8192
    gpu_memory_utilization: 0.8
```

### Production Environment

Create `conf/env/prod.yaml`:

```yaml
# @package _global_
defaults:
  - override /hydra/launcher: slurm_monitor

runtime:
  debug: false
  sample_n: null  # Full dataset
  output_root: /production/outputs/${now:%Y-%m-%d}

wandb:
  enabled: true
  project: UAIR-production
  group: prod-${now:%Y%m%d}
  tags: [production, ${experiment.name}]

model:
  engine_kwargs:
    max_model_len: 16384
    gpu_memory_utilization: 0.9
    max_num_seqs: 16
```

---

## Advanced Patterns

### Pattern 1: Config Inheritance Chain

```yaml
# conf/pipeline/base_pipeline.yaml
# @package _global_
pipeline:
  output_root: ${runtime.output_root}
  allow_partial: false
  sources:
    articles:
      path: ${data.parquet_path}
      type: parquet
```

```yaml
# conf/pipeline/extended_pipeline.yaml
# @package _global_
defaults:
  - base_pipeline  # Inherit from base

# Add/override nodes
pipeline:
  graph:
    nodes:
      # ... additional nodes
```

### Pattern 2: Conditional Configuration

```yaml
# Use environment variable to control GPU usage
runtime:
  use_gpu: ${oc.env:USE_GPU,true}

nodes:
  topic:
    overrides:
      topic.embed.device: ${oc.decode:"{\"cuda\" if runtime.use_gpu else \"cpu\"}"}
```

### Pattern 3: Dynamic Output Paths

```yaml
experiment:
  name: ${oc.env:EXPERIMENT_NAME,default_experiment}
  run_id: ${now:%Y%m%d_%H%M%S}

runtime:
  output_root: /outputs/${experiment.name}/${experiment.run_id}
```

### Pattern 4: Shared Overrides

```yaml
# Define shared overrides once
_shared_llm_config: &llm_config
  model.engine_kwargs.max_model_len: 4096
  model.engine_kwargs.gpu_memory_utilization: 0.7
  model.batch_size: 8

nodes:
  classify:
    overrides:
      <<: *llm_config  # Reuse shared config
      runtime.sample_n: 100
  
  taxonomy:
    overrides:
      <<: *llm_config  # Reuse again
      runtime.sample_n: 200
```

### Pattern 5: Multi-Dataset Processing

```yaml
pipeline:
  sources:
    dataset_a:
      path: /data/dataset_a.parquet
      type: parquet
    dataset_b:
      path: /data/dataset_b.parquet
      type: parquet
  
  graph:
    nodes:
      process_a:
        stage: classify
        inputs:
          dataset: dataset_a
        outputs:
          results: outputs/process_a/results.parquet
      
      process_b:
        stage: classify
        inputs:
          dataset: dataset_b
        outputs:
          results: outputs/process_b/results.parquet
      
      merge:
        stage: merge
        depends_on: [process_a, process_b]
        inputs:
          data_a: process_a.results
          data_b: process_b.results
        outputs:
          merged: outputs/merge/merged.parquet
```

---

## Configuration Validation

### Pre-Flight Checks

```bash
# Validate configuration without running
python -m pipelines.uair.cli \
  pipeline=my_pipeline \
  --cfg job  # Print resolved config

# Check for missing files/references
python -m pipelines.uair.cli \
  pipeline=my_pipeline \
  hydra.verbose=true
```

### Common Configuration Errors

**Error**: `KeyError: 'unknown artifact reference'`
```yaml
# ❌ Bad: Typo in dependency reference
taxonomy:
  inputs:
    dataset: classfy.relevant  # Typo: 'classfy' instead of 'classify'

# ✅ Good: Correct reference
taxonomy:
  inputs:
    dataset: classify.relevant
```

**Error**: `ValueError: Cycle detected in pipeline graph`
```yaml
# ❌ Bad: Circular dependency
stage_a:
  depends_on: [stage_b]
stage_b:
  depends_on: [stage_a]  # Circular!

# ✅ Good: Acyclic dependency graph
stage_a:
  depends_on: []
stage_b:
  depends_on: [stage_a]
```

**Error**: `InterpolationKeyError`
```yaml
# ❌ Bad: Referencing undefined variable
output_path: ${undefined_var}/outputs

# ✅ Good: Define variable first or provide default
runtime:
  base_path: /outputs
output_path: ${runtime.base_path}/outputs

# Or with default:
output_path: ${oc.env:OUTPUT_PATH,/default/path}/outputs
```

---

## Summary

This guide covered:

- Organizing configuration files hierarchically  
- Composing configs with defaults and overrides  
- Building common pipeline patterns (linear, parallel, filter-then-process)  
- Overriding parameters per-node  
- Configuring models for different GPU setups  
- Selecting appropriate SLURM launchers  
- Creating environment-specific configurations  
- Using advanced patterns (inheritance, conditionals, dynamic paths)  

For additional information:
- See [Custom Stages Guide](CUSTOM_STAGES_GUIDE.md) for implementing new stages
- See [SLURM Guide](SLURM_GUIDE.md) for distributed execution details
- See [Complete Examples](EXAMPLES.md) for full pipeline walkthroughs

---

*Last updated: 2025-10-02*

