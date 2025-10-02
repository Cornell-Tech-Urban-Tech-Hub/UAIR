# UAIR Pipeline Framework - User Guide

Building Custom dagspaces for Urban AI Risk Assessment

---

## Table of Contents

1. [Introduction](#introduction)
2. [Quick Start](#quick-start)
3. [Core Concepts](#core-concepts)
4. [Building Custom Stages](#building-custom-stages)
5. [Configuration Guide](#configuration-guide)
6. [Advanced Topics](#advanced-topics)
7. [Complete Examples](#complete-examples)
8. [Best Practices](#best-practices)
9. [Reference](#reference)
10. [Troubleshooting](#troubleshooting)

---

## Introduction

### What is UAIR?

UAIR (Urban AI Risks) is a pipeline framework designed for assessing AI-related risks in urban contexts by running large-scale AI inference over news article datasets. The framework provides:

- **DAG-based Pipeline Orchestration**: Define complex multi-stage workflows as directed acyclic graphs
- **Distributed Execution**: Scale to large datasets using Ray Data and SLURM clusters
- **LLM Integration**: Built-in support for vLLM inference with automatic GPU management
- **Experiment Tracking**: Automatic logging to Weights & Biases
- **Configuration-Driven**: Modular, composable configs using Hydra

### Core Philosophy

UAIR follows three key principles:

1. **Configuration over Code**: Define dagspaces declaratively in YAML, not Python
2. **Composability**: Mix and match stages, models, and datasets
3. **Reproducibility**: Every run is tracked with full configuration snapshots

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    UAIR Pipeline Framework                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │   Sources    │───▶│   Stage 1    │───▶│   Stage 2    │      │
│  │ (Parquet/CSV)│    │  (classify)  │    │  (taxonomy)  │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│         │                    │                    │              │
│         │                    ▼                    ▼              │
│         │            ┌──────────────┐    ┌──────────────┐      │
│         └───────────▶│   Stage 3    │───▶│   Outputs    │      │
│                      │   (verify)   │    │  (Parquet)   │      │
│                      └──────────────┘    └──────────────┘      │
│                                                                   │
├─────────────────────────────────────────────────────────────────┤
│  Execution Layer: Orchestrator + Artifact Registry              │
├─────────────────────────────────────────────────────────────────┤
│  Compute Layer: Ray Data + vLLM + SLURM                         │
├─────────────────────────────────────────────────────────────────┤
│  Config Layer: Hydra + OmegaConf                                │
└─────────────────────────────────────────────────────────────────┘
```

### Key Components

| Component | Purpose | Technology |
|-----------|---------|------------|
| **Orchestrator** | Executes pipeline DAG, manages artifacts | Python |
| **Stage Runners** | Execute individual processing stages | Python + Ray |
| **Config System** | Manage complex hierarchical configs | Hydra + OmegaConf |
| **Data Processing** | Scalable data transformations | Ray Data |
| **LLM Inference** | High-performance model serving | vLLM |
| **Job Scheduling** | Distributed job execution | SLURM (optional) |
| **Experiment Tracking** | Log metrics, artifacts, configs | Weights & Biases |

---

## Quick Start

### Prerequisites

**Required:**
- Python 3.9+
- 16GB+ RAM (for local runs)
- Access to input data (Parquet format)

**Optional:**
- SLURM cluster (for distributed execution)
- NVIDIA GPU (for LLM stages)
- Weights & Biases account (for tracking)

### Installation

```bash
# Clone the repository
cd /path/to/UAIR

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import ray, hydra, vllm; print('All dependencies installed')"
```

### Your First Pipeline: Topic Modeling

This section demonstrates a simple single-stage pipeline that performs topic modeling on news articles.

#### Step 1: Prepare Your Data

Ensure you have a Parquet file with an `article_text` column:

```python
import pandas as pd

# Example: Create sample data
articles = pd.DataFrame({
    'article_text': [
        'AI surveillance cameras deployed in city centers...',
        'Machine learning predicts urban traffic patterns...',
        'Smart city initiative raises privacy concerns...',
        # ... more articles
    ]
})

articles.to_parquet('/path/to/your/articles.parquet', index=False)
```

#### Step 2: Configure the Pipeline

Create `conf/pipeline/my_first_pipeline.yaml`:

```yaml
# @package _global_
defaults:
  - override /hydra/launcher: null  # Run locally (no SLURM)

runtime:
  debug: true
  sample_n: 100  # Process only 100 articles for testing
  output_root: ./outputs/my_first_run

pipeline:
  output_root: ${runtime.output_root}
  allow_partial: false
  
  sources:
    articles:
      path: /path/to/your/articles.parquet
      type: parquet
  
  graph:
    nodes:
      topic:
        stage: topic
        depends_on: []
        inputs:
          dataset: articles
        outputs:
          docs: outputs/topic/docs_topics.parquet
        overrides:
          topic.embed.device: cpu  # Use CPU for embeddings
        wandb_suffix: topic_modeling
```

#### Step 3: Run the Pipeline

```bash
# Run from the dagspaces/uair directory
python -m dagspaces.uair.cli \
  pipeline=my_first_pipeline \
  data.parquet_path=/path/to/your/articles.parquet
```

#### Step 4: Examine Outputs

After completion, you'll find:

```
outputs/my_first_run/
├── pipeline_manifest.json      # Execution metadata
└── outputs/
    └── topic/
        └── docs_topics.parquet  # Results with topic assignments
```

Load and inspect results:

```python
import pandas as pd

results = pd.read_parquet('outputs/my_first_run/outputs/topic/docs_topics.parquet')
print(results.columns)
# ['article_text', 'topic_id', 'topic_prob', 'topic_top_terms', 'plot_x', 'plot_y', ...]

# View topic distribution
print(results['topic_id'].value_counts())

# Check top terms for each topic
print(results.groupby('topic_id')['topic_top_terms'].first())
```

### Understanding What Happened

1. **Configuration Loading**: Hydra loaded your pipeline config and merged it with defaults
2. **Pipeline Graph**: The orchestrator parsed the DAG (single node: `topic`)
3. **Data Loading**: Articles were loaded from Parquet into a pandas DataFrame
4. **Topic Stage Execution**:
   - Text embeddings generated using `nomic-embed-text-v1.5`
   - UMAP dimensionality reduction
   - HDBSCAN clustering
   - Top terms extracted via TF-IDF
5. **Output Saving**: Results saved as Parquet with topic assignments
6. **W&B Logging**: If configured, run logged to your W&B project

### Next Steps

With a working pipeline, consider the following extensions:

- **Add More Stages**: Chain topic modeling with classification
- **Use GPU**: Set `topic.embed.device: cuda` for faster embeddings
- **Scale Up**: Remove `sample_n` to process full dataset
- **Deploy to SLURM**: Use `override /hydra/launcher: g2_slurm_cpu`
- **Customize**: Adjust UMAP/HDBSCAN parameters in `topic` config

Refer to [Core Concepts](#core-concepts) for detailed framework understanding, or [Complete Examples](#complete-examples) for more complex dagspaces.

---

## Core Concepts

### 1. Pipeline Architecture

A UAIR pipeline is defined as a **Directed Acyclic Graph (DAG)** of processing stages.

#### Components of a Pipeline

**1. Sources**: Input data definitions

```yaml
pipeline:
  sources:
    articles:           # Logical name
      path: /data/articles.parquet
      type: parquet
```

**2. Nodes**: Processing stages with dependencies

```yaml
pipeline:
  graph:
    nodes:
      classify:
        stage: classify              # Which stage implementation to use
        depends_on: []               # No dependencies (root node)
        inputs:
          dataset: articles          # Reference to source
        outputs:
          all: outputs/classify/all.parquet
          relevant: outputs/classify/relevant.parquet
```

**3. Artifact Registry**: Tracks data flow between stages

The orchestrator maintains an internal registry that maps logical names to file paths:

```
Registry:
  articles → /data/articles.parquet
  classify.all → /outputs/classify/all.parquet
  classify.relevant → /outputs/classify/relevant.parquet
```

When a downstream stage references `classify.relevant`, the registry resolves it to the actual file path.

#### Execution Flow

1. **Topological Sort**: Orchestrator orders nodes based on dependencies
2. **Sequential Execution**: Each node runs after all dependencies complete
3. **Artifact Resolution**: Inputs resolved from registry
4. **Output Registration**: New outputs added to registry for downstream use

Example execution order:

```
classify (depends on: [])
   ↓
taxonomy (depends on: [classify])
   ↓
verification (depends on: [taxonomy])
```

#### Multi-Output Stages

Stages can produce multiple outputs that downstream stages can reference independently:

```yaml
nodes:
  classify:
    outputs:
      all: outputs/classify/all.parquet          # All classified articles
      relevant: outputs/classify/relevant.parquet # Only relevant ones
  
  taxonomy:
    inputs:
      dataset: classify.relevant  # Use only relevant articles
  
  analyze_all:
    inputs:
      dataset: classify.all       # Use all articles
```

### 2. Configuration System

UAIR uses **Hydra** for hierarchical configuration management with **OmegaConf** for variable interpolation.

#### Configuration Structure

```
dagspaces/uair/conf/
├── config.yaml                    # Base config
├── data/
│   ├── inputs.yaml               # Input data configs
│   └── flattened_rules.yaml
├── model/
│   └── vllm_qwen3-30b.yaml      # Model configs
├── prompt/
│   ├── classify.yaml             # Prompt templates
│   ├── taxonomy.yaml
│   └── decompose.yaml
├── pipeline/
│   ├── taxonomy_full.yaml        # Complete pipeline configs
│   └── topic_modeling_of_relevant_classifications.yaml
├── taxonomy/
│   └── weitz.yaml                # Domain taxonomies
└── hydra/
    └── launcher/
        ├── g2_slurm_cpu.yaml     # SLURM configs
        └── g2_slurm_gpu_4x.yaml
```

#### Config Composition with Defaults

The `defaults` list specifies which configs to compose:

```yaml
# config.yaml
defaults:
  - _self_                    # Include this file
  - data: inputs              # Load conf/data/inputs.yaml
  - prompt: classify          # Load conf/prompt/classify.yaml
  - model: vllm_qwen3-30b     # Load conf/model/vllm_qwen3-30b.yaml
  - optional pipeline: null   # Optionally load a pipeline config

experiment:
  name: UAIR

runtime:
  debug: false
  sample_n: null
```

**Note**: The order of items in the defaults list determines precedence, with later items overriding earlier ones. The `_self_` keyword controls when the current file's values are applied.

#### Variable Interpolation

OmegaConf supports powerful variable interpolation:

```yaml
# Reference other config values
runtime:
  output_root: /outputs
  debug: false

pipeline:
  output_root: ${runtime.output_root}  # → /outputs

# Environment variables with defaults
taxonomy_json: ${oc.env:TAXONOMY_JSON,/default/path/taxonomy.yaml}

# Nested references
data:
  parquet_path: /data/articles.parquet

pipeline:
  sources:
    articles:
      path: ${data.parquet_path}  # → /data/articles.parquet
```

#### Command-Line Overrides

Override any config value from the command line:

```bash
# Simple overrides
python -m dagspaces.uair.cli runtime.debug=true

# Nested overrides
python -m dagspaces.uair.cli model.engine_kwargs.max_model_len=4096

# Change config group
python -m dagspaces.uair.cli data=flattened_rules

# Multiple overrides
python -m dagspaces.uair.cli \
  runtime.debug=true \
  runtime.sample_n=500 \
  model.batch_size=8
```

#### Config Groups and Selection

Config groups allow you to swap entire configuration blocks:

```bash
# Use different data source
python -m dagspaces.uair.cli data=flattened_rules

# Use different model
python -m dagspaces.uair.cli model=different_model

# Use different pipeline
python -m dagspaces.uair.cli pipeline=taxonomy_full
```

### 3. Stage Runners & Registry

#### The StageRunner Interface

Every stage implements the `StageRunner` base class:

```python
from dataclasses import dataclass
from typing import Dict, Any
import pandas as pd
from omegaconf import DictConfig

@dataclass
class StageExecutionContext:
    cfg: DictConfig              # Full config for this stage
    node: PipelineNodeSpec       # Node definition from pipeline config
    inputs: Dict[str, str]       # Resolved input paths: {alias: path}
    output_paths: Dict[str, str] # Resolved output paths: {name: path}
    output_dir: str              # Directory for outputs
    output_root: str             # Root output directory
    logger: Optional[WandbLogger] = None  # W&B logger

@dataclass
class StageResult:
    outputs: Dict[str, str]      # Actual output paths created
    metadata: Dict[str, Any]     # Stage execution metadata (rows, etc.)

class StageRunner:
    stage_name: str  # Must match the stage name in config
    
    def run(self, context: StageExecutionContext) -> StageResult:
        raise NotImplementedError
```

#### Built-in Stage Runners

UAIR provides five built-in stage runners:

| Stage Runner | Stage Name | Purpose |
|--------------|------------|---------|
| `ClassificationRunner` | `classify` | LLM-based relevance classification |
| `TaxonomyRunner` | `taxonomy` | Hierarchical category assignment |
| `DecomposeRunner` | `decompose` | Structured information extraction |
| `TopicRunner` | `topic` | Unsupervised topic modeling |
| `VerificationRunner` | `verification` | Multi-strategy claim verification |

#### Stage Registry

The orchestrator maintains a registry mapping stage names to implementations:

```python
_STAGE_REGISTRY: Dict[str, StageRunner] = {
    "classify": ClassificationRunner(),
    "decompose": DecomposeRunner(),
    "taxonomy": TaxonomyRunner(),
    "topic": TopicRunner(),
    "verification": VerificationRunner(),
}
```

Custom stages are registered by adding them to this dictionary.

#### Stage Execution Lifecycle

```
1. Pipeline graph parsed
    ↓
2. Node dependencies resolved
    ↓
3. For each node in topological order:
    a. Prepare stage config (base + overrides)
    b. Resolve inputs from artifact registry
    c. Resolve output paths
    d. Create StageExecutionContext
    e. Initialize W&B logger
    f. Call stage_runner.run(context)
    g. Register outputs in artifact registry
    h. Log metrics and artifacts to W&B
    ↓
4. Pipeline complete
```

### 4. Data Processing Patterns

#### Input Formats

Stages can process data in two formats:

**1. Pandas DataFrame** (default for small datasets)

```python
def run_my_stage(df: pd.DataFrame, cfg: DictConfig):
    # Process in-memory
    df['new_column'] = df['article_text'].apply(process_text)
    return df
```

**2. Ray Dataset** (for large datasets with `runtime.streaming_io=true`)

```python
def run_my_stage(ds: ray.data.Dataset, cfg: DictConfig):
    # Process in parallel, streaming
    ds = ds.map_batches(process_batch, batch_format="pandas")
    return ds
```

The orchestrator automatically handles conversion based on `runtime.streaming_io` flag.

#### Canonical Column Names

UAIR expects certain canonical column names:

| Column | Description | Required |
|--------|-------------|----------|
| `article_text` | Full text of article | Yes (for most stages) |
| `article_id` | Unique identifier | Auto-generated if missing |
| `article_path` | Original file path | Optional |
| `chunk_text` | Text chunk (for chunked processing) | Optional |
| `is_relevant` | Binary relevance flag | Output of classify stage |

The orchestrator performs automatic column mapping based on `data.columns` config.

#### Handling Nested Data

Parquet doesn't support nested Python objects natively. UAIR provides serialization helpers:

```python
from dagspaces.uair.stages.classify import _to_json_str

# Serialize nested structures
row['messages'] = _to_json_str(messages_list)
row['usage'] = _to_json_str(usage_dict)

# Set runtime.serialize_nested_json=true to auto-serialize
```

---

## Building Custom Stages

Refer to the [Custom Stages Guide](CUSTOM_STAGES_GUIDE.md) for detailed information on implementing custom processing stages.

---

For additional topics including Configuration Recipes, Advanced Topics, Complete Examples, and Troubleshooting, consult the [Configuration Guide](CONFIGURATION_GUIDE.md) and [Quick Reference](QUICK_REFERENCE.md).

---

*Last updated: 2025-10-02*

