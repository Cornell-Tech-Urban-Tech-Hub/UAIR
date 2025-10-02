# UAIR: Urban AI Risks Assessment Framework

Large-scale AI-powered analysis of urban AI risks in news media.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## Overview

UAIR (Urban AI Risks) is a scalable pipeline framework for assessing AI-related risks in urban contexts through large-scale inference over news article datasets. The framework enables researchers to:

- **Classify** articles for AI relevance using LLM-powered or heuristic methods
- **Categorize** articles into risk taxonomies (climate adaptation, governance, ethics)
- **Extract** structured information about AI deployments and impacts
- **Cluster** articles by topic to discover emerging patterns
- **Verify** claims and validate extracted information

### Key Features

- **Configuration-Driven**: Define complex multi-stage pipelines in YAML, no code changes needed
- **Scalable**: Process millions of articles using Ray Data and SLURM clusters
- **LLM-Integrated**: Built-in vLLM support with automatic GPU management
- **Modular**: Mix and match stages, models, and datasets
- **Tracked**: Automatic experiment logging with Weights & Biases
- **Extensible**: Easy to add custom processing stages

---

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/your-org/UAIR.git
cd UAIR

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt
```

### Run Your First Pipeline

```bash
# Topic modeling on a sample of articles
python -m pipelines.uair.cli \
  runtime.debug=true \
  runtime.sample_n=100 \
  data.parquet_path=/path/to/articles.parquet
```

### Example: Full Risk Assessment Pipeline

```bash
# Complete pipeline: classify → taxonomy → verify
python -m pipelines.uair.cli \
  pipeline=taxonomy_full \
  data.parquet_path=/path/to/articles.parquet
```

Results are saved to `outputs/` with full experiment tracking in W&B.

---

## Documentation

Complete documentation is available in `docs/`:

### Getting Started

- **[Documentation Hub](docs/README.md)** - Documentation navigation and index
- **[User Guide](docs/USER_GUIDE.md)** - Complete introduction with Quick Start
- **[Quick Reference](docs/QUICK_REFERENCE.md)** - Command cheat sheet

### Building Pipelines

- **[Configuration Guide](docs/CONFIGURATION_GUIDE.md)** - Pipeline recipes and config patterns
- **[Custom Stages Guide](docs/CUSTOM_STAGES_GUIDE.md)** - Building custom processing stages

### Learning Path

| Level | Time | What to Read |
|-------|------|--------------|
| **Beginner** | 1-2 hours | [User Guide](docs/USER_GUIDE.md) (Intro + Quick Start + Core Concepts) |
| **Intermediate** | 3-4 hours | [Configuration Guide](docs/CONFIGURATION_GUIDE.md) (Pipeline Recipes) |
| **Advanced** | 5+ hours | [Custom Stages Guide](docs/CUSTOM_STAGES_GUIDE.md) (Build custom stages) |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Pipeline Definition (YAML)                    │
├─────────────────────────────────────────────────────────────────┤
│  Sources → Stage 1 → Stage 2 → Stage 3 → Outputs               │
│  (Data)    (classify) (taxonomy) (verify)  (Parquet)           │
├─────────────────────────────────────────────────────────────────┤
│              Orchestrator (DAG Execution Engine)                 │
├─────────────────────────────────────────────────────────────────┤
│  Ray Data (Distributed) | vLLM (GPU) | SLURM (Cluster) | W&B   │
├─────────────────────────────────────────────────────────────────┤
│            Hydra (Configuration Management)                      │
└─────────────────────────────────────────────────────────────────┘
```

### Built-in Processing Stages

| Stage | Purpose | Input | Output |
|-------|---------|-------|--------|
| **classify** | Relevance filtering | Raw articles | `is_relevant` flag |
| **taxonomy** | Risk categorization | Articles | `chunk_label` (risk category) |
| **decompose** | Info extraction | Articles | Structured fields |
| **topic** | Topic modeling | Articles | `topic_id`, cluster info |
| **verification** | Claim validation | Labeled articles | Verification scores |

---

## Project Structure

```
UAIR/
├── docs/                          # Documentation
│   ├── README.md                  # Documentation hub
│   ├── USER_GUIDE.md              # Complete user guide
│   ├── CUSTOM_STAGES_GUIDE.md     # Build custom stages
│   ├── CONFIGURATION_GUIDE.md     # Config recipes
│   └── QUICK_REFERENCE.md         # Cheat sheet
├── pipelines/uair/                # Core framework
│   ├── cli.py                     # CLI entry point
│   ├── orchestrator.py            # Pipeline orchestrator
│   ├── config_schema.py           # Configuration schemas
│   ├── wandb_logger.py            # W&B integration
│   ├── conf/                      # Configuration files
│   │   ├── config.yaml            # Base config
│   │   ├── data/                  # Data source configs
│   │   ├── model/                 # Model configs
│   │   ├── prompt/                # Prompt templates
│   │   ├── pipeline/              # Pipeline definitions
│   │   └── hydra/launcher/        # SLURM configs
│   └── stages/                    # Processing stages
│       ├── classify.py            # Relevance classification
│       ├── taxonomy.py            # Risk categorization
│       ├── topic.py               # Topic modeling
│       ├── verify.py              # Verification
│       └── decompose.py           # Information extraction
├── scripts/                       # Utility scripts
├── data/                          # Data directory
├── outputs/                       # Pipeline outputs
└── requirements.txt               # Python dependencies
```

---

## Use Cases

### Urban AI Risks Assessment (Primary)

Analyze news coverage of AI deployments in urban contexts:

```yaml
# conf/pipeline/urban_risks.yaml
pipeline:
  graph:
    nodes:
      classify:  # Filter AI-relevant articles
        stage: classify
      taxonomy:  # Categorize by risk type
        stage: taxonomy
        depends_on: [classify]
      verify:    # Validate claims
        stage: verification
        depends_on: [taxonomy]
```

### Custom Domain Analysis

Adapt for other domains (medical, legal, scientific):

1. Define your taxonomy in `conf/taxonomy/my_domain.yaml`
2. Create custom prompts in `conf/prompt/my_prompts.yaml`
3. Build pipeline in `conf/pipeline/my_pipeline.yaml`
4. Run: `python -m pipelines.uair.cli pipeline=my_pipeline`

See [Custom Stages Guide](docs/CUSTOM_STAGES_GUIDE.md) for details.

---

## Example Pipelines

### Topic Modeling

```bash
# Discover topics in your dataset
python -m pipelines.uair.cli \
  pipeline=cluster_topic \
  topic.embed.device=cuda \
  data.parquet_path=/data/articles.parquet
```

### Multi-Stage Analysis

```bash
# Full pipeline with classification, taxonomy, and verification
python -m pipelines.uair.cli \
  pipeline=taxonomy_full \
  runtime.sample_n=1000 \
  data.parquet_path=/data/articles.parquet
```

### Custom Configuration

```bash
# Override GPU and batch settings
python -m pipelines.uair.cli \
  pipeline=my_pipeline \
  model.engine_kwargs.max_model_len=8192 \
  model.batch_size=16 \
  model.engine_kwargs.tensor_parallel_size=4
```

More examples in [Configuration Guide](docs/CONFIGURATION_GUIDE.md#pipeline-recipes).

---

## Deployment

### Local Execution

```bash
# Run locally (no SLURM)
python -m pipelines.uair.cli \
  hydra/launcher=null \
  runtime.sample_n=100
```

### SLURM Cluster

```bash
# Submit to SLURM with GPU
python -m pipelines.uair.cli \
  pipeline=my_pipeline \
  hydra/launcher=g2_slurm_gpu_4x
```

See [Configuration Guide - SLURM Launchers](docs/CONFIGURATION_GUIDE.md#slurm-launcher-configuration) for details.

---

## Configuration

UAIR uses [Hydra](https://hydra.cc/) for hierarchical configuration:

```yaml
# config.yaml
defaults:
  - data: inputs
  - model: vllm_qwen3-30b
  - prompt: classify
  - pipeline: null

runtime:
  debug: false
  sample_n: null
  output_root: ./outputs

pipeline:
  sources:
    articles:
      path: ${data.parquet_path}
  graph:
    nodes:
      # Define your stages here
```

**Override from command line**:
```bash
python -m pipelines.uair.cli \
  runtime.debug=true \
  model.batch_size=8 \
  data.parquet_path=/path/to/data.parquet
```

See [Configuration Guide](docs/CONFIGURATION_GUIDE.md) for comprehensive patterns.

---

## Development

### Running Tests

```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Run with coverage
pytest --cov=pipelines tests/
```

### Creating a Custom Stage

1. **Implement stage function** in `pipelines/uair/stages/mystage.py`:
```python
def run_mystage(df, cfg):
    # Your processing logic
    return df
```

2. **Register stage** in `orchestrator.py`:
```python
class MyStageRunner(StageRunner):
    stage_name = "mystage"
    def run(self, context):
        # ...
```

3. **Add to registry**:
```python
_STAGE_REGISTRY["mystage"] = MyStageRunner()
```

Full guide: [Custom Stages Guide](docs/CUSTOM_STAGES_GUIDE.md)

---

## Research

This framework supports the Urban AI Risks research project, assessing AI deployment risks in urban contexts through large-scale news analysis.

**Related Work**:
- Climate adaptation taxonomy (Weitz et al.)
- AI risk frameworks
- Urban AI governance

---

## Contributing

We welcome contributions! Areas of interest:

- **New Stages**: Additional processing capabilities
- **Taxonomies**: Domain-specific risk categorizations
- **Optimizations**: Performance improvements
- **Documentation**: Examples, tutorials, guides

See [Custom Stages Guide](docs/CUSTOM_STAGES_GUIDE.md) for implementation guidelines.

---

## License

MIT License - see [LICENSE](LICENSE) file for details.

---

## Getting Help

### Documentation

- Start with [User Guide](docs/USER_GUIDE.md)
- Check [Quick Reference](docs/QUICK_REFERENCE.md) for common issues
- Browse [Configuration Guide](docs/CONFIGURATION_GUIDE.md) for recipes

### Common Issues

**GPU Out of Memory**:
```bash
python -m pipelines.uair.cli \
  model.engine_kwargs.gpu_memory_utilization=0.6 \
  model.batch_size=2
```

**Ray Object Store Full**:
```bash
python -m pipelines.uair.cli \
  runtime.rows_per_block=1000
```

**Debug Mode**:
```bash
python -m pipelines.uair.cli \
  runtime.debug=true \
  runtime.sample_n=10
```

More troubleshooting: [Quick Reference - Troubleshooting](docs/QUICK_REFERENCE.md#troubleshooting)

---

## Acknowledgments

Built with:
- [Hydra](https://hydra.cc/) - Configuration management
- [Ray Data](https://docs.ray.io/en/latest/data/data.html) - Distributed processing
- [vLLM](https://docs.vllm.ai/) - LLM inference
- [Weights & Biases](https://wandb.ai/) - Experiment tracking
- [SLURM](https://slurm.schedmd.com/) - Cluster scheduling

---

## Contact

For questions about the framework:
- Check the [documentation](docs/)
- Review [examples](docs/CONFIGURATION_GUIDE.md#pipeline-recipes)
- Consult [troubleshooting guide](docs/QUICK_REFERENCE.md#troubleshooting)

---

For additional information, consult the [User Guide - Quick Start](docs/USER_GUIDE.md#quick-start).

---

*Project maintained by the Urban AI Risks research team.*
