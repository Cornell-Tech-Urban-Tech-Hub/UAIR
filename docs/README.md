# UAIR Pipeline Framework Documentation

**Build scalable, configuration-driven pipelines for Urban AI Risk assessment**

---

## Welcome! 👋

This documentation will help you build custom experimental pipelines using the UAIR framework. Whether you're analyzing news articles, extracting risk indicators, or building entirely new AI risk assessment workflows, this guide has you covered.

---

## 📚 Documentation Structure

### For Beginners

**Start here if you're new to UAIR:**

1. **[User Guide](USER_GUIDE.md)** - Complete introduction
   - What is UAIR?
   - Quick Start (your first pipeline in 10 minutes)
   - Core Concepts (architecture, configuration, stages)
   - Required reading for understanding the framework

### For Pipeline Builders

**Once you understand the basics:**

2. **[Configuration Guide](CONFIGURATION_GUIDE.md)** - Master Hydra configs
   - Pipeline recipes (linear, parallel, filter-then-process)
   - Per-node configuration
   - Model and SLURM launcher configuration
   - Advanced patterns and best practices

3. **[Custom Stages Guide](CUSTOM_STAGES_GUIDE.md)** - Extend the framework
   - Stage implementation templates
   - LLM-powered stages with vLLM
   - Advanced patterns (caching, multi-output, conditional)
   - Testing strategies

### For Quick Lookup

4. **[Quick Reference](QUICK_REFERENCE.md)** - Cheat sheet
   - Common commands
   - Configuration snippets
   - Stage reference
   - Troubleshooting tips

### Advanced Topics (Coming Soon)

5. **SLURM Guide** - Distributed execution on clusters
6. **Complete Examples** - End-to-end pipeline walkthroughs
7. **API Reference** - Detailed API documentation
8. **Troubleshooting** - Common issues and solutions

---

## 🚀 Quick Start

### 1. Installation

```bash
cd /share/pierson/matt/UAIR
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run Your First Pipeline

```bash
# Simple topic modeling on 100 articles
python -m pipelines.uair.cli \
  runtime.debug=true \
  runtime.sample_n=100 \
  data.parquet_path=/path/to/articles.parquet
```

### 3. Explore the Documentation

- **New to pipelines?** → [User Guide](USER_GUIDE.md#quick-start)
- **Building a custom pipeline?** → [Configuration Guide](CONFIGURATION_GUIDE.md#pipeline-recipes)
- **Need a new stage?** → [Custom Stages Guide](CUSTOM_STAGES_GUIDE.md#simple-stage-template)
- **Need quick help?** → [Quick Reference](QUICK_REFERENCE.md)

---

## 🎯 What Can You Build?

### Example Use Cases

**✅ Multi-Stage Classification Pipeline**
```
Articles → Relevance Filter → Taxonomy Classification → Verification
```

**✅ Parallel Analysis**
```
Articles → Relevance Filter → [Topic Modeling | Sentiment Analysis | Risk Scoring]
```

**✅ Iterative Refinement**
```
Articles → Coarse Classification → Fine-Grained Analysis → Quality Check
```

**✅ Custom Domain Pipelines**
- Medical text analysis
- Legal document processing
- Code vulnerability detection
- Scientific literature mining
- ... and more!

---

## 🏗️ Framework Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Your Pipeline Definition                      │
│                         (YAML Config)                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  📊 Data Sources    →  🔧 Processing Stages  →  💾 Outputs      │
│  (Parquet/CSV)         (classify, taxonomy,     (Parquet)       │
│                         topic, custom, etc.)                     │
│                                                                   │
├─────────────────────────────────────────────────────────────────┤
│              Orchestrator (DAG Execution Engine)                 │
├─────────────────────────────────────────────────────────────────┤
│  Ray Data (Distributed)  |  vLLM (GPU Inference)  |  W&B (Logs) │
├─────────────────────────────────────────────────────────────────┤
│              Hydra (Configuration Management)                    │
└─────────────────────────────────────────────────────────────────┘
```

### Key Features

- **🔄 DAG-based Pipelines**: Express complex workflows as directed graphs
- **⚙️ Configuration-Driven**: No code changes needed for new pipelines
- **⚡ Distributed Execution**: Scale to large datasets with Ray + SLURM
- **🤖 LLM Integration**: Built-in vLLM support with automatic GPU management
- **📊 Experiment Tracking**: Automatic W&B logging
- **🧩 Modular**: Mix and match stages, models, and datasets

---

## 📖 Learning Path

### Beginner Track (1-2 hours)

1. Read [User Guide - Introduction](USER_GUIDE.md#introduction) (15 min)
2. Follow [Quick Start](USER_GUIDE.md#quick-start) tutorial (30 min)
3. Understand [Core Concepts](USER_GUIDE.md#core-concepts) (45 min)

**Goal**: Run a basic pipeline and understand the framework

### Intermediate Track (3-4 hours)

1. Study [Configuration Guide - Pipeline Recipes](CONFIGURATION_GUIDE.md#pipeline-recipes) (60 min)
2. Build a custom pipeline using recipes (90 min)
3. Learn [Per-Node Configuration](CONFIGURATION_GUIDE.md#per-node-configuration) (30 min)

**Goal**: Build production-ready pipelines from templates

### Advanced Track (5+ hours)

1. Create a custom stage with [Custom Stages Guide](CUSTOM_STAGES_GUIDE.md) (2 hours)
2. Build an LLM-powered stage (2 hours)
3. Deploy to SLURM cluster (1+ hours)

**Goal**: Extend the framework for your domain

---

## 🔍 Find What You Need

### By Task

- **"I want to run a pipeline"** → [User Guide - Quick Start](USER_GUIDE.md#quick-start)
- **"I want to build a new pipeline"** → [Configuration Guide - Recipes](CONFIGURATION_GUIDE.md#pipeline-recipes)
- **"I want to create a new stage"** → [Custom Stages Guide](CUSTOM_STAGES_GUIDE.md#simple-stage-template)
- **"I need to debug a config"** → [Quick Reference - Troubleshooting](QUICK_REFERENCE.md#troubleshooting)
- **"I want to use different GPUs"** → [Configuration Guide - Model Config](CONFIGURATION_GUIDE.md#model-configuration)
- **"I need to run on SLURM"** → [Configuration Guide - SLURM Launchers](CONFIGURATION_GUIDE.md#slurm-launcher-configuration)

### By Concept

- **Pipeline Architecture** → [User Guide - Core Concepts](USER_GUIDE.md#1-pipeline-architecture)
- **Configuration System** → [User Guide - Config System](USER_GUIDE.md#2-configuration-system)
- **Stage Runners** → [User Guide - Stage Runners](USER_GUIDE.md#3-stage-runners--registry)
- **Data Processing** → [User Guide - Data Patterns](USER_GUIDE.md#4-data-processing-patterns)

### By Stage Type

- **Classification** → Built-in `classify` stage
- **Taxonomy** → Built-in `taxonomy` stage  
- **Topic Modeling** → Built-in `topic` stage
- **Verification** → Built-in `verification` stage
- **Custom Stage** → [Custom Stages Guide](CUSTOM_STAGES_GUIDE.md)

---

## 💡 Common Workflows

### Workflow 1: Test a Pipeline Locally

```bash
# Quick test with 10 articles on CPU
python -m pipelines.uair.cli \
  pipeline=my_pipeline \
  runtime.debug=true \
  runtime.sample_n=10 \
  hydra/launcher=null
```

### Workflow 2: Run Full Pipeline on SLURM

```bash
# Production run with full dataset on GPU cluster
python -m pipelines.uair.cli \
  pipeline=production_pipeline \
  runtime.debug=false \
  runtime.sample_n=null \
  data.parquet_path=/data/full_dataset.parquet
```

### Workflow 3: Iterate on Configuration

```bash
# Try different model parameters
python -m pipelines.uair.cli \
  pipeline=my_pipeline \
  model.batch_size=8 \
  model.engine_kwargs.max_model_len=4096 \
  runtime.sample_n=100
```

### Workflow 4: Build and Test Custom Stage

1. Create stage: `pipelines/uair/stages/mystage.py`
2. Register: Add to `_STAGE_REGISTRY` in `orchestrator.py`
3. Test: `python -m pipelines.uair.cli pipeline=test_mystage`

---

## 🤝 Contributing

### Adding a New Stage

See [Custom Stages Guide - Registering Your Stage](CUSTOM_STAGES_GUIDE.md#registering-your-stage)

### Improving Documentation

Documentation lives in `/share/pierson/matt/UAIR/docs/`

To contribute:
1. Edit relevant markdown file
2. Follow existing structure and style
3. Add examples and code snippets
4. Update this README's table of contents if needed

---

## 🆘 Getting Help

### Documentation

- Start with [User Guide](USER_GUIDE.md)
- Check [Quick Reference](QUICK_REFERENCE.md) for common issues
- Search this documentation (Ctrl+F or Cmd+F)

### Debugging

1. **Enable debug mode**: `runtime.debug=true`
2. **Use small sample**: `runtime.sample_n=10`
3. **Check logs**: `outputs/.../stage_name/*.log`
4. **Verify config**: `python -m pipelines.uair.cli --cfg job`

### Common Issues

- **GPU OOM**: Reduce `model.engine_kwargs.max_model_len` or `model.batch_size`
- **Ray OOM**: Increase `runtime.object_store_proportion` or reduce `runtime.rows_per_block`
- **Config errors**: Check for typos in `depends_on` or input references
- **SLURM failures**: Check `.slurm_jobs/` logs

---

## 📊 What's Inside Each Stage?

### Built-in Stages

| Stage | Input | Output | Purpose |
|-------|-------|--------|---------|
| **classify** | Articles | `is_relevant` flag | Filter relevant articles |
| **taxonomy** | Articles | `chunk_label` | Assign risk categories |
| **decompose** | Articles | `ci_*` fields | Extract structured info |
| **topic** | Articles | `topic_id` | Unsupervised clustering |
| **verification** | Labeled articles | `verify_*` scores | Validate claims |

See [Quick Reference - Stage Reference](QUICK_REFERENCE.md#stage-reference) for details.

---

## 🎓 Examples Gallery (Coming Soon)

### Urban AI Risks Pipeline
- Filter relevant news articles
- Classify into risk taxonomy
- Verify high-impact claims
- Generate risk reports

### Custom Domain Pipelines
- Medical literature mining
- Legal document analysis
- Code security scanning
- Scientific paper screening

---

## 📝 Version History

- **v1.0** (2025-10-02): Initial documentation release
  - User Guide
  - Configuration Guide
  - Custom Stages Guide
  - Quick Reference

---

## 🔗 External Resources

### Technologies Used

- [Hydra](https://hydra.cc/) - Configuration management
- [OmegaConf](https://omegaconf.readthedocs.io/) - Config interpolation
- [Ray Data](https://docs.ray.io/en/latest/data/data.html) - Distributed data processing
- [vLLM](https://docs.vllm.ai/) - LLM inference
- [SLURM](https://slurm.schedmd.com/) - Job scheduling
- [Weights & Biases](https://wandb.ai/) - Experiment tracking

### Related Projects

- **UAIR Research**: Urban AI Risks assessment project
- **Weitz Taxonomy**: Climate adaptation AI risks

---

## 📬 Contact

For questions about UAIR framework:
- Check existing documentation first
- Review code examples in `pipelines/uair/`
- Consult [Quick Reference](QUICK_REFERENCE.md) troubleshooting section

---

**Ready to build your first pipeline?** → [Start with the User Guide](USER_GUIDE.md)

---

*Documentation last updated: 2025-10-02*

