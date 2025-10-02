# Building Custom Stages - Practical Guide

This guide provides instructions for creating custom processing stages for the UAIR pipeline framework, from simple transformations to complex LLM-powered analysis.

---

## Table of Contents

1. [Stage Implementation Basics](#stage-implementation-basics)
2. [Simple Stage Template](#simple-stage-template)
3. [LLM-Powered Stages](#llm-powered-stages)
4. [Advanced Patterns](#advanced-patterns)
5. [Registering Your Stage](#registering-your-stage)
6. [Testing Strategies](#testing-strategies)

---

## Stage Implementation Basics

### Anatomy of a Stage

Every custom stage consists of three components:

1. **Stage Function**: Core processing logic (`run_*_stage`)
2. **Stage Runner Class**: Wrapper that interfaces with orchestrator
3. **Registration**: Adding to the global stage registry

### Minimal Stage Checklist

- [ ] Accepts DataFrame or Ray Dataset as input
- [ ] Accepts Hydra config object
- [ ] Returns DataFrame or Ray Dataset
- [ ] Handles missing/malformed data gracefully
- [ ] Respects `runtime.debug` and `runtime.sample_n` flags
- [ ] Generates expected output columns

---

## Simple Stage Template

### Example: Sentiment Analysis Stage

This example demonstrates a custom stage that adds sentiment scores to articles.

#### Step 1: Create the Stage Function

Create `dagspaces/uair/stages/sentiment.py`:

```python
"""Sentiment analysis stage for UAIR pipeline."""
from typing import Any, Dict
import pandas as pd
from omegaconf import DictConfig

try:
    import ray
    _RAY_OK = True
except Exception:
    _RAY_OK = False


def run_sentiment_stage(df, cfg: DictConfig):
    """Run sentiment analysis on article text.
    
    Args:
        df: pandas DataFrame or Ray Dataset with 'article_text' column
        cfg: Hydra configuration object
        
    Returns:
        DataFrame/Dataset with added columns:
        - sentiment_score: float [-1, 1], negative to positive
        - sentiment_label: str, one of 'negative', 'neutral', 'positive'
        - sentiment_confidence: float [0, 1]
    """
    # Check if input is Ray Dataset (streaming mode)
    is_ray_ds = hasattr(df, "map_batches") and hasattr(df, "count") and _RAY_OK
    
    if not is_ray_ds:
        # Pandas path: simple in-memory processing
        if df is None or len(df) == 0:
            # Return empty DataFrame with expected schema
            return pd.DataFrame(columns=[
                "article_text", 
                "sentiment_score", 
                "sentiment_label",
                "sentiment_confidence"
            ])
        
        return _process_sentiment_pandas(df, cfg)
    else:
        # Ray Data path: distributed processing
        return _process_sentiment_ray(df, cfg)


def _process_sentiment_pandas(df: pd.DataFrame, cfg: DictConfig) -> pd.DataFrame:
    """Process sentiment using pandas (in-memory)."""
    import nltk
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    
    # Initialize sentiment analyzer
    try:
        nltk.data.find('vader_lexicon')
    except LookupError:
        nltk.download('vader_lexicon', quiet=True)
    
    sia = SentimentIntensityAnalyzer()
    
    # Get configuration parameters
    try:
        threshold_positive = float(getattr(cfg.sentiment, "threshold_positive", 0.05))
        threshold_negative = float(getattr(cfg.sentiment, "threshold_negative", -0.05))
    except Exception:
        threshold_positive = 0.05
        threshold_negative = -0.05
    
    def analyze_sentiment(text: str) -> Dict[str, Any]:
        """Analyze sentiment of a single text."""
        if not text or not isinstance(text, str):
            return {
                "sentiment_score": 0.0,
                "sentiment_label": "neutral",
                "sentiment_confidence": 0.0
            }
        
        scores = sia.polarity_scores(text)
        compound = scores['compound']
        
        # Determine label based on thresholds
        if compound >= threshold_positive:
            label = "positive"
        elif compound <= threshold_negative:
            label = "negative"
        else:
            label = "neutral"
        
        # Confidence is the absolute value of compound score
        confidence = abs(compound)
        
        return {
            "sentiment_score": compound,
            "sentiment_label": label,
            "sentiment_confidence": confidence
        }
    
    # Apply sentiment analysis
    out = df.copy()
    sentiment_results = out['article_text'].apply(analyze_sentiment)
    
    # Expand dict results into columns
    out['sentiment_score'] = sentiment_results.apply(lambda x: x['sentiment_score'])
    out['sentiment_label'] = sentiment_results.apply(lambda x: x['sentiment_label'])
    out['sentiment_confidence'] = sentiment_results.apply(lambda x: x['sentiment_confidence'])
    
    return out


def _process_sentiment_ray(ds, cfg: DictConfig):
    """Process sentiment using Ray Data (distributed)."""
    import nltk
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    
    # Get configuration parameters
    try:
        threshold_positive = float(getattr(cfg.sentiment, "threshold_positive", 0.05))
        threshold_negative = float(getattr(cfg.sentiment, "threshold_negative", -0.05))
    except Exception:
        threshold_positive = 0.05
        threshold_negative = -0.05
    
    def process_batch(batch: pd.DataFrame) -> pd.DataFrame:
        """Process a batch of articles (runs on Ray workers)."""
        # Initialize analyzer in worker
        try:
            nltk.data.find('vader_lexicon')
        except LookupError:
            nltk.download('vader_lexicon', quiet=True)
        
        sia = SentimentIntensityAnalyzer()
        
        def analyze_text(text: str) -> Dict[str, Any]:
            if not text or not isinstance(text, str):
                return {
                    "sentiment_score": 0.0,
                    "sentiment_label": "neutral",
                    "sentiment_confidence": 0.0
                }
            
            scores = sia.polarity_scores(text)
            compound = scores['compound']
            
            if compound >= threshold_positive:
                label = "positive"
            elif compound <= threshold_negative:
                label = "negative"
            else:
                label = "neutral"
            
            return {
                "sentiment_score": compound,
                "sentiment_label": label,
                "sentiment_confidence": abs(compound)
            }
        
        # Process batch
        results = batch['article_text'].apply(analyze_text)
        batch['sentiment_score'] = results.apply(lambda x: x['sentiment_score'])
        batch['sentiment_label'] = results.apply(lambda x: x['sentiment_label'])
        batch['sentiment_confidence'] = results.apply(lambda x: x['sentiment_confidence'])
        
        return batch
    
    # Apply transformation in parallel
    return ds.map_batches(process_batch, batch_format="pandas")
```

#### Step 2: Create the Stage Runner

Add to `dagspaces/uair/orchestrator.py`:

```python
from .stages.sentiment import run_sentiment_stage

class SentimentRunner(StageRunner):
    stage_name = "sentiment"
    
    def run(self, context: StageExecutionContext) -> StageResult:
        # Get input dataset path
        dataset_path = context.inputs.get("dataset")
        if not dataset_path:
            raise ValueError(f"Node '{context.node.key}' requires 'dataset' input")
        
        # Update config with input path
        cfg = context.cfg
        OmegaConf.update(cfg, "data.parquet_path", dataset_path, merge=True)
        
        # Prepare input (pandas or ray dataset)
        df, ds, use_streaming = prepare_stage_input(cfg, dataset_path, self.stage_name)
        in_obj = ds if use_streaming and ds is not None else df
        
        # Run stage function
        out = run_sentiment_stage(in_obj, cfg)
        
        # Convert Ray Dataset to pandas if needed
        if hasattr(out, "to_pandas"):
            out = out.to_pandas()
        
        # Save outputs to disk
        if isinstance(out, pd.DataFrame):
            for output_name, output_path in context.output_paths.items():
                out.to_parquet(output_path, index=False)
        
        # Log results table to wandb (optional but recommended)
        if isinstance(out, pd.DataFrame) and context.logger:
            try:
                prefer_cols = ["article_id", "sentiment_label", "sentiment_score", "article_text"]
                context.logger.log_table(
                    out, 
                    "sentiment/results", 
                    prefer_cols=prefer_cols,
                    panel_group="inspect_results"
                )
            except Exception as e:
                print(f"Warning: Failed to log sentiment results: {e}", flush=True)
        
        # Prepare metadata
        metadata: Dict[str, Any] = {
            "rows": len(out) if isinstance(out, pd.DataFrame) else None,
            "streaming": bool(use_streaming),
        }
        
        # Collect outputs
        outputs = _collect_outputs(
            context,
            {name: spec.optional for name, spec in context.node.outputs.items()},
        )
        
        return StageResult(outputs=outputs, metadata=metadata)
```

#### Step 3: Register the Stage

Add to the `_STAGE_REGISTRY` in `dagspaces/uair/orchestrator.py`:

```python
_STAGE_REGISTRY: Dict[str, StageRunner] = {
    "classify": ClassificationRunner(),
    "decompose": DecomposeRunner(),
    "taxonomy": TaxonomyRunner(),
    "topic": TopicRunner(),
    "verification": VerificationRunner(),
    "sentiment": SentimentRunner(),  # ← Add your custom stage
}
```

#### Step 4: Create Configuration

Create `conf/sentiment/default.yaml`:

```yaml
# Sentiment analysis configuration
threshold_positive: 0.05  # Scores >= this are positive
threshold_negative: -0.05  # Scores <= this are negative
```

#### Step 5: Use in a Pipeline

Create `conf/pipeline/with_sentiment.yaml`:

```yaml
# @package _global_
defaults:
  - override /hydra/launcher: null

runtime:
  debug: true
  sample_n: 100
  output_root: ./outputs/sentiment_test

pipeline:
  output_root: ${runtime.output_root}
  allow_partial: false
  
  sources:
    articles:
      path: ${data.parquet_path}
      type: parquet
  
  graph:
    nodes:
      sentiment:
        stage: sentiment
        depends_on: []
        inputs:
          dataset: articles
        outputs:
          results: outputs/sentiment/results.parquet
        overrides:
          sentiment.threshold_positive: 0.1
          sentiment.threshold_negative: -0.1
```

#### Step 6: Run It

```bash
python -m dagspaces.uair.cli \
  pipeline=with_sentiment \
  data.parquet_path=/path/to/articles.parquet
```

---

## LLM-Powered Stages

### Using vLLM for Inference

For stages that require LLM inference (classification, extraction, generation), UAIR provides Ray Data integration with vLLM.

### Template: LLM Classification Stage

This template demonstrates a stage that classifies articles into risk categories.

```python
"""Risk classification stage using vLLM."""
from typing import Any, Dict
import os
import logging
import pandas as pd
from omegaconf import OmegaConf

try:
    import ray
    from ray.data.llm import build_llm_processor, vLLMEngineProcessorConfig
    _RAY_OK = True
except Exception:
    _RAY_OK = False


def run_risk_classification_stage(df, cfg: DictConfig):
    """Classify articles into risk categories using LLM.
    
    Adds columns:
    - risk_category: str, the predicted risk category
    - risk_confidence: float, model confidence
    - llm_output: str, raw model output
    """
    is_ray_ds = hasattr(df, "map_batches") and hasattr(df, "count") and _RAY_OK
    
    if not is_ray_ds:
        if df is None or len(df) == 0:
            return pd.DataFrame(columns=["article_text", "risk_category", "risk_confidence"])
        out = df.copy()
    
    # Load risk categories from config
    try:
        risk_categories = list(getattr(cfg, "risk_categories", [
            "Data Privacy",
            "Algorithmic Bias",
            "Surveillance",
            "Misinformation",
            "Job Displacement",
            "None"
        ]))
    except Exception:
        risk_categories = ["Data Privacy", "Algorithmic Bias", "Surveillance", "None"]
    
    # Build system and user prompts
    system_prompt = """You are an expert AI risk analyst. Classify the following news article into one of the risk categories. Return only the category name, nothing else."""
    
    categories_str = "\n".join([f"- {cat}" for cat in risk_categories])
    
    def format_user_prompt(article_text: str) -> str:
        return f"""Categories:
{categories_str}

Article:
{article_text}

Category:"""
    
    # Configure vLLM engine
    ek = dict(getattr(cfg.model, "engine_kwargs", {}))
    ek.setdefault("max_model_len", 4096)
    ek.setdefault("max_num_seqs", 16)
    ek.setdefault("gpu_memory_utilization", 0.85)
    
    # Auto-detect tensor parallelism from GPUs
    if "tensor_parallel_size" not in ek:
        try:
            num_gpus = _detect_num_gpus()
            ek["tensor_parallel_size"] = num_gpus
        except Exception:
            ek.setdefault("tensor_parallel_size", 1)
    
    # Apply best-practice defaults
    ek.setdefault("enable_prefix_caching", True)
    ek.setdefault("use_v2_block_manager", True)
    ek.setdefault("tokenizer_mode", "auto")
    ek.setdefault("trust_remote_code", True)
    ek.setdefault("dtype", "auto")
    
    engine_config = vLLMEngineProcessorConfig(
        model_source=str(getattr(cfg.model, "model_source")),
        runtime_env={
            "env_vars": {
                "VLLM_LOGGING_LEVEL": "WARNING",
            }
        },
        engine_kwargs=ek,
        concurrency=int(getattr(cfg.model, "concurrency", 1)),
        batch_size=int(getattr(cfg.model, "batch_size", 16)),
    )
    
    # Sampling parameters with guided decoding
    sampling_params = {
        "temperature": 0.1,
        "max_tokens": 32,
        "guided_decoding": {
            "choice": risk_categories  # Constrain output to valid categories
        }
    }
    
    # Preprocessing: Format prompts
    def preprocess(row: Dict[str, Any]) -> Dict[str, Any]:
        article_text = str(row.get("article_text", ""))
        # Trim text if too long (simplified - use tokenizer in production)
        if len(article_text) > 10000:
            article_text = article_text[:10000]
        
        return {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": format_user_prompt(article_text)}
            ],
            "sampling_params": sampling_params,
            **row  # Pass through original columns
        }
    
    # Postprocessing: Extract category and confidence
    def postprocess(row: Dict[str, Any]) -> Dict[str, Any]:
        generated = str(row.get("generated_text", "")).strip()
        
        # Check if output matches a valid category
        risk_category = "None"
        for cat in risk_categories:
            if cat.lower() in generated.lower():
                risk_category = cat
                break
        
        # Confidence from token probabilities (if available)
        # Simplified - extract from usage metadata in production
        risk_confidence = 0.9 if risk_category != "None" else 0.5
        
        return {
            **row,
            "risk_category": risk_category,
            "risk_confidence": risk_confidence,
            "llm_output": generated,
        }
    
    # Build LLM processor
    processor = build_llm_processor(
        engine_config,
        preprocess=preprocess,
        postprocess=postprocess
    )
    
    # Apply to dataset
    if is_ray_ds:
        return processor(df)
    else:
        # Convert pandas to Ray Dataset, process, convert back
        ds = ray.data.from_pandas(out)
        out_ds = processor(ds).materialize()
        return out_ds.to_pandas()


def _detect_num_gpus() -> int:
    """Detect number of allocated GPUs."""
    try:
        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if cuda_visible and cuda_visible.strip():
            return len([x for x in cuda_visible.split(",") if x.strip()])
    except Exception:
        pass
    
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.device_count()
    except Exception:
        pass
    
    return 1
```

### Key LLM Stage Patterns

#### 1. GPU Memory Management

```python
# Conservative settings for 2-GPU setup
ek = {
    "max_model_len": 2048,
    "max_num_seqs": 2,
    "gpu_memory_utilization": 0.45,
    "tensor_parallel_size": 2,
}

# Aggressive settings for 4-GPU setup
ek = {
    "max_model_len": 16384,
    "max_num_seqs": 8,
    "gpu_memory_utilization": 0.9,
    "tensor_parallel_size": 4,
}
```

#### 2. Text Trimming for Context Window

```python
def trim_text_for_model(text: str, max_tokens: int) -> str:
    """Trim text to fit model context window."""
    from transformers import AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_source, 
        trust_remote_code=True
    )
    
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) <= max_tokens:
        return text
    
    trimmed_tokens = tokens[:max_tokens]
    return tokenizer.decode(trimmed_tokens, skip_special_tokens=True)
```

#### 3. Guided Decoding for Structured Outputs

```python
# Choice-based (classification)
sampling_params = {
    "guided_decoding": {
        "choice": ["YES", "NO"]
    }
}

# JSON schema (extraction)
from pydantic import BaseModel

class RiskExtraction(BaseModel):
    risk_type: str
    severity: int
    affected_population: str

sampling_params = {
    "guided_decoding": {
        "json": RiskExtraction.model_json_schema()
    }
}
```

#### 4. Batch Size Optimization

```python
# GPU-aware batch sizes
GPU_BATCH_SETTINGS = {
    "rtx_a6000": {"batch_size": 4, "max_num_seqs": 4},
    "rtx_a5000": {"batch_size": 2, "max_num_seqs": 2},
    "a100": {"batch_size": 8, "max_num_seqs": 8},
}

gpu_type = detect_gpu_type()
settings = GPU_BATCH_SETTINGS.get(gpu_type, {"batch_size": 4})
```

---

## Advanced Patterns

### Pattern 1: Multi-Output Stage

```python
def run_multitask_stage(df, cfg):
    """Stage that produces multiple outputs."""
    # Process data
    results = process_data(df, cfg)
    
    # Split into multiple outputs
    high_confidence = results[results['confidence'] > 0.8]
    low_confidence = results[results['confidence'] <= 0.8]
    
    # Save both (orchestrator handles this via output_paths)
    return {
        'high_conf': high_confidence,
        'low_conf': low_confidence,
        'all': results
    }
```

### Pattern 2: Conditional Processing

```python
def run_conditional_stage(df, cfg):
    """Only process relevant articles."""
    # Skip processing if no relevant articles
    if 'is_relevant' in df.columns:
        df = df[df['is_relevant'] == True]
    
    if len(df) == 0:
        print("No relevant articles to process, returning empty")
        return df
    
    # Continue with processing
    return process_relevant_articles(df, cfg)
```

### Pattern 3: Caching Expensive Computations

```python
import hashlib
import pickle
from pathlib import Path

def run_cached_stage(df, cfg):
    """Cache expensive embeddings."""
    cache_dir = Path(cfg.runtime.output_root) / "cache"
    cache_dir.mkdir(exist_ok=True)
    
    # Compute cache key from input data
    cache_key = hashlib.md5(
        str(df['article_text'].values).encode()
    ).hexdigest()
    cache_file = cache_dir / f"embeddings_{cache_key}.pkl"
    
    if cache_file.exists():
        print(f"Loading cached embeddings from {cache_file}")
        with open(cache_file, 'rb') as f:
            embeddings = pickle.load(f)
    else:
        print("Computing embeddings...")
        embeddings = compute_embeddings(df['article_text'].values)
        with open(cache_file, 'wb') as f:
            pickle.dump(embeddings, f)
    
    df['embedding'] = embeddings.tolist()
    return df
```

---

## Registering Your Stage

### Complete Registration Checklist

1. **Create stage module**: `dagspaces/uair/stages/mystage.py`
2. **Implement `run_*_stage` function**
3. **Create `MyStageRunner` class** in `orchestrator.py`
4. **Add to `_STAGE_REGISTRY`**
5. **Create default config**: `conf/mystage/default.yaml`
6. **Add stage import** to `dagspaces/uair/stages/__init__.py`

### Example: Complete Registration

```python
# dagspaces/uair/stages/__init__.py
from .classify import run_classification_stage
from .taxonomy import run_taxonomy_stage
from .mystage import run_my_custom_stage  # Add this

# dagspaces/uair/orchestrator.py
from .stages.mystage import run_my_custom_stage

class MyCustomStageRunner(StageRunner):
    stage_name = "mystage"
    
    def run(self, context: StageExecutionContext) -> StageResult:
        # Implementation...
        pass

_STAGE_REGISTRY: Dict[str, StageRunner] = {
    # ... existing stages
    "mystage": MyCustomStageRunner(),
}
```

---

## Testing Strategies

### Unit Testing a Stage

```python
# tests/test_sentiment_stage.py
import pytest
import pandas as pd
from omegaconf import OmegaConf
from dagspaces.uair.stages.sentiment import run_sentiment_stage

def test_sentiment_basic():
    """Test basic sentiment analysis."""
    # Create test data
    test_df = pd.DataFrame({
        'article_text': [
            'This is wonderful news!',
            'Terrible disaster strikes city',
            'The weather is okay today'
        ]
    })
    
    # Create minimal config
    cfg = OmegaConf.create({
        'sentiment': {
            'threshold_positive': 0.05,
            'threshold_negative': -0.05
        }
    })
    
    # Run stage
    result = run_sentiment_stage(test_df, cfg)
    
    # Assertions
    assert 'sentiment_label' in result.columns
    assert result.iloc[0]['sentiment_label'] == 'positive'
    assert result.iloc[1]['sentiment_label'] == 'negative'
    assert result.iloc[2]['sentiment_label'] == 'neutral'

def test_sentiment_empty_input():
    """Test handling of empty input."""
    empty_df = pd.DataFrame(columns=['article_text'])
    cfg = OmegaConf.create({})
    
    result = run_sentiment_stage(empty_df, cfg)
    
    assert len(result) == 0
    assert 'sentiment_label' in result.columns
```

### Integration Testing with Pipeline

```bash
# Test with small sample
python -m dagspaces.uair.cli \
  pipeline=test_mystage \
  runtime.debug=true \
  runtime.sample_n=10 \
  data.parquet_path=tests/fixtures/sample_articles.parquet
```

### Debugging Tips

1. **Enable debug logging**:
```yaml
runtime:
  debug: true
  sample_n: 5  # Process only 5 articles
```

2. **Check intermediate outputs**:
```python
def run_my_stage(df, cfg):
    print(f"Input shape: {df.shape}")
    print(f"Input columns: {df.columns.tolist()}")
    
    result = process(df)
    
    print(f"Output shape: {result.shape}")
    print(f"Output columns: {result.columns.tolist()}")
    print(f"Sample output:\n{result.head()}")
    
    return result
```

3. **Use fake LLM mode for testing**:
```yaml
runtime:
  fake_llm: true  # Skip actual LLM inference
```

---

## Summary

This guide covered:

- Creating simple transformation stages  
- Implementing LLM-powered stages with vLLM  
- Handling both pandas and Ray Data inputs  
- Registering stages in the orchestrator  
- Configuring and overriding stage parameters  
- Testing stages in isolation and in dagspaces  

For additional information:
- See [Configuration Guide](CONFIGURATION_GUIDE.md) for advanced config patterns
- See [Complete Examples](EXAMPLES.md) for full pipeline walkthroughs
- See [SLURM Guide](SLURM_GUIDE.md) for distributed execution

---

*Last updated: 2025-10-02*

