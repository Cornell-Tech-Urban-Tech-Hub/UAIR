# W&B Table Organization and Sampling

## Overview

All results tables in the UAIR pipeline are now organized into a dedicated `inspect_results` panel group in W&B. This provides a clean, organized view of data inspection tables separate from metrics and other logged content.

## Changes Implemented

### 1. Panel Group Organization

All stage results tables are now logged to the `inspect_results` panel group:

- `inspect_results/classify/results` - Classification results
- `inspect_results/taxonomy/results` - Taxonomy classification results  
- `inspect_results/decompose/results` - Decomposition results (CI tuples)
- `inspect_results/topic/results` - Topic modeling results
- `inspect_results/verification/results` - Verification results

### 2. Random Sampling Guarantee

When tables exceed the configured row limit (default: 1000 rows), they are **randomly sampled** rather than taking the first N rows. This ensures:

- Representative samples across the dataset
- No bias toward early-processed articles
- Reproducible samples (uses configured seed: default 777)

### 3. Sampling Metadata

When sampling occurs, additional metadata is logged:

- `{table_key}/rows`: Number of rows in the logged sample
- `{table_key}/total_rows`: Total number of rows in the full dataset
- `{table_key}/sampled`: Boolean flag indicating if sampling was applied
- `{table_key}/sample_seed`: Seed used for random sampling (for reproducibility)

## Usage in W&B UI

### Viewing Results Tables

1. Navigate to your W&B run
2. Go to the **Tables** or **Data** section
3. Look for the **inspect_results** panel group
4. All stage results tables will be organized under this group

### Identifying Sampled Tables

Console output during logging will indicate if sampling occurred:

```
[wandb] ✓ Logged table 'inspect_results/classify/results': 1,000 rows (randomly sampled from 15,234)
```

Or for tables under the limit:

```
[wandb] ✓ Logged table 'inspect_results/topic/results': 823 rows
```

## Configuration

### Table Sample Size

Configure the maximum rows per table in `conf/config.yaml`:

```yaml
wandb:
  table_sample_rows: 1000  # Max rows before sampling
  table_sample_seed: 777    # Seed for reproducible sampling
```

### Disabling Panel Groups (if needed)

To log tables without panel groups, set `panel_group=None` when calling `log_table`:

```python
logger.log_table(df, "my_table", panel_group=None)  # Logs to top level
```

## Implementation Details

### `WandbLogger.log_table()` Method

Updated signature:

```python
def log_table(
    self,
    df: pd.DataFrame,
    key: str,
    prefer_cols: Optional[List[str]] = None,
    max_rows: Optional[int] = None,
    panel_group: Optional[str] = None,
) -> None:
    """Log pandas DataFrame as wandb table with random sampling when needed.
    
    Args:
        df: Pandas DataFrame to log
        key: Table name/key (e.g., "classify/results")
        prefer_cols: Optional list of preferred columns to include
        max_rows: Max rows to sample (default from config)
        panel_group: Optional panel group name (e.g., "inspect_results")
                    Will prefix the key as "panel_group/key"
    """
```

### Sampling Strategy

```python
if total_rows > max_rows:
    df_sample = df.sample(
        n=max_rows,
        random_state=self.wb_config.table_sample_seed
    ).reset_index(drop=True)
    sampled = True
else:
    df_sample = df.reset_index(drop=True)
    sampled = False
```

Key points:
- Uses `pandas.DataFrame.sample()` for random sampling
- Applies configured seed for reproducibility
- Resets index to maintain clean row numbers in W&B

### All Results Tables Use `inspect_results`

Updated in `orchestrator.py`:

```python
# Classification stage
context.logger.log_table(
    out, 
    "classify/results", 
    prefer_cols=prefer_cols,
    panel_group="inspect_results"
)

# Taxonomy stage
context.logger.log_table(
    out, 
    "taxonomy/results", 
    prefer_cols=prefer_cols,
    panel_group="inspect_results"
)

# Similar for decompose, topic, verification stages
```

## Benefits

### 1. Organized UI
- All data inspection tables grouped together
- Cleaner separation from metrics and plots
- Easier to find specific results

### 2. Representative Samples
- Random sampling prevents early-data bias
- Better reflects overall dataset distribution
- Maintains statistical properties of the data

### 3. Reproducibility
- Sampling uses fixed seed by default
- Can reproduce exact same sample
- Seed value logged with table for traceability

### 4. Performance
- Reduces upload time for large datasets
- Faster W&B UI rendering
- Lower storage costs

## Example: Classification Results

### Before (head sampling, no panel group)
```
Tables/
├── classify/results (first 1000 rows)
├── taxonomy/results (first 1000 rows)
└── topic/results (first 1000 rows)
```

### After (random sampling, organized)
```
inspect_results/
├── classify/results (random 1000 from 15,234 rows)
├── taxonomy/results (random 1000 from 8,456 rows)
└── topic/results (823 rows - under limit)
```

## Querying Sampled Tables via W&B API

```python
import wandb

api = wandb.Api()
run = api.run("UAIR/run-id")

# Access table metadata
table_data = run.history(keys=["inspect_results/classify/results/total_rows"])
total_rows = table_data["inspect_results/classify/results/total_rows"].iloc[0]

# Check if sampling was applied
sampled = run.history(keys=["inspect_results/classify/results/sampled"])
if sampled["inspect_results/classify/results/sampled"].iloc[0]:
    print(f"Table was randomly sampled from {total_rows} total rows")
```

## Migration Notes

### Existing Runs
- Old runs will still have tables at the top level (no panel group)
- New runs will use the `inspect_results` panel group
- No data loss or compatibility issues

### Custom Table Logging
If you add custom tables, you can:
- Use `inspect_results` for data inspection tables
- Use other panel groups for different purposes (e.g., "diagnostics", "debug")
- Or log to top level by omitting `panel_group` parameter

## Files Modified

1. **`pipelines/uair/wandb_logger.py`**:
   - Added `panel_group` parameter to `log_table()`
   - Added sampling metadata logging
   - Added console output for sampling information
   - Confirmed random sampling implementation

2. **`pipelines/uair/orchestrator.py`**:
   - Updated all `log_table()` calls to use `panel_group="inspect_results"`
   - Applied to classify, taxonomy, decompose, topic, and verification stages

## References

- W&B Tables Documentation: https://docs.wandb.ai/guides/data-vis/tables
- W&B Panel Customization: https://docs.wandb.ai/guides/app/features/panels
- Pandas Random Sampling: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.sample.html


