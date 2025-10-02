# W&B Run Grouping Fix

## Problem

Previously, each SLURM job (monitor + child stages) was creating its own W&B group based on its individual job ID. This meant that stages from a single pipeline run were scattered across different groups, making it difficult to view all related runs together in the W&B UI.

**Before Fix:**
```
Group: slurm-12345 (monitor job)
  ├── orchestrator run

Group: slurm-12346 (classify job)
  ├── classify run

Group: slurm-12347 (taxonomy job)
  ├── taxonomy run
```

**After Fix:**
```
Group: slurm-12345 (parent/monitor job)
  ├── orchestrator run (monitor)
  ├── classify run (child)
  ├── taxonomy run (child)
  ├── decompose run (child)
  ├── topic run (child)
  └── verification run (child)
```

## Solution

The orchestrator now explicitly passes its W&B group ID to all child stages, ensuring they all use the same group.

### Implementation

#### 1. Get Parent Group ID

When `run_experiment()` starts, it extracts the group ID from the orchestrator's `WandbLogger`:

```python
parent_group = logger.wb_config.group if logger.wb_config else None
if parent_group:
    os.environ["WANDB_GROUP"] = parent_group
    print(f"[orchestrator] Setting WANDB_GROUP={parent_group} for child stages")
```

#### 2. Inject into Child Jobs

Before submitting each child job via submitit, the orchestrator injects an explicit `export WANDB_GROUP=<parent_group>` command into the SLURM setup script:

```python
if parent_group:
    # Get current setup commands from launcher config
    current_setup = list(launcher_cfg.get("setup", []))
    
    # Find insertion point (after source/init commands)
    insert_idx = 0
    for i, cmd in enumerate(current_setup):
        if "source" in cmd or "export HYDRA_FULL_ERROR" in cmd:
            insert_idx = i + 1
    
    # Insert WANDB_GROUP export
    wandb_group_export = f"export WANDB_GROUP={parent_group}"
    current_setup.insert(insert_idx, wandb_group_export)
    executor.update_parameters(slurm_setup=current_setup)
```

This ensures that:
- The WANDB_GROUP environment variable is set in the child SLURM job's environment
- The launcher config's conditional export (`if [ -n "$WANDB_GROUP" ]; then export WANDB_GROUP="$WANDB_GROUP"; fi`) propagates it properly

#### 3. Priority Order for Group Resolution

The `_get_group_from_config()` function uses this priority order:

1. **Explicit config** (`cfg.wandb.group`)
2. **WANDB_GROUP env var** ← Child jobs use this (set by parent)
3. **SUBMITIT_JOB_ID** (parent job ID, if available)
4. **SLURM_JOB_ID** (current job ID)

Child jobs now get the parent's group via priority #2, preventing them from falling back to their own SLURM_JOB_ID.

## Console Output

### Monitor (Parent) Job
```
[orchestrator] Setting WANDB_GROUP=slurm-12345 for child stages
{
  "node": "classify",
  "stage": "classify",
  "status": "submitting",
  "launcher": "g2_slurm_pierson"
}
{
  "debug": "injected_wandb_group",
  "group": "slurm-12345",
  "node": "classify"
}
```

### Child Job (e.g., Classify)
```
[wandb] Starting run: UAIR-classify-2025-10-02_14-30-00
[wandb] Run group: slurm-12345
[wandb] ✓ Run started: UAIR-classify-2025-10-02_14-30-00 (ONLINE)
```

## Verification

### In W&B UI

1. Navigate to your W&B project
2. Look at the **Groups** view
3. All stages from a single pipeline run should appear under the same group (parent job ID)

### Via W&B API

```python
import wandb

api = wandb.Api()

# Get all runs in a specific group
runs = api.runs(
    "UAIR",
    filters={"group": "slurm-12345"}
)

print(f"Runs in group slurm-12345:")
for run in runs:
    print(f"  - {run.name} (job_type={run.job_type})")

# Expected output:
# Runs in group slurm-12345:
#   - UAIR-monitor-... (job_type=orchestrator)
#   - UAIR-classify-... (job_type=classify)
#   - UAIR-taxonomy-... (job_type=taxonomy)
#   - UAIR-topic-... (job_type=topic)
#   - UAIR-verification-... (job_type=verification)
```

## Edge Cases

### 1. W&B Disabled
If W&B is disabled (`wandb.enabled=false`), `parent_group` will be `None` and no injection occurs. This is fine - no grouping needed.

### 2. Local Execution (No Launcher)
If stages run locally (no launcher specified), they inherit the environment directly from the parent process, so `WANDB_GROUP` is automatically available.

### 3. Explicit Group in Config
If you explicitly set `wandb.group` in the config, that takes priority over auto-generated groups:

```yaml
wandb:
  group: "my-experiment-run-1"
```

All stages will use "my-experiment-run-1" regardless of job IDs.

### 4. WANDB_GROUP Environment Variable
If you set `WANDB_GROUP` explicitly before running the pipeline:

```bash
export WANDB_GROUP="custom-group"
python -m pipelines.uair.cli ...
```

All stages (parent + children) will use "custom-group".

## Benefits

### 1. Better Organization
- All stages from one pipeline run are grouped together
- Easy to find related runs
- Clear hierarchy in W&B UI

### 2. Easier Analysis
- Compare metrics across stages in the same run
- View complete pipeline execution in one place
- Identify which stages belong to which pipeline execution

### 3. Proper Distributed Execution
- Works correctly with submitit/SLURM
- Handles complex multi-job pipelines
- Respects job dependencies and parallelism

## Files Modified

**`pipelines/uair/orchestrator.py`**:
1. Extract parent group ID from monitor's WandbLogger
2. Set `WANDB_GROUP` in orchestrator's environment
3. Inject `export WANDB_GROUP=<parent_group>` into child job setup scripts

## Testing

### Quick Test

Run a multi-stage pipeline and verify all runs appear in the same group:

```bash
cd /share/pierson/matt/UAIR
source .venv/bin/activate

# Run a pipeline with multiple stages
python -m pipelines.uair.cli \
    runtime.debug=true \
    runtime.sample_n=100 \
    +pipeline=topic_modeling_of_relevant_classifications

# Check console output for:
# [orchestrator] Setting WANDB_GROUP=slurm-XXXXX for child stages
# {"debug": "injected_wandb_group", "group": "slurm-XXXXX", "node": "classify"}
```

Then in W&B:
1. Go to your project
2. Switch to **Groups** view
3. Find the group `slurm-XXXXX`
4. Verify it contains runs for: orchestrator, classify, topic

### Manual Group Test

Force a specific group and verify all stages use it:

```bash
export WANDB_GROUP="test-grouping-fix"
python -m pipelines.uair.cli \
    runtime.debug=true \
    runtime.sample_n=50 \
    +pipeline=topic_modeling_of_relevant_classifications
```

Check W&B for group "test-grouping-fix" containing all stage runs.

## Rollback

If this causes issues, you can disable group propagation by:

1. Setting an explicit empty group in config:
   ```yaml
   wandb:
     group: ""
   ```

2. Or unset WANDB_GROUP before running:
   ```bash
   unset WANDB_GROUP
   python -m pipelines.uair.cli ...
   ```

## Related Documentation

- W&B Grouping: https://docs.wandb.ai/guides/track/log/intro#organize-runs-with-groups
- Submitit Executor: https://github.com/facebookincubator/submitit
- SLURM Job Arrays: https://slurm.schedmd.com/job_array.html


