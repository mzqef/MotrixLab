---
name: training-pipeline
description: Index skill for VBot quadruped RL training. Routes to specialized skills for curriculum learning, hyperparameter optimization, reward/penalty engineering, and campaign management.
---

## Purpose

**Entry point** for RL training tasks. Routes to specialized skills.

### 0. Check AutoML and Run Directories

```powershell
# 1. List all AutoML runs and their outcomes
Get-ChildItem starter_kit_log/automl_* -Directory | ForEach-Object {
    Write-Host "`n=== $($_.Name) ===" -ForegroundColor Cyan
    $state = Join-Path $_.FullName "state.yaml"
    if (Test-Path $state) { Get-Content $state | Select-Object -First 30 }
}

# 2. Check current automl progress state
if (Test-Path starter_kit_schedule/progress/automl_state.yaml) {
    Get-Content starter_kit_schedule/progress/automl_state.yaml
}

# 3. Read task-specific reference (reward scales, search spaces, terrain, etc.)
Get-Content starter_kit_docs/<task-name>/Task_Reference.md

# 4. List training runs and their timestamps
Get-ChildItem runs/<env-name>/ -Directory | Sort-Object Name -Descending | Select-Object -First 10

# 4. Review reward library for tried components
Get-ChildItem starter_kit_schedule/reward_library/ -Recurse -Filter "*.yaml" -ErrorAction SilentlyContinue

# 5. Check WAKE_UP.md if progress_watcher was running  
if (Test-Path starter_kit_schedule/progress/WAKE_UP.md) {
    Get-Content starter_kit_schedule/progress/WAKE_UP.md
}
```

**What to look for:**
- Best reward/composite score achieved so far
- Which HP configurations worked best
- Known failure modes already diagnosed
- Whether anti-laziness mechanisms are active in the current code

**Only after reviewing**, proceed to Quick Start commands below.

## Quick Start — Just Run Training

> **🔴 AutoML-First Policy (MANDATORY):** See `.github/copilot-instructions.md` for the full policy.
> **NEVER** use `train.py` for parameter exploration — use `automl.py` batch search.
> `train.py` is ONLY for: smoke tests (<500K steps), `--render` visual debug, or final deployment runs with known-good config.

```powershell
# === PRIMARY: AUTOML PIPELINE (USE THIS for all parameter exploration) ===
uv run starter_kit_schedule/scripts/automl.py `
    --mode stage `
    --budget-hours 8 `
    --hp-trials 15

# === MONITOR AUTOML STATE ===
Get-Content starter_kit_schedule/progress/automl_state.yaml

# === READ AUTOML RESULTS ===
Get-Content starter_kit_log/automl_*/report.md

# === SMOKE TEST ONLY (<500K steps, verify code compiles) ===
uv run scripts/train.py --env <env-name> --train-backend torch --max-env-steps 200000 --check-point-interval 50

# === VISUAL DEBUGGING ONLY ===
uv run scripts/train.py --env <env-name> --train-backend torch --render

# === FINAL DEPLOYMENT RUN (after AutoML found best config) ===
uv run scripts/train.py --env <env-name> --train-backend torch

# === EVALUATE A CHECKPOINT ===
uv run scripts/play.py --env <env-name>

# === TENSORBOARD ===
uv run tensorboard --logdir runs/<env-name>
```

## Hardware Performance (measured)

| Metric | Value |
|--------|-------|
| Backend | PyTorch (JAX NOT available) |
| GPU | NVIDIA (torch_gpu=True) |
| Training speed | ~7,500-12,500 steps/sec |
| 200K steps | ~16s |
| 2M steps | ~4 min |
| 5M steps (HP trial) | ~7-8 min |
| 50M steps (full run) | ~70 min |
| 100M steps | ~2.2 hours |

## Task-Specific Reference

> **Environment-specific details** (reward scales, search spaces, terrain descriptions, known exploits, empirical findings) are in:
> `starter_kit_docs/{task-name}/Task_Reference.md`
>
> **Experiment history** is in:
> `starter_kit_docs/{task-name}/REPORT_*.md`
>
> These contain critical knowledge about reward hacking exploits (e.g., lazy robot), anti-laziness mechanisms, dead gradient fixes, sprint-crash exploits, and LR scheduler findings.

## Skill Routing

| Task | Skill |
|------|-------|
| Multi-stage curriculum | → `curriculum-learning` |
| Hyperparameter search | → `hyperparameter-optimization` |
| Reward/penalty tuning | → `reward-penalty-engineering` |
| Training execution | → `training-campaign` |
| Competition strategy | → `quadruped-competition-tutor` |
| Robot model configuration | → `mjcf-xml-reasoning` |
| Visual debugging | → `subagent-copilot-cli` |

## Key Pipeline Scripts

| Script | Purpose |
|--------|---------|
| `scripts/train.py` | Single training run |
| `scripts/play.py` | Evaluate / play a checkpoint |
| `scripts/view.py` | View environment (no training) |
| `scripts/capture_vlm.py` | VLM visual analysis (frame capture + gpt-4.1) |
| `starter_kit_schedule/scripts/automl.py` | **AutoML entry point** — HP search, reward search, curriculum |
| `starter_kit_schedule/scripts/train_one.py` | Single trial subprocess (called by automl) |
| `starter_kit_schedule/scripts/evaluate.py` | Read TensorBoard logs for metrics |
| `starter_kit_schedule/scripts/monitor_training.py` | **Training monitor & TB analyzer** — live dashboard, deep analysis, run comparison |
| `starter_kit_schedule/scripts/eval_checkpoint.py` | **Checkpoint evaluator & ranker** — headless multi-trial eval, rank by metric |
| `starter_kit_schedule/scripts/smoke_test.py` | **Smoke test & reward budget audit** — env step test, budget ratios |
| `starter_kit_schedule/scripts/progress_watcher.py` | Progress monitoring + WAKE_UP.md generation |
| `starter_kit_schedule/scripts/analyze.py` | Compare experiment results |
| `starter_kit_schedule/scripts/status.py` | Check training status |

## When to Read Each Skill

| Question | Read |
|----------|------|
| "How do I progress from flat to obstacles?" | `curriculum-learning` |
| "What learning rate should I use?" | `hyperparameter-optimization` |
| "Why is my robot bouncing?" | `reward-penalty-engineering` → diagnose phase |
| "How do I resume interrupted training?" | `training-campaign` |
| "What reward ideas have we already tried?" | `starter_kit_schedule/reward_library/` |
| "How do I systematically test a reward idea?" | `reward-penalty-engineering` |
| "How do I analyze training screenshots?" | `subagent-copilot-cli` |
| "Where are the robot's feet?" | `mjcf-xml-reasoning` |
| "What are the important competition rules?" | `quadruped-competition-tutor` |
| "What are the concrete reward scales/search spaces?" | `starter_kit_docs/{task}/Task_Reference.md` |
| "What experiments have been tried?" | `starter_kit_docs/{task}/REPORT_*.md` |
| "Launch automl / HP search" | **This skill** — use Quick Start above |

## Known Issues (Already Fixed)

These are **pipeline-level** issues that have been resolved. Do NOT re-investigate:

| Issue | Fix Applied | Location |
|-------|-------------|----------|
| numpy int64/float64 not JSON serializable | Added `_NumpyEncoder` class + `sample_from_space()` returns native Python types | `automl.py` |
| Import order: `@rlcfg` fails if env not registered | `train_one.py` imports `vbot` before `motrix_rl` | `train_one.py` |
| AutoML no TensorBoard data | `check_point_interval` not passed to rl_overrides. Now auto-calculated to ensure ≥5 data points | `automl.py` |
| Wrong TensorBoard tag for evaluation | Tag is `Reward / Instantaneous reward (mean)` not `Total reward` | `evaluate.py` |

> **Task-specific known issues** (reward exploits, dead gradients, LR scheduler findings) are documented in `starter_kit_docs/{task}/Task_Reference.md`.

## Managed Directories

| Directory | Purpose |
|-----------|---------|
| `starter_kit_schedule/` | Plans, configs, progress, automl state |
| `starter_kit_schedule/reward_library/` | Archived reward/penalty components & configs |
| `starter_kit_schedule/progress/` | AutoML state, run tracking |
| `starter_kit_log/` | Experiment logs, metrics |
| `runs/` | Training outputs (checkpoints, TensorBoard logs) |
```
