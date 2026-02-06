---
name: training-pipeline
description: Index skill for VBot quadruped RL training. Routes to specialized skills for curriculum learning, hyperparameter optimization, reward/penalty engineering, and campaign management.
---

## Purpose

**Entry point** for RL training tasks. Routes to specialized skills.

## Quick Start — Just Run Training

```powershell
# === SINGLE TRAINING RUN (simplest) ===
uv run scripts/train.py --env vbot_navigation_section001

# === AUTOML / HP SEARCH (recommended for optimization) ===
# This is the main pipeline entry point. It handles HP sampling, subprocess
# training, TensorBoard evaluation, and result archiving automatically.
uv run starter_kit_schedule/scripts/automl.py `
    --mode stage `
    --budget-hours 12 `
    --hp-trials 8

# === MONITOR AUTOML STATE ===
# State is saved to: starter_kit_schedule/progress/automl_state.yaml
Get-Content starter_kit_schedule/progress/automl_state.yaml

# === MANUAL TRAINING WITH RENDERING ===
uv run scripts/train.py --env vbot_navigation_section001 --render

# === EVALUATE A CHECKPOINT ===
uv run scripts/play.py --env vbot_navigation_section001

# === TENSORBOARD ===
uv run tensorboard --logdir runs/vbot_navigation_section001
```

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
| `starter_kit_schedule/scripts/automl.py` | **AutoML entry point** — HP search, reward search, curriculum |
| `starter_kit_schedule/scripts/train_one.py` | Single trial subprocess (called by automl) |
| `starter_kit_schedule/scripts/evaluate.py` | Read TensorBoard logs for metrics |
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
| "Launch automl / HP search" | **This skill** — use Quick Start above |

## Known Issues (Already Fixed)

These issues have been resolved. Do NOT re-investigate or re-fix them:

| Issue | Fix Applied | Location |
|-------|-------------|----------|
| numpy int64/float64 not JSON serializable | Added `_NumpyEncoder` class + `sample_from_space()` returns native Python types | `starter_kit_schedule/scripts/automl.py` |
| Import order: `@rlcfg` fails if env not registered | `train_one.py` imports `vbot` before `motrix_rl` | `starter_kit_schedule/scripts/train_one.py` |
| Zero reward (robot doesn't move) | Full reward function implemented | `starter_kit/navigation1/vbot/vbot_section001_np.py` |
| Dual env registration confusion | Only `vbot_navigation_section001` is registered (old `vbot_navigation_flat` removed) | `cfg.py`, `cfgs.py`, `__init__.py` |

## Managed Directories

| Directory | Purpose |
|-----------|---------|
| `starter_kit_schedule/` | Plans, configs, progress, automl state |
| `starter_kit_schedule/reward_library/` | Archived reward/penalty components & configs |
| `starter_kit_schedule/progress/` | AutoML state, run tracking |
| `starter_kit_log/` | Experiment logs, metrics |
| `runs/` | Training outputs (checkpoints, TensorBoard logs) |
```
