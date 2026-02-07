---
name: training-pipeline
description: Index skill for VBot quadruped RL training. Routes to specialized skills for curriculum learning, hyperparameter optimization, reward/penalty engineering, and campaign management.
---

## Purpose

**Entry point** for RL training tasks. Routes to specialized skills.

## Quick Start — Just Run Training

```powershell
# === FAST PROBE RUN (benchmark hardware + validate changes) ===
# 200K steps: ~16s. Good for smoke testing reward changes.
uv run scripts/train.py --env vbot_navigation_section001 --train-backend torch --max-env-steps 200000 --check-point-interval 50

# === 2M PROBE (see actual learning trends) ===
# ~4 min. Enough to confirm reward gradient is alive.
uv run scripts/train.py --env vbot_navigation_section001 --train-backend torch --max-env-steps 2000000 --check-point-interval 100

# === SINGLE TRAINING RUN (full) ===
uv run scripts/train.py --env vbot_navigation_section001 --train-backend torch

# === AUTOML / HP + REWARD SEARCH (recommended for optimization) ===
uv run starter_kit_schedule/scripts/automl.py `
    --mode stage `
    --budget-hours 8 `
    --hp-trials 15

# === MONITOR AUTOML STATE ===
Get-Content starter_kit_schedule/progress/automl_state.yaml

# === PROGRESS WATCHER (generates WAKE_UP.md for agent context) ===
uv run starter_kit_schedule/scripts/progress_watcher.py --snapshot
uv run starter_kit_schedule/scripts/progress_watcher.py --watch --interval 120

# === MANUAL TRAINING WITH RENDERING ===
uv run scripts/train.py --env vbot_navigation_section001 --train-backend torch --render

# === EVALUATE A CHECKPOINT ===
uv run scripts/play.py --env vbot_navigation_section001

# === TENSORBOARD ===
uv run tensorboard --logdir runs/vbot_navigation_section001
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

## Critical Reward Function Knowledge

These issues were root-caused and fixed. **Do NOT revert**:

| Problem | Root Cause | Fix |
|---------|-----------|-----|
| Robot never reaches target | `exp(-12.6/0.5)≈0` — dead gradient at far distance | Widened sigma to 5.0: `exp(-d/5.0)` |
| Robot learns "don't move" | `termination=-200` dominates per-step reward (~2-5) | Reduced to `-50` |
| No exploration | `entropy_loss_scale=0.0` in base PPOCfg | Set to `0.005` in VBotSection001PPOConfig |
| Only fixed 12.6m target | `pose_command_range=[0,10.2,0,0,10.2,0]` — no randomization | Randomized: `[-3,3,-π,3,12,π]` |
| Hardcoded reward scales | `approach_scale`, `arrival_bonus` etc not in scales dict | Moved all to `RewardConfig.scales` |
| No gradient at distance | Only exponential decay reward | Added `distance_progress` linear reward (1-d/d_max) |
| Robot falls and stops learning | Only penalties, no incentive to stay alive | Added `alive_bonus=0.5` per step |

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
| "Launch automl / HP search" | **This skill** — use Quick Start above |

## Known Issues (Already Fixed)

These issues have been resolved. Do NOT re-investigate or re-fix them:

| Issue | Fix Applied | Location |
|-------|-------------|----------|
| numpy int64/float64 not JSON serializable | Added `_NumpyEncoder` class + `sample_from_space()` returns native Python types | `automl.py` |
| Import order: `@rlcfg` fails if env not registered | `train_one.py` imports `vbot` before `motrix_rl` | `train_one.py` |
| Dead reward gradient at distance | Changed sigma from 0.5 to 5.0, added `distance_progress` linear reward | `vbot_section001_np.py` |
| Zero entropy prevents exploration | Set `entropy_loss_scale=0.005` | `motrix_rl/cfgs.py` |
| Termination penalty too harsh | Reduced from -200 to -50, dominates per-step reward | `cfg.py` |
| Hardcoded reward scales | Moved `approach_scale`, `arrival_bonus`, `stop_scale`, `zero_ang_bonus` into RewardConfig.scales | `cfg.py` + `vbot_section001_np.py` |
| AutoML no TensorBoard data | `check_point_interval` not passed to rl_overrides. Now auto-calculated to ensure ≥5 data points | `automl.py` |
| Wrong TensorBoard tag for evaluation | Tag is `Reward / Instantaneous reward (mean)` not `Total reward` | `evaluate.py` |
| Fixed training target | `pose_command_range` was [0,10.2,0,0,10.2,0] (no randomization) | `cfg.py` |

## Managed Directories

| Directory | Purpose |
|-----------|---------|
| `starter_kit_schedule/` | Plans, configs, progress, automl state |
| `starter_kit_schedule/reward_library/` | Archived reward/penalty components & configs |
| `starter_kit_schedule/progress/` | AutoML state, run tracking |
| `starter_kit_log/` | Experiment logs, metrics |
| `runs/` | Training outputs (checkpoints, TensorBoard logs) |
```
