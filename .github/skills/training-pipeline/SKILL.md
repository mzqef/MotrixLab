---
name: training-pipeline
description: Index skill for VBot quadruped RL training. Routes to specialized skills for curriculum learning, hyperparameter optimization, reward/penalty engineering, and campaign management.
---

## Purpose

**Entry point** for RL training tasks. Routes to specialized skills.

## ⚠️ Step 0: Review Experiment History (MANDATORY)

> **Before running ANY training, AutoML, or reward experiment**, review existing results to avoid repeating work and to build on prior discoveries.

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

# 3. List training runs and their timestamps
Get-ChildItem runs/vbot_navigation_section001/ -Directory | Sort-Object Name -Descending | Select-Object -First 10

# 4. Review reward library for tried components
Get-ChildItem starter_kit_schedule/reward_library/ -Recurse -Filter "*.yaml" -ErrorAction SilentlyContinue

# 5. Check WAKE_UP.md if progress_watcher was running  
if (Test-Path starter_kit_schedule/progress/WAKE_UP.md) {
    Get-Content starter_kit_schedule/progress/WAKE_UP.md
}
```

**What to look for:**
- Best reward/composite score achieved so far
- Which HP configurations worked best (lr, rollouts, termination, arrival_bonus)
- Known failure modes already diagnosed (see "Diagnosed & Fixed Issues" below)
- Whether anti-laziness mechanisms are active in the current code

**Only after reviewing**, proceed to Quick Start commands below.

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
| Robot learns to stand still ("lazy robot") | `alive_bonus × ~3800 steps ≈ 1900` dwarfs `arrival_bonus = 15` | **Anti-laziness trifecta** (see below) |
| No exploration | `entropy_loss_scale=0.0` in base PPOCfg | Set to `0.005` in VBotSection001PPOConfig |
| Only fixed 12.6m target | `pose_command_range=[0,10.2,0,0,10.2,0]` — no randomization | Randomized: `[-3,3,-π,3,12,π]` |
| Hardcoded reward scales | `approach_scale`, `arrival_bonus` etc not in scales dict | Moved all to `RewardConfig.scales` |
| No gradient at distance | Only exponential decay reward | Added `distance_progress` linear reward (1-d/d_max) |
| Robot reaches but never stops | No episode end after reaching target | Added `_update_truncate()`: 50 steps of reached+stopped → truncate |

### Anti-Laziness Trifecta (Critical Discovery)

At 50M steps, the robot discovered it could maximize reward by **standing still** and collecting alive_bonus every step (~1900 total per episode), completely ignoring the arrival_bonus (15). Three mechanisms fix this:

1. **Conditional alive_bonus:** `alive_bonus = np.where(ever_reached, 0.0, scale)` — No reward for existing after reaching target. Forces the robot to seek the goal, not exploit survival.

2. **Time decay on navigation rewards:** `time_decay = clip(1 - 0.5 * steps/max_steps, 0.5, 1.0)` — Early steps are worth more, creating urgency. Navigation rewards (position_tracking, approach, etc.) multiply by this factor. Penalties are NOT multiplied (termination stays full strength).

3. **Massive arrival_bonus:** Increased from 15 → 50 (now searched in range [20, 100]). Must dominate the total alive_bonus accumulation to incentivize actual goal-reaching.

**Detection signal:** Robot distance goes UP during training (1.0m → 1.6m), episode length near max (3800/4000), reached% drops to ~0.3%. Reward still looks "good" because alive_bonus accumulates.

### Current Reward Configuration (Round 2)

```python
# cfg.py RewardConfig.scales — current values
position_tracking: 1.5       # exp(-d/5.0)
fine_position_tracking: 5.0   # exp(-d/0.3) when d < 1.5m
heading_tracking: 0.8
forward_velocity: 1.5
distance_progress: 2.0       # linear: 1 - d/initial_distance
alive_bonus: 0.5              # CONDITIONAL — only when NOT ever_reached
approach_scale: 8.0
arrival_bonus: 50.0           # Large enough to beat alive_bonus accumulation
stop_scale: 2.0
zero_ang_bonus: 6.0
termination: -100.0
lin_vel_z: -0.3
ang_vel_xy: -0.03
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
| **Lazy robot (reward hacking)** | Conditional alive_bonus + time_decay + arrival_bonus=50 | `vbot_section001_np.py` + `cfg.py` |
| Hardcoded reward scales | Moved `approach_scale`, `arrival_bonus`, `stop_scale`, `zero_ang_bonus` into RewardConfig.scales | `cfg.py` + `vbot_section001_np.py` |
| AutoML no TensorBoard data | `check_point_interval` not passed to rl_overrides. Now auto-calculated to ensure ≥5 data points | `automl.py` |
| Wrong TensorBoard tag for evaluation | Tag is `Reward / Instantaneous reward (mean)` not `Total reward` | `evaluate.py` |
| Fixed training target | `pose_command_range` was [0,10.2,0,0,10.2,0] (no randomization) | `cfg.py` |
| Robot reaches but doesn't stop | Added `_update_truncate()` override — 50 consecutive steps of reached+stopped → truncate | `vbot_section001_np.py` |

## Experiment History Summary

### Round 1 AutoML (15 trials at 5M steps each)
- Best: reward=6.75, reached=6.45%, composite=0.3733
- Best HP: lr=2.42e-04, entropy=1.33e-03, rollouts=32, termination=-200
- **Outcome:** Full 50M training → lazy robot (killed at 30min)

### Round 2 AutoML (anti-laziness, 10 trials at 10M steps each)
- Trial 1: reward=1.90, reached=0.33%, composite=0.316
- Late learning surge detected (reward jumped at step 4819)
- **Status:** Search still in progress — needs more trials and longer training

### Key Findings
1. Learning rate ~2e-4 works well; very low (<5e-5) too slow
2. Rollouts 16-32 all viable; 32 slightly better
3. Termination -100 to -200 range is correct; -10/-25 too lenient
4. arrival_bonus must be >> alive_bonus × max_episode_steps / 2
5. Anti-laziness mechanisms are ESSENTIAL for >10M step training

## Managed Directories

| Directory | Purpose |
|-----------|---------|
| `starter_kit_schedule/` | Plans, configs, progress, automl state |
| `starter_kit_schedule/reward_library/` | Archived reward/penalty components & configs |
| `starter_kit_schedule/progress/` | AutoML state, run tracking |
| `starter_kit_log/` | Experiment logs, metrics |
| `runs/` | Training outputs (checkpoints, TensorBoard logs) |
```
