---
name: training-pipeline
description: Index skill for VBot quadruped RL training. Routes to specialized skills for curriculum learning, hyperparameter optimization, reward/penalty engineering, and campaign management.
---

## Purpose

**Entry point** for RL training tasks. Routes to specialized skills.

## ⚠️ Step 0: Review Experiment History (MANDATORY)

> **Before running ANY training, AutoML, or reward experiment**, review existing results to avoid repeating work and to build on prior discoveries.

### 0a. Read Experiment Reports (ALWAYS DO THIS FIRST)

```powershell
# Check which REPORT files exist
Get-ChildItem REPORT_NAV*.md | Select-Object Name, Length, LastWriteTime

# Read the nav1 report — contains ALL experiment history, config state, and TODO items
Get-Content REPORT_NAV1.md
```

**REPORT_NAV*.md files are the canonical experiment record.** They contain:
- Complete experiment history with per-step metrics tables
- Current VERIFIED configuration state (reward scales, spawn radii, LR scheduler)
- Diagnosed failure modes and their fixes
- Active TODO items and next steps
- Curriculum plan with stage definitions
- Lessons learned across all sessions

> **After completing any experiment or making significant changes**, append results to the relevant REPORT_NAV*.md file. Never overwrite existing content — the history is a chronological record.

### 0b. Check AutoML and Run Directories

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
uv run scripts/train.py --env vbot_navigation_section001 --train-backend torch --max-env-steps 200000 --check-point-interval 50

# === VISUAL DEBUGGING ONLY ===
uv run scripts/train.py --env vbot_navigation_section001 --train-backend torch --render

# === FINAL DEPLOYMENT RUN (after AutoML found best config) ===
uv run scripts/train.py --env vbot_navigation_section001 --train-backend torch

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

### Current Reward Configuration (Phase5 — Session 2 verified)

> **IMPORTANT**: Always verify current values by reading `REPORT_NAV1.md` → Section 17.
> The values below may be outdated if changed after this skill was last updated.

```python
# cfg.py VBotSection001EnvCfg.RewardConfig.scales — Phase5/6 values
# (override in subclass, NOT inherited from base — learned lesson!)
position_tracking: 1.5       # exp(-d/5.0)
fine_position_tracking: 8.0  # sigma=0.5, range<2.5m
heading_tracking: 1.0
forward_velocity: 0.8        # Down from 1.5 — prevents sprint-crash exploit
distance_progress: 1.5       # Down from 2.0
alive_bonus: 0.15            # Down from 0.5 — CONDITIONAL on NOT ever_reached
approach_scale: 5.0           # Up from 4.0
arrival_bonus: 100.0          # Up from 50 — must dominate alive budget
inner_fence_bonus: 40.0       # One-time at d<0.75m
stop_scale: 5.0               # Up from 2.0 — sustained stopping
zero_ang_bonus: 10.0          # Up from 6.0
near_target_speed: -0.5       # Penalize speed when d<0.5m (activation 0.5m, not 2.0m!)
boundary_penalty: -3.0        # Penalize near platform edge
termination: -200.0           # Session 3 value (was -150, tested -250 too harsh)
lin_vel_z: -0.3
ang_vel_xy: -0.03
# Speed cap: forward_velocity clipped to 0.6 m/s in vbot_section001_np.py
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

> **For complete experiment history**, see `REPORT_NAV1.md` at workspace root.
> The summary below may be outdated. Always check the report first.

### Session 1: KL-Adaptive LR Experiments (Exp1-4)
- Exp1: kl=0.016, peaked 67% reached at step 5000, collapsed to 18% by step 15K (KL LR instability)
- Exp2: Warm-start from Exp1, stagnant (poisoned optimizer)
- Exp3: kl=0.008, peaked 32%, LR crushed to 0.000167
- Exp4: kl=0.012, peaked 50%, still declined — **no KL threshold works for this task**

### Session 2: Config Drift Discovery & Fixes (Exp5-6b)
- **Critical finding**: Phase5 reward changes were NEVER persisted in VBotSection001EnvCfg
- Exp5: Linear LR scheduler, peaked 59%, sprint-crash exploit discovered
- Exp6b: Config drift confirmed, killed at 7K steps (0% reached)
- **Fixes applied**: RewardConfig override in subclass, annular spawning, 3 new reward terms, linear LR scheduler

### Session 3: Manual Reward Tuning via train.py (Exp7-12) — LESSON LEARNED
- **Mistake**: Used `train.py` for one-at-a-time reward parameter search instead of `automl.py`.
- 6 experiments run manually, each killed after a few K steps, changing one variable at a time.
- Exp7: near_target_speed at d<2.0m → "deceleration moat" at 1m (19.8% peaked)
- Exp8: near_target_speed at d<0.5m → **52% reached at step 4K** (best ever!) but sprint-crash returned at 12K+
- Exp9: forward_velocity=0.2 → 0% (too weak, robot lazy)
- Exp10: forward_velocity=0.5 → 8.6% (still too weak)
- Exp11: speed cap + termination=-250 → 7% (penalty too harsh)
- Exp12: termination=-200, speed cap=0.6 → interrupted before results

**Key lesson**: This manual iteration wasted time. All 6 experiments should have been a single `automl.py --hp-trials 8` batch search with `REWARD_SEARCH_SPACE` covering forward_velocity, near_target_speed, termination. The AutoML pipeline would have found the best combination automatically with structured comparison.

### Key Findings
1. KL-adaptive LR scheduler is fundamentally unstable for this task — use linear anneal
2. Sprint-crash exploit: forward_velocity≥0.8 causes sprint→crash→reset cycle after ~12K steps
3. "Deceleration moat": near_target_speed at d<2.0m blocks approach. Use d<0.5m
4. Config persistence across sessions is fragile — always verify runtime config
5. Anti-laziness mechanisms are ESSENTIAL for >10M step training
6. **NEVER use train.py for parameter search — ALWAYS use automl.py batch search**

## Managed Directories

| Directory | Purpose |
|-----------|---------|
| `starter_kit_schedule/` | Plans, configs, progress, automl state |
| `starter_kit_schedule/reward_library/` | Archived reward/penalty components & configs |
| `starter_kit_schedule/progress/` | AutoML state, run tracking |
| `starter_kit_log/` | Experiment logs, metrics |
| `runs/` | Training outputs (checkpoints, TensorBoard logs) |
```
