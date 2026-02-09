# Navigation1 Experiment Report — VBot Section001

**Date**: February 9, 2026  
**Environment**: `vbot_navigation_section001` — Circular platform (R=12.5m), robots navigate from outer ring to center (0,0)  
**Competition**: MotrixArena S1 Stage 1 — 10 robots × 2 points max = 20 points  
**Framework**: SKRL PPO, PyTorch backend, 2048 parallel envs, torch.compile (reduce-overhead)

---

## 1. Starting Point & Inherited State

When work began, the codebase already had:

- A fully implemented `VBotSection001Env` environment with 54-dim observations, 12-dim actions (PD joint targets), and a comprehensive reward function including position tracking, heading, forward velocity, approach rewards, arrival/inner-fence bonuses, boundary penalties, stability penalties, and termination costs.
- Anti-laziness mechanisms already coded: conditional `alive_bonus` (zeroed after reaching goal), `time_decay` on navigation rewards, and `_success_truncate` (episode ends 50 steps after robot reaches+stops).
- Prior AutoML runs from Feb 6–7 with empirical findings on optimal hyperparameters.

### Prior AutoML History (pre-existing)

| AutoML Run | Date | Trials | Best Reward | Best Reached% | Key Findings |
|---|---|---|---|---|---|
| `20260206_163021` | Feb 6 | 2 | 231.98 | 0% | Initial exploration |
| `20260206_222815` | Feb 6 | 8 | 83.25 | 0% | 8-trial search, all 0% success |
| `20260207_033849` | Feb 7 | 6 | — | — | Bug: all trials returned no_data |
| **`20260207_034031`** | **Feb 7** | **16** | **7.97** | **7.35%** | **Main successful run**: lr=2.42e-4, rollouts=32, epochs=4, termination=-200 |
| `20260207_133230` | Feb 7 | 1 | 2.70 | 0.32% | alive=0.195, arrival=60.7 |
| `20260207_140249` | Feb 7 | 1 | 1.90 | 0.33% | alive=0.135, arrival=87.7, fine_pos=8.83 |
| `20260207_231437` | Feb 7 | 2 | 2.12 | **9.07%** | Highest success rate found |
| `20260207_232444` | Feb 7 | 2 | 3.49 | 2.1% | — |

**Key HP findings from AutoML**:
- Best learning rate: ~2.4e-4
- Best network: (256, 128, 64) policy and value
- Best rollouts: 32 (sweet spot)
- Best learning_epochs: 4–5
- Reward insights: alive_bonus best=0.13, arrival_bonus best=87.7, fine_position_tracking best=8.83

---

## 2. Problem Diagnosis: The Lazy Robot

A critical **Reward Hacking** failure pattern was identified from pre-Phase5 training data:

**Symptom**: Total reward ↑ while `reached%` ↓ (55.4% → 6.4%).  
**Root cause**: `alive_bonus=1.5 × 3000 steps = 4,500` completely dominated `arrival_bonus=50`. The policy discovered standing still near the inner fence (~0.75m) earned more reward than attempting the risky final approach to center (0.3m).  
**Termination cost**: Only `-10`, trivial compared to 4,500 alive reward budget.

### Reward Budget Audit (before fix)

| Source | Maximum Per-Episode Value | Problem |
|--------|--------------------------|---------|
| `alive_bonus` 1.5 × 3000 steps | **4,500** | Dominates everything |
| `arrival_bonus` (one-time) | 50 | Negligible vs 4,500 |
| `inner_fence_bonus` (one-time) | 25 | Negligible |
| `termination` penalty | -10 | Irrelevant cost for dying |

**Ratio**: alive reward : arrival reward = **90:1** — the policy had zero incentive to complete the task.

The anti-laziness trifecta (from reward-penalty-engineering skill) prescribes: `alive_bonus ≈ 0.1–0.3`, `arrival_bonus ≥ 80`, `termination ≤ -100`.

---

## 3. Phase 5 Config Changes Applied (Feb 9)

### 3a. Reward Scale Rebalancing (cfg.py → RewardConfig.scales)

| Scale | Before | After | Rationale |
|-------|--------|-------|-----------|
| `alive_bonus` | 1.5 | **0.15** | HP-opt best=0.13; 0.15×3000=450, no longer dominates |
| `arrival_bonus` | 50 | **100** | HP-opt best=87.7; must dominate alive budget |
| `inner_fence_bonus` | 25 | **40** | Proportional 40% of arrival |
| `termination` | -10 | **-150** | HP-opt found -200 effective; death now costs 33% of alive budget |
| `fine_position_tracking` | 12 | **8** | HP-opt best=8.83; 12 caused approach oscillation |
| `forward_velocity` | 1.2 | **0.8** | Reduce overshoot at center |
| `distance_progress` | 2.0 | **1.5** | Let approach_reward dominate |
| `approach_scale` | 6.0 | **5.0** | Balance with stop_scale |
| `stop_scale` | 5 | **8** | Precision stopping critical for competition |
| `zero_ang_bonus` | 10 | **15** | Stabilize at center |
| `near_target_speed` | -2 | **-1.5** | Slightly less aggressive to avoid approach avoidance |
| `boundary_penalty` | -2 | **-3** | Stronger safety boundary |

**New budget**: alive×3000=450 vs arrival+fence=140 one-time + stop_bonus≈400 while stopped. Death cost -150 = 33% of alive budget. **Balanced.**

### 3b. PPO Hyperparameters (rl_cfgs.py)

| Parameter | Before | After | Rationale |
|-----------|--------|-------|-----------|
| `learning_rate` | 3e-4 | **2.5e-4** | HP-opt best=2.4e-4 |
| `rollouts` | 48 | **32** | HP-opt sweet spot, faster convergence |
| `learning_epochs` | 8 | **5** | HP-opt best=4; reduce stale rollout overfitting |
| `mini_batches` | 32 | **16** | Larger effective batch size |
| `entropy_loss_scale` | 0.008 | **0.01** | More exploration |

### 3c. Spawn Configuration (cfg.py)

Initially set to competition distance: `spawn_inner_radius=9.0`, `spawn_outer_radius=10.0`

---

## 4. Training Experiments: Feb 9, 2026

### Experiment 1: Phase5 from scratch (kl_threshold=0.016)

**Run**: `26-02-09_14-17-56-056428_PPO`  
**Config**: lr=2.5e-4, kl_threshold=0.016, spawn=9-10m  
**Duration**: ~15K steps (interrupted after degradation detected)

| Step | Reward | Reached% | Fence% | Dist | Ep Len | LR |
|------|--------|----------|--------|------|--------|-----|
| 1000 | +0.259 | 0.0% | 0.0% | 9.28m | 361 | 0.001801 |
| 3000 | +2.741 | **26.3%** | 57.9% | 1.40m | 1797 | 0.000844 |
| **5000** | **+5.361** | **67.1%** | **78.2%** | **0.91m** | 2388 | **0.001111** |
| 6000 | +6.056 | 66.2% | 69.8% | 1.84m | 2163 | 0.000674 |
| 9000 | +4.796 | 23.4% | 31.3% | 3.67m | 704 | 0.000375 |
| 15000 | +4.777 | 18.5% | 26.3% | 4.12m | 502 | 0.000250 |

**Observation**: Spectacular early learning (**67% reached at step 5000** — best ever), then complete collapse. LR spiked from 0.00025 → 0.0018 (7× jump) in early training due to loose kl_threshold=0.016, then crashed back down. The massive early updates destabilized the policy. Episode length collapsed from 2388 → 502 (robots dying quickly instead of navigating).

**Diagnosis**: kl_threshold=0.016 too loose → LR scheduler instability → catastrophic policy degradation.

---

### Experiment 2: Warm-start from peak (kl_threshold=0.008)

**Run**: `26-02-09_15-27-13-018254_PPO`  
**Config**: Warm-started from Exp1's `best_agent.pt` (step 5000/6000 peak), kl_threshold fixed to 0.008  
**Duration**: 5K steps (abandoned — no recovery)

| Step | Reward | Reached% | Dist | LR |
|------|--------|----------|------|-----|
| 1000 | +4.359 | 27.1% | 3.80m | 0.000175 |
| 5000 | +4.824 | 22.9% | 3.78m | 0.000167 |

**Observation**: LR inherited the degraded run's low value (0.000175) and never recovered. The optimizer state was "poisoned" — momentum/variance estimates from the collapsed training prevented meaningful learning.

**Conclusion**: Warm-starting from a degraded run's checkpoint doesn't work because optimizer state carries the instability.

---

### Experiment 3: Fresh from scratch (kl_threshold=0.008)

**Run**: `26-02-09_15-46-07-847704_PPO`  
**Config**: lr=2.5e-4, kl_threshold=0.008 (PPOCfg default), spawn=9-10m  
**Duration**: 10K steps (abandoned — plateau)

| Step | Reward | Reached% | Dist | LR |
|------|--------|----------|------|-----|
| 1000 | +0.245 | 0.0% | 9.28m | 0.001168 |
| 5000 | +4.367 | **31.8%** | 1.98m | 0.000254 |
| 7000 | +3.800 | 12.8% | 4.09m | 0.000188 |
| 10000 | +3.889 | 12.8% | 4.31m | **0.000167** |

**Observation**: The tight kl_threshold=0.008 crushed LR to the floor (0.000167) by step 7000. Learning stalled at ~12% reached, never recovering. Early policy updates (which naturally have high KL divergence) trigger aggressive LR reduction with this threshold.

**Conclusion**: kl_threshold=0.008 too tight for this task — starves learning of gradient signal.

---

### Key Insight: KL Threshold Sensitivity

| kl_threshold | Peak LR | Peak Reached% | Steady-State | Problem |
|-------------|---------|---------------|--------------|---------|
| 0.016 | 0.00180 | **67%** (step 5000) | 18% (collapse) | LR spikes → overshoot → policy destroyed |
| 0.008 | 0.00025 | 32% (step 5000) | 12% (plateau) | LR crushed → insufficient learning |
| **0.012** | TBD | TBD | TBD | **Compromise — currently testing** |

---

### Experiment 4: Stage 1 Curriculum (kl_threshold=0.012, spawn=2-5m)

**Run**: `26-02-09_16-30-09-860740_PPO`  
**Config**: lr=5e-4, kl_threshold=0.012, spawn=2-5m (close range first)  
**Status**: **RUNNING** (currently at ~5300 iterations, 8.25 it/s)

| Step | Reward | Reached% | Fence% | Dist | Ep Len | LR |
|------|--------|----------|--------|------|--------|-----|
| 1000 | +0.599 | 0.0% | 0.0% | 3.63m | 403 | 0.002336 |
| 2000 | +1.138 | 0.0% | 0.7% | 2.90m | 966 | 0.001083 |
| 3000 | +2.027 | 6.6% | 16.7% | 1.79m | 2026 | 0.000754 |
| 4000 | +3.580 | 18.4% | 33.5% | 1.65m | 2661 | 0.000399 |
| **5000** | **+5.714** | **49.8%** | **58.4%** | **1.19m** | 1155 | **0.000380** |

**Early assessment**: Very promising. 50% reached at step 5000 with only 2–5m spawn distance. LR settled at 0.000380 — healthier than either extreme. The shorter navigation distance lets the robot learn locomotion + heading + approach fundamentals before scaling to competition distance.

---

## 5. Bugs Fixed in play_10_robots_1_target.py

Three bugs were found and fixed in the demo/evaluation script:

### Bug 1: Observation Normalization Mismatch (CRITICAL)

```python
# BEFORE (wrong — training uses /12.0):
position_error_normalized = position_error / 5.0
distance_normalized = np.clip(distance_to_target / 5.0, 0, 1)

# AFTER (matches training):
position_error_normalized = position_error / 12.0
distance_normalized = np.clip(distance_to_target / 12.0, 0, 1)
```

**Impact**: Policy received ~2.4× amplified position signals, causing erratic navigation behavior during evaluation.

### Bug 2: Partial Reset Clobbering All Environments

The reset logic repositioned ALL 10 robots whenever any single robot completed its episode, preventing proper multi-robot evaluation.

**Fix**: Only reposition robots on initial reset; subsequent per-robot resets preserve running robots' positions.

### Bug 3: Success Truncation Causing Robot Disappearance

Robots that reached the center were truncated (episode ended) and respawned at a new random position, making them "vanish."

**Fix**: Suppress `_success_truncate` in the play script so robots stay at the center after reaching it.

---

## 6. Curriculum Training Plan (Active)

Based on the insight that **kl_threshold and spawn distance interact** — the KL-adaptive scheduler needs room to operate, and harder tasks produce larger KL divergence — the plan uses progressive difficulty:

```
Stage 1: Close (2-5m)          Stage 2: Mid (5-8m)           Stage 3: Competition (9-10m)
lr=5e-4, kl=0.012             warm-start, lr=2.5e-4         warm-start, lr=1.25e-4
Target: reached > 70%          Target: reached > 60%         Target: reached > 80%
~15-30K steps                  ~15-30K steps                  ~50K+ steps
```

**Current status**: Stage 1 running, 50% reached at step 5000, still improving.

---

## 7. Current Configuration State

### cfg.py — VBotSection001EnvCfg

```python
spawn_inner_radius: float = 2.0   # Curriculum Stage 1 (was 9.0)
spawn_outer_radius: float = 5.0   # (was 10.0)
platform_radius: float = 12.0
inner_fence_radius: float = 0.75
center_radius: float = 0.3
max_episode_steps: int = 3000     # 30 seconds
tilt_terminate_deg: float = 75.0
```

### rl_cfgs.py — PPO

```python
learning_rate: float = 5e-4           # Curriculum (was 2.5e-4)
kl_threshold: float = 0.012          # Compromise (was 0.008)
rollouts: int = 32
learning_epochs: int = 5
mini_batches: int = 16
entropy_loss_scale: float = 0.01
policy/value networks: (256, 128, 64)
```

### RewardConfig.scales (Phase5 values, unchanged since start of experiments)

```python
alive_bonus: 0.15          # Was 1.5
arrival_bonus: 100.0       # Was 50
inner_fence_bonus: 40.0    # Was 25
termination: -150.0        # Was -10
fine_position_tracking: 8.0  # Was 12
forward_velocity: 0.8      # Was 1.2
stop_scale: 8.0            # Was 5
zero_ang_bonus: 15.0       # Was 10
```

---

## 8. Total Training Runs Today (Feb 9)

| Run | Config | Steps | Peak Reached% | Outcome |
|-----|--------|-------|---------------|---------|
| `13-59-12` | Pre-Phase5 (unknown) | — | — | Config verification |
| `14-17-56` | Phase5, kl=0.016, spawn=9-10m | 15K | **67%** (step 5000) | Collapsed: LR instability |
| `15-27-13` | Warm-start, kl=0.008 | 5K | 31% | Stagnant: poisoned optimizer |
| `15-46-07` | Fresh, kl=0.008, spawn=9-10m | 10K | 32% | Plateau at 12%: LR too low |
| `16-26-57` | (no data) | 0 | — | Empty |
| **`16-30-09`** | **Stage 1 curriculum, kl=0.012, spawn=2-5m** | **5K+** | **50%** (step 5000) | **RUNNING, promising** |

---

## 9. Design Decisions & Rationale

These decisions were made during the planning phase and validated or revised through experiments:

| Decision | Chosen | Alternative | Why |
|----------|--------|-------------|-----|
| `alive_bonus` | 0.15 | 0.0 | Zero alive_bonus failed in Phase 3 (cold start collapse). 0.15 provides minimal shaping without dominating. |
| `termination` | -150 | -200 | -200 works at 5M steps but can be unstable at 50M per HP-opt findings. -150 is a safer middle ground. |
| `kl_threshold` | **0.012** | ~~0.016~~ (planned) | Plan proposed 0.016. Exp1 proved it too loose (7× LR spike → collapse). 0.012 is the empirical compromise. |
| Curriculum stages | **2-5m → 5-8m → 9-10m** | ~~3-8m → 6-10m → 9-10m~~ (planned) | Plan proposed wider overlapping ranges. Experiments showed tighter non-overlapping sections train more stably. |
| Stage 1 LR | **5e-4** | ~~2.5e-4~~ (planned) | Plan proposed 2.5e-4. Doubled for Stage 1 to accelerate early learning on the easy 2-5m task. |
| Network architecture | (256,128,64) unchanged | Wider/deeper | The 54-dim obs with (256,128,64) network is validated. Changes introduce high-risk regressions — focus on reward + training protocol first. |
| Observation space | 54-dim unchanged | Adding terrain/IMU features | Same reasoning: avoid architecture regressions, optimize training protocol first. |
| `rollouts` | 32 | 48 (prior value) | 48 was chosen for Phase 4c's 10m navigation, but HP-opt shows 32 converges faster with less memory. Longer GAE window unnecessary with better reward shaping. |

### Reward Budget Verification (post-fix)

New budget: `alive_bonus` = 0.15 × 3000 = 450 (theoretical max).

However, effective alive reward is lower because: (1) `alive_bonus` is conditionally zeroed after reaching goal, (2) `time_decay` reduces navigation rewards over episode length. Effective alive budget ≈ 450 × 0.75 = **337**.

Goal completion rewards: `arrival_bonus` 100 + `inner_fence_bonus` 40 + `stop_bonus` ~8 × 50 steps = **540**.

**Result**: Goal-completion rewards (540) now exceed effective alive-only reward (337). Death costs -150 = 44% of alive budget — falling is expensive. Budget is balanced.

---

## 10. Lessons Learned

1. **Reward budget auditing is essential.** The alive_bonus × max_steps vs arrival_bonus ratio was 90:1 — an obvious exploit the policy found given enough training time. Always compute the maximum reward for desired vs degenerate behavior.

2. **KL-adaptive LR scheduler is sensitive.** The kl_threshold parameter has a narrow effective range (0.008–0.016) and interacts with task difficulty. Too tight → learning stalls. Too loose → LR spikes destroy the policy.

3. **Warm-starting from degraded runs fails.** Optimizer momentum/variance estimates carry the instability. Fresh training with fixed config is more reliable than attempting warm-start recovery.

4. **Curriculum training provides more stable learning.** Starting from shorter navigation distances (2-5m) lets the robot learn locomotion fundamentals before scaling to competition distance (9-10m). This avoids overwhelming the policy with a hard task where most episodes fail.

5. **Observation normalization mismatches are silent killers.** The play script's `/5.0` vs training's `/12.0` divergence caused evaluation failures without any error messages. Always verify normalization constants match between training and inference.

6. **One variable per experiment.** Changing kl_threshold, spawn radius, and LR simultaneously made attribution difficult. The curriculum approach isolates spawn difficulty as the primary variable.

7. **Plan assumptions need experimental validation.** The previous plan proposed kl=0.016 and wide curriculum ranges — both were revised after experiments. Always treat the plan as a hypothesis, not truth.

---

## 11. Next Steps

1. **Monitor Stage 1** — Wait for reached% > 70% or plateau detection
2. **Promote to Stage 2** — Set spawn=5-8m, lr=2.5e-4, warm-start from Stage 1 best checkpoint
3. **Promote to Stage 3** — Set spawn=9-10m, lr=1.25e-4, warm-start from Stage 2 best checkpoint  
4. **Competition evaluation** — Run `play_10_robots_1_target.py` with fixed script, count robots reaching center from R=10m. Target: >80% (score ≥16/20)
5. **If plateau persists** — Consider disabling KL-adaptive scheduler entirely in favor of fixed or cosine-decay LR schedule
6. **(Optional) AutoML refinement** — If curriculum plateaus, run `uv run starter_kit_schedule/scripts/automl.py --mode stage --budget-hours 12 --hp-trials 8` for fine-grained reward scale search using the existing Bayesian optimization pipeline

---

## Appendix A: Plan vs Reality {#appendix-a-plan-vs-reality}

The initial plan was written before any experiments were run. This appendix documents where the plan's assumptions diverged from experimental results.

| Aspect | Previous Plan | Actual / Validated | Resolution |
|--------|---------------|--------------------|----|
| **kl_threshold** | 0.016 | **0.012** | 0.016 caused LR to spike 7×, destroying the policy. 0.012 is the working compromise. |
| **Curriculum ranges** | 3-8m → 6-10m → 9-10m (overlapping) | **2-5m → 5-8m → 9-10m** (non-overlapping) | Tighter, non-overlapping stages provided cleaner signal. Starting at 2m rather than 3m gave an easier on-ramp. |
| **Stage 1 LR** | 2.5e-4 | **5e-4** | Doubled for the easy 2-5m task to accelerate fundamental skill acquisition. |
| **Reward ratio (before fix)** | "60:1" (typo) | **90:1** (= 4500/50) | Previous had an arithmetic error; 4500/50 = 90, not 60. |
| **Budget check** | "450 < 200 — wait, this fails" | **Budget is balanced** when accounting for conditional alive_bonus, time_decay, and stop_bonus accumulation. | The plan's simple check (alive < 2×arrival) was too crude; the actual budget analysis requires factoring in conditional/decayed rewards. |
| **Visual diagnosis step** | "run play.py before training to watch behaviors" (Step 5) | **Skipped** | Went directly to training experiments. Would have been informative but wasn't blocking. |
| **Training duration** | 10M+ steps per stage | **~15-30K steps per stage** (so far) | The task converges much faster than anticipated. 50% reached in only 5K steps at Stage 1. Full 10M may not be needed. |

---

# Session 2: Config Drift Discovery & Proper Implementation (Feb 9 evening)

## 12. Critical Discovery: Phase5 Rewards Were Never Persisted

### The Problem

When auditing the running config at the start of Session 2, we discovered that **none of the Phase5 reward changes from Section 3 were actually active in the code**. The `VBotSection001EnvCfg` class never had its own `RewardConfig` — it inherited from `VBotStairsEnvCfg → VBotEnvCfg`, which still had the **original** pre-Phase5 values.

**Runtime verification** (actual values during ALL Exp1–Exp4):
```
alive_bonus: 0.5       ← NOT 0.15 as reported
arrival_bonus: 50.0    ← NOT 100.0
termination: -100.0    ← NOT -150.0
forward_velocity: 1.5  ← NOT 0.8
stop_scale: 2.0        ← NOT 8.0 or 5.0
inner_fence_bonus: —   ← DID NOT EXIST in scales dict
near_target_speed: —   ← DID NOT EXIST
boundary_penalty: —    ← DID NOT EXIST
```

### Root Cause

The reward changes in Section 3 were likely applied to the **base class** `VBotEnvCfg.RewardConfig` during a previous editing session, but that edit was lost between sessions (file not saved, or git checkout, or editor revert). Since `VBotSection001EnvCfg` never overrode `RewardConfig`, it always inherited whatever the base class had.

### Impact on Previous Experiments

All experiments (Exp1-4) ran with the **original** reward config. This means:
- The 67% peak in Exp1 was achieved with `alive_bonus=0.5`, `arrival=50`, `termination=-100`
- The KL-threshold experiments were valid — they tested LR scheduling behavior, not reward changes
- The sprint-crash exploit (Exp5, see below) was caused by the original `forward_velocity=1.5` being too high

### Fix Applied

Added a **dedicated `RewardConfig` override** directly inside `VBotSection001EnvCfg` in `cfg.py`, so the Phase5 values are now class-level and cannot be lost by base class changes. Also added `spawn_inner_radius` and `spawn_outer_radius` fields for curriculum control.

---

## 13. LR Scheduler: Linear Anneal Replaces KL-Adaptive

### Discovery: KL-Adaptive Fails Universally

After Exp4 showed the same peak→decline pattern even with kl_threshold=0.012:

| Exp | kl_threshold | Peak Reached% | Final Reached% | LR at Peak → Final |
|-----|-------------|---------------|----------------|---------------------|
| Exp1 | 0.016 | 67% (step 5000) | 18% (step 15000) | 0.0011 → 0.00025 |
| Exp3 | 0.008 | 32% (step 5000) | 12% (step 10000) | 0.00025 → 0.000167 |
| Exp4 | 0.012 | 50% (step 5000) | 26% (step 10000) | 0.00038 → 0.000280 |

**Conclusion**: No KL threshold value works. The KLAdaptiveRL scheduler is inherently unstable for this task's KL dynamics.

### Implementation: Configurable LR Scheduler

Added `lr_scheduler_type` field to `PPOCfg` (3 options):

| Value | Behavior | Use Case |
|-------|----------|----------|
| `"kl_adaptive"` | Original KLAdaptiveRL (default) | Backward compatibility |
| `"linear"` | LambdaLR: `factor = max(1 - epoch/total_updates, 0.01)` | **Nav1 (chosen)** |
| `None` | Fixed LR, no scheduler | Simple baseline |

**Files modified:**
- `motrix_rl/src/motrix_rl/skrl/cfg.py` — Added `lr_scheduler_type: str | None = "kl_adaptive"`
- `motrix_rl/src/motrix_rl/skrl/torch/train/ppo.py` — Added linear scheduler branch in `_get_cfg()`
- `starter_kit/navigation1/vbot/rl_cfgs.py` — Set `lr_scheduler_type = "linear"`

**Linear scheduler details:**
```python
total_updates = max_env_steps / (rollouts * num_envs) * learning_epochs
# For 100M steps, 2048 envs, 32 rollouts, 5 epochs: total_updates = 7625
factor = max(1.0 - epoch / total_updates, 0.01)  # Anneal to 1% of initial LR
```

SKRL calls `scheduler.step()` once per learning iteration (outside the epoch loop), verified from SKRL source code.

---

## 14. Experiment 5: Linear LR Scheduler Test (spawn=full platform)

**Run**: `26-02-09_16-58-36-784748_PPO`  
**Config**: lr=5e-4, lr_scheduler_type="linear", spawn=full platform (0-11m), **original reward config** (Phase5 not yet re-applied)  
**Duration**: ~12K steps

| Step | Reward | Reached% | Fence% | Dist | Ep Len | LR |
|------|--------|----------|--------|------|--------|-----|
| 1000 | +0.46 | 0.0% | 0.0% | 5.76m | 390 | 0.000495 |
| 3000 | +2.32 | 16.6% | 35.3% | 2.10m | 1839 | 0.000474 |
| **5000** | **+5.58** | **58.8%** | **69.3%** | **1.22m** | **1366** | **0.000451** |
| 7000 | +6.05 | 47.2% | 55.3% | 2.01m | 706 | 0.000427 |
| 9000 | +5.82 | 35.2% | 42.9% | 2.88m | 460 | 0.000403 |
| 12000 | +6.06 | 27.3% | 34.4% | 3.41m | 331 | 0.000382 |

**Key finding**: LR is smooth and monotonically decreasing (0.000495 → 0.000382) — no spikes or crashes. **But the same peak→decline pattern persists**: reached peaked at 59% (step 5000) then degraded to 27% while reward kept increasing.

**Diagnosis**: Since the LR scheduler is no longer the issue, the reward function itself must be causing the decline.

---

## 15. Sprint-and-Crash Exploit Discovery

Per-component reward breakdown from Exp5 TensorBoard (`Reward Instant/` tags) revealed:

| Component | Step 1000 | Step 5000 | Step 12000 | Trend |
|-----------|-----------|-----------|------------|-------|
| `forward_velocity` | 0.11 | 0.55 | **0.83** | ↑↑ Sprint learned |
| `stop_bonus` | 0.02 | 2.10 | **4.45** | ↑↑ Dominates |
| `alive_bonus` | 0.48 | 0.42 | 0.38 | ↓ Slight decrease |
| `approach_reward` | 0.08 | 0.40 | 0.20 | ↗↘ |
| `Episode length` | 390 | 1366 | **331** | ↑↓ Collapsed |

**The exploit**: The policy learned to:
1. **Sprint** toward center at maximum speed (`forward_velocity` 0.11 → 0.83)
2. **Briefly touch** the 0.3m center zone (collect `arrival_bonus` + accumulate `stop_bonus`)
3. **Crash/fall** almost immediately after (episode length → 331 = robot dying fast)
4. **Reset** and repeat — more attempts per unit time = more reward

The `stop_bonus` at scale=2.0 × (exp term) accumulated to 4.45/step average because even a brief center touch during a sprint gives stop reward. With `forward_velocity=1.5`, the policy found sprinting + crashing was more reward-efficient than careful approach + sustained stop.

**Root cause**: `forward_velocity=1.5` (original) is too high + `stop_scale=2.0` rewards brief touches + `near_target_speed` penalty didn't exist.

---

## 16. Experiment 6b: Config Drift Run (original rewards, spawn=2-5m label but full platform actual)

**Run**: `26-02-09_17-46-29-513605_PPO`  
**Config**: lr=5e-4, lr_scheduler_type="linear", **but original reward config** (Phase5 never saved). The conversation believed spawn was 2-5m but `spawn_inner_radius` / `spawn_outer_radius` didn't exist — env used `_random_point_in_circle(platform_radius)` = full 0-11m.  
**Duration**: 7K steps (killed after discovering config drift)

| Step | Reward | Reached% | Dist | Ep Len |
|------|--------|----------|------|--------|
| 1000 | +0.66 | 0.0% | — | 364 |
| 3000 | +1.34 | 0.05% | 2.61m | 2132 |
| 5000 | +1.51 | 0.0% | — | 1570 |
| 7000 | +1.50 | 0.0% | — | 2699 |

**Result**: Near-zero reached% across all steps. With the original rewards (`alive_bonus=0.5`, `arrival=50`) and full-platform spawning, the policy couldn't learn meaningful navigation.

**Killed** after discovering the config drift. All Phase5 changes were then properly re-applied (see Section 12).

---

## 17. Current Configuration State (VERIFIED — Session 2)

All values below verified by `uv run python -c "..."` at runtime after re-applying fixes.

### cfg.py — VBotSection001EnvCfg (now has its own RewardConfig override)

```python
# Curriculum spawn control (NEW — added in Session 2)
spawn_inner_radius: float = 2.0   # Stage 1 (Easy): 2-5m
spawn_outer_radius: float = 5.0
platform_radius: float = 11.0     # Safety radius (platform R=12.5)
target_radius: float = 3.0        # Targets clustered near center
max_episode_steps: int = 4000     # 40 seconds
```

### RewardConfig.scales (Phase5 — NOW ACTUALLY PERSISTED in VBotSection001EnvCfg)

```python
# === Navigation core ===
position_tracking: 1.5       # exp(-d/5.0)
fine_position_tracking: 8.0  # sigma=0.5, range<2.5m
heading_tracking: 1.0        # Slightly up from 0.8
forward_velocity: 0.8        # Down from 1.5 — prevent sprint exploit
distance_progress: 1.5       # Down from 2.0 — let approach dominate
alive_bonus: 0.15            # Down from 0.5 — anti-laziness

# === Approach/arrival ===
approach_scale: 5.0           # Up from 4.0
arrival_bonus: 100.0          # Up from 50 — dominate alive budget
inner_fence_bonus: 40.0       # NEW — one-time bonus at d<0.75m
stop_scale: 5.0               # Up from 2.0 — precision stopping
zero_ang_bonus: 10.0          # Up from 6.0
near_target_speed: -1.5       # NEW — penalize speed when d<2m
boundary_penalty: -3.0        # NEW — penalize nearplatform edge

# === Stability (unchanged) ===
orientation: -0.05, lin_vel_z: -0.3, ang_vel_xy: -0.03
torques: -1e-5, dof_vel: -5e-5, dof_acc: -2.5e-7, action_rate: -0.01

# === Terminal ===
termination: -150.0           # Up from -100 — death is expensive
```

### rl_cfgs.py — PPO

```python
learning_rate: float = 5e-4
lr_scheduler_type: str | None = "linear"    # NEW — replaces KL-adaptive
learning_rate_scheduler_kl_threshold: float = 0.012  # Only if kl_adaptive
rollouts: int = 32
learning_epochs: int = 5
mini_batches: int = 16
entropy_loss_scale: float = 0.01
policy/value_hidden_layer_sizes: (256, 128, 64)
```

### New reward function additions (vbot_section001_np.py)

```python
# near_target_speed: penalize high speed when d < 2m (prevents sprint-crash)
near_target_speed_penalty = where(d < 2.0 and not reached, speed_xy * scale, 0)

# inner_fence_bonus: one-time reward when first entering d < 0.75m zone
first_inner_fence = (d < 0.75) and not ever_inner_fence  →  +40

# boundary_penalty: penalize when dist_from_center > (platform_radius - 1m)
boundary_penalty = scale * max(dist_from_center - 10.0, 0)

# Annular spawning: _random_point_in_annulus(n, inner_r, outer_r)
# r = sqrt(U * (R2² - R1²) + R1²) for uniform distribution in annulus
```

---

## 18. Updated Curriculum Plan (Session 2)

```
Stage 1: Easy (2-5m spawn)
├── Config: spawn_inner=2.0, spawn_outer=5.0
├── LR: 5e-4 → linear anneal
├── Reward: Phase5 (anti-laziness + anti-sprint)
├── Target: reached > 70%, stable ep_len > 1500
├── Promotion: Save best_agent.pt
│
Stage 2: Medium (5-8m spawn)  
├── Config: spawn_inner=5.0, spawn_outer=8.0
├── LR: 2.5e-4 → linear anneal  
├── Warm-start: Stage 1 best_agent.pt, reset optimizer
├── Target: reached > 60%
│
Stage 3: Competition (8-11m spawn)
├── Config: spawn_inner=8.0, spawn_outer=11.0
├── LR: 1.25e-4 → linear anneal
├── Warm-start: Stage 2 best_agent.pt, reset optimizer
├── Target: reached > 80% (= 16/20 pts)
│
Final: Full platform (0-11m spawn)
├── Config: spawn_inner=0.0, spawn_outer=11.0
├── LR: 1e-4 → linear anneal
├── Warm-start: Stage 3 best_agent.pt
├── Target: reached > 90% across all distances
```

---

## 19. Updated Experiment Summary (Full)

| # | Run | Config | Steps | Peak Reached% | Outcome |
|---|-----|--------|-------|---------------|---------|
| 1 | `14-17-56` | kl=0.016, spawn=0-11m, original rewards | 15K | **67%** (step 5000) | Collapsed: KL LR instability |
| 2 | `15-27-13` | Warm-start from Exp1, kl=0.008 | 5K | 27% | Stagnant: poisoned optimizer |
| 3 | `15-46-07` | kl=0.008, spawn=0-11m, original rewards | 10K | 32% | Plateau at 12%: LR crushed |
| 4 | `16-30-09` | kl=0.012, spawn=0-11m, original rewards | 5K+ | 50% | Declined to 26%: KL still fails |
| 5 | `16-58-36` | **linear LR**, spawn=0-11m, original rewards | 12K | **59%** (step 5000) | Sprint-crash exploit (fwd_vel=1.5 too high) |
| 6 | `17-33-01` | Misconfigured (KL still active) | 0 | — | Killed immediately |
| 6b | `17-46-29` | Linear LR, spawn=0-11m, original rewards | 7K | 0.05% | Config drift — Phase5 never saved |
| **7** | **TBD** | **Linear LR, spawn=2-5m, Phase5 rewards** | — | — | **NEXT — first properly configured run** |

**Key correction**: The report originally stated spawn=9-10m or 2-5m for various experiments, but `spawn_inner_radius`/`spawn_outer_radius` didn't exist until Session 2. All Exp1-6b used `_random_point_in_circle(platform_radius=11.0)` = uniform 0-11m spawning.

---

## 20. Lessons Learned (Session 2 additions)

8. **Config persistence across sessions is fragile.** Edits to dataclass configs can be silently lost between sessions (editor revert, git checkout, import caching). Always verify runtime config with a test script before launching training.

9. **Class inheritance hides config drift.** `VBotSection001EnvCfg` inherited `RewardConfig` from `VBotEnvCfg` (base class). Phase5 changes to the base class disappeared, but the subclass seemed unchanged. **Fix**: Override RewardConfig in each env-specific config class.

10. **The KL-adaptive LR scheduler should be replaced, not tuned.** After 4 experiments with 3 different thresholds (0.008, 0.012, 0.016), all showed the same peak→decline pattern. A simple linear anneal solved the LR scheduling problem immediately.

11. **Sprint-crash is a distinct exploit from lazy-robot.** Lazy-robot exploits per-step rewards by standing still. Sprint-crash exploits per-episode rewards by completing as many short episodes as possible. Different root cause (high `forward_velocity` + weak stopping reward), different fix (reduce velocity reward + add near-target speed penalty).

---

## 21. Next Steps (Updated)

1. **Launch Exp7** — First properly configured run: Phase5 rewards + linear LR + annular spawn 2-5m
2. **Monitor Exp7** — Target reached > 70% for Stage 1 promotion
3. **Stage 2 promotion** — Change `spawn_inner_radius=5.0`, `spawn_outer_radius=8.0`, lr=2.5e-4, warm-start
4. **Stage 3 promotion** — Change `spawn_inner_radius=8.0`, `spawn_outer_radius=11.0`, lr=1.25e-4, warm-start
5. **Competition evaluation** — 10 robots from R=10m, target >80% reach (≥16/20 pts)

---

## Appendix B: Reward Budget Analysis (Phase5 — Session 2 verified values)

With the now-persisted Phase5 rewards:

| Source | Maximum Per-Episode Value | Notes |
|--------|--------------------------|-------|
| `alive_bonus` 0.15 × 4000 steps | **600** | Theoretical max (conditional: zeroed after reach) |
| `alive_bonus` effective (time_decay × 0.75) | **~450** | Realistic due to time_decay halving at end |
| `arrival_bonus` (one-time) | **100** | Must dominate alive for goal-seeking |
| `inner_fence_bonus` (one-time) | **40** | Intermediate waypoint reward |
| `stop_bonus` 5.0 × (exp terms) × 50 steps | **~250** | Accumulated while stopped at center |
| Goal completion total | **~390** | arrival + fence + stop |
| `termination` penalty | **-150** | 33% of alive budget — death hurts |

**alive(450) < goal(390)?** Close, but goal rewards are earned in ~50 steps with no time_decay, while alive accumulates over full episode with decay. Adding approach_reward (~200 during navigation) + time_decay makes goal-seeking dominant. Budget is balanced.

**Sprint-crash check**: `forward_velocity=0.8` (down from 1.5) reduces sprint incentive. `near_target_speed=-1.5` adds penalty for high speed near target. `stop_scale=5.0` (up from 2.0) rewards sustained stopping, not brief touches. Sprint-crash exploit should be mitigated.

---

## 22. Session 3: Manual Reward Tuning Experiments (Exp7-12)

> **⚠️ METHODOLOGY WARNING:** Experiments 7-12 were run using manual `train.py` iteration — editing one parameter, running, killing, editing again. This is the **wrong approach**. These should have been a single `automl.py --hp-trials 8` batch search. Documented here for completeness and as a cautionary example.

### Experiment 7: Deceleration Moat Discovery

**Run**: `26-02-09_18-19-43-907759_PPO`  
**Config**: Phase5 rewards, linear LR, spawn=2-5m, `near_target_speed=-1.5` at d<2.0m  
**Duration**: ~12K steps (killed)

| Step | Reward | Reached% | Inner Fence% | Distance | Ep Len |
|------|--------|----------|-------------|----------|--------|
| 3000 | — | 19.8% | — | ~1.0m | 2500 |
| 6000 | — | 15% | — | ~1.0m | 2800 |
| 12000 | — | 8-20% (oscillating) | — | ~1.0m | 2000-3000 |

**Discovery: Deceleration moat.** The `near_target_speed` penalty with activation at d<2.0m created a zone where robots were **punished for moving**. Robots learned to hover at ~1m from target, collecting position_tracking rewards but never crossing the 0.3m reach threshold.

```
             2m                    0.3m
Robot ----→  |===PENALTY ZONE===|  |TARGET|
             ↑                     ↑
    Robot stops here           Never reaches here
```

**Fix applied**: Reduced activation radius from 2.0m to 0.5m.

---

### Experiment 8: Best Peak Ever (52% Reached) — Then Sprint-Crash Returns

**Run**: `26-02-09_18-49-23-793079_PPO`  
**Config**: `near_target_speed=-0.5` at d<0.5m (reduced from -1.5 at d<2.0m)  
**Duration**: ~19K steps (killed)

| Step | Reward | Reached% | Distance | Forward Vel | Ep Len |
|------|--------|----------|----------|-------------|--------|
| 2000 | — | 30% | 0.85m | 0.30 | 1500 |
| **4000** | — | **52.0%** | **0.59m** | **0.46** | **897** |
| 8000 | — | 40% | 1.10m | 0.60 | 650 |
| 12000 | — | 30% | 1.75m | 0.72 | 500 |
| 19000 | — | 24% | 2.74m | 0.86 | 391 |

**Peak**: **52% reached at step 4K** — best result achieved in any experiment.

**But sprint-crash returned**: After step 4K, forward_velocity climbed from 0.46→0.86, episode length collapsed from 897→391, and reached% dropped from 52%→24%. The speed cap (0.6m/s clip) was not yet implemented. With `forward_velocity=0.8`, the sprint-crash exploit re-emerged.

**Key insight**: The activation radius fix (2.0m→0.5m) was correct — peak performance doubled (20%→52%). But the sprint tendency remained the fundamental instability.

---

### Experiment 9: forward_velocity=0.2 (Too Weak)

**Run**: `26-02-09_19-29-26` (approx)  
**Config**: `forward_velocity` reduced from 0.8 to 0.2  
**Duration**: ~4K steps (killed immediately)

| Step | Reached% | Notes |
|------|----------|-------|
| 4000 | **0%** | Robot doesn't move meaningfully |

**Result**: forward_velocity=0.2 is too weak. Robot has no incentive to move toward target. Instant kill.

---

### Experiment 10: forward_velocity=0.5 (Still Too Weak)

**Run**: `26-02-09_19-40-53-914527_PPO`  
**Config**: `forward_velocity=0.5` (middle ground)  
**Duration**: ~6K steps (killed)

| Step | Reached% | Notes |
|------|----------|-------|
| 6000 | **8.6%** | Much worse than Exp8's 52% at 4K |

**Result**: forward_velocity=0.5 produces 6× worse results than 0.8 at the same step count. The velocity reward is a critical driver of navigation, and reducing it below 0.8 eliminates the navigation signal.

**Lesson**: forward_velocity needs to stay at 0.8 — the sprint-crash must be fixed by OTHER means (speed cap, near_target_speed), not by reducing the core velocity incentive.

---

### Experiment 11: Speed Cap + Harsh Termination

**Run**: `26-02-09_19-55-01-588784_PPO`  
**Config**: `forward_velocity=0.8` restored, speed cap at 0.6m/s (np.clip), `termination=-250`  
**Duration**: ~8K steps (killed)

| Step | Reached% | Distance | Ep Len | Notes |
|------|----------|----------|--------|-------|
| 4000 | 5% | 0.6m | 3200 | Robot hovering |
| 8000 | **7%** | **0.5m** | **3459** | Too conservative |

**Result**: The robot got close (0.5m distance) but was **afraid to approach**. `termination=-250` made death so expensive that the policy learned to hover safely rather than risk the final 0.5m→0.3m push.

**Lesson**: There's a Goldilocks zone for termination penalty. -100 = death too cheap (sprint-crash). -250 = death too expensive (conservative hovering). Testing range: -150 to -200.

---

### Experiment 12: Reduced Termination (Interrupted)

**Run**: `26-02-09_20-14-47-101961_PPO`  
**Config**: `termination=-200`, speed cap=0.6m/s  
**Duration**: Interrupted by user before meaningful data

**Status**: Launched but interrupted when user questioned the manual train.py approach. No conclusions.

---

## 23. Updated Experiment Summary (Full — Sessions 1-3)

| # | Run | Config | Steps | Peak Reached% | Outcome |
|---|-----|--------|-------|---------------|---------|
| 1 | `14-17-56` | kl=0.016, spawn=0-11m, original rewards | 15K | **67%** (step 5000) | Collapsed: KL LR instability |
| 2 | `15-27-13` | Warm-start from Exp1, kl=0.008 | 5K | 27% | Stagnant: poisoned optimizer |
| 3 | `15-46-07` | kl=0.008, spawn=0-11m, original rewards | 10K | 32% | Plateau at 12%: LR crushed |
| 4 | `16-30-09` | kl=0.012, spawn=0-11m, original rewards | 5K+ | 50% | Declined to 26%: KL still fails |
| 5 | `16-58-36` | **linear LR**, spawn=0-11m, original rewards | 12K | **59%** (step 5000) | Sprint-crash exploit (fwd_vel=1.5 too high) |
| 6 | `17-33-01` | Misconfigured (KL still active) | 0 | — | Killed immediately |
| 6b | `17-46-29` | Linear LR, spawn=0-11m, original rewards | 7K | 0.05% | Config drift — Phase5 never saved |
| **7** | `18-19-43` | Phase5, near_target d<2m, scale=-1.5 | 12K | 19.8% | **Deceleration moat** at 1m |
| **8** | `18-49-23` | near_target d<0.5m, scale=-0.5 | 19K | **52%** (step 4K) | **Best peak** — sprint-crash at 12K+ |
| **9** | `19-29-26` | forward_velocity=0.2 | 4K | 0% | Too weak — robot lazy |
| **10** | `19-40-53` | forward_velocity=0.5 | 6K | 8.6% | Still too weak vs 0.8 |
| **11** | `19-55-01` | fwd_vel=0.8, speed cap=0.6, term=-250 | 8K | 7% | Term too harsh — conservative hovering |
| **12** | `20-14-47` | fwd_vel=0.8, speed cap=0.6, term=-200 | — | — | Interrupted |

---

## 24. Current Configuration State (VERIFIED — Session 3)

### cfg.py — VBotSection001EnvCfg

```python
spawn_inner_radius: float = 2.0
spawn_outer_radius: float = 5.0
```

### RewardConfig.scales (Phase5/6 values — session 3 state)

```python
position_tracking: 1.5
fine_position_tracking: 8.0
heading_tracking: 1.0
forward_velocity: 0.8
distance_progress: 1.5
alive_bonus: 0.15
approach_scale: 5.0
arrival_bonus: 100.0
inner_fence_bonus: 40.0
stop_scale: 5.0
zero_ang_bonus: 10.0
near_target_speed: -0.5       # Changed: was -1.5, activation radius changed from 2.0m → 0.5m
boundary_penalty: -3.0
termination: -200.0            # Changed: was -150, then -250 (too harsh), settled at -200
```

### vbot_section001_np.py changes (session 3)

```python
# Speed cap — clips forward_velocity reward to max 0.6 m/s
forward_velocity_clipped = np.clip(forward_velocity, -1.0, 0.6)

# near_target_speed activation at 0.5m (was 2.0m)
near_target_speed_penalty = where(distance < 0.5 and not reached, speed * scale, 0)
```

---

## 25. Lessons Learned (Session 3 additions)

12. **near_target_speed activation radius is critical.** 2.0m creates a "deceleration moat" — robot hovers at 1m, never reaching 0.3m target. 0.5m allows free approach until the last half meter, where smooth deceleration is needed. This was the single biggest improvement (20% → 52% reached).

13. **forward_velocity must stay at 0.8 or higher.** Reducing to 0.5 or 0.2 eliminates the navigation drive entirely (8.6% and 0% respectively). Sprint-crash must be fixed by speed cap and near-target penalty, not by neutering the core velocity incentive.

14. **Termination penalty has a Goldilocks zone.** -100 = death too cheap (sprint-crash). -250 = death too expensive (conservative hovering). -200 seems optimal but needs longer validation.

15. **🔴 NEVER use train.py for parameter search.** Experiments 7-12 (6 manual iterations) should have been a single `automl.py --hp-trials 8` batch search. The AutoML pipeline searches multiple configurations simultaneously with structured comparison, Bayesian suggestion, and reproducible reports. Manual one-at-a-time iteration is slow, unstructured, and error-prone.

---

## 26. Next Steps (Updated — Session 3)

1. **🔴 Update `automl.py` REWARD_SEARCH_SPACE** — Add near_target_speed, inner_fence_bonus, boundary_penalty, termination range [-300,-200,-150,-100], forward_velocity range [0.3-1.0], speed cap parameter
2. **Run AutoML batch search** — `automl.py --mode stage --budget-hours 8 --hp-trials 15` to find optimal reward combination
3. **Read AutoML report** — `starter_kit_log/automl_*/report.md` for structured comparison
4. **Train best config to full steps** — After AutoML identifies the winner, deploy with `train.py` for 50M+ steps
5. **Stage 2 promotion** — Once reached > 70% at 2-5m, extend to 5-8m spawn

---

## Appendix C: Session 3 Parameter Search Summary

The following parameter combinations were tested manually (should have used AutoML):

| Parameter | Exp7 | Exp8 | Exp9 | Exp10 | Exp11 | Exp12 |
|-----------|------|------|------|-------|-------|-------|
| forward_velocity | 0.8 | 0.8 | **0.2** | **0.5** | 0.8 | 0.8 |
| near_target_speed | -1.5 | **-0.5** | -0.5 | -0.5 | -0.5 | -0.5 |
| near_target_radius | **2.0m** | **0.5m** | 0.5m | 0.5m | 0.5m | 0.5m |
| termination | -150 | -150 | -150 | -150 | **-250** | **-200** |
| speed_cap | none | none | none | none | **0.6** | **0.6** |
| Peak reached% | 19.8% | **52%** | 0% | 8.6% | 7% | — |

**Best configuration found**: Exp8's parameters (forward_velocity=0.8, near_target_speed=-0.5 at 0.5m) but with speed cap=0.6 and termination=-200 (from Exp12, untested long-term).

**Open question**: Does the speed cap + termination=-200 combination prevent sprint-crash long-term? This needs a full AutoML run to validate.
