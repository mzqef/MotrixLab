# Navigation1 Task — Final Presentation

## MotrixArena S1 Quadruped Navigation Competition — Flat Ground

**Task**: VBot quadruped robot navigates from random spawn (8-11m from center) to the center of a circular platform (R=12.5m).  
**Scoring**: 10 robots × 2 points each = 20 max. +1 inner fence, +1 center. Any fall = lose both points.  
**Status**: ✅ **COMPLETED — Expected score: 20/20**

---

## 1. Final Results

| Metric | Value |
|--------|-------|
| **Reach rate** | **100.00%** (4608/4608 episodes, 0 failures) |
| **Fall rate** | **0%** |
| **Avg steps to reach** | **479** / 1000 max |
| **Speed at 8-12m** | **1.65 m/s** |
| **Speed at 5-8m** | **1.95 m/s** |
| **Speed at 3-5m** | **2.32 m/s** |
| **Speed at 1-3m** | **2.55 m/s** |
| **Speed at 0-1m** | **0.58 m/s** (braking) |
| **Expected competition score** | **20/20** |

### Competition Score Analysis

With 100% reach rate and 0% fall rate across 4608 test episodes:
- P(all 10 robots succeed) ≈ **99.5%+**
- P(≥9 robots succeed) ≈ **99.99%**
- Expected score: **20/20**

For tiebreakers: avg ~479 steps × 10ms sim_dt = ~4.79 seconds per robot. Fast traversal provides a tiebreaker advantage if multiple competitors achieve 20/20.

---

## 2. Best Checkpoint

| File | Location |
|------|----------|
| **`stage3_continue_agent1600_reached100_4608.pt`** | `starter_kit_schedule/checkpoints/` |
| Backup: `stage3_frozen_agent8800_reached9998.pt` | `starter_kit_schedule/checkpoints/` |

### Source Run Chain
```
R16 agent_9600 (Stage 1, seed=2026, 66.58% TB reached)
  └→ Stage 2 warm-start → agent_1000 (97.76% reached)
      └→ Stage 3 warm-start → agent_1000 (98% reached)
          └→ Speed-opt rewards + reset optimizer → agent_1000 (99.9%, 1.44 m/s)
              └→ Frozen normalizer → agent_8800 (99.95%, 1.65 m/s)
                  └→ Continuation (LR=3e-5) → agent_1600 (100%, 1.65 m/s) ← FINAL
```

---

## 3. Training Journey Summary (5 days, 11 sessions)

### Day 1 — Foundation (Feb 9, Sessions 1-3)
- Set up environment, first training runs
- Discovered KL-adaptive LR instability → switched to linear LR anneal
- Established evaluation methodology

### Day 2 — Infrastructure (Feb 9-10, Sessions 4-6)
- Built AutoML pipeline for batch HP search
- Ran 15-trial + 10-trial AutoML sweeps
- Round7: 50-step stop_bonus budget cap → 32.9% peak
- **Discovered 11 reward hacking patterns** (documented taxonomy)

### Day 3 AM — Breakthrough (Feb 11, Session 7)
- **Reward Budget Audit**: uncovered 2 critical exploits:
  1. `time_decay` → "die early for fresh high-value steps" incentive
  2. Ungated `fine_position_tracking` → "hover at d=0.52m" earns 97% of reaching reward
- **R11 Fix**: removed time_decay + gated fine_tracking behind `ever_reached`
- **Result**: 35% → 65% (+1.83×) — largest single improvement
- **Multi-seed validation**: 5 seeds, all peaked at 59-67% (σ=2.8%)

### Day 3 Mid — Curriculum (Feb 11, Sessions 8-10)
- Stage 2 (5-8m): warm-start from R16 → 97.76% reached
- Stage 3 (8-11m): warm-start from Stage 2 → 98% reached

### Day 3 PM — Final Push (Feb 11, Session 11)
- **Speed optimization**: doubled `forward_velocity`, halved penalties → +41% speed
- **TensorBoard cycling diagnosis**: proved "collapse" was metrics sampling artifact
- **Frozen normalizer**: monkey-patched SKRL RunningStandardScaler
- **Rigorous testing**: 12,288 episodes → 99.95% reached
- **Continuation training**: → **100.00% (4608/4608)**

---

## 4. Key Technical Innovations

### 4.1 Reward Budget Audit Method
Before training, compute maximum cumulative reward for:
- **Desired behavior**: walk to target, stop, collect bonuses
- **Degenerate behavior**: stand still, hover, sprint-crash-reset

If degenerate ≥ desired, fix before training. Caught 2 critical exploits.

### 4.2 Curriculum Learning (3-Stage)
Transfer knowledge from easy distances to hard ones:
```
Stage 1 (2-5m) → Stage 2 (5-8m) → Stage 3 (8-11m)
```
Each stage: warm-start from previous best, reduce LR, reset optimizer.

### 4.3 Frozen RunningStandardScaler
SKRL's obs normalizer has no built-in freeze. Solution:
```python
agent._state_preprocessor._parallel_variance = lambda *a, **kw: None
agent._value_preprocessor._parallel_variance = lambda *a, **kw: None
```
Zero cost, prevents normalizer drift during warm-start training.

### 4.4 TensorBoard Cycling Diagnosis
All 2048 envs synchronize → `reached_fraction` oscillates 3%-98% based on episode phase. **Not a real collapse.** Lesson: ALWAYS evaluate checkpoints independently.

---

## 5. Reward Engineering — 11 Hacking Patterns Discovered

| # | Pattern | Root Cause | Fix |
|---|---------|-----------|-----|
| 1 | Lazy Robot | alive×max_steps >> arrival | Increase arrival bonus |
| 2 | Standing Dominance | max_steps too long | Shorten episodes |
| 3 | Sprint-Crash | Episode reset farming | Speed cap |
| 4 | Touch-and-Die | No survival after reaching | Always-active alive_bonus |
| 5 | Fly-Through Stop | Stop bonus at any speed | Speed-gate v<0.3 |
| 6 | Deceleration Moat | Penalty zone too wide | Reduce radius |
| 7 | Conservative Hovering | Termination too harsh | Reduce penalty |
| 8 | Negative Walk | Penalties > movement | Increase fwd_velocity |
| 9 | Reach-and-Farm | Stop bonus accumulation | 50-step budget cap |
| 10 | **Die-Early** | **time_decay favors short episodes** | **Remove time_decay** |
| 11 | **Hover-Near-Boundary** | **fine_tracking gives 97% of reaching** | **Gate behind ever_reached** |

---

## 6. Final Configuration

### Reward Scales
```python
forward_velocity = 3.5        # Speed incentive
approach_scale = 50.0         # Distance improvement per step
arrival_bonus = 160.0         # One-time reach bonus
inner_fence_bonus = 30.26     # One-time inner fence bonus
stop_scale = 5.97             # Precision stopping
alive_bonus = 0.08            # Low base survival
termination = -75.0           # Fall penalty
# + stability penalties (halved for dynamic gait)
# + R11 fixes: no time_decay, gated fine_tracking
```

### PPO Hyperparameters
```python
policy_net = [256, 128, 64]   # 3-layer MLP
value_net = [512, 256, 128]   # Wider value network
learning_rate = 3e-5          # Low for fine-tuning
ratio_clip = 0.12             # Tight updates
entropy = 0.008               # Moderate exploration
rollouts = 24, epochs = 6, mini_batches = 32
freeze_preprocessor = True    # Frozen normalizer
```

---

## 7. Files Delivered

| File | Purpose |
|------|---------|
| `starter_kit_schedule/checkpoints/stage3_continue_agent1600_reached100_4608.pt` | **Competition submission checkpoint** |
| `starter_kit_schedule/checkpoints/stage3_frozen_agent8800_reached9998.pt` | Backup checkpoint |
| `starter_kit_docs/navigation1/REPORT_NAV1.md` | Full experiment history (95 sections, 11 sessions) |
| `starter_kit_docs/navigation1/Task_Reference.md` | Task-specific reference (updated) |
| `starter_kit_docs/navigation1/FINAL_PRESENTATION.md` | This document |
| `_eval_stage3.py` | Headless evaluation script with velocity profiling |
| `starter_kit_schedule/scripts/train_one.py` | Training script with frozen normalizer support |

---

## 8. Lessons for Navigation2

Key transferable insights for the obstacle course:

1. **Reward Budget Audit is mandatory** — compute degenerate vs desired before training
2. **Curriculum learning works** — start easy, transfer to hard
3. **Frozen normalizer for all warm-starts** — zero cost safety net
4. **TensorBoard instantaneous metrics lie** — always evaluate checkpoints
5. **Speed optimization via reward rebalancing** — increase speed incentive + reduce penalties proportionally
6. **11 reward hacking patterns** — check each applies differently with terrain
