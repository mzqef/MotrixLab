# Navigation2 Experiment Report — VBot Obstacle Course (Stage 2)

**Date**: February 2026  
**Environments**: `vbot_navigation_section011` (slopes), `vbot_navigation_section012` (stairs+bridge), `vbot_navigation_section013` (balls+ramp), `vbot_navigation_long_course` (full 30m)  
**Competition**: MotrixArena S1 Stage 2 — 105 points max (Section 1: 20pts, Section 2: 60pts, Section 3: 25pts)  
**Framework**: SKRL PPO, PyTorch backend, 2048 parallel envs, torch.compile (reduce-overhead)

---

## 1. Starting Point & Inherited State

### Task Overview

Navigation2 is fundamentally different from Navigation1:

| Aspect | Navigation1 | Navigation2 |
|--------|------------|------------|
| **Terrain** | Flat circular platform (R=12.5m) | 30m linear course: slopes → stairs → bridge → balls → ramp |
| **Navigation** | Radial (any direction to center) | Linear (Y-axis forward, ~34m total) |
| **Elevation** | z=0 everywhere | z=0 → 1.294 → 2.794 → 1.294 → 1.494 |
| **Obstacles** | None | Spheres (R=0.75), cones, gold balls, 0.75m wall |
| **Scoring** | 20 pts (10 dogs × 2) | 105 pts (3 sections, checkpoints, smileys, red packets) |
| **Episode length** | 1000 steps (10s) | 4000-9000 steps (40-90s) |
| **Sections** | 1 environment | 5 environments (3 sections + stairs + long_course) |

### Codebase State

When work begins, the codebase has:

- **Section environments**: `VBotSection011Env`, `VBotSection012Env`, `VBotSection013Env` each with 54-dim observations, 12-dim actions
- **Long course environment**: `VBotLongCourseEnv` with 7-waypoint navigation system (WAYPOINTS list + auto-switching)
- **Reward configs**: Each section has its own `RewardConfig.scales` in `cfg.py`
- **RL configs**: Each section has its own PPO hyperparameters in `rl_cfgs.py`
- **No prior training runs** for navigation2 (clean slate)

### Environment Configs Summary

| Environment | max_steps | Spawn | Target | Distance | Reward Notes |
|-------------|-----------|-------|--------|----------|--------------|
| section011 | 4000 (40s) | (0, -2.4, 0.5) ±0.5m | (0, 7.8, 1.294) | ~12.6m | alive=1.0, arrival=50, term=-50 |
| section012 | 6000 (60s) | (0, 9.5, 1.8) ±0.3m | (0, 24.0, 1.294) | ~14.5m | alive=0.3, arrival=80, term=-200 |
| section013 | 5000 (50s) | (0, 26.0, 1.8) ±0.5m | (0, 32.3, 1.494) | ~6.3m | alive=0.3, arrival=60, term=-200 |
| long_course | 9000 (90s) | (0, -2.4, 0.5) ±0.5m | 7 waypoints | ~34m | waypoint=30, arrival=100, term=-100 |

---

## 2. Terrain Analysis

### Section 01 — Slopes + High Platform (20 pts)

```
Y: -3.5    0    4.5   7.8
    |---flat---|--ramp--|--high platform--|
    z=0        z=0.41   z=1.294
    
    Challenge: 15° upslope → step up to 1.294m platform
    Key obstacle: Ramp gradient + platform edge transition
```

**Predicted difficulty**: Medium. The 15° slope is manageable for a quadruped, but the transition from ramp to platform top requires foot clearance and balance.

### Section 02 — Stairs + Bridge + Spheres (60 pts)

```
Y: 8.8   12.4  14.2  15~20  21.4  23.2  24.3
    |--entry--|--stairs up--|--bridge/spheres--|--stairs down--|--exit--|
    z=1.294   z=1.37→2.79    z≈2.86              z=2.79→1.37    z=1.294

    Left route: 10-step stairs (steep ΔZ=0.15) → arch bridge → 10-step stairs down
    Right route: 10-step stairs (gentle ΔZ=0.10) → 5 spheres + 8 cones → stairs down
```

**Predicted difficulty**: Very Hard. Stairs require precise foot placement and knee lift. Bridge is narrow (~2.64m) with railings. Sphere obstacles (R=0.75m) block right path.

### Section 03 — Gold Balls + Steep Ramp (25 pts)

```
Y: 24.3  27.6  29.3  31.2  32.3  34.3
    |--entry--|--wall+ramp--|--platform--|--gold balls--|--final--|--wall--|
    z=1.294   z=1.301→?      z=1.294     z=0.844(balls) z=1.494

    Challenge: 0.75m high step → 21.8° steep ramp → 3 gold balls (R=0.75, spaced 3m)
```

**Predicted difficulty**: Hard. The 0.75m step is significant (VBot is ~0.35m tall). 21.8° steep ramp harder than Section 01's 15°. Gold balls block the path with only 2.5m gaps.

---

## 3. Reward Budget Pre-Audit

### Section 011 (Current Config)

```
STANDING STILL for 4000 steps (alive=1.0):
  alive = 1.0 × 4000 = 4,000
  position_tracking = exp(-12.6/5) × 2.0 ≈ 0.16/step × 4000 = 640
  Total standing ≈ 4,640

COMPLETING TASK:
  arrival_bonus = 50 (one-time)
  approach + forward ≈ 200
  Total completing ≈ 250 (+ standing reward while walking)

⚠️ STANDING WINS BY ~4,390!
```

**Assessment**: `alive_bonus=1.0` with `max_episode_steps=4000` creates a massive lazy-robot incentive. Same issue as Navigation1 pre-Phase5, but worse (alive budget 4000 vs arrival 50 = 80:1 ratio).

### Section 012 (Current Config)

```
STANDING STILL for 6000 steps (alive=0.3):
  alive = 0.3 × 6000 = 1,800
  
COMPLETING TASK:
  arrival_bonus = 80 (one-time)

Ratio: 1800/80 = 22.5:1 — still heavily favoring laziness.
```

### Section 013 (Current Config)

```
STANDING STILL for 5000 steps (alive=0.3):
  alive = 0.3 × 5000 = 1,500
  
COMPLETING TASK:
  arrival_bonus = 60 (one-time)

Ratio: 1500/60 = 25:1 — lazy robot strongly favored.
```

### Long Course (Current Config)

```
STANDING STILL for 9000 steps (alive=0.5):
  alive = 0.5 × 9000 = 4,500
  
COMPLETING ALL WAYPOINTS:
  waypoint_bonus = 30 × 7 = 210
  arrival_bonus = 100
  Total = 310

Ratio: 4500/310 = 14.5:1 — lazy robot still wins easily.
```

**Conclusion**: ALL environments have broken reward budgets. The `alive_bonus × max_steps >> arrival + waypoint` pattern will produce lazy robots. This must be fixed before training begins.

---

## 4. Key Differences from Navigation1 Reward Engineering

Navigation2 introduces challenges not present in Navigation1:

| Challenge | Navigation1 | Navigation2 | Impact on Rewards |
|-----------|------------|------------|-------------------|
| **Elevation changes** | None | 15° slope, 21.8° slope, stairs | Need height progress reward, z-aware approach |
| **Non-monotonic path** | Straight-line to center | Up → down → up (z oscillates) | Cannot use simple distance-to-target; need waypoints |
| **Obstacles** | None | Spheres, cones, gold balls | Need collision avoidance or obstacle-specific rewards |
| **Multiple terrain types** | Flat only | Flat, slope, stairs, bridge, rough | May need terrain-adaptive stability penalties |
| **Long episodes** | 1000 steps | 4000-9000 steps | Passive rewards dominate even more at long horizons |
| **Waypoint navigation** | Single target | 7 sequential waypoints | Waypoint bonus structure + distance-to-next-waypoint |
| **Route choice** | N/A | Left (harder stairs + bridge) vs Right (easier + spheres) | Policy must discover optimal route; reward shouldn't bias |

### Terrain-Specific Reward Components to Consider

| Component | Section | Rationale |
|-----------|---------|-----------|
| **Height progress** | 011, 013 | Reward z-axis progress on slopes; distance alone misses vertical gain |
| **Knee lift bonus** | 012 | Higher knee clearance needed for stair steps |
| **Stability on slopes** | 011, 013 | Tighter orientation penalties on inclines |
| **Waypoint completion** | long_course | Progressive bonuses for reaching each waypoint |
| **Checkpoint distance** | All | Distance-to-checkpoint as continuous signal (not just binary) |

---

## 5. Training Experiments

*(No experiments run yet. This section will be populated as training begins.)*

---

## 6. Current Configuration State

### cfg.py — Per-Section Reward Scales

#### Section 011 (slopes + high platform)
```python
position_tracking: 2.0
fine_position_tracking: 2.0
heading_tracking: 1.0
forward_velocity: 1.5
distance_progress: 2.0
alive_bonus: 1.0          # ⚠️ Too high for 4000 steps
approach_scale: 8.0
arrival_bonus: 50.0        # ⚠️ Too low vs alive budget
stop_scale: 2.0
zero_ang_bonus: 6.0
orientation: -0.05
lin_vel_z: -0.5
ang_vel_xy: -0.05
torques: -1e-5
dof_vel: -5e-5
dof_acc: -2.5e-7
action_rate: -0.01
termination: -50.0         # ⚠️ Death too cheap
```

#### Section 012 (stairs + bridge + spheres)
```python
position_tracking: 1.5
fine_position_tracking: 5.0
heading_tracking: 0.8
forward_velocity: 1.5
distance_progress: 2.0
alive_bonus: 0.3
approach_scale: 8.0
arrival_bonus: 80.0
stop_scale: 1.5
zero_ang_bonus: 6.0
orientation: -0.05
lin_vel_z: -0.3
ang_vel_xy: -0.03
torques: -1e-5
dof_vel: -5e-5
dof_acc: -2.5e-7
action_rate: -0.01
termination: -200.0
```

#### Section 013 (gold balls + steep ramp)
```python
position_tracking: 1.5
fine_position_tracking: 5.0
heading_tracking: 0.8
forward_velocity: 1.5
distance_progress: 2.0
alive_bonus: 0.3
approach_scale: 8.0
arrival_bonus: 60.0
stop_scale: 1.5
zero_ang_bonus: 6.0
orientation: -0.05
lin_vel_z: -0.3
ang_vel_xy: -0.03
torques: -1e-5
dof_vel: -5e-5
dof_acc: -2.5e-7
action_rate: -0.01
termination: -200.0
```

#### Long Course (all sections)
```python
position_tracking: 1.5
fine_position_tracking: 5.0
heading_tracking: 0.8
forward_velocity: 1.5
distance_progress: 2.0
alive_bonus: 0.5           # ⚠️ 0.5 × 9000 = 4500 >> waypoints
approach_scale: 8.0
waypoint_bonus: 30.0        # 30 × 7 = 210 total
arrival_bonus: 100.0
stop_scale: 2.0
zero_ang_bonus: 6.0
orientation: -0.05
lin_vel_z: -0.3
ang_vel_xy: -0.03
torques: -1e-5
dof_vel: -5e-5
dof_acc: -2.5e-7
action_rate: -0.01
termination: -100.0
```

### rl_cfgs.py — PPO Hyperparameters

| Parameter | Section 011 | Section 012 | Section 013 | Long Course |
|-----------|-------------|-------------|-------------|-------------|
| learning_rate | 3e-4 | 2e-4 | 2.5e-4 | 2e-4 |
| rollouts | 24 | 32 | 28 | 48 |
| learning_epochs | 8 | 8 | 8 | 8 |
| mini_batches | 32 | 32 | 32 | 32 |
| entropy_loss_scale | 0.005 | 0.008 | 0.006 | 0.01 |
| max_env_steps | 100M | 200M | 150M | 300M |
| discount_factor | 0.99 | 0.99 | 0.99 | 0.995 |
| network | (256,128,64) | (256,128,64) | (256,128,64) | (256,128,64) |

---

## 7. Curriculum Training Plan

Based on Navigation1 lessons and Navigation2 terrain analysis:

```
Stage 2A: Section 011 (slopes + high platform)
├── Environment: vbot_navigation_section011
├── Goal: Learn slope climbing + platform transitions
├── Steps: 30-50M
├── LR: 3e-4, linear anneal
├── Target: reached > 60% at full distance
│
Stage 2B: Section 012 (stairs + bridge)
├── Environment: vbot_navigation_section012
├── Goal: Learn stair climbing/descending + bridge traversal
├── Steps: 40-80M (hardest section, 60 pts)
├── LR: 2e-4, warm-start from Stage 2A, reset optimizer
├── Target: forward progress > 10m (past stairs)
│
Stage 2C: Section 013 (balls + steep ramp)
├── Environment: vbot_navigation_section013
├── Goal: Learn steep ramp + ball avoidance
├── Steps: 30-50M
├── LR: 2.5e-4, warm-start from Stage 2B, reset optimizer
├── Target: reached > 50%
│
Final: Long Course (all sections)
├── Environment: vbot_navigation_long_course
├── Goal: End-to-end 30m navigation with waypoint switching
├── Steps: 50-100M
├── LR: 2e-4, warm-start from Stage 2C, reset optimizer
├── Target: reach ≥5/7 waypoints consistently
```

**Key curriculum design decisions**:
- Start with Section 011 (easiest terrain, builds locomotion + slope skills)
- Section 012 requires the most training time (complex terrain, highest point value)
- Warm-start transfers locomotion skills between sections
- Long course training last — needs all section skills combined

---

## 8. Pre-Training TODO

Based on Navigation1 lessons learned:

1. **Fix reward budgets** — All sections have alive_bonus too high relative to arrival. Apply anti-laziness trifecta from Navigation1.
2. **Consider max_episode_steps reduction** — Navigation1's key fix was 4000→1000. For Navigation2, balance between allowing enough time for complex terrain and preventing passive reward accumulation.
3. **Add linear LR scheduler** — Navigation1 proved KL-adaptive is unstable; use linear anneal.
4. **Verify config persistence** — Runtime-verify all reward scales before training (Navigation1 had config drift bug).
5. **Add step-delta approach** — Navigation1's step-delta approach (vs min-distance) provided better gradient signal.
6. **Design terrain-specific rewards** — Height progress for slopes, knee lift for stairs, waypoint bonuses for long course.
7. **Run VLM visual analysis** — Before designing rewards, capture frames of current random behavior to identify gaps.

---

## 9. Lessons Inherited from Navigation1

> See `starter_kit_docs/navigation1/REPORT_NAV1.md` for full context.

| # | Lesson | Relevance to NAV2 |
|---|--------|-------------------|
| 1 | Reward budget audit before training | **CRITICAL** — NAV2 has longer episodes (4000-9000 steps), making budget even more important |
| 2 | KL-adaptive LR is unstable | Use linear anneal for all NAV2 environments |
| 3 | Don't warm-start from degraded runs | Reset optimizer when transferring between sections |
| 4 | Curriculum easier→harder | Section 011 → 012 → 013 → long_course |
| 5 | Config persistence across sessions | Verify runtime config before every NAV2 training run |
| 6 | forward_velocity must stay ≥0.8 | Navigation drive essential; fix sprint-crash via speed coupling, not by reducing velocity |
| 7 | Step-delta approach > min-distance | Continuous gradient for both approaching and retreating |
| 8 | near_target_speed radius matters | 0.5m activation, not 2.0m (deceleration moat) |
| 9 | Use AutoML, not manual train.py | Batch search for reward weights + HP tuning |
| 10 | VLM visual analysis before reward design | Capture frames to diagnose behavior before coding fixes |

---

## 10. Next Steps

1. **Fix reward budgets for all sections** — Apply anti-laziness trifecta
2. **Launch Stage 2A training** — Section 011 (slopes) with fixed rewards
3. **Monitor and report results** — Append to this report
4. **VLM analysis of Stage 2A policy** — Before progressing to Stage 2B
5. **Design Section 012 terrain-specific rewards** — After understanding slope behavior

---

*This report will be updated chronologically as experiments are conducted. Never overwrite existing content — the history is a permanent record.*
