# Tutorial: VBot Long Course Navigation — Full 34m Three-Section Obstacle Course

> This tutorial covers the **Long Course** environment — the competition submission task combining all three sections into a single 34m run with a waypoint navigation system.
> For per-section guides, see:
> - [Section 011 Tutorial](../section011/Tutorial.md) — slopes, hfield, ramp
> - [Section 012 Tutorial](../section012/Tutorial.md) — stairs, bridge, obstacles
> - [Section 013 Tutorial](../section013/Tutorial.md) — high step, steep ramp, gold balls

---

## 1. Overview

**Goal**: Train VBot to navigate the complete 34m obstacle course from START to FINISH, collecting maximum competition points (105 pts).

| Aspect | Value |
|--------|-------|
| Environment | `vbot_navigation_long_course` |
| Total distance | ~34m (y: -2.4 → 32.3) |
| Sections | 3 (slopes → stairs/bridge → balls/ramp) |
| Waypoints | 7 (auto-advancing navigation system) |
| Episode | 9000 steps (90 seconds) |
| Competition points | 105 max (20 + 60 + 25) |

### Why This Is Hard

The long course is the **hardest RL problem** in MotrixArena S1:

1. **Long horizon**: 9000 steps = 90s. Sparse rewards at waypoints are drowned by per-step passive rewards.
2. **Terrain diversity**: Bumps, 15° slopes, stairs, bridges, 21.8° ramps, heavy balls — one policy must handle all.
3. **Navigation complexity**: Path includes lateral movement (x: 0 → -3 → 0), elevation changes (z: 0.5 → 1.294 → 0 → 1.494), and tight gaps.
4. **Catastrophic forgetting**: Skills learned for early sections may degrade when training on later sections.

---

## 2. Terrain Map & Route

```
                    FINISH ★ WP6 (0, 32.3)     z≈1.494
                    │
     Section 03     │  3 gold balls → hfield → 21.8° ramp → 0.75m step
     (25 pts)       │
                    │
                    WP5 (0, 24.5)               z≈0
                    │
     Section 02     WP4 (-3, 23.0)  ← stair down
     (60 pts)       │
                    WP3 (-3, 20.5)  ← bridge end
                    │
                    WP2 (-3, 15.0)  ← bridge start
                    │
                    WP1 (-3, 12.0)  ← left stair entrance
                    │
                    │ wave hfield + left/right stair choice
                    │
     Section 01     WP0 (0, 6.0)    ← ramp top   z≈1.294
     (20 pts)       │
                    │ 15° ramp + red packets
                    │ hfield + smiley zones
                    │
                    START (0, -2.4)              z=0.5
```

### Default Route (Left Path)

The 7 waypoints trace the **LEFT route** through Section 02:
1. **START → WP0**: Forward through hfield, up 15° ramp to high platform
2. **WP0 → WP1**: Navigate across wave hfield, turn left to left stairway
3. **WP1 → WP2**: Climb left stairs to bridge start
4. **WP2 → WP3**: Cross the bridge
5. **WP3 → WP4**: Descend left stairs on far side
6. **WP4 → WP5**: Navigate to Section 02 exit platform
7. **WP5 → WP6**: Through Section 03 (balls, step, ramp) to FINISH

### Alternative: Right Route

Not implemented in default waypoints. Would use right stairs → riverbed → right stairs. Potentially easier (no bridge) but collects fewer points (no bridge red packets).

---

## 3. Waypoint Navigation System

### How It Works

```python
WAYPOINTS = [
    (0.0, 6.0),      # WP0: section01 exit
    (-3.0, 12.0),     # WP1: left stair entrance
    (-3.0, 15.0),     # WP2: bridge start
    (-3.0, 20.5),     # WP3: bridge end
    (-3.0, 23.0),     # WP4: stair descent
    (0.0, 24.5),      # WP5: section02 exit
    (0.0, 32.3),      # WP6: FINISH
]
WAYPOINT_THRESHOLD = 1.5   # Distance to auto-advance
FINAL_THRESHOLD = 0.8      # Tighter threshold for final target
```

- Robot always navigates toward the **current waypoint**
- When within `WAYPOINT_THRESHOLD` (1.5m), the waypoint index advances
- Final waypoint uses tighter `FINAL_THRESHOLD` (0.8m)
- Waypoint bonus (30.0) awarded at each advance
- Observation space is identical to per-section environments (54-dim)

### Key Observation: Target Tracking

The 54-dim observation includes **position error** and **heading error** relative to the *current waypoint* — the same observation fields used in per-section training. This means:
- The robot doesn't need to know the full course layout
- It only sees the direction and distance to the next waypoint
- Skills transfer naturally from per-section training

---

## 4. Training Strategy

### 4.1 Curriculum Transfer Chain

```
section011 (slopes)     → checkpoint
section012 (stairs)     → warm-start from section011 → checkpoint
section013 (balls/ramp) → warm-start from section012 → checkpoint
long_course (full)      → warm-start from section013 → final policy
```

Each warm-start should:
- Reduce LR to 0.3× of the source training LR
- Reset optimizer state (stale momentum/variance from different task)
- Run a 1M-step smoke test before committing to full training

### 4.2 Phase 1: Smoke Test (5M steps)

```powershell
uv run scripts/train.py --env vbot_navigation_long_course --max-env-steps 5000000
```

**Success criteria**:
- Robot spawns correctly and doesn't immediately fall
- At least WP0 reached occasionally
- Reward curve shows some learning signal

### 4.3 Phase 2: Warm-Start Training (50-100M steps)

```powershell
# After section013 completes:
uv run scripts/train.py --env vbot_navigation_long_course \
  --policy runs/vbot_navigation_section013/<best_run>/checkpoints/best_agent.pickle \
  --max-env-steps 100000000
```

### 4.4 Phase 3: Full Training (300M steps)

Use AutoML for hyperparameter search, then run the winning config to full 300M steps.

```powershell
uv run starter_kit_schedule/scripts/automl.py --mode stage --budget-hours 8 --hp-trials 15
```

---

## 5. Long-Horizon Challenges

### 5.1 The Passive Reward Trap

With 9000 steps per episode and `alive_bonus=0.5`, standing still earns:
```
0.5 × 9000 = 4,500 reward (just from alive)
```

This **dwarfs** the completion bonuses (7×30 + 100 = 310). The robot will learn to stand still.

**Fix**: See [Tutorial_RL_Reward_Engineering.md](Tutorial_RL_Reward_Engineering.md) for anti-laziness strategies.

### 5.2 Discount Factor

With `discount_factor=0.995`, a reward 9000 steps in the future is worth:
```
0.995^9000 ≈ 0.000000000000000000019 (essentially 0)
```

Even with `discount_factor=0.999`:
```
0.999^9000 ≈ 0.000012
```

This means the final arrival_bonus is invisible to the robot at the start. Solutions:
1. **Waypoint bonuses** create intermediate reward signals (every ~5m)
2. **Dense progress rewards** (forward_velocity, distance_progress) provide per-step signal
3. **Curriculum**: Train sections first, then combine — the robot already knows the path

### 5.3 Section Transition Failures

Critical failure points where skills from one section don't transfer to the next:
- **Section01 → Section02 transition** (y≈10.3): High platform → wave hfield. Risk of falling off platform edge.
- **Section02 stair descent** (y≈22-24): Robot must descend stairs without falling. Different skill from ascending.
- **Section02 → Section03 transition** (y≈24.3): Platform → new terrain type. Heading change.

### 5.4 Waypoint-Induced Circling

If the robot reaches a waypoint at high speed, it may overshoot and circle back, wasting time. The `WAYPOINT_THRESHOLD=1.5m` helps, but high-speed policies may still struggle.

---

## 6. Debugging Workflow

### Step 1: Visual Inspection

```powershell
# Watch the robot navigate (or fail)
uv run scripts/play.py --env vbot_navigation_long_course --render

# VLM analysis for behavior diagnosis
uv run scripts/capture_vlm.py --env vbot_navigation_long_course --max-frames 30 \
  --vlm-prompt "Which section does the robot fail at? What terrain causes falls?"
```

### Step 2: Waypoint Progress Tracking

Monitor which waypoints the robot consistently reaches:
- **WP0 only**: Section01 skills OK, but can't transition to Section02
- **WP0-WP1**: Navigation works but stairs are blocking
- **WP0-WP4**: Stairs+bridge OK, but stair descent fails
- **WP0-WP5**: Section02 complete, Section03 is the bottleneck
- **All 7**: Success! Check timing — must complete in 90s

### Step 3: TensorBoard Metrics

```powershell
uv run tensorboard --logdir runs/vbot_navigation_long_course
```

Key metrics:
- `Reward / Total`: Should trend upward (not plateau too early)
- Waypoint-specific metrics (if logged): Track which WPs are consistently reached
- Episode length: Should decrease over training (robot gets faster, or terminates less)

---

## 7. Quick Start

```powershell
# 1. Check prerequisite training is complete
Test-Path runs/vbot_navigation_section011/*/checkpoints/best_agent.pickle
Test-Path runs/vbot_navigation_section012/*/checkpoints/best_agent.pickle
Test-Path runs/vbot_navigation_section013/*/checkpoints/best_agent.pickle

# 2. Verify environment loads
uv run scripts/view.py --env vbot_navigation_long_course

# 3. Check reward budget (should warn about lazy robot)
uv run python -c "
from starter_kit.navigation2.vbot import cfg as _
from motrix_envs.registry import make
env = make('vbot_navigation_long_course', num_envs=1)
cfg = env._cfg
s = cfg.reward_config.scales
alive = s.get('alive_bonus', 0) * cfg.max_episode_steps
wp = s.get('waypoint_bonus', 0) * 7
arrival = s.get('arrival_bonus', 0)
total = wp + arrival
print(f'alive_budget={alive:.0f}  completion_bonuses={total:.0f}  ratio={alive/max(total,0.01):.1f}:1')
"

# 4. Fix reward budget, then train
uv run scripts/train.py --env vbot_navigation_long_course --max-env-steps 5000000
```
