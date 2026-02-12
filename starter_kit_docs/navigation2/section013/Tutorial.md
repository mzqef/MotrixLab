# Section 013 Tutorial — Gold Balls + Steep Ramp + High Step

Welcome! This tutorial covers the **Section 013** environment in Navigation2 — training a quadruped robot (VBot) to overcome a 0.75m high step, climb a 21.8° steep ramp, and navigate between 3 gold balls to reach the final platform.

> **Prerequisite**: Read `starter_kit_docs/navigation1/Tutorial.md` for framework fundamentals, and `starter_kit_docs/navigation2/long_course/Tutorial.md` for the overall Navigation2 course overview.

---

## 1. Section 013 Overview

| Aspect | Value |
|--------|-------|
| **Environment** | `vbot_navigation_section013` |
| **Terrain** | Entry → 0.75m step → 21.8° ramp → height field → 3 gold balls → final platform (z=1.494) |
| **Distance** | ~6.3m (y=26.0 → y=32.33) |
| **Episode** | 5000 steps (50s) |
| **Points** | 25 pts |
| **Status** | Default config — reward budget needs fixing before training |

### What Skills Does This Section Train?

- Overcoming extreme obstacles (0.75m step — taller than the robot)
- Steep slope climbing (21.8° — significantly steeper than Section 011's 15°)
- Obstacle avoidance (3 gold balls, R=0.75m, ~2.5m spacing)
- Precision navigation through narrow gaps (~1.0m usable gap between balls)
- Height field traversal (mild undulation at y≈29.33)

---

## 2. Terrain Map — Section 03

```
Y →  24.3  26.3  27.6  29.3  31.2  32.3  34.3
      |--entry--|--step+ramp--|--hfield--|--balls--|--final--|--wall--|
      z=1.294   z: step 0.75m  z=1.294   🟡🟡🟡   z=1.494
                  ramp 21.8°
```

### Terrain Elements

| Element | Y Location | Z-height | Challenge Level |
|---------|----------|----------|-----------------|
| Entry platform | y≈26.3 | z=1.294 | Easy — flat transition |
| **0.75m high step** | y≈27.6 | 0.75m wall | **Extreme** — 2.14× robot height |
| **21.8° steep ramp** | y≈27.6 | tilted 21.8° | **Hard** — 1.45× steeper than Section 011 |
| Height field | y≈29.3 | z≈1.294 | Medium — undulation |
| **3 gold balls** | y≈31.2 | R=0.75 each | **Hard** — precise gap navigation |
| Final platform | y≈32.3 | **z=1.494** | Course finish (highest point) |

### Gold Ball Layout Detail

```
  x: -5    -3    -1.5   0    1.5    3    5
      |     🟡    GAP   🟡   GAP    🟡   |
      wall   R=0.75          R=0.75       wall
      
  Ball spacing: 3.0m center-to-center
  Ball-to-ball distance: 3.0 - 2×0.75 = 1.5m edge-to-edge
  Usable gap width: ~1.0m (with robot body clearance)
  Gap centers: x ≈ -1.5 and x ≈ +1.5
```

**Robot spawn**: (0, 26.0, 1.8) on entry platform, ±0.5m randomization.

---

## 3. The 0.75m Step Problem

### Physical Analysis

| Parameter | Value |
|-----------|-------|
| Step height | 0.75m |
| Robot standing height | ~0.35m |
| Step-to-robot ratio | **2.14×** |
| Leg reach (max extension) | ~0.4m |

**Critical question**: Can VBot physically climb a 0.75m step?

With a standing height of ~0.35m and max leg reach of ~0.4m, the robot cannot simply step up 0.75m. Possible strategies:

1. **Use the adjacent ramp**: The 21.8° ramp starts near the step — it may be the intended path to bypass the step entirely.
2. **Jumping/lunging**: Some RL policies discover jumping behaviors, but this requires very high action_scale and carries high fall risk.
3. **Climbing motion**: Sequential front-leg placement on step lip, then push up with hind legs. Very complex behavior to learn.

**Recommendation**: Focus training on the ramp route first. Only attempt step climbing if ramp fails or if VLM analysis shows the step position makes the ramp inaccessible.

---

## 4. Configuration

### Where to Edit

| What | File |
|------|------|
| Spawn, target, episode length, reward scales | `starter_kit/navigation2/vbot/cfg.py` (class `VBotSection013EnvCfg`) |
| PPO hyperparameters | `starter_kit/navigation2/vbot/rl_cfgs.py` |
| Environment logic (reward computation) | `starter_kit/navigation2/vbot/vbot_section013_np.py` |
| Terrain geometry | `starter_kit/navigation2/vbot/xmls/scene_section013.xml` |

### Current Config Highlights

```python
# Spawn
pos = [0.0, 26.0, 1.8]  # Entry platform
max_episode_steps = 5000  # 50 seconds

# ⚠️ Reward budget BROKEN — needs fixing before training:
alive_bonus = 0.3         # 0.3 × 5000 = 1,500 >> arrival(60)
arrival_bonus = 60.0
termination = -200.0
```

---

## 5. Reward Engineering — Step + Ramp + Balls

### 5.1 Reward Budget (Must Fix)

```
STANDING STILL for 5000 steps:
  alive = 0.3 × 5000 = 1,500
  Total standing ≈ 1,800+

COMPLETING TASK:
  arrival_bonus = 60

⚠️ Ratio 25:1 — LAZY ROBOT GUARANTEED
```

**Fix template**:
```python
alive_bonus = 0.05        # 0.05 × 5000 = 250
arrival_bonus = 150.0     # > alive_budget
height_progress = 10.0    # Higher than Section 011 (steeper ramp)
step_milestone = 20.0     # One-time: successfully passing the step/ramp zone
ball_gap_bonus = 15.0     # One-time per gap navigated
termination = -100.0
```

### 5.2 Steep Ramp Rewards (21.8°)

Adapt Section 011's slope rewards for a steeper angle:

```python
# Height progress — scale higher for steeper ramp
z_progress = current_z - last_z
height_reward = height_progress_scale * np.clip(z_progress, -0.5, 1.0)  # scale=10.0

# Orientation penalty — relaxed further for 21.8° body tilt
orientation = -0.02  # Lower than Section 011's -0.03 (steeper tilt is correct)
lin_vel_z = -0.10    # Lower than Section 011's -0.15 (more z-velocity expected)
```

### 5.3 Gold Ball Avoidance

```python
# Gap center reward: encourage robot to use gaps between balls
gap_centers = [-1.5, 1.5]  # x-coordinates of gaps
if 30.5 < robot_y < 32.0:  # In gold ball zone
    min_gap_dist = min(abs(robot_x - gc) for gc in gap_centers)
    gap_reward = gap_scale * max(0, 1.0 - min_gap_dist / 1.0)

# Contact penalty for ball collision
ball_contact = get_contact_forces_with_balls()
if any(ball_contact > threshold):
    reward -= ball_collision_penalty  # e.g., -10.0
```

---

## 6. Training Workflow

### This Section Uses Curriculum — Train After Section 012

```
Stage 2A: Section 011 (slopes)
  ↓ warm-start
Stage 2B: Section 012 (stairs + bridge) — MUST COMPLETE FIRST
  ↓ warm-start best checkpoint
Stage 2C: Section 013 (step + ramp + balls) — THIS SECTION
```

### Smoke Test

```powershell
uv run scripts/train.py --env vbot_navigation_section013 --max-env-steps 200000
```

### Visual Debug

```powershell
uv run scripts/train.py --env vbot_navigation_section013 --render
```

### Full Training (Warm-Start from Section 012)

```powershell
uv run scripts/train.py --env vbot_navigation_section013 \
    --checkpoint runs/vbot_navigation_section012/<best_run>/checkpoints/best_agent.pt
```

### Evaluation

```powershell
uv run scripts/play.py --env vbot_navigation_section013
uv run scripts/capture_vlm.py --env vbot_navigation_section013 --max-frames 25
```

### TensorBoard

```powershell
uv run tensorboard --logdir runs/vbot_navigation_section013
```

**Key metrics to watch**:
- `max_y`: Forward progress (should approach 32.33)
- `max_z`: Height reached (should approach 1.494)
- `height_progress`: Ramp climbing behavior
- `ep_len_mean`: Episode length trends
- Ball-related metrics (contact frequency, gap navigation success)

---

## 7. Debugging Tips

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| Robot stands still at entry | alive_bonus dominates (lazy robot) | Fix reward budget |
| Robot can't get past 0.75m step | Step too tall for direct climb | Focus on ramp route; verify ramp accessibility in XML |
| Robot falls on 21.8° ramp | Orientation penalty too tight | Reduce orientation to -0.02, reduce lin_vel_z to -0.10 |
| Robot stops before gold balls | Collision avoidance too strong vs forward reward | Balance ball_collision_penalty vs arrival_bonus |
| Robot collides with gold balls | No avoidance signal | Add contact penalty, consider gap_reward |
| Robot reaches balls but can't find gap | No gap navigation signal | Add gap center bonus, consider extending observation |

---

## 8. File Reference

| File | Purpose |
|------|---------|
| `starter_kit/navigation2/vbot/cfg.py` | Section013 config + reward scales |
| `starter_kit/navigation2/vbot/vbot_section013_np.py` | Section 03 environment implementation |
| `starter_kit/navigation2/vbot/rl_cfgs.py` | Section013 PPO hyperparameters |
| `starter_kit/navigation2/vbot/xmls/scene_section013.xml` | Section 03 MJCF scene |
| `starter_kit_docs/navigation2/section013/Task_Reference.md` | Terrain, scoring, reward config |
| `starter_kit_docs/navigation2/section013/REPORT_NAV2_section013.md` | Experiment history |
| `starter_kit_docs/navigation2/section013/Tutorial_RL_Reward_Engineering.md` | Reward engineering guide |
