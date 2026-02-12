# Section 012 Tutorial — Stairs + Bridge + Spheres + Cones

Welcome! This tutorial covers the **Section 012** environment in Navigation2 — training a quadruped robot (VBot) to climb stairs, cross a narrow arch bridge, navigate past sphere/cone obstacles, and descend stairs.

> **Prerequisite**: Read `starter_kit_docs/navigation1/Tutorial.md` for framework fundamentals, and `starter_kit_docs/navigation2/long_course/Tutorial.md` for the overall Navigation2 course overview.

---

## 1. Section 012 Overview

| Aspect | Value |
|--------|-------|
| **Environment** | `vbot_navigation_section012` |
| **Terrain** | Entry platform → stairs (left/right) → bridge/spheres → stairs down → exit platform |
| **Distance** | ~14.5m (y=9.5 → y=24.0) |
| **Episode** | 6000 steps (60s) |
| **Points** | **60 pts** (highest value section — 57% of total Stage 2 score) |
| **Status** | Default config — reward budget needs fixing before training |

### What Skills Does This Section Train?

- Stair climbing (10 steps, ΔZ=0.10-0.15 per step)
- Stair descending (balance and controlled descent)
- Narrow-path traversal (arch bridge, ~2.64m wide)
- Obstacle avoidance (5 spheres R=0.75, 8 cones)
- Route selection (left steep stairs + bridge vs right gentle stairs + obstacles)
- Height-aware navigation (z: 1.294 → 2.794 → 1.294, up-then-down)

---

## 2. Terrain Map — Section 02

### Two Routes

```
              LEFT ROUTE (harder)              RIGHT ROUTE (easier)
              x ≈ -3.0                         x ≈ +2.0
              
Y=12.4  ── 10-step stairs UP (ΔZ=0.15) ──    10-step stairs UP (ΔZ=0.10) ──
            z: 1.37 → 2.79                    z: 1.32 → 2.29
            
Y=15~20 ── Arch bridge ──────────────────    5 spheres (R=0.75) + 8 cones ──
            23 segments, peak z≈2.86          scattered obstacles
            width ~2.64m, railings
            
Y=21~23 ── 10-step stairs DOWN ──────────    10-step stairs DOWN ───────────
            z: 2.79 → 1.37                    z: 2.29 → 1.32
            
Y=24.3  ──────────────── EXIT PLATFORM (z≈1.294) ────────────────────────────
```

### Terrain Details

| Element | Left Route | Right Route |
|---------|-----------|-------------|
| Stairs up | 10 steps, ΔZ≈0.15/step (steep) | 10 steps, ΔZ≈0.10/step (gentle) |
| Middle | Arch bridge (narrow, elevated) | Spheres + cones (obstacles) |
| Stairs down | 10 steps, ΔZ≈0.15/step | 10 steps, ΔZ≈0.10/step |
| Difficulty | Hard (steep stairs + narrow bridge) | Medium (gentle stairs + obstacle avoidance) |

**Robot spawn**: (0, 9.5, 1.8) on entry platform, ±0.3m randomization.

---

## 3. Route Selection

The policy must discover the optimal route on its own. **Do not bias the reward toward left or right.** Let the RL agent learn from experience.

**Route trade-offs**:
- **Left**: Steeper stairs but clear bridge path (no obstacles on bridge)
- **Right**: Gentler stairs but 5 spheres (R=0.75m) blocking the path

Both routes reach the same exit platform at y≈24.3, z≈1.294.

---

## 4. Terrain-Specific Challenges

### 4.1 Stair Climbing

Stairs require fundamentally different locomotion than flat ground or slopes:

| Parameter | Flat Ground | 15° Slope (Section 011) | Stairs (Section 012) |
|-----------|------------|------------------------|---------------------|
| Foot clearance needed | Minimal | Low | **High** (step edge) |
| Knee flexion | Standard | Slightly increased | **Significantly increased** |
| Speed | Fast | Moderate | **Slow** |
| Foot placement precision | Low | Low | **High** |
| Fall risk | Low | Medium | **High** |

**Key reward components for stairs**:
- **Knee lift bonus**: Reward higher calf joint flexion when on stairs
- **Height progress**: Reward z-axis gain per step (like slopes but more granular)
- **Foot slip penalty**: Penalize lateral/backward foot sliding on stair edges
- **Reduced forward_velocity**: Stability over speed

### 4.2 Arch Bridge

The bridge is only ~2.64m wide with railings on both sides:

```
|  railing  |--- 2.64m usable width ---|  railing  |
```

VBot's body width is ~0.3m, but with leg spread it occupies ~0.5m. The margin is sufficient but requires tight lateral control.

**Key reward components for bridge**:
- **Lateral deviation penalty**: Penalize deviation from bridge centerline
- **Reduced angular velocity**: Limit turning on the bridge
- **Height stability**: Bridge is at z≈2.86 — falls are catastrophic

### 4.3 Obstacle Avoidance

Right route has 5 spheres (R=0.75m) and 8 cones scattered in the path:

**Key reward components for obstacles**:
- **Contact penalty**: Penalize collision with obstacle bodies
- **Edge preference**: Encourage paths along terrain edges where obstacles are fewer
- **Reactive avoidance**: Robot must learn to detour around obstacles

---

## 5. Configuration

### Where to Edit

| What | File |
|------|------|
| Spawn, target, episode length, reward scales | `starter_kit/navigation2/vbot/cfg.py` (class `VBotSection012EnvCfg`) |
| PPO hyperparameters | `starter_kit/navigation2/vbot/rl_cfgs.py` |
| Environment logic (reward computation) | `starter_kit/navigation2/vbot/vbot_section012_np.py` |
| Terrain geometry | `starter_kit/navigation2/vbot/xmls/scene_section012.xml` |

### Current Config Highlights

```python
# Spawn
pos = [0.0, 9.5, 1.8]  # Entry platform (z≈1.294 + 0.5m)
max_episode_steps = 6000  # 60 seconds

# ⚠️ Reward budget BROKEN — needs fixing before training:
alive_bonus = 0.3         # 0.3 × 6000 = 1,800 >> arrival(80)
arrival_bonus = 80.0
termination = -200.0
```

---

## 6. Reward Engineering — Stairs & Bridge

### 6.1 Reward Budget (Must Fix)

```
STANDING STILL for 6000 steps:
  alive = 0.3 × 6000 = 1,800
  Total standing ≈ 2,200+

COMPLETING TASK:
  arrival_bonus = 80

⚠️ Ratio 27:1 — LAZY ROBOT GUARANTEED
```

**Fix template** (apply before training):
```python
alive_bonus = 0.05        # 0.05 × 6000 = 300
arrival_bonus = 200.0     # > alive_budget
# Add stair milestones: 10.0 × N checkpoints
# Add bridge crossing bonus: 30.0
# Add height_progress: 8.0
termination = -100.0
```

### 6.2 Stair-Specific Rewards

```python
# Knee lift bonus (only on stairs, detected by Y-position or z-gradient)
if on_stairs:
    for leg in ['FR', 'FL', 'RR', 'RL']:
        calf_angle = joint_pos[f'{leg}_calf_joint']
        knee_lift = -calf_angle  # More negative = higher lift
        if knee_lift > 1.5:
            reward += 0.2 * (knee_lift - 1.5)

# Foot slip penalty
foot_velocities = get_foot_velocities()
for foot_vel in foot_velocities:
    if foot_in_contact[foot]:
        lateral_slip = np.linalg.norm(foot_vel[:2])
        reward -= 0.1 * lateral_slip
```

### 6.3 Bridge-Specific Rewards

```python
# Lateral deviation penalty on bridge (y≈15.3 to y≈20.3)
if on_bridge:
    bridge_center_x = -3.0  # Left route bridge centerline
    lateral_error = abs(robot_x - bridge_center_x)
    reward -= 0.3 * max(lateral_error - 0.5, 0.0)  # Free zone: ±0.5m
```

---

## 7. Training Workflow

### This Section Uses Curriculum — Train After Section 011

```
Stage 2A: Section 011 (slopes) — MUST COMPLETE FIRST
  ↓ warm-start best checkpoint
Stage 2B: Section 012 (stairs + bridge) — THIS SECTION
```

### Smoke Test

```powershell
uv run scripts/train.py --env vbot_navigation_section012 --max-env-steps 200000
```

### Visual Debug

```powershell
uv run scripts/train.py --env vbot_navigation_section012 --render
```

### Full Training (Warm-Start from Section 011)

```powershell
uv run scripts/train.py --env vbot_navigation_section012 \
    --checkpoint runs/vbot_navigation_section011/<best_run>/checkpoints/best_agent.pt
```

### AutoML Batch Search

```powershell
uv run starter_kit_schedule/scripts/automl.py --mode stage --budget-hours 8 --hp-trials 15
```

### Evaluation

```powershell
uv run scripts/play.py --env vbot_navigation_section012
uv run scripts/capture_vlm.py --env vbot_navigation_section012 --max-frames 25
```

### TensorBoard

```powershell
uv run tensorboard --logdir runs/vbot_navigation_section012
```

**Key metrics to watch**:
- `max_y`: Forward progress (should approach 24.0)
- `max_z`: Height reached (should approach 2.79 for left route, 2.29 for right)
- `height_progress`: Stair climbing behavior
- `ep_len_mean`: Episode length (too short = falling on stairs, max = lazy)

---

## 8. Debugging Tips

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| Robot stands still at entry | alive_bonus dominates (lazy robot) | Fix reward budget |
| Robot falls on first step | Insufficient knee lift / wrong foot placement | Add knee lift bonus, reduce forward_velocity on stairs |
| Robot reaches stair top but falls on bridge | Lateral instability | Add bridge lateral penalty, reduce speed |
| Robot avoids stairs entirely | Stairs too punishing / termination too harsh | Reduce orientation penalty on stairs |
| Robot can't descend stairs | Front-heavy gait | Train descent separately, reduce forward_velocity |
| Robot collides with spheres (right route) | No obstacle avoidance signal | Add contact penalty for sphere bodies |

---

## 9. File Reference

| File | Purpose |
|------|---------|
| `starter_kit/navigation2/vbot/cfg.py` | Section012 config + reward scales |
| `starter_kit/navigation2/vbot/vbot_section012_np.py` | Section 02 environment implementation |
| `starter_kit/navigation2/vbot/rl_cfgs.py` | Section012 PPO hyperparameters |
| `starter_kit/navigation2/vbot/xmls/scene_section012.xml` | Section 02 MJCF scene |
| `starter_kit_docs/navigation2/section012/Task_Reference.md` | Terrain, scoring, reward config |
| `starter_kit_docs/navigation2/section012/REPORT_NAV2_section012.md` | Experiment history |
| `starter_kit_docs/navigation2/section012/Tutorial_RL_Reward_Engineering.md` | Reward engineering guide |
