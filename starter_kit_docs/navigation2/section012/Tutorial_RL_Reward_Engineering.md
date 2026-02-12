# Tutorial: RL Reward Engineering for Section 012 — Stairs + Bridge + Obstacles

**Case Study: VBot navigating Section 02 of the MotrixArena S1 obstacle course**

> This tutorial covers reward engineering specific to Section 012 — stair climbing/descending, arch bridge traversal, and sphere/cone obstacle avoidance.

> **Prerequisite**: Read `starter_kit_docs/navigation1/Tutorial_RL_Reward_Engineering.md` for foundational lessons.
> For slope-specific reward engineering, see `starter_kit_docs/navigation2/section011/Tutorial_RL_Reward_Engineering.md`.
> For full-course reward engineering, see `starter_kit_docs/navigation2/long_course/Tutorial_RL_Reward_Engineering.md`.

---

## 1. The Task

| Aspect | Value |
|--------|-------|
| Environment | `vbot_navigation_section012` |
| Terrain | Stairs (left/right routes) → bridge/spheres → stairs down → exit |
| Distance | ~14.5m |
| Episode | 6000 steps (60s) |
| Points | **60 pts** (57% of Stage 2 total) |

**Architecture**: Single-target navigation with potential for waypoint guidance through stairs/bridge. Two route choices (left steep stairs + bridge vs right gentle stairs + obstacles).

---

## 2. Reward Budget Audit (CRITICAL — Not Yet Fixed)

> **Core Principle** (from Navigation1): Before training, compute max reward for desired vs degenerate behavior.

### Current Config (BROKEN)

```
STANDING STILL for 6000 steps (alive=0.3):
  alive = 0.3 × 6000 = 1,800
  position_tracking = exp(-14.5/5) × 1.5 ≈ 0.08/step → 480
  heading_tracking ≈ 0.5/step → 3,000
  Total standing ≈ 5,280+

COMPLETING TASK:
  arrival_bonus = 80

⚠️ STANDING (5,280) >> ARRIVAL (80) — Ratio: 66:1
LAZY ROBOT ABSOLUTELY GUARANTEED
```

### Required Fix

```python
# Target anti-laziness config for section012:
alive_bonus = 0.05        # 0.05 × 6000 = 300
arrival_bonus = 200.0     # > alive_budget
termination = -100.0

# Add stair progression rewards:
stair_step_bonus = 5.0     # per stair step climbed (10 × 5 = 50)
bridge_crossing_bonus = 30.0  # one-time for completing bridge
height_progress = 8.0      # z-delta climbing reward (proven in Section 011)
descent_progress = 4.0     # z-delta descending reward (controlled descent)

# Add terrain-specific penalties:
knee_lift_bonus = 0.2      # on stairs only
foot_slip_penalty = -0.1   # stance phase lateral sliding
bridge_lateral_penalty = -0.3  # deviation from bridge centerline
```

### Fixed Budget Projection

```
STANDING STILL for 6000 steps:
  alive = 0.05 × 6000 = 300
  Total standing ≈ 800

COMPLETING TASK:
  arrival_bonus = 200
  stair_step_bonus = 10 × 5 × 2 = 100 (up + down, both routes)
  bridge_crossing_bonus = 30
  height_progress ≈ 40-80
  forward ≈ 200-400
  Total completing ≈ 570-810 + navigation rewards

✅ Budget becomes balanced (with navigation rewards, completing >> standing)
```

---

## 3. Stair-Specific Reward Engineering

### 3.1 The Stair Climbing Challenge

Stairs are fundamentally harder than slopes because each step is a discrete obstacle:

| Factor | Slope (Section 011) | Stairs (Section 012) |
|--------|---------------------|---------------------|
| Surface | Continuous incline | Discrete steps (ΔZ=0.10-0.15m) |
| Foot clearance | Minimal | Must clear step edge |
| Gait | Smooth forward walk | Step-by-step with high knee lift |
| Fall pattern | Slide backward | Trip on step lip, fall forward/backward |
| Speed | Moderate | Slow (precision > speed) |

### 3.2 Knee Lift Bonus

Standard flat-ground gait drags feet — on stairs, this catches the step edge and trips the robot.

```python
# Reward higher calf flexion when on stairs
# Detect stairs by Y-position range or terrain gradient
if 12.4 < robot_y < 14.3 or 21.4 < robot_y < 23.2:  # On stair zones
    for leg_idx in range(4):
        calf_angle = joint_pos[calf_indices[leg_idx]]
        knee_lift = -calf_angle  # More negative angle = higher lift
        if knee_lift > 1.5:  # Default calf angle = -1.8, so lift = 1.8
            reward += knee_lift_bonus * (knee_lift - 1.5)
```

**Warning**: Don't make knee_lift_bonus too high — it can encourage "marching in place" instead of forward progress. Balance with forward_velocity.

### 3.3 Height Progress for Stairs

Reuse Section 011's height_progress concept but adapt for stair geometry:

```python
# Reward z-axis gain (climbing stairs)
z_progress = current_z - last_z
# For stairs: clip more tightly (each step is ΔZ ≈ 0.10-0.15, not continuous)
height_reward = height_progress_scale * np.clip(z_progress, -0.02, 0.2)
```

### 3.4 Stair Step Milestones

One-time bonuses for reaching specific heights on stairs:

```python
# Left stairs: z checkpoints at 1.5, 1.8, 2.1, 2.4, 2.7
# Right stairs: z checkpoints at 1.4, 1.6, 1.8, 2.0, 2.2
stair_milestones = [1.5, 1.8, 2.1, 2.4, 2.7]  # Adjust per route
for i, z_target in enumerate(stair_milestones):
    if robot_z > z_target and i not in reached_milestones:
        reward += stair_step_bonus
        reached_milestones.add(i)
```

### 3.5 Foot Slip Penalty

On stair edges, feet can slide backward — penalize this:

```python
# Penalize lateral/backward foot sliding during stance
for foot_idx in range(4):
    if foot_in_contact[foot_idx]:
        foot_vel_xy = foot_velocities[foot_idx, :2]
        slip_magnitude = np.linalg.norm(foot_vel_xy)
        reward -= foot_slip_penalty * slip_magnitude
```

---

## 4. Bridge-Specific Reward Engineering

### 4.1 Narrow Path Challenge

Bridge width (~2.64m) vs robot effective width (~0.5m with legs) = ~1m margin per side. Railings prevent falling off but colliding with them wastes energy and slows progress.

### 4.2 Lateral Deviation Penalty

```python
# On bridge (detected by y-position and z-height):
if 15.3 < robot_y < 20.3 and robot_z > 2.5:
    bridge_center_x = -3.0  # Left route bridge centerline
    lateral_error = abs(robot_x - bridge_center_x)
    # Free zone: ±0.5m from center (1m total = safe corridor)
    penalty = bridge_lateral_penalty * max(lateral_error - 0.5, 0.0)
    reward += penalty  # Negative — it's a penalty
```

### 4.3 Bridge Crossing Bonus

One-time reward for successfully crossing the bridge:

```python
# Detect bridge crossing: reach y > 20.0 at z > 2.3
if robot_y > 20.0 and robot_z > 2.3 and not bridge_crossed:
    reward += bridge_crossing_bonus  # 30.0
    bridge_crossed = True
```

---

## 5. Obstacle Avoidance Rewards

### 5.1 Contact Penalty (Right Route)

```python
# Penalize collision with sphere/cone obstacle bodies
obstacle_contacts = get_contact_forces_with_obstacles()
if any(obstacle_contacts > threshold):
    reward -= obstacle_collision_penalty  # e.g., -5.0 per collision step
```

### 5.2 Observation Extension (Advanced)

To add obstacle awareness to the policy, extend the observation space:

```python
# Relative positions of nearest obstacles
obstacle_rel_pos = obstacle_positions - robot_position  # [N, 3]
extended_obs = np.concatenate([base_obs, obstacle_rel_pos.flatten()])
```

**Tradeoff**: Changes observation dim → breaks warm-start from section011. Only use if obstacle avoidance is the bottleneck.

---

## 6. Stair Descent Reward Engineering

Descending stairs is often harder than ascending — the robot must control its forward momentum to avoid tumbling.

### 6.1 Controlled Descent

```python
# Reward controlled z-decrease (not free-fall)
z_decrease = last_z - current_z  # Positive when descending
if on_descending_stairs:
    # Reward slow, controlled descent
    controlled = z_decrease > 0 and z_decrease < 0.2  # Not too fast
    descent_reward = descent_progress * np.where(controlled, z_decrease, 0.0)
```

### 6.2 Forward Speed Limiting on Stairs

```python
# Reduce target forward velocity on stairs
if on_stairs:
    effective_forward_vel_target = 0.3  # Much slower than flat ground (1.0+)
    vel_error = abs(forward_vel - effective_forward_vel_target)
    speed_penalty = -0.5 * max(vel_error - 0.2, 0.0)
```

---

## 7. Non-Monotonic Distance Problem

Section 012's elevation profile goes up then down. Pure 2D distance-to-target can be misleading:

```
Robot at entry (y=9.5) → Target at exit (y=24.0)
Distance = 14.5m ✅

Robot climbs stairs to (y=13, z=2.3) → Target at exit (y=24.0)
2D distance = 11.0m — looks closer ✅

BUT: Robot is at z=2.3, target is at z=1.294
The robot must DESCEND before reaching the target.
3D distance = sqrt(11² + 1²) ≈ 11.05m — barely different from 2D.
```

**Solution**: Use Y-axis forward progress as the primary signal, not distance-to-target:

```python
y_progress = current_y - last_y
forward_reward = forward_scale * np.clip(y_progress, -0.1, 0.5)
```

---

## 8. Predicted Exploits (Section 012-Specific)

| Exploit | Description | Prevention |
|---------|-------------|------------|
| **Stair-base camper** | Robot stands at stair base, collects passive rewards | Conditional alive_bonus, Y-axis checkpoints |
| **Bridge bouncer** | Robot oscillates on bridge start/end | Step-delta with no-retreat clip |
| **Stair-top sitter** | Robot climbs stairs then sits at top | Bridge crossing bonus incentivizes forward progress |
| **Obstacle hugger** | Robot pushes against spheres slowly (contact + forward) | Contact penalty > forward reward per step |
| **Route oscillator** | Robot wanders between left and right routes | Let RL decide; no bias. Y-progress rewards both routes |
| **Descent sprinter** | Robot runs full speed down stairs = tumble | Speed limiting on stairs, controlled descent reward |

### Exploit Detection Signals

| Signal | Metric to Watch | Healthy Range |
|--------|----------------|---------------|
| Forward progress | max_y per episode | Should approach 24.0 |
| Height achieved | max_z per episode | Should reach 2.3+ (stairs top) |
| Episode length pattern | ep_len trend | Increasing = learning | Maxed = lazy |
| Stair milestones | stair_step_bonus | Should become non-zero |

---

## 9. Config Verification Script

```powershell
uv run python -c "
from starter_kit.navigation2.vbot import cfg as _
from motrix_envs.registry import make
env = make('vbot_navigation_section012', num_envs=1)
cfg = env._cfg
s = cfg.reward_config.scales
max_steps = cfg.max_episode_steps
alive = s.get('alive_bonus', 0) * max_steps
arrival = s.get('arrival_bonus', 0)
ratio = alive / max(arrival, 0.01)
print(f'max_steps={max_steps}  alive_budget={alive:.0f}  arrival={arrival:.0f}')
print(f'ratio={ratio:.1f}:1  (should be <2)')
print(f'term={s.get(\"termination\",\"?\")}  forward={s.get(\"forward_velocity\",\"?\")}')
if ratio > 5:
    print('⚠️  WARNING: Lazy robot likely! Fix reward budget before training.')
"
```
