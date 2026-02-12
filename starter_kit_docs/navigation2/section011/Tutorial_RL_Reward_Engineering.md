# Tutorial: RL Reward Engineering for Section 011 — Slopes + Multi-Waypoint + Celebration

**Case Study: VBot navigating Section 01 of the MotrixArena S1 obstacle course**

> This tutorial covers reward engineering specific to Section 011 — height field traversal, 15° slope climbing, scoring zone collection, multi-waypoint navigation, and celebration spin.

> **Prerequisite**: Read `starter_kit_docs/navigation1/Tutorial_RL_Reward_Engineering.md` for foundational lessons.
> For full-course reward engineering, see `starter_kit_docs/navigation2/long_course/Tutorial_RL_Reward_Engineering.md`.

---

## 1. The Task

| Aspect | Value |
|--------|-------|
| Environment | `vbot_navigation_section011` |
| Terrain | Flat → 15° slope → high platform |
| Distance | ~10.3m |
| Episode | 3000 steps (30s) |
| Points | 20 pts (smileys 12 + red packets 6 + celebration 2) |

**Architecture**: 3-waypoint navigation (smiley zone → red packet zone → platform) + celebration spin state machine.

---

## 2. Reward Budget Audit (v3 Multi-Waypoint + Celebration)

> **Core Principle** (from Navigation1): Before training, compute max reward for desired vs degenerate behavior.

```
STANDING STILL for 3000 steps at spawn (d≈10.3m, alive=0.05):
  alive = 0.05 × 3000 = 150
  position_tracking = exp(-10.3/5) × 1.5 ≈ 0.19/step → 570
  Total standing ≈ 720

COMPLETING FULL COURSE (START → smileys → ramp → red packets → platform → celebration):
  waypoint_bonus = 3 × 25 = 75 (one-time per waypoint)
  smiley_bonus = 3 × 20 = 60 (pass-through)
  red_packet_bonus = 3 × 10 = 30 (pass-through)
  celebration_bonus = 30 (one-time)
  arrival_bonus = 160 (one-time)
  traversal = 2 × 15 = 30 (milestones)
  spin rewards ≈ 450 (spin_progress + spin_hold)
  wp_approach + forward ≈ 300-500
  Total completing ≈ 1,135-1,335

✅ COMPLETING > STANDING — incentive aligned (strong discrete bonuses)
```

### Anti-Laziness Fix Applied

```python
# Section 011 current config:
alive_bonus = 0.05       # 0.05 × 3000 = 150
arrival_bonus = 160.0    # > alive_budget
waypoint_bonus = 25.0    # 3 × 25 = 75
smiley_bonus = 20.0      # 3 × 20 = 60
red_packet_bonus = 10.0  # 3 × 10 = 30
celebration_bonus = 30.0
termination = -75.0      # 50% of alive_budget
# Now: alive=150, goal_rewards=355 → ratio 0.42:1 ✅
```

---

## 3. Slope-Specific Rewards

### 3.1 Height Progress Reward

Standard `forward_velocity` only rewards Y-axis movement. On a 15° slope, moving forward also means climbing — this effort should be explicitly rewarded.

```python
# Reward z-axis gain (climbing)
z_progress = current_z - last_z  # Positive when climbing
height_reward = np.clip(z_progress * height_scale, -0.5, 1.0)  # scale=8.0
```

### 3.2 Slope-Aware Orientation

Don't penalize pitch on slopes — the robot SHOULD lean forward:

```python
# Standard flat-ground: orientation = -0.05
# Section 011 (15° slope): orientation = -0.03 (reduced)
# Also reduced: lin_vel_z = -0.15 (climbing has z-velocity)
#               ang_vel_xy = -0.02 (ramp stability is different)
```

### 3.3 Traversal Milestones

One-time bonuses confirm the robot is making terrain progress:

| Milestone | Condition | Reward |
|-----------|-----------|--------|
| Mid-ramp | y > 4.0 AND z > 0.3 | 15.0 |
| Ramp top | y > 6.5 AND z > 0.8 | 15.0 |

---

## 4. Multi-Waypoint Navigation Rewards

### Step-Delta Approach

```python
# Reward closing distance to current waypoint (not just any forward movement)
last_wp_dist = info.get("last_wp_distance", distance_to_target.copy())
wp_delta = last_wp_dist - distance_to_target
info["last_wp_distance"] = distance_to_target.copy()
wp_approach = np.clip(wp_delta * scales["waypoint_approach"], 0.0, 1.5)  # scale=40.0
```

### Waypoint Facing

```python
# Reward facing current waypoint
target_bearing = np.arctan2(position_error[:, 1], position_error[:, 0])
facing_error = wrap_angle(target_bearing - robot_heading)
heading_tracking = np.exp(-np.abs(facing_error) / 0.5)
wp_facing = scales["waypoint_facing"] * heading_tracking  # scale=0.6
```

### One-Time Waypoint Bonus

```python
# Per-waypoint one-time bonus (3 × 25.0 = 75 total)
wp_bonus = np.where(first_reach, scales["waypoint_bonus"], 0.0)  # scale=25.0
```

### Navigation vs Celebration Phase

```python
# During celebration phase: suppress navigation rewards, use spin rewards instead
nav_reward = np.where(in_celeb, alive_bonus, full_nav_reward)
```

---

## 5. Celebration Spin Rewards

### Spin Progress (Continuous)

```python
# Reward progress toward target heading during spin phases
heading_delta = abs(wrap_angle(celeb_target_heading - robot_heading))
spin_progress_reward = scales["spin_progress"] * (1.0 - heading_delta / π)  # scale=3.0
```

### Hold Reward (Stillness)

```python
# Reward stillness after both spins complete
is_still = (speed_xy < 0.15) & (abs(gyro_z) < 0.3)
spin_hold_reward = np.where(holding & is_still, scales["spin_hold"], 0.0)  # scale=5.0
```

### Completion Bonus

```python
# One-time when all phases done
celeb_bonus = np.where(done_hold, scales["celebration_bonus"], 0.0)  # scale=30.0
```

### Speed Penalty During Spin

```python
# Penalize fast translation during spin (should rotate in place)
excess_speed = np.maximum(speed_xy - celeb_speed_limit, 0.0)
celeb_speed_penalty = np.where(is_spinning, -2.0 * excess_speed ** 2, 0.0)
```

---

## 6. Scoring Zone Passive Collection

```python
# Smileys and red packets are collected passively as robot passes through
# (not waypoint-gated — side zones collected by proximity)
for i in range(num_smileys):
    d = np.linalg.norm(robot_xy - smiley_centers[i], axis=1)
    first = (d < smiley_radius) & ~smileys_reached[:, i]
    smileys_reached[:, i] |= (d < smiley_radius)
    smiley_total += np.where(first, scales["smiley_bonus"], 0.0)  # 20.0 each
```

---

## 7. Swing-Phase Contact Penalty

**Problem**: On height field bumps, feet moving at high speed can catch = tripping.

```python
# Penalty triggers when: contact_force > 1N AND calf_vel > 2rad/s
# Penalty = Σ(contact × force × velocity / 100)
# Scale: -0.15 (moderate; avoids suppressing all walking gait)
```

---

## 8. Predicted Exploits (Section 011-Specific)

| Exploit | Description | Prevention |
|---------|-------------|------------|
| **Slope hoverer** | Robot climbs partway up slope, stands collecting passive rewards | Time-decay + checkpoint bonuses on Y-axis |
| **Ramp-avoiding lazy robot** | Robot stays on flat area before ramp | Large arrival bonus at high platform |
| **Platform edge camper** | Robot reaches platform but doesn't celebrate | celebration_bonus (30.0) incentivizes completion |
| **Celebration wiggler** | Robot oscillates during spin phases instead of clean rotation | Speed penalty during spin + heading tolerance (0.3 rad) |

### Exploit Detection Signals

| Signal | Metric to Watch | Healthy Range |
|--------|----------------|---------------|
| Forward progress | max_y per episode | Should increase over training |
| Waypoint progression | wp_idx_mean | Should reach 2+ (all waypoints) |
| Celebration success | celeb_state_mean | Should rise above 0 |
| Z-position progress | max_z per episode | Should approach 1.294 (platform height) |

---

## 9. Config Verification Script

```powershell
uv run python -c "
from starter_kit.navigation2.vbot import cfg as _
from motrix_envs.registry import make
env = make('vbot_navigation_section011', num_envs=1)
cfg = env._cfg
s = cfg.reward_config.scales
max_steps = cfg.max_episode_steps
alive = s.get('alive_bonus', 0) * max_steps
arrival = s.get('arrival_bonus', 0)
wp = s.get('waypoint_bonus', 0) * 3
smileys = s.get('smiley_bonus', 0) * 3
red_pkt = s.get('red_packet_bonus', 0) * 3
celeb = s.get('celebration_bonus', 0)
goal_total = arrival + wp + smileys + red_pkt + celeb
ratio = alive / max(goal_total, 0.01)
print(f'max_steps={max_steps}  alive_budget={alive:.0f}  goal_rewards={goal_total:.0f}')
print(f'ratio={ratio:.1f}:1  (should be <1)')
print(f'arrival={arrival:.0f}  wp={wp:.0f}  smileys={smileys:.0f}  red_pkt={red_pkt:.0f}  celeb={celeb:.0f}')
print(f'term={s.get(\"termination\",\"?\")}  forward={s.get(\"forward_velocity\",\"?\")}')
"
```
