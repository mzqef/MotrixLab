# Tutorial: RL Reward Engineering for Navigation2 — Multi-Terrain Obstacle Course

**Case Studies from MotrixArena S1 Stage 2 Competition**

> This tutorial adapts the reward engineering methodology proven in Navigation1 (flat ground) to the much harder Navigation2 obstacle course. While the abstract principles are the same, the multi-terrain, long-horizon, multi-section nature of Navigation2 introduces unique challenges.

> **Prerequisite**: Read `starter_kit_docs/navigation1/Tutorial_RL_Reward_Engineering.md` for the foundational lessons on reward budget audits, lazy-robot/sprint-crash exploits, LR scheduling, and curriculum learning.

---

## Table of Contents

1. [The Task](#1-the-task)
2. [Why Navigation2 Reward Engineering Is Harder](#2-why-navigation2-reward-engineering-is-harder)
3. [Lesson 1: Reward Budget Audit — Multi-Section Edition](#3-lesson-1-reward-budget-audit-multi-section-edition)
4. [Lesson 2: The Long-Horizon Passive Reward Trap](#4-lesson-2-the-long-horizon-passive-reward-trap)
5. [Lesson 3: Terrain-Specific Rewards](#5-lesson-3-terrain-specific-rewards)
6. [Lesson 4: Waypoint Navigation Rewards](#6-lesson-4-waypoint-navigation-rewards)
7. [Lesson 5: Height-Aware Progress Signals](#7-lesson-5-height-aware-progress-signals)
8. [Lesson 6: Obstacle-Specific Rewards](#8-lesson-6-obstacle-specific-rewards)
9. [Lesson 7: Curriculum Transfer & Reward Compatibility](#9-lesson-7-curriculum-transfer-and-reward-compatibility)
10. [Design Principles (Summary)](#10-design-principles-summary)
11. [Predicted Reward Exploits (Navigation2-Specific)](#11-predicted-reward-exploits)
12. [Appendix: Key Code Patterns](#appendix-key-code-patterns)

---

## 1. The Task

**Goal**: Train a 12-joint quadruped robot to traverse a 30-meter obstacle course consisting of three sections with slopes, stairs, bridges, spheres, gold balls, and steep ramps.

**Environments** (5 total, shared 54-dim obs / 12-dim action):

| Environment | Terrain | Distance | Episode | Points |
|-------------|---------|----------|---------|--------|
| `vbot_navigation_section011` | Flat → 15° slope → high platform | ~12.6m | 4000 steps (40s) | 20 pts |
| `vbot_navigation_section012` | Stairs (10-step) → bridge/spheres | ~14.5m | 6000 steps (60s) | 60 pts |
| `vbot_navigation_section013` | 0.75m wall → 21.8° ramp → gold balls | ~6.3m | 5000 steps (50s) | 25 pts |
| `vbot_navigation_long_course` | All combined with 7 waypoints | ~34m | 9000 steps (90s) | 105 pts |

**Key differences from Navigation1**:
- Navigation1 was a single flat environment, 1000 steps, one target, 20 pts
- Navigation2 has 3D terrain (z varies from 0 to 2.86m), longer episodes, multiple targets
- Passive reward accumulation is a much bigger problem at 4000-9000 step horizons

---

## 2. Why Navigation2 Reward Engineering Is Harder

### 2.1 Longer Episodes Amplify the Lazy Robot

Navigation1's critical fix was reducing `max_episode_steps` from 4000 to 1000. But Navigation2 **requires** long episodes — the robot needs 40-90 seconds to physically traverse complex terrain. You can't just shorten the episode.

**The dilemma**: Long episodes are necessary for the task but create massive passive reward budgets.

```
Navigation1: alive_bonus=0.15 × 1000 steps = 150 (manageable)
Navigation2 Section012: alive_bonus=0.3 × 6000 steps = 1,800 (dominates everything)
```

**Solution approaches**:
- Time-decay on passive rewards (reward decreases over episode)
- Conditional alive_bonus (only active while making forward progress)
- Dramatically increase one-time bonuses (arrival, waypoints) to outweigh passive accumulation
- Checkpoint-based rewards that accumulate as robot progresses

### 2.2 3D Terrain Breaks Flat-Ground Assumptions

Navigation1's approach_reward used 2D distance. On flat ground, distance monotonically decreases as the robot approaches the target. On 3D terrain:

```
Section 012 (stairs + bridge, left route):
  2D distance decreases...
  but robot must go UP stairs (z: 1.29 → 2.79)...
  then ACROSS bridge...
  then DOWN stairs (z: 2.79 → 1.29)...
  
  The 2D target is "behind the stairs" — going forward (up) temporarily 
  INCREASES 2D distance to the target at the end of Section 02.
```

**Problem**: A pure 2D distance-to-target reward may penalize correct behavior (climbing stairs) because the 2D distance to the exit sometimes increases.

**Solution approaches**:
- Use waypoints as intermediate targets
- Reward Y-axis forward progress instead of target distance
- Use 3D distance (including z) for approach calculations

### 2.3 Route Choice Is Implicit

Section 02 offers two routes (left stairs+bridge vs right stairs+spheres). The reward function shouldn't bias the policy toward one route — let the RL agent discover which is easier. But waypoints for the long course currently encode the **left route** (see WAYPOINTS in `vbot_long_course_np.py`).

### 2.4 Multiple Terrain Types Need Different Stability Tolerances

On flat ground, any body tilt is "wrong." On a 15° slope, the body naturally tilts 15° — penalizing tilt penalizes correct behavior.

| Terrain | Body Pitch | Standard Orientation Penalty | Correct Behavior |
|---------|-----------|----------------------------|-------------------|
| Flat ground | ~0° | ✅ Low penalty | Stand upright |
| 15° slope | ~15° | ❌ High penalty incorrectly | Lean into slope |
| 21.8° ramp | ~22° | ❌ Very high penalty incorrectly | Lean steeply |
| Stairs | Variable | ❌ Oscillating penalty | Step-by-step adjustment |

---

## 3. Lesson 1: Reward Budget Audit — Multi-Section Edition

> **Core Principle** (from Navigation1): Before training, compute the maximum reward for desired vs degenerate behavior. If degenerate wins, the policy WILL exploit it.

### Section 011 Budget (Current Config)

```
STANDING STILL for 4000 steps at spawn (d≈12.6m, alive=1.0):
  alive = 1.0 × 4000 = 4,000
  position_tracking = exp(-12.6/5) × 2.0 ≈ 0.16/step → 640
  Total standing ≈ 4,640

TRAVERSING TO TARGET:
  arrival_bonus = 50 (one-time)
  forward + approach ≈ 500 (higher due to active movement)
  Total completing ≈ 550 + partial standing reward

⚠️ STANDING BUDGET (4,640) >> MOVEMENT REWARD (550)
   Ratio: 8.4:1 in favor of laziness
```

### Section 012 Budget (Current Config)

```
STANDING STILL for 6000 steps (alive=0.3):
  alive = 0.3 × 6000 = 1,800
  Total standing ≈ 2,500+ (including passive position/heading)

COMPLETING TASK:
  arrival_bonus = 80

⚠️ STANDING (2,500) >> ARRIVAL (80) — Ratio: 31:1
```

### Section 013 Budget

```
STANDING for 5000 steps (alive=0.3):
  alive = 0.3 × 5000 = 1,500

COMPLETING:
  arrival_bonus = 60

⚠️ Ratio: 25:1
```

### Long Course Budget

```
STANDING for 9000 steps (alive=0.5):
  alive = 0.5 × 9000 = 4,500

COMPLETING ALL 7 WAYPOINTS:
  waypoints = 30 × 7 = 210
  arrival = 100
  Total = 310

⚠️ Ratio: 14.5:1
```

### The Fix Template for Navigation2

Apply the anti-laziness trifecta with Navigation2-specific adjustments:

```python
# Navigation2 anti-laziness template:
# 1. alive_bonus × max_steps < 2 × (arrival + sum(waypoint_bonuses))
# 2. |termination| > 0.25 × alive_budget
# 3. arrival + waypoints > 50% of alive_budget

# Section 011 example fix:
alive_bonus = 0.1       # 0.1 × 4000 = 400
arrival_bonus = 100.0    # Must exceed alive_budget × 0.5
termination = -150.0     # 37.5% of alive_budget
# Now: alive=400, arrival=100, ratio=4:1 → much better (add waypoints to flip)
```

---

## 4. Lesson 2: The Long-Horizon Passive Reward Trap

### The Problem

Navigation2's long episodes (4000–9000 steps) make passive rewards accumulate far more than in Navigation1 (1000 steps). Even after fixing alive_bonus, other passive components add up:

```
Per-step passive rewards (available while standing still):
  position_tracking: exp(-d/5) × 2.0 ≈ 0.16/step (at d=12.6m)
  heading_tracking: cos(err) × 1.0 ≈ 0.5/step (already facing target)
  alive_bonus: 0.1/step (after fix)
  Total passive: ~0.76/step

Over 4000 steps: 0.76 × 4000 = 3,040 passive reward while doing nothing!
```

### The Solutions

**Option A: Time-Decay Passive Rewards**

```python
# Reduce passive rewards over episode lifetime
time_decay = max(1.0 - step / max_steps, 0.1)  # Linear decay to 10%
alive_bonus *= time_decay
position_tracking *= time_decay
```

**Option B: Conditional Passive Rewards**

```python
# Only give alive_bonus while making forward progress
y_progress = current_y - last_y  # Y-axis forward progress this step
alive_bonus = np.where(y_progress > 0.001, 0.1, 0.0)  # Only reward if moving forward
```

**Option C: Checkpoint-Based Progressive Rewards**

Instead of per-step passive rewards, use checkpoint-based one-time rewards that accumulate only through progress:

```python
# Define Y-axis checkpoints every 2m
checkpoints = np.arange(0, 35, 2.0)  # [0, 2, 4, 6, ..., 34]
checkpoint_reward = 5.0  # One-time bonus per checkpoint

# In update_state:
for i, cp_y in enumerate(checkpoints):
    if robot_y > cp_y and i not in reached_checkpoints:
        reward += checkpoint_reward
        reached_checkpoints.add(i)
```

**Navigation2 recommendation**: Use a combination — time-decayed passive rewards + checkpoint bonuses. The checkpoints provide continuous incentive to move forward without being exploitable by standing.

---

## 5. Lesson 3: Terrain-Specific Rewards

### 5.1 Slope Climbing (Section 011)

Standard forward_velocity rewards the robot for moving fast on flat ground. On a 15° slope, moving forward also means climbing — which is harder and should be explicitly rewarded.

**Height progress reward**:

```python
# Reward z-axis gain (climbing)
z_progress = current_z - last_z  # Positive when climbing
height_reward = np.clip(z_progress * height_scale, -0.5, 1.0)
```

**Slope-aware orientation**:

```python
# Don't penalize pitch on slopes — the robot SHOULD lean forward
if terrain_slope_angle > 5.0:  # degrees
    # Reduce orientation penalty proportional to expected tilt
    expected_pitch = np.radians(terrain_slope_angle)
    deviation = abs(body_pitch - expected_pitch)
    orientation_penalty = -0.05 * deviation**2
else:
    orientation_penalty = -0.05 * body_pitch**2  # Standard flat-ground penalty
```

### 5.2 Stair Climbing (Section 012)

Stairs require higher knee lift than flat-ground walking. The standard gait learned on flat ground will drag feet and trip.

**Knee lift bonus**:

```python
# Reward higher calf flexion during stair ascent
# When robot is on stairs (detected by z-gradient or y-position range):
if 12.4 < robot_y < 14.3:  # On left stairs
    for leg in ['FR', 'FL', 'RR', 'RL']:
        calf_angle = joint_pos[f'{leg}_calf_joint']
        knee_lift = -calf_angle  # More negative = higher lift
        if knee_lift > 1.5:  # Threshold: above default (-1.8 → lift=1.8)
            reward += 0.2 * (knee_lift - 1.5)
```

**Foot slip penalty**:

```python
# Penalize lateral foot sliding on stair edges
foot_velocities = get_foot_velocities()  # [4, 3]
for foot_vel in foot_velocities:
    lateral_slip = np.linalg.norm(foot_vel[:2])  # XY velocity
    if foot_in_contact[foot]:  # Only penalize during stance
        reward -= 0.1 * lateral_slip
```

### 5.3 Bridge Traversal (Section 012)

The arch bridge is only ~2.64m wide with railings. The robot needs tight lateral control.

**Lateral deviation penalty**:

```python
# On bridge (y≈15.3 to y≈20.3, x should be near -3.0):
if 15.3 < robot_y < 20.3:
    # Penalize deviation from bridge centerline
    bridge_center_x = -3.0
    lateral_error = abs(robot_x - bridge_center_x)
    reward -= 0.3 * max(lateral_error - 0.5, 0.0)  # Free zone: ±0.5m
```

---

## 6. Lesson 4: Waypoint Navigation Rewards

### The Long Course Challenge

The long course (34m, 7 waypoints) requires the policy to navigate a non-trivial path. Unlike Navigation1's single target, the robot must:
1. Walk forward (Section 01)
2. Turn left toward stairs (x: 0 → -3)
3. Climb stairs
4. Cross bridge
5. Descend stairs
6. Turn right to exit (x: -3 → 0)
7. Navigate Section 03

### Waypoint Reward Structure

```python
WAYPOINTS = [
    (0.0, 6.0),      # WP0: Section 01 exit
    (-3.0, 12.0),    # WP1: Left staircase entrance
    (-3.0, 15.0),    # WP2: Bridge start
    (-3.0, 20.5),    # WP3: Bridge end
    (-3.0, 23.0),    # WP4: Left staircase 2 bottom
    (0.0, 24.5),     # WP5: Section 02 exit
    (0.0, 32.3),     # WP6: Final platform
]
```

**Progressive waypoint bonuses** (higher reward for later, harder waypoints):

```python
waypoint_rewards = [20, 25, 30, 35, 30, 25, 50]  # Total: 215
# Later waypoints = harder to reach → higher reward
# Final waypoint (WP6) gets largest bonus (completing the course)
```

**Distance-to-current-waypoint** (continuous signal):

```python
current_wp = waypoints[current_wp_index]
dist_to_wp = np.linalg.norm(robot_xy - current_wp)
wp_approach_reward = approach_scale * (last_dist_to_wp - dist_to_wp)  # Step-delta
```

### Budget Check for Long Course

```
Waypoint bonuses: 215 total (one-time, progressive)
Arrival bonus: 100 (at WP6)
Total goal rewards: 315

Alive budget (after fix): 0.1 × 9000 = 900
Passive budget: ~0.76 × 9000 × 0.75(time_decay) ≈ 5,130

⚠️ Passive still dominates!
   Need: checkpoint + waypoint rewards ≈ 315
   vs passive ≈ 5,130

Fix: Add distance-based checkpoints every 2m:
   checkpoints = 17 × 5.0 = 85 additional one-time rewards
   Or: reduce position_tracking and heading_tracking scales
```

---

## 7. Lesson 5: Height-Aware Progress Signals

### The 2D Distance Problem

On the long course, pure 2D distance to the next waypoint can be misleading:

```
Robot at (0, 12.0, 1.3) → WP2 at (-3, 15.0)
2D distance = sqrt(9 + 9) = 4.24m

Robot climbs stairs to (0, 13.0, 2.0)
2D distance = sqrt(9 + 4) = 3.61m  ← improved

Robot at top of stairs (-3, 14.5, 2.79)
2D distance = sqrt(0 + 0.25) = 0.5m  ← almost there!

But... the robot still needs to cross the entire bridge and descend stairs
before reaching the section exit. 2D distance to WP5 may INCREASE
as the robot correctly climbs stairs toward WP1-WP4.
```

### Solution: Y-Axis Forward Progress

For a linear course (Y-axis is forward), reward Y-axis progress directly:

```python
# Simple Y-progress reward (robust to 3D terrain)
y_progress = current_y - last_y
forward_reward = forward_scale * np.clip(y_progress, -0.1, 0.5)
```

This is robust because the course always progresses along the Y-axis, regardless of elevation changes.

### Combined Height + Distance Signal

For slope/stair sections where z-progress matters:

```python
# 3D progress: weighted combination of Y-forward and Z-up
y_progress = current_y - last_y
z_progress = current_z - last_z  # Positive = climbing
progress_3d = y_progress + 0.5 * max(z_progress, 0)  # Reward climbing, don't penalize descending
```

**Why `max(z_progress, 0)`?** On the descent portion (after bridge, Section 02 exit), the robot must go down. Penalizing negative z-progress would fight correct behavior.

---

## 8. Lesson 6: Obstacle-Specific Rewards

### 8.1 Static Obstacles (Spheres, Cones — Section 012)

The right path through Section 02 has 5 spheres (R=0.75m) and 8 cones. Options:

**Option A: Contact penalty (simple, no observation change)**

```python
# Penalize collision with obstacle bodies
obstacle_contacts = get_contact_forces(obstacle_bodies)
collision_penalty = -5.0 * np.any(obstacle_contacts > 0.1)
```

**Option B: Obstacle-aware observation (complex, better performance)**

```python
# Extend observation with relative obstacle positions
obstacle_rel_pos = obstacle_positions - robot_position  # [N, 3] relative
# Flatten and append to standard 54-dim obs
extended_obs = np.concatenate([base_obs, obstacle_rel_pos.flatten()])
# observation_space now = 54 + N*3 dimensions
```

**Tradeoff**: Option B breaks warm-start compatibility with standard environments. Use Option A first, only switch to Option B if collision avoidance is the bottleneck.

### 8.2 Gold Balls (Section 013)

Three gold balls (R=0.75m) at y≈31.2, spaced 3m apart (x={-3, 0, 3}). The gaps between them are ~2.5m — just wide enough for VBot (body width ~0.3m) but requires precise navigation.

```
  x: -5    -3    -1.5   0    1.5    3    5
      |     🟡    gap   🟡   gap    🟡   |
      wall                              wall
```

**Strategy**: Reward the robot for entering the gap regions and avoiding ball contact:

```python
# Gap centers at x = {-1.5, 1.5} (between balls)
gap_centers = [-1.5, 1.5]
if 30.5 < robot_y < 32.0:  # In gold ball zone
    # Reward being near a gap center
    min_gap_dist = min(abs(robot_x - gc) for gc in gap_centers)
    gap_reward = 1.0 * max(0, 1.0 - min_gap_dist / 1.0)  # Linear falloff
    
    # Penalize ball contact
    ball_contact = any(dist(robot, ball) < R+margin for ball in gold_balls)
    if ball_contact:
        reward -= 10.0
```

### 8.3 High Step (Section 013)

The 0.75m wall at y≈27.6 is a significant obstacle (VBot stands ~0.35m tall). The robot must somehow get over it.

**Considerations**:
- Direct step-up may be impossible at this height
- The ramp (21.8°) is adjacent — may be the intended path
- Reward should not penalize height loss when backing up to find the ramp
- May need explicit "ramp discovery" reward or waypoint at ramp base

---

## 9. Lesson 7: Curriculum Transfer & Reward Compatibility

### 9.1 Warm-Start Between Sections

The curriculum goes: Section 011 → Section 012 → Section 013 → Long Course. Each warm-start transfers the policy weights but should reset the optimizer.

**Key constraint**: All environments must share the same observation (54-dim) and action (12-dim) spaces. If you extend observations for obstacle avoidance, you can't warm-start from the standard policy.

**Recommendation**: Keep the 54-dim observation space for all curriculum stages. Add obstacle-specific rewards as external signals (contact penalties, position bonuses) rather than observation extensions. Only consider observation extension for the final long-course training.

### 9.2 Reward Scale Compatibility

When warm-starting from Section 011 to Section 012, the policy has learned a reward "scale." If Section 012's rewards are 10× larger or smaller, the value function will be wildly miscalibrated.

**Best practice**: Keep overall reward magnitude similar across sections:

```python
# Target: total per-step reward ~ 1-5 for all sections
# Section 011: alive(0.1) + pos_tracking(0.16) + heading(0.5) + forward(~0.2) ≈ 1.0/step
# Section 012: alive(0.1) + pos_tracking(0.08) + heading(0.5) + forward(~0.2) ≈ 0.9/step
# Section 013: alive(0.1) + pos_tracking(0.15) + heading(0.5) + forward(~0.2) ≈ 0.95/step
```

### 9.3 Skills Transfer

| Skill | Learned In | Transfers To | Compatibility |
|-------|-----------|-------------|---------------|
| Basic locomotion | Section 011 | All sections | ✅ Direct transfer |
| Slope climbing | Section 011 | Section 013 (steeper) | ✅ Transfer + refinement |
| Stair climbing | Section 012 | Long course | ✅ If same stair geometry |
| Bridge walking | Section 012 | Long course | ✅ Narrow traversal skill |
| Obstacle avoidance | Section 013 | Long course | ⚠️ Only if same obs space |
| Waypoint switching | Long course only | N/A | ❌ Not in section envs |

---

## 10. Design Principles (Summary)

### Reward Engineering for Multi-Terrain

1. **Audit the budget for EVERY section separately.** Each has different max_steps and reward scales. A budget that works for 1000 steps fails at 6000.

2. **Use time-decay or conditional passive rewards.** Long episodes (4000–9000 steps) make unconditional alive_bonus, position_tracking, and heading_tracking dominant.

3. **Checkpoint-based > continuous distance rewards on complex terrain.** Y-axis checkpoints every 2m provide reliable progress signal regardless of elevation changes.

4. **Height progress matters on slopes and stairs.** Reward z-axis gain separately from Y-axis forward progress.

5. **Don't penalize correct tilt on slopes.** Standard orientation penalties fight the physics of slope traversal. Consider terrain-aware orientation or simply reducing the penalty scale for slope environments.

6. **Waypoint bonuses should be progressive.** Later waypoints = harder to reach = higher reward. The final waypoint should be the largest bonus.

7. **Keep observation space consistent across curriculum.** All 5 environments share 54-dim obs. Don't extend for one section unless you'll extend for all (or accept losing warm-start).

### Training Protocol

8. **Train sections in order of increasing difficulty.** Section 011 (slopes) → 012 (stairs) → 013 (balls) → long course.

9. **Reset optimizer on warm-start.** Transfer policy weights, discard optimizer state. Same lesson from Navigation1.

10. **Match reward magnitude across sections.** Prevent value function miscalibration during warm-start.

11. **Use AutoML batch search, not manual train.py iteration.** This is even more important for Navigation2 because the search space is larger (more terrain-specific parameters).

### Experimental Methodology

12. **VLM analysis at every section transition.** Before moving from Section 011 to 012, capture frames of the 011 policy to verify locomotion quality.

13. **One section at a time.** Don't try to fix all sections simultaneously. Master each terrain type before combining.

14. **Terrain-specific metrics matter more than total reward.** Track: forward_y_progress, max_z_reached, stair_steps_climbed, waypoints_reached — not just total reward.

---

## 11. Predicted Reward Exploits (Navigation2-Specific)

Based on Navigation1 experience and terrain analysis, anticipate these exploit patterns:

| Exploit | Section | Description | Prevention |
|---------|---------|-------------|------------|
| **Slope hoverer** | 011 | Robot climbs partway up slope, stands on slope collecting passive rewards | Time-decay + checkpoint bonuses on Y-axis |
| **Ramp-avoiding lazy robot** | 011 | Robot stays on flat area before ramp | Large arrival bonus at high platform |
| **Stair-base camper** | 012 | Robot stands at stair base, collects heading/position rewards | Conditional alive_bonus requires Y-progress |
| **Bridge bouncer** | 012 | Robot bounces between bridge start/end collecting approach/retreat rewards | Step-delta with no-retreat clip (learned from NAV1 Round5) |
| **Ball-zone avoider** | 013 | Robot stops before gold balls to avoid collision penalty | Balance collision penalty vs forward progress reward |
| **Waypoint circler** | long | Robot circles near a waypoint threshold, triggering/un-triggering switchover | Hysteresis on waypoint switching (larger exit than entry threshold) |
| **First-waypoint farmer** | long | Robot reaches WP0 quickly, stops (collecting "I'm at a waypoint" reward) | One-time waypoint bonuses, not per-step |
| **Passive elevation rider** | All | Robot at higher z earns more position_tracking if target is elevated | Use 2D distance for position_tracking, separate z-progress |

### Exploit Detection Signals

| Signal | Metric to Watch | Healthy Range |
|--------|----------------|---------------|
| Forward progress | max_y per episode | Should increase over training |
| Section completion rate | % episodes reaching target | Should be >0 by step 10K |
| Episode length + reward | ep_len → max AND reward → flat | If both: lazy robot |
| Episode length collapsing | ep_len << max_steps | Sprint-crash or harsh termination |
| Z-position stagnation | max_z per episode | Should increase for slope/stair sections |
| Waypoints reached | mean waypoints per episode | Should increase monotonically |

---

## Appendix: Key Code Patterns

### A. Per-Section Reward Budget Audit Script

```powershell
uv run python -c "
from starter_kit.navigation2.vbot import cfg as _
from motrix_envs.registry import make

configs = [
    ('vbot_navigation_section011', 'Section 01 (slopes)'),
    ('vbot_navigation_section012', 'Section 02 (stairs)'),
    ('vbot_navigation_section013', 'Section 03 (balls)'),
    ('vbot_navigation_long_course', 'Long Course'),
]

for env_name, label in configs:
    env = make(env_name, num_envs=1)
    cfg = env._cfg
    s = cfg.reward_config.scales
    max_steps = cfg.max_episode_steps
    alive = s.get('alive_bonus', 0) * max_steps
    arrival = s.get('arrival_bonus', 0)
    waypoints = s.get('waypoint_bonus', 0) * 7 if 'waypoint_bonus' in s else 0
    goal_total = arrival + waypoints
    death = s.get('termination', 0)
    ratio = alive / max(goal_total, 0.01)
    
    print(f'=== {label} ({env_name}) ===')
    print(f'  max_steps={max_steps}  alive_budget={alive:.0f}  goal_rewards={goal_total:.0f}')
    print(f'  ratio={ratio:.1f}:1  death={death}')
    if ratio > 5:
        print(f'  ⚠️  LAZY ROBOT LIKELY')
    elif ratio > 2:
        print(f'  ⚠️  Budget marginal')
    else:
        print(f'  ✅ Budget OK')
    print()
"
```

### B. Y-Axis Progress Reward Implementation

```python
def compute_y_progress_reward(self, state, info):
    """Reward Y-axis forward progress (robust to 3D terrain)."""
    robot_y = state.data.body_pos[self._body_idx, 1]
    last_y = info.get("last_y", robot_y.copy())
    
    y_progress = robot_y - last_y
    reward = self._cfg.reward_config.scales.get("y_progress", 1.0) * np.clip(y_progress, -0.1, 0.5)
    
    info["last_y"] = robot_y.copy()
    return reward
```

### C. Checkpoint Reward Implementation

```python
def compute_checkpoint_rewards(self, robot_y, info):
    """One-time bonuses for reaching Y-axis checkpoints."""
    checkpoint_spacing = 2.0  # meters
    checkpoint_reward = 5.0   # per checkpoint
    
    reached = info.get("checkpoints_reached", set())
    reward = np.zeros(self._num_envs, dtype=np.float32)
    
    max_checkpoints = int(34.0 / checkpoint_spacing)
    for cp_idx in range(max_checkpoints):
        cp_y = cp_idx * checkpoint_spacing
        mask = (robot_y > cp_y) & np.array([cp_idx not in reached])
        reward += mask * checkpoint_reward
        if mask.any():
            reached.add(cp_idx)
    
    info["checkpoints_reached"] = reached
    return reward
```

### D. Time-Decay Passive Reward

```python
def compute_time_decay(self, step_count, max_steps):
    """Linear time decay: 1.0 at step 0, 0.1 at max_steps."""
    return np.maximum(1.0 - 0.9 * step_count / max_steps, 0.1)
```

### E. Config Verification for All NAV2 Environments

```powershell
# Quick check — run before every training launch
uv run python -c "
from starter_kit.navigation2.vbot import cfg as _
from motrix_envs.registry import make
for name in ['vbot_navigation_section011','vbot_navigation_section012','vbot_navigation_section013','vbot_navigation_long_course']:
    env = make(name, num_envs=1)
    s = env._cfg.reward_config.scales
    print(f'{name}: alive={s.get(\"alive_bonus\",\"?\")} arrival={s.get(\"arrival_bonus\",\"?\")} term={s.get(\"termination\",\"?\")} fwd={s.get(\"forward_velocity\",\"?\")} steps={env._cfg.max_episode_steps}')
"
```
