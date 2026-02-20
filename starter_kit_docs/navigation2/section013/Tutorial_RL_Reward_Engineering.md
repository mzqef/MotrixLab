# Tutorial: RL Reward Engineering for Section 013 — Gold Balls + Steep Ramp + High Step

**Case Study: VBot navigating Section 03 of the MotrixArena S1 obstacle course**

> This tutorial covers reward engineering specific to Section 013 — the 0.75m high step, 21.8° steep ramp, and stable traversal through 3 gold balls.

> **Prerequisite**: Read `starter_kit_docs/navigation1/Tutorial_RL_Reward_Engineering.md` for foundational lessons.
> For slope-specific rewards, see `starter_kit_docs/navigation2/section011/Tutorial_RL_Reward_Engineering.md`.
> For stair/bridge rewards, see `starter_kit_docs/navigation2/section012/Tutorial_RL_Reward_Engineering.md`.
> For full-course reward engineering, see `starter_kit_docs/navigation2/long_course/Tutorial_RL_Reward_Engineering.md`.

---

## 1. The Task

| Aspect | Value |
|--------|-------|
| Environment | `vbot_navigation_section013` |
| Terrain | Entry → 0.75m step → 21.8° ramp → hfield → 3 gold balls → final platform |
| Distance | ~6.3m |
| Episode | 5000 steps (50s) |
| Points | 25 pts |

**Unique challenges**:
- 0.75m step is 2.14× robot height — may require ramp route or extreme locomotion
- 21.8° ramp is 1.45× steeper than Section 011's 15° slope
- 3 gold balls (R=0.75m) with ~1.0m usable gaps — precision navigation + stability under possible contact

**Section3 rule clarification**:
- 不碰滚球通过随机地形：+10
- 碰滚球且不摔倒、不出界通过随机地形：+15（更高）
- 因此奖励口径应是“稳定通过（可含受控接触）”，不是“必须避球”。

---

## 2. Reward Budget Audit (CRITICAL — 当前已实现)

### 当前配置（已实现）

```
STANDING STILL for 5000 steps (alive=0.05):
  alive = 0.05 × 5000 = 250
  other shaping/track terms are constrained by no-progress/no-milestone condition
  Total standing = controlled baseline

COMPLETING TASK:
  arrival_bonus = 120
  milestones = step_or_ramp(25) + ball_zone_pass(20) + celebration(80)
  continuous shaping includes forward/distance/height/ball-gap terms

✅ completion path (arrival + milestones + shaping) > standing
```

### 当前已实现口径

```python
alive_bonus = 0.05          # 0.05 × 5000 = 250
arrival_bonus = 120.0
step_or_ramp_bonus = 25.0
ball_zone_pass_bonus = 20.0
celebration_bonus = 80.0
ball_gap_alignment = 2.0
ball_contact_reward = 4.0
ball_unstable_contact_penalty = -8.0
height_progress = 10.0
score_clear_factor = 0.3
termination = -120.0

forward_velocity = 1.8
orientation = -0.03
lin_vel_z = -0.12
```

### Fixed Budget Projection

```
STANDING STILL for 5000 steps:
  alive = 0.05 × 5000 = 250
  Total standing = baseline (no milestone accumulation)

COMPLETING TASK:
  arrival_bonus = 120
  step_or_ramp_bonus = 25
  ball_zone_pass_bonus = 20
  celebration_bonus = 80
  shaping (forward + distance + height + ball_gap_alignment) > 0
  Total completing = 245 + shaping (and dominates standing baseline)

✅ COMPLETING > STANDING — incentive aligned
```

---

## 3. Steep Ramp Reward Engineering (21.8°)

### 3.1 Comparison with Section 011 (15°)

| Parameter | Section 011 (15°) | Section 013 (21.8°) | Adjustment |
|-----------|-------------------|---------------------|------------|
| Body pitch | ~15° | ~22° | Relax orientation penalty further |
| Height gain per meter forward | 0.27m | 0.40m | Increase height_progress scale |
| Fall risk | Medium | High | Tighter tilt termination |
| Forward velocity | 3.5 (fast) | 1.5-2.0 (slow, cautious) | Reduce speed target |
| Z-velocity | Normal | Higher (steeper climb) | Relax lin_vel_z penalty |

### 3.2 Height Progress Reward (Scaled Up)

```python
# 21.8° ramp requires more climbing effort per meter
# Section 011 used height_progress=8.0 for 15° → scale up for 21.8°
z_progress = current_z - last_z
height_reward = 10.0 * np.clip(z_progress, -0.5, 1.0)  # Higher scale
```

### 3.3 Slope-Aware Stability

```python
# For 21.8° ramp, body naturally tilts ~22°
# Standard orientation penalty (-0.05) would heavily penalize correct behavior
orientation_penalty = -0.02 * orientation_error  # Very relaxed
lin_vel_z_penalty = -0.10 * abs(z_velocity)      # Relaxed (steep climbing = z-velocity)
ang_vel_xy_penalty = -0.02 * angular_velocity     # Keep similar to Section 011
```

---

## 4. Gold Ball Stable Traversal

### 4.1 Gap Navigation Strategy

The 3 gold balls create 2 navigable gaps:

```
Gap 1: x ≈ -1.5 (between LEFT and CENTER balls)
Gap 2: x ≈ +1.5 (between CENTER and RIGHT balls)
```

Each gap is ~1.0m usable width (3.0m spacing - 2×0.75m radius ≈ 1.5m, minus robot clearance).

### 4.2 Recommended: Stable Contact Reward + Unstable Contact Penalty

```python
# No observation change — use contact proxy + stability decomposition
ball_contact_proxy = proximity_to_ball_center()
stable_factor = f(projected_gravity, gyro)  # [0, 1]
stable_contact_reward = ball_contact_reward * ball_contact_proxy * stable_factor * in_ball_zone
unstable_contact_penalty = ball_unstable_contact_penalty * ball_contact_proxy * (1 - stable_factor) * in_ball_zone
```

**Pro**: Matches official scoring (15>10), keeps warm-start compatibility, and avoids over-penalizing useful contact.
**Con**: Requires careful calibration of stable/unstable scales to avoid reckless collision exploits.

### 4.3 Gap Alignment Reward (Keep)

```python
# Reward approaching gap centers when in ball zone
if 30.5 < robot_y < 32.0:  # In gold ball zone
    gap_centers = [-1.5, 1.5]
    min_gap_dist = min(abs(robot_x - gc) for gc in gap_centers)
    gap_reward = ball_gap_scale * max(0, 1.0 - min_gap_dist / 1.0)  # Linear falloff
```

**Pro**: Provides positive guidance toward safe path.
**Con**: Biases toward gap centers; robot might stop at gap instead of passing through.

### 4.4 Option C: Obstacle-Aware Observation (Advanced)

```python
# Add relative ball positions to observation
ball_positions_rel = gold_ball_positions - robot_position  # [3, 3]
extended_obs = np.concatenate([base_obs, ball_positions_rel.flatten()])
# New obs dim: 54 + 9 = 63
```

**Pro**: Best long-term performance — robot can see and plan around balls.
**Con**: Breaks warm-start from section012 (different obs dim). Only use as last resort.

### 4.5 Recommendation

Use **stable-contact decomposition + gap alignment**: keep `ball_gap_alignment` for path guidance, reward stable contact in ball zone, and penalize unstable contact. Consider observation extension only if this shaping still underperforms.

---

## 5. The 0.75m Step Problem

### 5.1 Analysis

The 0.75m step is arguably the hardest single obstacle in the entire Navigation2 course:

| Comparison | Height |
|-----------|--------|
| Robot standing height | ~0.35m |
| Maximum leg reach | ~0.4m |
| 0.75m step | **0.75m** |
| Step/robot ratio | **2.14×** |

Direct step-up is likely physically impossible for VBot.

### 5.2 Strategies

1. **Ramp route** (most likely): The 21.8° ramp is adjacent to the step. The intended path may be to go around via the ramp, not over the step directly.

2. **Verify with XML**: Read `scene_section013.xml` to understand the spatial relationship between the step and ramp — are they blocking the same path or offering alternatives?

3. **VLM analysis**: Run `capture_vlm.py` with random/section012 policy to see how the robot interacts with the step/ramp geometry.

### 5.3 Reward Design for Step Zone

```python
# Don't reward "getting over the step" directly
# Instead, reward Y-axis progress PAST the step zone
if robot_y > 28.0:  # Past step zone
    if not step_milestone_reached:
        reward += step_milestone  # 20.0 one-time
        step_milestone_reached = True
```

---

## 6. Predicted Exploits (Section 013-Specific)

| Exploit | Description | Prevention |
|---------|-------------|------------|
| **Step-base camper** | Robot idles at entry before step | Y-axis milestones, forward_velocity reward |
| **Ramp-avoiding idle** | Robot stays on entry platform | Conditional alive_bonus, large arrival bonus |
| **Ball-zone avoider** | Robot stops right before gold balls | arrival_bonus must dominate passive rewards at y~31 |
| **Gap camping** | Robot enters gap but sits between balls | One-time gap_bonus only, arrival_bonus pulls forward |
| **Ball rider** | Robot pushes against ball (contact + slow forward) | Contact penalty > forward reward per step at ball zone |
| **Height farmer** | Robot repeatedly climbs/descends ramp for z-progress | Only reward upward z-progress, use milestones |

### Exploit Detection Signals

| Signal | Metric | Healthy Range |
|--------|--------|---------------|
| Y-axis progress | max_y per episode | Should approach 32.33 |
| Step zone passage | step_milestone count | Should become non-zero after initial training |
| Ball interaction | ball_collision_count | Should decrease over training |
| Height reached | max_z per episode | Should approach 1.494 |
| Episode length | ep_len trend | Increasing = learning; maxed = lazy |

---

## 7. Curriculum Transfer Considerations

### From Section 012 → Section 013

| Skill | Transfer Quality | Notes |
|-------|-----------------|-------|
| Basic locomotion | ✅ Good | Core walking skill transfers |
| Stair climbing | ⚠️ Partial | Step height clearance may help with ramp |
| Balance at elevation | ✅ Good | Bridge experience helps with elevated terrain |
| Obstacle avoidance | ⚠️ Partial | Only if section012 had sphere contact training |
| Speed control | ✅ Good | Slower speed useful for steep ramp + ball navigation |

### Warm-Start LR

```
Section 012 LR: 2e-4
Section 013 warm-start LR: 2e-4 × 0.3 = 6e-5
```
Reduce LR to prevent catastrophic forgetting of locomotion skills.

---

## 8. Config Verification Script

```powershell
uv run python -c "
from starter_kit.navigation2.vbot import cfg as _
from motrix_envs.registry import make
env = make('vbot_navigation_section013', num_envs=1)
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
