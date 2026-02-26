# Tutorial: RL Reward Engineering for Section 012 — Ordered Multi-Waypoint Navigation

**Case Study: VBot collecting ALL rewards on Section 02 via a strict ordered route**

> This tutorial covers reward engineering specific to Section 012 — the ordered multi-waypoint full-collection strategy through stones, under-bridge, bridge out-and-back, and celebration.

> **Prerequisite**: Read `starter_kit_docs/navigation1/Tutorial_RL_Reward_Engineering.md` for foundational lessons.
> For slope-specific reward engineering, see `starter_kit_docs/navigation2/section011/Tutorial_RL_Reward_Engineering.md`.
> For full-course reward engineering, see `starter_kit_docs/navigation2/long_course/Tutorial_RL_Reward_Engineering.md`.

---

## 1. The Task

| Aspect | Value |
|--------|-------|
| Environment | `vbot_navigation_section012` |
| Strategy | Ordered multi-waypoint (14 WPs), right-side-first fixed route |
| Terrain | Entry → right stairs + stones → under-bridge → bridge out-and-back → exit |
| Distance | ~14.5m straight, ~25m+ actual route |
| Episode | 6000 steps (60s) |
| Points | **60 pts** (57% of Stage 2 total) |
| Celebration | 3 configurable right turns at exit |

**Architecture**: Generic ordered waypoint progression. The `_update_waypoint_state` function handles all waypoints uniformly — no per-waypoint special cases in the reward function.

---

## 2. Reward Budget Audit (Verified ✅)

> **Core Principle** (from Navigation1): Before training, compute max reward for desired vs degenerate behavior.

### Current Config (Fixed)

```
STANDING STILL for 6000 steps (alive=0.05, conditional):
  alive = 0.05 × 3000 (upright fraction) = 150
  Total standing ≈ 150

COMPLETING ALL 14 WAYPOINTS + 3 RIGHT TURNS:
  alive = 150
  Milestones: ~217 (14 waypoint bonuses)
  Celebration: 15×3 + 80 = 125
  waypoint_approach: ~200 cumulative
  forward_velocity: ~150
  Total completing ≈ 700+

✅ Completing (700+) >> Standing (150) — budget is sound
```

---

## 3. Ordered Waypoint Reward Engineering

### 3.1 Generic Waypoint Progression

The core reward signal is **waypoint_approach** (step-delta to current WP) + **milestone_bonus** (one-time on first arrival). This is uniform across all 14 waypoints — no per-WP special cases needed.

```python
# step-delta approach reward (generic)
wp_delta = last_wp_distance - distance_to_current_wp
wp_approach = clip(wp_delta * waypoint_approach_scale, -0.5, 5.0)

# milestone bonus (generic, from Waypoint.bonus_key → scales)
if first_arrival_at_wp:
    milestone_bonus = scales[wp.bonus_key]  # e.g., 8.0 for stone, 30.0 for bridge
```

### 3.2 Z-Constraint Waypoints

Some waypoints have altitude constraints that prevent cheating:
- **Under-bridge (WP6-7)**: `z_max=2.2` — must be below bridge deck
- **Bridge (WP9-11)**: `z_min=2.3` — must be on the bridge, not underneath

The arrival check is: `dist < radius AND z_min ≤ current_z ≤ z_max`.

### 3.3 Virtual Waypoints as Route Guides

Virtual waypoints don't correspond to competition scoring zones but guide the route:
- **WP0 (right_approach)**: Pulls robot toward right side before stones
- **WP8 (bridge_climb_base)**: Guides robot to far stair base (avoids near-end approach)
- **WP9, WP11**: Enforce robot is on bridge (z>2.3) before/after hongbao

Virtual WP bonuses are smaller (10-20) than reward WP bonuses (8-30) to keep focus on scoring zones.

### 3.4 Celebration Multi-Jump

After reaching WP13 (exit), the celebration FSM activates:
```
IDLE → TURNING → SETTLING → TURNING → ... (3 times) → DONE
```
- **jump_reward**: Continuous reward for `z_above_standing` during JUMP state
- **per_jump_bonus**: One-time bonus per successful jump peak (z > threshold)
- **celebration_bonus**: Large bonus when all jumps complete

---

## 4. Terrain-Specific Challenges

### 4.1 Stair Climbing/Descending

Handled by terrain-zone-driven rewards (`_terrain_scale`):
- `foot_clearance × stair_boost`: Amplified on stair zones to encourage knee lift
- `swing_contact_penalty × stair_scale`: Reduced on stairs (foot-edge contact is expected)
- `height_progress`: Generic z-delta reward, naturally rewards stair climbing

### 4.2 Narrow Bridge (out-and-back)

| Challenge | Reward Signal |
|-----------|---------------|
| Narrow path (~2.64m) | Generic waypoint_approach keeps robot on route |
| Altitude requirement (z>2.3) | Z-constraint on WP9-11 prevents cheating |
| Out-and-back pattern | WP9→WP10→WP11 forces walk-to-center, then turn around |
| Descent from same end | WP12 at far stair base guides return path |

---

## 5. Reusable Design Pattern

### 5.1 Waypoint & OrderedRoute Dataclasses

The `Waypoint` and `OrderedRoute` dataclasses in `cfg.py` are designed to be reusable across any section:

```python
@dataclass
class Waypoint:
    xy: tuple[float, float]      # Target position
    label: str                    # Human-readable name
    kind: str = "reward"          # "reward" | "virtual" | "goal"
    radius: float = 1.2          # Arrival detection radius
    z_min: float = -10.0         # Altitude constraint (lower bound)
    z_max: float = 100.0         # Altitude constraint (upper bound)
    bonus_key: str = ""          # Key in reward_config.scales
    bonus_default: float = 0.0   # Fallback if key missing
```

To create a new section's route, subclass `OrderedRoute` and define waypoints. The environment logic (`_init_ordered_route`, `_update_waypoint_state`, `_get_current_target`) works generically.

### 5.2 Adding New Waypoints

To add a waypoint to the route:
1. Add a `Waypoint(...)` entry in `Section012Route.__post_init__` at the desired position
2. Add the `bonus_key` to `reward_config.scales` with a value
3. No changes needed in `vbot_section012_np.py` — the vectorized numpy arrays auto-resize

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
