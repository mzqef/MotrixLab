# Section 012 Task Reference

**Environment ID**: `vbot_navigation_section012`
**Config class**: `VBotSection012EnvCfg` in `starter_kit/navigation2/vbot/cfg.py`
**Env class**: `VBotSection012Env` in `starter_kit/navigation2/vbot/vbot_section012_np.py`
**Last updated**: March 1, 2026

---

## Competition Scoring (Section 2 = 60 pts max)

| Points | Condition | Waypoint(s) |
|--------|-----------|-------------|
| +10 | Pass wave terrain to stairs | — (transit WP0-1) |
| +5 | From stairs to riverbed/bridge | — (transit) |
| +3 ×5 | Stone hongbaos (5 on right riverbed) | WP2-6 |
| +5 ×2 | Under-bridge hongbaos (2) | WP7-8 |
| +10 | Cross bridge via hongbao to stair exit | WP11 (bridge_hongbao) |
| +5 | Descend stairs to exit platform | WP14 (exit_platform) |
| +5 | Celebration at exit (walk + sit) | CELEB_DONE |
| **60** | **Total** | |

## Terrain Geometry (from XML: 0131_C_section02_hotfix1.xml)

| Parameter | Value |
|-----------|-------|
| Entry platform z | 1.294 |
| Right stair step height | ΔZ≈0.10m per step (10 steps) |
| Right stair top z | ~2.29 |
| Left stair step height | ΔZ≈0.15m per step (10 steps) |
| Left stair top z | 2.794 |
| Bridge z range | 2.51 → 2.71 (peak ≈2.86) |
| Bridge width | ~2.64m |
| Bridge y range | 15.31 → 20.33 |
| Exit platform z | 1.294 |
| Exit platform center | (0, 24.33) |
| Course bounds | x: ±5.2, y: 8.5~25.5, z_min: 0.5 |

**Robot spawn**: (2.0, 12.0, 1.8), ±(0.5, 0.3)m randomization. At right stair base.

## Observation Space (69-dim)

Aligned with section011 for warm-start checkpoint loading:

| Index | Dims | Component | Notes |
|-------|------|-----------|-------|
| 0-2 | 3 | `linear_velocity` | base frame |
| 3-5 | 3 | `angular_velocity` (gyro) | base frame |
| 6-8 | 3 | `projected_gravity` | orientation signal |
| 9-20 | 12 | `joint_positions` (relative) | pos - default |
| 21-32 | 12 | `joint_velocities` | × dof_vel scale |
| 33-44 | 12 | `last_actions` | previous step |
| 45-46 | 2 | `position_error` (xy) | to current WP |
| 47 | 1 | `heading_error` | to current WP |
| 48 | 1 | `base_height` | z coordinate |
| 49 | 1 | `celebration_progress` | 0→1 during walk+sit celebration |
| 50-53 | 4 | `foot_contact` | binary contact flags |
| 54-56 | 3 | `trunk_acceleration` | impact detection |
| 57-68 | 12 | `actuator_torques` | normalized |
| **Total** | **69** | | |

## Ordered Route Waypoints (15 WPs, from cfg.py: Section012Route)

| WP# | Label | Position (x,y) | Kind | Radius | Z Constraint | Bonus Key | Default | Competition Pts |
|-----|-------|----------------|------|--------|-------------|-----------|---------|----------------|
| 0 | right_approach | (2.0, 12.0) | virtual | 1.5 | — | `transit_bonus` | 10.0 | — |
| 1 | stair_top | (2.0, 14.5) | virtual | 1.2 | — | `transit_bonus` | 20.0 | — |
| 2 | stone_1_near_left | (0.36, 15.84) | reward | 1.2 | — | `stone_bonus` | 10.0 | +3 |
| 3 | stone_2_near_right | (3.50, 15.84) | reward | 1.2 | — | `stone_bonus` | 10.0 | +3 |
| 4 | stone_3_center | (2.00, 17.83) | reward | 1.2 | — | `stone_bonus` | 10.0 | +3 |
| 5 | stone_4_far_left | (0.36, 19.72) | reward | 1.2 | — | `stone_bonus` | 10.0 | +3 |
| 6 | stone_5_far_right | (3.50, 19.72) | reward | 1.2 | — | `stone_bonus` | 10.0 | +3 |
| 7 | under_bridge_far | (-3.0, 19.5) | reward | 1.5 | z < 2.2 | `under_bridge_bonus` | 15.0 | +5 |
| 8 | under_bridge_near | (-3.0, 16.0) | reward | 1.5 | z < 2.2 | `under_bridge_bonus` | 15.0 | +5 |
| 9 | bridge_climb_base | (-3.0, 22.5) | virtual | 1.5 | — | `transit_bonus` | 10.0 | — |
| 10 | bridge_far_entry | (-3.0, 20.0) | virtual | 1.5 | z > 2.3 | `bridge_entry_bonus` | 20.0 | — |
| 11 | bridge_hongbao | (-3.0, 17.83) | reward | 2.0 | z > 2.3 | `bridge_hongbao_bonus` | 30.0 | +10 |
| 12 | bridge_turnaround | (-3.0, 20.0) | virtual | 1.5 | z > 2.3 | `transit_bonus` | 5.0 | — |
| 13 | bridge_descent | (-3.0, 22.5) | virtual | 1.5 | — | `transit_bonus` | 10.0 | — |
| 14 | exit_platform | (0.0, 24.33) | goal | 0.8 | — | `exit_bonus` | 30.0 | +5 (celebration) |

`wp_idx` = count of reached waypoints (0 → 15, monotonic).

## Celebration Configuration (v58: Walk + Sit)

Identical to section011. Triggered after reaching WP14 (exit_platform).

| Parameter | Value | Notes |
|-----------|-------|-------|
| `celeb_x_offset` | 4.0 | X target = goal_x + 4.0 |
| `celeb_walk_radius` | 1.0 | Arrival detection radius |
| `celeb_sit_z` | 1.40 | z threshold (standing ≈1.56, platform z ≈1.294) |
| `celeb_sit_steps` | 30 | Hold sit for 30 steps (0.3s) |
| `celeb_x_target` | (4.0, 24.33) | Computed: exit_platform XY + X offset |

**FSM**: `CELEB_IDLE → CELEB_WALKING → CELEB_SITTING → CELEB_DONE`

**Rewards**:
- `celeb_walk_approach` (200.0): Delta-based approach to X endpoint
- `celeb_walk_bonus` (30.0): One-time bonus on reaching X endpoint
- `celeb_sit_reward` (5.0): Per-step sitting reward (× z_below_threshold)
- `celebration_bonus` (50.0): All-done final bonus

## Current Reward Config (BASE_REWARD_SCALES)

All sections share `BASE_REWARD_SCALES`. Key values:

```python
# ===== Navigation (per-step) =====
forward_velocity:       3.163
waypoint_approach:    280.534   # Dominant signal (step-delta toward WP)
waypoint_facing:        0.637
position_tracking:      0.259
alive_bonus:            1.013   # Decayed (alive_decay_horizon=2383)
zone_approach:         74.727

# ===== Waypoint milestone bonuses (one-time) =====
# Values set per-waypoint via bonus_key → scales mapping
# See Waypoint table above for bonus_key and defaults

# ===== Celebration =====
celeb_walk_approach:  200.0
celeb_walk_bonus:      30.0
celeb_sit_reward:       5.0
celebration_bonus:     50.0

# ===== Terrain-specific =====
foot_clearance:         0.219
foot_clearance_stair_boost: 20.0  # ×20 on stairs (extreme knee lift)
foot_clearance_valley_boost: 10.0  # v59: ×10 in riverbed (stones R=0.75)
foot_clearance_wave_boost:   3.0
slope_orientation:      0.04    # Compensate forward-lean (stairs+valley+far-end)
lin_vel_z:             -0.005   # Near-zero (allow vertical push)

# ===== Gait & stability =====
stance_ratio:           0.070
swing_contact_penalty: -0.003
swing_contact_stair_scale: 0.5
swing_contact_valley_scale: 0.3   # v59: lenient in riverbed (frequent stone contacts)
drag_foot_penalty:     -0.15
stagnation_penalty:    -0.5
crouch_penalty:        -1.5
impact_penalty:        -0.100
torque_saturation:     -0.012

# ===== Stability penalties =====
orientation:           -0.026
ang_vel_xy:            -0.038
torques:               -5e-6
dof_pos:               -0.008
dof_vel:               -3e-5
dof_acc:               -1.5e-7
action_rate:           -0.007
termination:         -150.0
score_clear_factor:     0.0
```

### Reward Budget Audit

```
STANDING STILL for 6000 steps:
  alive_bonus (decayed, horizon=2383): ~300 max
  Total standing ≈ 300

COMPLETING ALL 15 WPs + CELEBRATION:
  alive: ~300
  waypoint_approach: ~500+ cumulative
  Milestones (15 WPs): ~230
  Celebration: 30 + 50 + 150 = 230
  forward_velocity: ~200
  Total completing ≈ 1,400+

  Ratio: 4.5:1+ — completing dominates ✅
```

## Termination Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| `hard_tilt_deg` | 70.0 | Immediate termination |
| `soft_tilt_deg` | 50.0 | Terminates after grace period |
| `enable_base_contact_term` | True | Body touches ground → terminate |
| `enable_stagnation_truncate` | True | Truncate on stagnation |
| `grace_period_steps` | 100 | 1s protection at episode start |
| `termination` reward | -150.0 | Penalty on terminate |

## Terrain Zones (action_scale modulation)

| Zone | Y range | action_scale | Clearance Boost | Swing Scale | Pre/Post Margin | Notes |
|------|---------|-------------|----------------|-------------|----------------|-------|
| s012_wave | 8.83-11.83 | 0.40 | wave_boost=3.0 | wave_scale=0.5 | 0/0 | Entry hfield |
| s012_stairs_up | 11.83-14.33 | **0.80** | stair_boost=**20.0** | stair_scale=0.5 | 1.0/0.3 | Max leg amplitude |
| s012_bridge_valley | 14.33-21.33 | **0.50** | valley_boost=**10.0** | valley_scale=**0.3** | 0.5/0.3 | v59: stones R=0.75, slopes 11.6° |
| s012_stairs_down | 21.33-23.33 | **0.80** | stair_boost=**20.0** | stair_scale=0.5 | 1.0/0 | v59: climb UP left stairs to bridge |

## PPO Hyperparameters (warm-start aligned with section011)

| Parameter | Value |
|-----------|-------|
| learning_rate | 5e-5 |
| rollouts | 24 |
| learning_epochs | 8 |
| mini_batches | 32 |
| discount_factor (γ) | 0.999 |
| lambda_param (λ) | 0.99 |
| entropy_loss_scale | 0.01 |
| ratio_clip | 0.2 |
| value_clip | 0.2 |
| max_env_steps | 200M |
| checkpoint_interval | 500 |
| policy_net | (256,128,64) |
| value_net | (512,256,128) |
| share_features | False |

## Warm-Start from Section011

```powershell
uv run scripts/train.py --env vbot_navigation_section012 --policy <section011_best.pt>
```

Requirements: 69-dim obs (identical layout), 12-dim action, same network architecture. Optimizer state resets on warm-start.

## Predicted Exploits

| Exploit | Description | Prevention |
|---------|-------------|------------|
| **Standing-still** | Collect alive at spawn | alive decayed, milestones dominate |
| **Waypoint-skip** | Skip ordered waypoints | Sequential enforcement |
| **Z-constraint cheat** | Bridge hongbao from below | z_min=2.3 on WP10-12 |
| **Under-bridge from above** | Get under-bridge from bridge | z_max=2.2 on WP7-8 |
| **Joint-dragging** | Drag along ground (relaxed term) | Hard termination enabled |

## Key Files

| File | Purpose |
|------|---------|
| `starter_kit/navigation2/vbot/cfg.py` | Config: OrderedRoute, Section012Route (15 WPs), CourseBounds |
| `starter_kit/navigation2/vbot/vbot_section012_np.py` | Ordered waypoint navigation + walk+sit celebration (~1200 lines) |
| `starter_kit/navigation2/vbot/rl_cfgs.py` | PPO hyperparameters |
| `starter_kit/navigation2/vbot/xmls/scene_section012.xml` | MJCF scene |
| `starter_kit_schedule/scripts/automl.py` | AutoML with REWARD_SEARCH_SPACE_SECTION012 |
