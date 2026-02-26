# Section 012 Task Reference — Ordered Multi-Waypoint Full-Collection Navigation

> **This file contains task-specific concrete values** for Section 012 (Stage 2B — stairs, arch bridge, hongbao collection).
> For abstract methodology, see `.github/copilot-instructions.md` and `.github/skills/`.
> For full-course reference, see `starter_kit_docs/navigation2/long_course/Task_Reference.md`.

---

## Environment ID

| Environment ID | Terrain | Status |
|----------------|---------|--------|
| `vbot_navigation_section012` | Section02: entry → right-side stones → under-bridge → bridge (out-and-back) → exit → celebrate | **IMPLEMENTED** — ordered waypoint navigation, 69-dim obs, warm-start ready |

## Strategy: Right-Side-First Ordered Route (14 waypoints)

The course is treated as a **multi-navigation problem**: every reward center (competition scoring zone) is an ordered waypoint. The robot follows a strict fixed route that collects ALL rewards.

```
Route: Right-side first, collect stones, go under bridge, climb up far end,
       cross bridge to collect hongbao, turn around, descend, exit → celebrate.

  WP 0: right_approach         (2.0, 12.0)   virtual  r=1.5
  WP 1: stone_hongbao_1        (0.36, 15.84) reward   r=1.2  → +3pts
  WP 2: stone_hongbao_2        (3.50, 15.84) reward   r=1.2  → +3pts
  WP 3: stone_hongbao_3        (2.00, 17.83) reward   r=1.2  → +3pts
  WP 4: stone_hongbao_4        (0.36, 19.72) reward   r=1.2  → +3pts
  WP 5: stone_hongbao_5        (3.50, 19.72) reward   r=1.2  → +3pts
  WP 6: under_bridge_far       (-3.0, 19.5)  reward   r=1.5  z<2.2 → +5pts
  WP 7: under_bridge_near      (-3.0, 16.0)  reward   r=1.5  z<2.2 → +5pts
  WP 8: bridge_climb_base      (-3.0, 22.5)  virtual  r=1.5
  WP 9: bridge_far_entry       (-3.0, 20.0)  virtual  r=1.5  z>2.3
  WP10: bridge_hongbao         (-3.0, 17.83) reward   r=2.0  z>2.3 → +10pts
  WP11: bridge_turnaround      (-3.0, 20.0)  virtual  r=1.5  z>2.3
  WP12: bridge_descent         (-3.0, 22.5)  virtual  r=1.5
  WP13: exit_platform          (0.0, 24.33)  goal     r=0.8  → +5pts (celebration)
  CELEBRATION: 3 right turns at exit platform
```

Total waypoints: 14 (7 reward + 6 virtual + 1 goal).
`wp_idx` = count of reached waypoints (0 → 14, monotonic). AutoML normalizes by `max_wp=14.0`.

### Waypoint Kinds

| Kind | Meaning | Bonus |
|------|---------|-------|
| `reward` | Competition scoring zone — awards milestone bonus on first arrival | Per-waypoint bonus from `scales` |
| `virtual` | Transit waypoint — guides route between reward zones | Per-waypoint bonus (smaller) |
| `goal` | Final destination — triggers celebration on arrival | Goal bonus |

## Competition Scoring — Section 2 (60 pts total)

Source: `MotrixArena_S1_计分规则讲解.md`

| Scoring Item | Points | Waypoint(s) | Reward Key |
|-------------|--------|-------------|------------|
| 河床石头上贺礼红包 (×5) | +3×5=15 | WP1-5 | `stone_hongbao_bonus` (8.0 each) |
| 桥底下拜年红包 (×2) | +5×2=10 | WP6-7 | `under_bridge_bonus` (15.0 each) |
| 经过吊桥途径拜年红包 | +10 | WP10 | `bridge_hongbao_bonus` (30.0) |
| 通过波浪地形到达楼梯 | +10 | implicit (terrain traversal) | `traversal_bonus` (20.0) |
| 从左楼梯到达吊桥 | +5 | WP8-9 transition | virtual WP bonuses |
| 从楼梯口下来到达平台 | +5 | WP12-13 transition | virtual WP bonuses |
| 庆祝动作 | +5 | WP13 (celebration) | `celebration_bonus` (80.0) |
| **Total** | **60** | | |

## Terrain Description — Section 02

### Overview

```
Y: 8.8   12.4  14.2  15.3  20.3  21.4  23.2  24.3
    |--entry--|--stairs up--|----bridge----|--stairs down--|--exit--|
    z=1.294   z→2.79         z≈2.51~2.71    z→1.37        z=1.294

Right-side-first route: x ≈ +2.0 for stones, x ≈ -3.0 for bridge/under-bridge
```

### Right Route — Stone Hongbaos (WP0-5)

| Element | Center/Range | Key Stats | Notes |
|---------|-------------|-----------|-------|
| Entry platform | (0, 10.33, 1.294) | Section01 exit | Robot spawns here |
| Right stairs up (10 steps) | x=2.0, y=12.4→14.2 | ΔZ≈0.10/step, z: 1.32→2.29 | Gentler slope |
| 5 spheres (stone hongbao) | see WP1-5 coordinates | R=0.75, +3 pts each on top | Main collection targets |
| Right stairs down (10 steps) | x=2.0 | ΔZ≈0.10/step | Gentler descent |

### Left Route — Bridge & Under-Bridge (WP6-12)

| Element | Center/Range | Key Stats | Notes |
|---------|-------------|-----------|-------|
| Under-bridge Hong Bao ×2 | (-3, 16) + (-3, 19.5) | r=1.5, z<2.2 | +5 pts each, collected before climbing |
| Left stairs up (10 steps) | x=-3.0, y=21.4→23.2 (far end) | ΔZ≈0.15/step, z: 1.37→2.79 | Steep — climb from far end |
| Arch bridge | x≈-3.0, y=15.31→20.33 | 23 segments, z≈2.51→2.71, width ~2.64m | Narrow with railings |
| Bridge Hong Bao | (-3.0, 17.83) | r=2.0, z>2.3 | +10 pts, out-and-back to collect |
| Left stairs down (10 steps) | x=-3.0, y=21.4→23.2 | ΔZ≈0.15/step, z: 2.79→1.37 | Descending same stairs |
| Exit platform | (0, 24.33, 1.294) | Final goal | Celebration zone |

### Key Terrain Parameters

| Parameter | Value |
|-----------|-------|
| Entry platform z | 1.294 |
| Left stair step height | ΔZ≈0.15m per step |
| Left stair top z | 2.794 |
| Bridge z range | 2.51 → 2.71 |
| Bridge width | ~2.64m |
| Bridge y range | 15.31 → 20.33 |
| Exit platform z | 1.294 |
| Exit platform center | (0, 24.33) |
| Course bounds | x: ±5.2, y: 8.5~25.5, z_min: 0.5 |

**Robot spawn**: (0, 9.5, 1.8), ±0.3m randomization. Distance to exit: ~14.5m.

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
| 49 | 1 | `celebration_progress` | 0=not started, 0→1 during jump |
| 50-53 | 4 | `foot_contact` | binary contact flags |
| 54-56 | 3 | `trunk_acceleration` | impact detection (v20) |
| 57-68 | 12 | `actuator_torques` | normalized (v20) |
| **Total** | **69** | | |

## Current Reward Config

```python
# ===== 导航主线奖励 =====
forward_velocity:       3.0     # 朝当前WP前进速度
waypoint_approach:    100.0     # step-delta接近当前WP
waypoint_facing:        0.15   # 面朝当前WP
position_tracking:      0.05   # 弱距离信号
alive_bonus:            0.05   # 条件式 (0.05×6000=300)

# ===== 一次性航点里程碑奖励 (对应 Section012Route 14个航点) =====
# 每个航点的bonus值通过 bonus_key → reward_config.scales 映射
# reward类航点:
stone_hongbao_bonus:    8.0    # 每个石头红包 ×5 (WP1-5)
under_bridge_bonus:    15.0    # 每个桥下红包 ×2 (WP6-7)
bridge_hongbao_bonus:  30.0    # 桥上红包 ×1 (WP10)
# virtual类航点:
right_approach_bonus:  10.0    # WP0: 右侧入口
climb_base_bonus:      12.0    # WP8: 楼梯底
bridge_far_entry_bonus: 20.0   # WP9: 桥远端入口
bridge_turnaround_bonus: 15.0  # WP11: 桥上掉头
bridge_descent_bonus:  10.0    # WP12: 下桥
# goal类航点:
exit_bonus:            20.0    # WP13: 终点平台

# ===== 庆祝右转 =====
per_turn_bonus:        15.0    # 每转一次的奖金 (×3次)
celebration_bonus:     80.0    # 完成全部右转的终极奖金
turn_reward:            8.0    # 庆祝右转连续奖励

# ===== 高度进步 & 地形里程碑 =====
height_progress:       12.0    # 爬楼梯z增量
traversal_bonus:       20.0    # Y轴地形里程碑 (4个)

# ===== 步态 & 抬脚 =====
foot_clearance:         0.02   # 摆动相抬脚
foot_clearance_stair_boost: 3.0  # 楼梯区放大
stance_ratio:           0.08   # ~2足着地
swing_contact_penalty: -0.025  # 摆动相触地
swing_contact_stair_scale: 0.5  # 楼梯区降低

# ===== v20: 传感器惩罚 =====
impact_penalty:        -0.02   # trunk冲击
torque_saturation:     -0.01   # 关节扭矩饱和

# ===== 稳定性惩罚 =====
orientation:           -0.015
lin_vel_z:             -0.06
ang_vel_xy:            -0.01
torques:               -5e-6
dof_vel:               -3e-5
dof_acc:               -1.5e-7
action_rate:           -0.005
termination:          -100.0

# ===== 终止得分清零 =====
score_clear_factor:     0.3    # 终止时扣除30%累积奖金
```

### Reward Budget Audit

```
STANDING STILL for 6000 steps:
  alive_bonus (conditional, ~50% upright): 0.05 × 3000 = 150
  No milestone bonuses (never moves)
  Total standing ≈ 150

COMPLETING ALL WAYPOINTS (est. 3000 steps):
  alive_bonus: 0.05 × 3000 = 150
  waypoint_approach: dominant per-step (up to ~200 cumulative)
  Milestones (14 WPs): 10+8×5+15×2+30+12+20+15+10+20 = 217
  Celebration: 15×3 + 80 = 125
  Total completing ≈ 700+
  
  Ratio: Completing (700) >> Standing (150) ✅
```

## PPO Hyperparameters (warm-start aligned)

| Parameter | Value | vs Section011 |
|-----------|-------|---------------|
| learning_rate | 5e-5 | same (warm-start) |
| rollouts | 24 | same |
| learning_epochs | 8 | same |
| mini_batches | 32 | same |
| discount_factor (γ) | 0.999 | same |
| lambda_param (λ) | 0.99 | same |
| entropy_loss_scale | 0.01 | same |
| ratio_clip | 0.2 | same |
| value_clip | 0.2 | same |
| max_env_steps | 200M | same |
| checkpoint_interval | 500 | same |
| policy_net | (256,128,64) | same |
| value_net | (512,256,128) | same |
| share_features | False | same |

## Warm-Start from Section011

```powershell
# Find best section011 checkpoint
uv run scripts/play.py --env vbot_navigation_section011  # auto-finds latest best

# Train section012 with warm-start
uv run scripts/train.py --env vbot_navigation_section012 --policy <section011_best.pt>
```

Requirements:
- **69-dim obs**: section012 obs layout matches section011 exactly
- **12-dim action**: same actuator count
- **Network architecture**: policy (256,128,64), value (512,256,128) — identical
- **LR**: already set to 5e-5 (same as section011, no further reduction needed for initial experiments)
- **Optimizer state**: reset on warm-start (stale momentum from different task)

## Ordered Route Waypoints (from cfg.py: Section012Route)

| WP# | Label | Position (x,y) | Kind | Radius | Z Constraint | Bonus Key | Default | Competition Pts |
|-----|-------|----------------|------|--------|-------------|-----------|---------|----------------|
| 0 | right_approach | (2.0, 12.0) | virtual | 1.5 | — | `right_approach_bonus` | 10.0 | — |
| 1 | stone_hongbao_1 | (0.36, 15.84) | reward | 1.2 | — | `stone_hongbao_bonus` | 8.0 | +3 |
| 2 | stone_hongbao_2 | (3.50, 15.84) | reward | 1.2 | — | `stone_hongbao_bonus` | 8.0 | +3 |
| 3 | stone_hongbao_3 | (2.00, 17.83) | reward | 1.2 | — | `stone_hongbao_bonus` | 8.0 | +3 |
| 4 | stone_hongbao_4 | (0.36, 19.72) | reward | 1.2 | — | `stone_hongbao_bonus` | 8.0 | +3 |
| 5 | stone_hongbao_5 | (3.50, 19.72) | reward | 1.2 | — | `stone_hongbao_bonus` | 8.0 | +3 |
| 6 | under_bridge_far | (-3.0, 19.5) | reward | 1.5 | z < 2.2 | `under_bridge_bonus` | 15.0 | +5 |
| 7 | under_bridge_near | (-3.0, 16.0) | reward | 1.5 | z < 2.2 | `under_bridge_bonus` | 15.0 | +5 |
| 8 | bridge_climb_base | (-3.0, 22.5) | virtual | 1.5 | — | `climb_base_bonus` | 12.0 | — |
| 9 | bridge_far_entry | (-3.0, 20.0) | virtual | 1.5 | z > 2.3 | `bridge_far_entry_bonus` | 20.0 | — |
| 10 | bridge_hongbao | (-3.0, 17.83) | reward | 2.0 | z > 2.3 | `bridge_hongbao_bonus` | 30.0 | +10 |
| 11 | bridge_turnaround | (-3.0, 20.0) | virtual | 1.5 | z > 2.3 | `bridge_turnaround_bonus` | 15.0 | — |
| 12 | bridge_descent | (-3.0, 22.5) | virtual | 1.5 | — | `bridge_descent_bonus` | 10.0 | — |
| 13 | exit_platform | (0.0, 24.33) | goal | 0.8 | — | `exit_bonus` | 20.0 | +5 (celebration) |

### Celebration Configuration

| Parameter | Value |
|-----------|-------|
| `required_turns` | 3 |
| `celebration_turn_threshold` | 1.55 (z above which a turn is counted) |
| `celebration_settle_z` | 1.50 (z below which settling is detected) |
| `per_turn_bonus` | 15.0 per successful right turn |
| `celebration_bonus` | 80.0 on completing all turns |

## Predicted Exploits

| Exploit | Description | Prevention |
|---------|-------------|------------|
| **Standing-still farmer** | Robot stays at spawn, collects alive | alive_bonus conditional on upright, milestones dominate |
| **Waypoint-skip jump** | Robot tries to skip ahead | Ordered route enforces sequential wp_current progression |
| **Z-constraint cheat** | Tries to collect bridge hongbao from below | z_min=2.3 on WP9-11 |
| **Under-bridge from above** | Tries to collect under-bridge from bridge | z_max=2.2 on WP6-7 |
| **Score-clear exploit** | Robot falls intentionally to reset score_clear | score_clear_factor=0.3 (loses 30% bonuses on termination) |

## AutoML Search Space (section012)

Defined in `starter_kit_schedule/scripts/automl.py` as `REWARD_SEARCH_SPACE_SECTION012`.
~35 searchable parameters including all milestone bonuses, gait rewards, stability penalties, and score_clear_factor.

```powershell
# Run section012 AutoML search
uv run starter_kit_schedule/scripts/automl.py --mode stage --budget-hours 8 --hp-trials 15
# (ensure env is set to vbot_navigation_section012 in automl config)
```

## Key Files

| File | Purpose |
|------|---------|
| [starter_kit/navigation2/vbot/cfg.py](../../../starter_kit/navigation2/vbot/cfg.py) | Section012 config: Waypoint, OrderedRoute, Section012Route (14 WPs), CourseBounds, RewardConfig |
| [starter_kit/navigation2/vbot/vbot_section012_np.py](../../../starter_kit/navigation2/vbot/vbot_section012_np.py) | Ordered waypoint navigation implementation (~900 lines) |
| [starter_kit/navigation2/vbot/rl_cfgs.py](../../../starter_kit/navigation2/vbot/rl_cfgs.py) | Section012 PPO hyperparameters (warm-start aligned) |
| [starter_kit/navigation2/vbot/xmls/scene_section012.xml](../../../starter_kit/navigation2/vbot/xmls/scene_section012.xml) | Section 02 MJCF scene |
| [starter_kit_schedule/scripts/automl.py](../../../starter_kit_schedule/scripts/automl.py) | AutoML with REWARD_SEARCH_SPACE_SECTION012 |
