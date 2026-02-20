# Section 012 Task Reference — Bridge-Priority Multi-Phase Navigation

> **This file contains task-specific concrete values** for Section 012 (Stage 2B — stairs, arch bridge, hongbao collection).
> For abstract methodology, see `.github/copilot-instructions.md` and `.github/skills/`.
> For full-course reference, see `starter_kit_docs/navigation2/long_course/Task_Reference.md`.

---

## Environment ID

| Environment ID | Terrain | Status |
|----------------|---------|--------|
| `vbot_navigation_section012` | Section02: entry → left stairs up → arch bridge → stairs down → under-bridge collect → exit | **IMPLEMENTED** — bridge-priority state machine, 69-dim obs, warm-start ready |

## Strategy: Bridge-Priority Fixed Route

```
  Phase 0: WAVE_TO_STAIR — 入口平台 → 左楼梯底  WP=[-3, 12.3]
  Phase 1: CLIMB_STAIR   — 左楼梯底 → 楼梯顶    WP=[-3, 14.5] + z>2.3
  Phase 2: CROSS_BRIDGE   — 过桥 (3个虚拟导航点)
           sub_idx 0: bridge_entry [-3, 15.8]
           sub_idx 1: bridge_mid   [-3, 17.83]  ← 桥上红包触发区
           sub_idx 2: bridge_exit  [-3, 20.0]   + z>2.3
  Phase 3: DESCEND_STAIR — 下左楼梯             WP=[-3, 23.2]
  Phase 4: COLLECT_UNDER_BRIDGE — 收集桥下红包   nearest-uncollected targeting
  Phase 5: REACH_EXIT    — 到达终点平台          WP=[0, 24.33] r=0.8
  Phase 6: CELEBRATION   — 庆祝跳跃             IDLE→JUMP→DONE
```

Total virtual waypoints: 9 (including 3 bridge sub-WPs + 2 under-bridge targets).
AutoML `compute_score` normalizes by `max_wp=9.0`.

## Competition Scoring — Section 2 (60 pts total)

Source: `MotrixArena_S1_计分规则讲解.md`

| Scoring Item | Points | Mapped Phase | Reward Key |
|-------------|--------|-------------|------------|
| 通过波浪地形到达楼梯 | +10 | Phase 0 → 1 | `wave_traversal_bonus` (30.0) |
| 从左楼梯到达吊桥 | +5 | Phase 1 → 2 | `stair_top_bonus` (25.0) |
| 经过吊桥途径拜年红包 | +10 | Phase 2 complete | `bridge_crossing_bonus` (50.0) |
| 从楼梯口下来到达平台 | +5 | Phase 3 → 5 | `stair_down_bonus` (20.0) |
| 庆祝动作 | +5 | Phase 6 | `celebration_bonus` (80.0) |
| 河床石头上贺礼红包 | +3×5=15 | (optional) | `stone_hongbao_bonus` (8.0 each) |
| 桥底下拜年红包 | +5×2=10 | Phase 4 | `under_bridge_bonus` (15.0 each) |
| **Total** | **60** | | |

## Terrain Description — Section 02

### Overview

```
Y: 8.8   12.4  14.2  15.3  20.3  21.4  23.2  24.3
    |--entry--|--stairs up--|----bridge----|--stairs down--|--exit--|
    z=1.294   z→2.79         z≈2.51~2.71    z→1.37        z=1.294

Bridge-priority route (LEFT): x ≈ -3.0 throughout
```

### Left Route — Steep Stairs + Arch Bridge (主线)

| Element | Center/Range | Key Stats | Notes |
|---------|-------------|-----------|-------|
| Entry platform | (0, 10.33, 1.294) | Section01 exit | Robot spawns here |
| Left stairs up (10 steps) | x=-3.0, y=12.43→14.23 | ΔZ≈0.15/step, z: 1.369→2.794 | Steep — higher per-step clearance needed |
| Arch bridge | x≈-3.0, y=15.31→20.33 | 23 segments, z≈2.51→2.71, width ~2.64m | Narrow with railings |
| Bridge Hong Bao | (-3.0, 17.83) | r=2.0, z>2.3 | +10 pts, collected naturally while crossing |
| Under-bridge Hong Bao ×2 | (-3, 16) + (-3, 19.5) | r=1.5, z<2.2 | +5 pts each, after descent |
| Left stairs down (10 steps) | x=-3.0, y=21.4→23.2 | ΔZ≈0.15/step, z: 2.794→1.369 | Descending — balance challenge |
| Exit platform | (0, 24.33, 1.294) | Final goal | Celebration zone |

### Right Route — Gentle Stairs + Obstacles (不走)

| Element | Center/Range | Key Stats | Notes |
|---------|-------------|-----------|-------|
| Right stairs up (10 steps) | x=2.0 | ΔZ≈0.10/step, z: 1.319→2.294 | Gentler but has obstacles |
| 5 spheres (stone hongbao) | R=0.75, y=15.8~19.7 | +3 pts each on top | Not on main route |
| Right stairs down (10 steps) | x=2.0 | ΔZ≈0.10/step | Gentler descent |

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

# ===== 一次性里程碑奖励 =====
wave_traversal_bonus:  30.0    # Phase 0→1, +10竞赛分
stair_top_bonus:       25.0    # Phase 1→2, +5竞赛分
bridge_crossing_bonus: 50.0    # Phase 2完成, +10竞赛分
stair_down_bonus:      20.0    # Phase 3→4, +5竞赛分
bridge_hongbao_bonus:  30.0    # 桥上红包 (过桥自然收集)
under_bridge_bonus:    15.0    # 每个桥下红包 ×2
stone_hongbao_bonus:    8.0    # 每个石头红包 ×5 (非主线)
celebration_bonus:     80.0    # 庆祝动作, +5竞赛分
phase_completion_bonus: 15.0   # 通用Phase完成

# ===== Zone吸引力 & 高度进步 =====
zone_approach:          5.0    # 红包/得分区接近
height_progress:       12.0    # 爬楼梯z增量
traversal_bonus:       20.0    # 地形里程碑

# ===== 步态 & 抬脚 =====
foot_clearance:         0.02   # 摆动相抬脚
foot_clearance_stair_boost: 3.0  # 楼梯区放大
stance_ratio:           0.08   # ~2足着地
swing_contact_penalty: -0.025  # 摆动相触地
swing_contact_stair_scale: 0.5  # 楼梯区降低

# ===== 跳跃庆祝 =====
jump_reward:            8.0    # 庆祝跳跃连续奖励

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

COMPLETING ALL PHASES (est. 3000 steps):
  alive_bonus: 0.05 × 3000 = 150
  waypoint_approach: dominant per-step (up to ~200 cumulative)
  Milestones: 30+25+50+20+30+15×2+80+15×6 = 370
  Total completing ≈ 720+
  
  Ratio: Completing (720) >> Standing (150) ✅
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

## Scoring Zones (from cfg.py)

| Zone | Center | Radius | Z Constraint | Points |
|------|--------|--------|-------------|--------|
| Bridge hongbao | (-3.0, 17.83) | 2.0 | z > 2.3 | +10 |
| Under-bridge #1 | (-3.0, 16.0) | 1.5 | z < 2.2 | +5 |
| Under-bridge #2 | (-3.0, 19.5) | 1.5 | z < 2.2 | +5 |
| Stone hongbao ×5 | see cfg.py | 1.0 | — | +3 each |
| Celebration | (0.0, 24.33) | 1.5 | z > 1.0 | +5 |

## Virtual Waypoints (BridgeNav)

| Phase | Waypoint | Position | Radius | Z Check | Info |
|-------|----------|----------|--------|---------|------|
| 0 | wave_to_stair | (-3.0, 12.3) | 1.2 | — | Left stair base |
| 1 | stair_top | (-3.0, 14.5) | 1.2 | z > 2.3 | Must climb stairs |
| 2.0 | bridge_entry | (-3.0, 15.8) | 1.5 | z > 2.3 | Bridge start |
| 2.1 | bridge_mid | (-3.0, 17.83) | 1.5 | z > 2.3 | Bridge center |
| 2.2 | bridge_exit | (-3.0, 20.0) | 1.5 | z > 2.3 | Bridge end |
| 3 | stair_down_bottom | (-3.0, 23.2) | 1.2 | — | Bottom of descent |
| 4 | under-bridge (nearest) | variable | 1.5 | z < 2.2 | 2 targets |
| 5 | exit_platform | (0.0, 24.33) | 0.8 | — | Final goal |
| 6 | celebration | at exit | — | — | Jump sequence |

## Predicted Exploits

| Exploit | Description | Prevention |
|---------|-------------|------------|
| **Stair-base camper** | Robot stays at stair base, collects alive + approach | alive_bonus conditional, milestones dominate |
| **Bridge bouncer** | Oscillates on bridge without crossing | step-delta approach (no-retreat), sub-WP progression |
| **Under-bridge farmer** | Stays below bridge collecting z-approach | z-constraint on bridge scoring zones (z>2.3) |
| **Phase-skip jump** | Robot tries to skip phases | State machine enforces sequential progression |
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
| [starter_kit/navigation2/vbot/cfg.py](../../../starter_kit/navigation2/vbot/cfg.py) | Section012 config: ScoringZones, BridgeNav, CourseBounds, RewardConfig |
| [starter_kit/navigation2/vbot/vbot_section012_np.py](../../../starter_kit/navigation2/vbot/vbot_section012_np.py) | Bridge-priority state machine implementation (~900 lines) |
| [starter_kit/navigation2/vbot/rl_cfgs.py](../../../starter_kit/navigation2/vbot/rl_cfgs.py) | Section012 PPO hyperparameters (warm-start aligned) |
| [starter_kit/navigation2/vbot/xmls/scene_section012.xml](../../../starter_kit/navigation2/vbot/xmls/scene_section012.xml) | Section 02 MJCF scene |
| [starter_kit_schedule/scripts/automl.py](../../../starter_kit_schedule/scripts/automl.py) | AutoML with REWARD_SEARCH_SPACE_SECTION012 |
