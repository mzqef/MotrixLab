# Long Course Task Reference — Full 34m Three-Section Navigation

> **This file contains task-specific concrete values** for the Long Course (all 3 sections combined).
> For abstract methodology, see `.github/copilot-instructions.md` and `.github/skills/`.
> For per-section details, see:
> - `starter_kit_docs/navigation2/section011/Task_Reference.md`
> - `starter_kit_docs/navigation2/section012/Task_Reference.md`
> - `starter_kit_docs/navigation2/section013/Task_Reference.md`

---

## Environment ID

| Environment ID | Terrain | Status |
|----------------|---------|--------|
| `vbot_navigation_long_course` | Full course: Section01→02→03 (spawn y=-2.4) | **NOT STARTED** — awaiting per-section completion |

## Competition Scoring — Full Course (105 pts total)

Source: `MotrixArena_S1_计分规则讲解.md`

```
Section 1 (20 pts):  START → "2026" platform
├── 3 × 笑脸区 (smiley zones)     = 3×4 = 12 pts
├── 3 × 红包区 (red packet zones)  = 3×2 = 6 pts
└── 庆祝动作 (celebration)         = 2 pts

Section 2 (60 pts):  "2026" → "丙午大吉" platform
├── 波浪地形到达楼梯              = 10 pts
├── 左楼梯→吊桥 or 右楼梯→河床   = 5 pts
├── 吊桥穿越拜年红包→楼梯口       = 10 pts
├── 楼梯口下来→"丙午大吉"平台     = 5 pts
├── 庆祝动作                      = 5 pts
├── 河床到达楼梯                   = 5 pts
├── 河床石头上贺礼红包 ×5         = 15 pts
└── 桥底下拜年红包 ×2             = 10 pts

Section 3 (25 pts):  "丙午大吉" → "中国结" platform
├── 穿过滚动球区到随机地形         = 10-15 pts (no collision=10, collision+survive=15)
├── 随机地形→"中国结"平台          = 5 pts
└── 庆祝动作                      = 5 pts
```

---

## Terrain Layout (Full Course ~34m)

```
Y=34.0  ┌─────────────────────────────┐  FINISH (中国结 platform)
        │         最终平台              │  z≈1.494
Y=32.3  │     WP6 ★ (0, 32.3)         │
        │                              │
Y=31.0  │   ◉ LEFT   ◉ CTR  ◉ RIGHT  │  3 gold balls R=0.75m
        │   (-3,31) (0,31)  (3,31)    │
Y=30.0  │                              │
Y=29.0  │     21.8° steep ramp         │  Section 03
Y=28.0  │     0.75m high step          │
Y=27.0  │     hfield (bumps)           │
        │                              │
Y=24.5  │     WP5 ★ (0, 24.5)         │  Exit Section 02
Y=24.3  ├──────────────────────────────┤
        │  丙午大吉 platform            │
Y=23.0  │     WP4 ★ (-3, 23.0)        │  Stair descent (left route)
        │                              │
Y=20.5  │     WP3 ★ (-3, 20.5)        │  Bridge end (left route)    Section 02
        │     Bridge / 河床             │
Y=15.0  │     WP2 ★ (-3, 15.0)        │  Bridge start
Y=12.0  │     WP1 ★ (-3, 12.0)        │  Left stair entrance
        │   LEFT stairs  RIGHT stairs  │  y≈10.3-14.3
        │        波浪地形(hfield)       │
Y=10.3  ├──────────────────────────────┤
        │   "2026" platform (z=1.294)  │
Y=6.0   │     WP0 ★ (0, 6.0)          │  Section 01 exit
        │     15° ramp (z→1.294)       │  Section 01
Y=4.4   │     红包 zones (3×)          │
Y=0.0   │     hfield (smiley zones 3×) │
Y=-2.4  │     START platform (z=0.5)   │  Spawn
        └──────────────────────────────┘
```

---

## Waypoint System

| Waypoint | Position (x, y) | Terrain | Threshold |
|----------|-----------------|---------|-----------|
| WP0 | (0.0, 6.0) | Section01 exit — top of ramp/high platform | 1.5m |
| WP1 | (-3.0, 12.0) | Left stair entrance | 1.5m |
| WP2 | (-3.0, 15.0) | Bridge start | 1.5m |
| WP3 | (-3.0, 20.5) | Bridge end | 1.5m |
| WP4 | (-3.0, 23.0) | Stair descent bottom (left route) | 1.5m |
| WP5 | (0.0, 24.5) | Section02 exit platform | 1.5m |
| WP6 | (0.0, 32.3) | FINISH — 中国结 platform | 0.8m |

**Route**: Default waypoints follow the **LEFT route** (left stair → bridge → left stair down).

---

## Spawn Configuration

```python
class InitState:
    pos = [0.0, -2.4, 0.5]
    pos_randomization_range = [-0.5, -0.5, 0.5, 0.5]  # ±0.5m in x,y
```

---

## Reward Scales (Current — BROKEN)

```python
# From VBotLongCourseEnvCfg.RewardConfig.scales:
"position_tracking": 1.5,
"fine_position_tracking": 5.0,
"heading_tracking": 0.8,
"forward_velocity": 1.5,
"distance_progress": 2.0,
"alive_bonus": 0.5,           # ⚠️ 0.5 × 9000 = 4,500 >> 310 completion
"approach_scale": 8.0,
"waypoint_bonus": 30.0,       # Per waypoint reached (7 WPs × 30 = 210)
"arrival_bonus": 100.0,       # Final target arrival
"stop_scale": 2.0,
"zero_ang_bonus": 6.0,
"orientation": -0.05,
"lin_vel_z": -0.3,
"ang_vel_xy": -0.03,
"torques": -1e-5,
"dof_vel": -5e-5,
"dof_acc": -2.5e-7,
"action_rate": -0.01,
"termination": -100.0,
```

### Budget Diagnosis

```
STANDING STILL:  alive=0.5 × 9000 = 4,500 + tracking ≈ 6,500–9,500
COMPLETING:      7×30 + 100 = 310 + navigation rewards ≈ 500–800
Ratio: ~14:1 to 21:1 → LAZY ROBOT GUARANTEED
```

---

## PPO Hyperparameters

```python
# From VBotLongCoursePPOConfig:
seed = 42
num_envs = 2048
max_env_steps = 300_000_000    # 300M steps
checkpoint_interval = 1000

learning_rate = 2e-4
rollouts = 48                  # Large rollout for long episodes
learning_epochs = 8
mini_batches = 32
discount_factor = 0.995        # Higher discount for long horizon
lambda_param = 0.95
grad_norm_clip = 1.0
entropy_loss_scale = 0.01

ratio_clip = 0.2
value_clip = 0.2
clip_predicted_values = True

share_policy_value_features = False
policy_hidden_layer_sizes = (256, 128, 64)
value_hidden_layer_sizes = (256, 128, 64)
```

**Key differences from per-section configs:**
- `rollouts = 48` (vs 24-32 for sections) — longer episodes need more rollout data
- `discount_factor = 0.995` (vs 0.99) — longer horizon requires higher discount
- `max_env_steps = 300M` (vs 100M-200M) — much more training needed for full course

---

## Training Estimates

| Steps | Duration (est.) |
|-------|-----------------|
| 5M (smoke test) | ~7 min |
| 50M (short run) | ~70 min |
| 100M (medium) | ~2.2 hrs |
| 300M (full) | ~6.6 hrs |

---

## Key Files

| File | Purpose |
|------|---------|
| `starter_kit/navigation2/vbot/cfg.py` | `VBotLongCourseEnvCfg` — config, reward scales |
| `starter_kit/navigation2/vbot/vbot_long_course_np.py` | Environment implementation, waypoint system |
| `starter_kit/navigation2/vbot/rl_cfgs.py` | `VBotLongCoursePPOConfig` — PPO hyperparameters |
| `starter_kit/navigation2/vbot/xmls/scene_world_full.xml` | Full course MJCF scene |
| `runs/vbot_navigation_long_course/` | Training outputs |

---

## AutoML Search Space (Not Yet Defined)

```python
# Placeholder — to be based on per-section best configs
REWARD_SEARCH_SPACE = {
    "alive_bonus": [0.02, 0.05, 0.1],
    "waypoint_bonus": [20.0, 30.0, 50.0],
    "arrival_bonus": [100.0, 200.0, 500.0],
    "forward_velocity": [1.0, 1.5, 2.5],
    "distance_progress": [1.5, 2.0, 3.0],
    "termination": [-50.0, -100.0, -200.0],
}
```
