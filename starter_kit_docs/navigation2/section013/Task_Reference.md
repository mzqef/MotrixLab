# Section 013 Task Reference — Gold Balls + Steep Ramp + High Step

> **This file contains task-specific concrete values** for Section 013 (Stage 2C — high step, steep ramp, gold ball obstacles).
> For abstract methodology, see `.github/copilot-instructions.md` and `.github/skills/`.
> For full-course reference, see `starter_kit_docs/navigation2/long_course/Task_Reference.md`.

---

## Environment ID

| Environment ID | Terrain | Status |
|----------------|---------|--------|
| `vbot_navigation_section013` | Section03: entry → 0.75m step → 21.8° ramp → hfield → 3 gold balls → final platform | **INITIALIZED** — baseline config initialized |

## Competition Scoring — Section 3 (25 pts total)

Source: `MotrixArena_S1_计分规则讲解.md`

```
Section 3 (25 pts):
├── Rolling balls traversal: 10-15 pts (navigate past 3 gold balls)
├── Random terrain: 5 pts
└── Final celebration: 5 pts
```

**Rule clarification (important)**:
- Pass random terrain **without** touching rolling balls: **+10**
- Pass random terrain **with ball contact but no fall / no out-of-bounds**: **+15** (higher)
- Therefore, training target is **stable traversal**; controlled contact is allowed and can be beneficial.

**Note**: Exact scoring zone positions for Section 3 still need extraction from scene XML / OBJ files.

## Terrain Description — Section 03

### Overview

```
Y: 24.3  26.3  27.6  29.3  31.2  32.3  34.3
    |--entry--|--step+ramp--|--hfield--|--gold balls--|--final--|--wall--|
    z=1.294   z↗?           z=1.294   z=0.844(balls) z=1.494
```

### Terrain Elements

| Element | Center (x, y, z) | Size | Top z | Notes |
|---------|-------------------|------|-------|-------|
| Entry platform | (0, 26.33, 1.044) | 5.0×1.0×0.25 box | 1.294 | From Section 02 exit |
| **0.75m high step** | (0, 27.58, 0.544) | 5.0×0.25×**0.75** box | ~1.294+ | Major obstacle |
| **21.8° steep ramp** | (0, 27.62, 1.301) | Tilted 21.8° | — | After high step |
| Middle platform + hfield | (0, 29.33, 0.794) | 5.0×1.5×0.5 box | 1.294 | With height field |
| **Gold ball LEFT** | (-3.0, 31.23, 0.844) | R=0.75 sphere | — | Blocking path |
| **Gold ball CENTER** | (0.0, 31.23, 0.844) | R=0.75 sphere | — | Blocking path |
| **Gold ball RIGHT** | (3.0, 31.23, 0.844) | R=0.75 sphere | — | Blocking path |
| **Final platform** | (0, 32.33, 0.994) | 5.0×1.5×0.5 box | **1.494** | Course finish |
| End wall | (0, 34.33, 2.564) | Blocking wall | — | Course boundary |

### Gold Ball Layout

```
  x: -5    -3    -1.5   0    1.5    3    5
      |     🟡    gap   🟡   gap    🟡   |
      wall                              wall
      
  Gap centers at x ≈ {-1.5, 1.5}
  Gap width ≈ 2.5m (ball-to-ball, minus 2×R=1.5m → usable gap ~1.0m)
```

**Robot spawn**: (0, 26.0, 1.8), ±0.5m randomization. **Target**: (0, 32.33, 1.494). Distance: ~6.3m.

### Key Terrain Challenges

| Challenge | Details | Impact |
|-----------|---------|--------|
| **0.75m high step** | Wall height vs robot height (~0.35m) = 2.14× robot height | May be physically impossible to step over directly |
| **21.8° steep ramp** | Steeper than Section 011's 15° | Requires aggressive forward lean |
| **3 gold balls** | R=0.75, spacing 3m, gap ~2.5m | Must navigate precisely between balls |
| **Height field** | At y≈29.33, surface undulation | Can trip robot after ramp descent |

## Current Reward Config

## 当前实现口径

- 单目标主线：从入口平台导航到最终平台中心（固定终点）。
- 三里程碑：
  - 通过 step/ramp 区域（`step_or_ramp_bonus`）
  - 通过 ball 区域（`ball_zone_pass_bonus`）
  - 终点停稳并庆祝（`arrival_bonus` + `stop_scale`/`zero_ang_bonus` + `celebration_bonus`）

## 连续shaping映射Section3得分

- `ball_gap_alignment`：在滚球区对齐可通行缝隙，提供连续导航梯度，服务于“滚球通过”得分。
- `ball_contact_reward`：在滚球区内，若接触代理信号存在且姿态稳定，则给予稳定接触奖励。
- `ball_unstable_contact_penalty`：在滚球区内，若接触伴随不稳定姿态/角速度，则施加惩罚。
- `height_progress`：强化坡道/台阶阶段的连续爬升信号，覆盖step+ramp关键地形。
- `termination` + `score_clear_factor`：将失败终止与清分机制绑定，防止通过后摔倒导致策略投机。

```python
position_tracking: 1.5
fine_position_tracking: 5.0
heading_tracking: 0.8
forward_velocity: 1.8
distance_progress: 2.0
alive_bonus: 0.05
approach_scale: 8.0
arrival_bonus: 120.0
step_or_ramp_bonus: 25.0
ball_zone_pass_bonus: 20.0
celebration_bonus: 80.0
ball_gap_alignment: 2.0
ball_contact_reward: 4.0
ball_unstable_contact_penalty: -8.0
height_progress: 10.0
stop_scale: 1.5
zero_ang_bonus: 6.0
orientation: -0.03
lin_vel_z: -0.12
ang_vel_xy: -0.03
torques: -1e-5
dof_vel: -5e-5
dof_acc: -2.5e-7
action_rate: -0.01
score_clear_factor: 0.3
termination: -120.0
```

**Budget audit**: Standing budget is constrained; completion path combines arrival + milestones + shaping and now dominates standing.

## PPO Hyperparameters

| Parameter | Value |
|-----------|-------|
| learning_rate | 2.5e-4 |
| lr_scheduler | — |
| rollouts | 28 |
| learning_epochs | 8 |
| mini_batches | 32 |
| entropy_loss_scale | 0.006 |
| ratio_clip | 0.2 |
| max_env_steps | 150M |
| discount_factor | 0.99 |
| policy_net | (256,128,64) |
| value_net | (256,128,64) |

## Curriculum Stage

```
Stage 2C: Section 013 (gold balls + steep ramp + high step)
├── Environment: vbot_navigation_section013
├── Warm-start: Stage 2B (section012) best checkpoint, optimizer reset
├── LR: × 0.3 of section012 LR (prevent catastrophic forgetting)
├── Steps: 30-50M
├── Goal: Navigate past step, climb ramp, pass through gold ball gaps, reach final platform
```

## Terrain Traversal Strategies

### 0.75m High Step

- **Physical feasibility**: Robot is ~0.35m tall — the step is 2.14× its height. Direct step-up may be impossible.
- **Alternative**: The 21.8° ramp is adjacent — may be the intended path to bypass the step.
- **Investigation needed**: VLM analysis to confirm whether step or ramp is navigable.

### 21.8° Steep Ramp

- **Steeper than Section 011's 15°**: Requires more aggressive forward lean.
- **Height progress reward**: Scale higher than Section 011 (more climbing effort per meter).
- **Orientation penalty**: Must be relaxed further for 21.8° (body tilt = 22°).

### Gold Ball Stable Traversal

- **Gaps at x ≈ {-1.5, 1.5}**: provide robust pass channels; keep `ball_gap_alignment`.
- **Controlled contact allowed**: Section3 rewards stable pass with contact higher (15 > 10).
- **Reward principle**: reward stable contact, penalize unstable contact; do not treat all contact as failure.
- **Primary objective**: pass the ball zone stably and continue to final platform.

## Predicted Exploits

| Exploit | Description | Prevention |
|---------|-------------|------------|
| **Step-base camper** | Robot stands before the 0.75m step | Y-axis milestones + large arrival bonus |
| **Ramp-avoiding idle** | Robot stays on entry platform | forward_velocity + conditional alive_bonus |
| **Ball-zone avoider** | Robot stops before gold balls to avoid contact | Increase stable-contact upside and keep unstable-contact penalty |
| **Gap camping** | Robot sits in gap between balls | Arrival bonus must dominate passive rewards |

## Key Files

| File | Purpose |
|------|---------|
| `starter_kit/navigation2/vbot/cfg.py` | Section013 config + reward scales (`VBotSection013EnvCfg`) |
| `starter_kit/navigation2/vbot/vbot_section013_np.py` | Section 03 environment implementation |
| `starter_kit/navigation2/vbot/rl_cfgs.py` | Section013 PPO hyperparameters |
| `starter_kit/navigation2/vbot/xmls/scene_section013.xml` | Section 03 MJCF scene |
| `starter_kit/navigation2/vbot/xmls/0126_C_section03.xml` | Section 03 collision model |
