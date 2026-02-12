# Section 013 Task Reference — Gold Balls + Steep Ramp + High Step

> **This file contains task-specific concrete values** for Section 013 (Stage 2C — high step, steep ramp, gold ball obstacles).
> For abstract methodology, see `.github/copilot-instructions.md` and `.github/skills/`.
> For full-course reference, see `starter_kit_docs/navigation2/long_course/Task_Reference.md`.

---

## Environment ID

| Environment ID | Terrain | Status |
|----------------|---------|--------|
| `vbot_navigation_section013` | Section03: entry → 0.75m step → 21.8° ramp → hfield → 3 gold balls → final platform | **PLANNED** — default config, not yet trained |

## Competition Scoring — Section 3 (25 pts total)

Source: `MotrixArena_S1_计分规则讲解.md`

```
Section 3 (25 pts):
├── Rolling balls traversal: 10-15 pts (navigate past 3 gold balls)
├── Random terrain: 5 pts
└── Final celebration: 5 pts
```

**Note**: Exact scoring zone positions for Section 3 need to be extracted from scene XML / OBJ files. This is a TODO.

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

```python
position_tracking: 1.5
fine_position_tracking: 5.0
heading_tracking: 0.8
forward_velocity: 1.5
distance_progress: 2.0
alive_bonus: 0.3              # ⚠️ BROKEN — 0.3×5000=1500 >> arrival(60)
approach_scale: 8.0
arrival_bonus: 60.0            # ⚠️ Too low relative to alive budget
stop_scale: 1.5
zero_ang_bonus: 6.0
orientation: -0.05
lin_vel_z: -0.3
ang_vel_xy: -0.03
torques: -1e-5
dof_vel: -5e-5
dof_acc: -2.5e-7
action_rate: -0.01
termination: -200.0
```

**Budget audit**: Standing (1,800+) >> Completing (60). Ratio 25:1. **Lazy robot guaranteed.**

### TODO: Fix Needed

Recommended targets:
- `alive_bonus: 0.05` → budget = 0.05 × 5000 = 250
- `arrival_bonus: 150.0` → exceeds alive budget
- Add height_progress (steeper = higher scale than Section 011)
- Add step traversal milestone bonus
- Add gold ball gap navigation reward
- Add contact penalty for gold ball collision

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

### Gold Ball Avoidance

- **Gaps at x ≈ {-1.5, 1.5}**: ~2.5m ball-to-ball, usable gap ~1.0m after subtracting radii.
- **Observation extension**: Consider adding ball positions to obs (if visible in simulation).
- **Prefer edges**: Gaps near x=±1.5 have more clearance.
- **Pause & proceed**: Wait for safe window (if balls roll — need to verify in simulation).

## Predicted Exploits

| Exploit | Description | Prevention |
|---------|-------------|------------|
| **Step-base camper** | Robot stands before the 0.75m step | Y-axis milestones + large arrival bonus |
| **Ramp-avoiding idle** | Robot stays on entry platform | forward_velocity + conditional alive_bonus |
| **Ball-zone avoider** | Robot stops before gold balls | Balance collision penalty vs forward progress |
| **Gap camping** | Robot sits in gap between balls | Arrival bonus must dominate passive rewards |

## Key Files

| File | Purpose |
|------|---------|
| `starter_kit/navigation2/vbot/cfg.py` | Section013 config + reward scales (`VBotSection013EnvCfg`) |
| `starter_kit/navigation2/vbot/vbot_section013_np.py` | Section 03 environment implementation |
| `starter_kit/navigation2/vbot/rl_cfgs.py` | Section013 PPO hyperparameters |
| `starter_kit/navigation2/vbot/xmls/scene_section013.xml` | Section 03 MJCF scene |
| `starter_kit/navigation2/vbot/xmls/0126_C_section03.xml` | Section 03 collision model |
