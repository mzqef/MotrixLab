# Section 012 Task Reference — Stairs + Bridge + Spheres + Cones

> **This file contains task-specific concrete values** for Section 012 (Stage 2B — stairs, arch bridge, sphere/cone obstacles).
> For abstract methodology, see `.github/copilot-instructions.md` and `.github/skills/`.
> For full-course reference, see `starter_kit_docs/navigation2/long_course/Task_Reference.md`.

---

## Environment ID

| Environment ID | Terrain | Status |
|----------------|---------|--------|
| `vbot_navigation_section012` | Section02: stairs (left/right) → bridge/spheres → stairs down → exit | **PLANNED** — default config, not yet trained |

## Competition Scoring — Section 2 (60 pts total)

Source: `MotrixArena_S1_计分规则讲解.md`

```
Section 2 (60 pts):
├── Wave terrain traversal: 8-12 pts
├── Stairs completion: 15-20 pts
├── Bridge/Riverbed crossing: 10-15 pts
└── Red packets (scattered): 6-12 pts
```

**Note**: Exact scoring zone positions for Section 2 need to be extracted from scene XML / OBJ files (similar to how Section 1 smileys/red packets were identified). This is a TODO.

## Terrain Description — Section 02

### Overview

Two routes lead from the Section 01 high platform (z=1.294) to the exit platform (z=1.294):

```
Y: 8.8   12.4  14.2  15~20  21.4  23.2  24.3
    |--entry--|--stairs up--|--bridge/spheres--|--stairs down--|--exit--|
    z=1.294   z→2.79         z≈2.86              z→1.37        z=1.294
```

### Left Route — Steep Stairs + Arch Bridge

| Element | Center/Range | Key Stats | Notes |
|---------|-------------|-----------|-------|
| Left stairs up (10 steps) | x=-3.0, y=12.43→14.23 | ΔZ≈0.15/step, z: 1.369→2.794 | Steep — higher per-step clearance needed |
| Arch bridge | x≈-3.0, y=15.31→20.33 | 23 segments, peak z≈2.86, width ~2.64m | Narrow with railings |
| Bridge supports | 4 cylindrical pillars (R=0.4) | 4 platform bases | Below bridge — not traversable |
| Left stairs down (10 steps) | x=-3.0, y=21.4→23.2 | ΔZ≈0.15/step, z: 2.794→1.369 | Descending — balance challenge |

### Right Route — Gentle Stairs + Obstacles

| Element | Center/Range | Key Stats | Notes |
|---------|-------------|-----------|-------|
| Right stairs up (10 steps) | x=2.0 | ΔZ≈0.10/step, z: 1.319→2.294 | Gentler slope than left |
| 5 spheres | R=0.75, y=15.8-19.7 | z=0.8-1.2, scattered | Path blocking obstacles |
| 8 cones (STL mesh) | Scattered | Variable positions | Smaller obstacles |
| Right stairs down (10 steps) | x=2.0 | ΔZ≈0.10/step, z: 2.294→1.319 | Gentler descent |

### Key Terrain Parameters

| Parameter | Value |
|-----------|-------|
| Entry platform z | 1.294 |
| Left stair step height | ΔZ≈0.15m per step |
| Right stair step height | ΔZ≈0.10m per step |
| Left stair top z | 2.794 |
| Right stair top z | 2.294 |
| Bridge peak z | ~2.86 |
| Bridge width | ~2.64m |
| Sphere radius | 0.75m |
| Exit platform z | ~1.294 |
| Exit platform center | (0, 24.33) |

**Robot spawn**: (0, 9.5, 1.8), ±0.3m randomization. **Target**: (0, 24.0, 1.294). Distance: ~14.5m.

## Current Reward Config

```python
position_tracking: 1.5
fine_position_tracking: 5.0
heading_tracking: 0.8
forward_velocity: 1.5
distance_progress: 2.0
alive_bonus: 0.3              # ⚠️ BROKEN — 0.3×6000=1800 >> arrival(80)
approach_scale: 8.0
arrival_bonus: 80.0            # ⚠️ Too low relative to alive budget
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

**Budget audit**: Standing (2,200+) >> Completing (80). Ratio 27:1. **Lazy robot guaranteed.**

### TODO: Fix Needed

Recommended targets:
- `alive_bonus: 0.05` → budget = 0.05 × 6000 = 300
- `arrival_bonus: 200.0` → exceeds alive budget
- Add stair milestone bonuses (per-stair or mid-stair checkpoints)
- Add bridge crossing bonus
- Add height_progress reward for stair climbing
- Add knee_lift bonus on stairs
- Add lateral stability on bridge

## PPO Hyperparameters

| Parameter | Value |
|-----------|-------|
| learning_rate | 2e-4 |
| lr_scheduler | — |
| rollouts | 32 |
| learning_epochs | 8 |
| mini_batches | 32 |
| entropy_loss_scale | 0.008 |
| ratio_clip | 0.2 |
| max_env_steps | 200M |
| discount_factor | 0.99 |
| policy_net | (256,128,64) |
| value_net | (256,128,64) |

## Curriculum Stage

```
Stage 2B: Section 012 (stairs + bridge + obstacles)
├── Environment: vbot_navigation_section012
├── Warm-start: Stage 2A (section011) best checkpoint, optimizer reset
├── LR: × 0.3 of section011 LR (prevent catastrophic forgetting)
├── Steps: 40-80M (hardest section, 60 pts)
├── Goal: Forward progress > 10m (past stairs), reach exit platform
```

## Terrain Traversal Strategies

### Stairs

- **Key challenge**: Step height clearance, balance, edge detection
- **Higher knee lift**: Increase calf joint flexion during ascent
- **Slower velocity**: Stability over speed on stairs
- **Left vs Right**: Left route is harder (ΔZ=0.15) but avoids sphere obstacles; right is gentler (ΔZ=0.10) but has 5 spheres

### Arch Bridge

- **Narrow traversal**: Only ~2.64m wide, with railings
- **Lateral stability**: Tight hip abduction control needed
- **Height awareness**: Peak z≈2.86, robot must maintain balance at elevation

### Sphere/Cone Obstacles

- **Contact penalty**: Penalize collision with obstacle bodies
- **Path planning**: 5 spheres (R=0.75m) scattered — need to navigate between them
- **Right route advantage**: Gentler stairs but obstacle avoidance required

## Predicted Exploits

| Exploit | Description | Prevention |
|---------|-------------|------------|
| **Stair-base camper** | Robot stands at stair base, collects heading/position rewards | Conditional alive_bonus, Y-axis checkpoints |
| **Bridge bouncer** | Robot bounces between bridge start/end | Step-delta with no-retreat clip |
| **Landing zone farmer** | Robot stays on entry platform | Large arrival bonus at exit platform |
| **Route confusion** | Robot oscillates between left and right routes | Let policy discover best route; don't bias waypoints |

## Key Files

| File | Purpose |
|------|---------|
| `starter_kit/navigation2/vbot/cfg.py` | Section012 config + reward scales (`VBotSection012EnvCfg`) |
| `starter_kit/navigation2/vbot/vbot_section012_np.py` | Section 02 environment implementation |
| `starter_kit/navigation2/vbot/rl_cfgs.py` | Section012 PPO hyperparameters |
| `starter_kit/navigation2/vbot/xmls/scene_section012.xml` | Section 02 MJCF scene |
| `starter_kit/navigation2/vbot/xmls/0126_C_section02.xml` | Section 02 collision model |
