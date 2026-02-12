# Section 011 Task Reference — Slopes + Multi-Waypoint + Celebration Spin

> **This file contains task-specific concrete values** for Section 011 (Stage 2A — slopes, height field, ramp, high platform).
> For abstract methodology, see `.github/copilot-instructions.md` and `.github/skills/`.
> For full-course reference, see `starter_kit_docs/navigation2/long_course/Task_Reference.md`.

---

## Environment ID

| Environment ID | Terrain | Status |
|----------------|---------|--------|
| `vbot_navigation_section011` | Section01: START→hfield→ramp→platform (spawn y=-2.5) | **ACTIVE** — v3 multi-waypoint + celebration spin |

## Competition Scoring — Section 1 (20 pts total)

Source: `MotrixArena_S1_计分规则讲解.md`

```
Section 1 (20 pts):
├── 3 × 笑脸区 (smiley zones)     = 3×4 = 12 pts
│   ├── LEFT  smiley: center=(-3, 0), OBJ=V_ScorePoint_001a.obj
│   ├── CENTER smiley: center=( 0, 0), OBJ=V_ScorePoint_001b.obj
│   └── RIGHT smiley: center=( 3, 0), OBJ=V_ScorePoint_001c.obj
│   (位于height field凹凸地形上, y≈0处, 机器人经过即可得分)
├── 3 × 红包区 (red packet zones)  = 3×2 = 6 pts
│   ├── LEFT  red pkt: center=(-3, 4.4), OBJ=V_ScorePoint_002a.obj
│   ├── CENTER red pkt: center=( 0, 4.4), OBJ=V_ScorePoint_002b.obj
│   └── RIGHT red pkt: center=( 3, 4.4), OBJ=V_ScorePoint_002c.obj
│   (悬浮在"GO"字样上, 位于15°坡道上, y≈4.4处)
└── 庆祝动作 (celebration)         = 2 pts
    (在"2026"平台=高台顶部做出庆祝动作)
```

**关键规则:**
- 笑脸区: "坑洼地形的趣味得分区有三个笑脸区域, 机器人经过笑脸时每经过一个+4分"
- 红包区: "斜坡地形的趣味得分区有三个悬浮在'GO'字样上的祝福红包, 机器人经过一个红包得分+2分"
- 庆祝: "在'2026'平台处做出庆祝动作+2分"
- 起点: "初始点位置随机分布在'START'平台区域" (Adiban_001, y∈[-3.5, -1.5])

### Scoring Zone Positions (from OBJ mesh vertex extraction)

| Zone | OBJ File | XY Center | Bounding Box | Detection Radius |
|------|----------|-----------|--------------|------------------|
| Smiley LEFT | V_ScorePoint_001a.obj | (-3, 0) | x∈[-4,-2], y∈[-1,1] | 1.2m |
| Smiley CENTER | V_ScorePoint_001b.obj | (0, 0) | x∈[-1,1], y∈[-1,1] | 1.2m |
| Smiley RIGHT | V_ScorePoint_001c.obj | (3, 0) | x∈[2,4], y∈[-1,1] | 1.2m |
| Red Pkt LEFT | V_ScorePoint_002a.obj | (-3, 4.4) | x∈[-4,-2], y∈[3.4,5.4] | 1.2m |
| Red Pkt CENTER | V_ScorePoint_002b.obj | (0, 4.4) | x∈[-1,1], y∈[3.4,5.4] | 1.2m |
| Red Pkt RIGHT | V_ScorePoint_002c.obj | (3, 4.4) | x∈[2,4], y∈[3.4,5.4] | 1.2m |
| Celebration | High platform (Adiban_004) | (0, 7.83) | x∈[-2.5,2.5], z>1.0 | 1.5m |

### Scoring Tactics

| Element | Points | Location | Strategy |
|---------|--------|----------|----------|
| Smileys | 4 pts each (×3=12) | height field at y≈0, x=-3/0/3 | Robot passes through each zone (radius~1.2m) |
| Red packets | 2 pts each (×3=6) | Ramp at y≈4.4, x=-3/0/3 | Touch/pass through on way up ramp |
| Celebration | 2 pts | High platform y≈7.83, z>1.0 | Perform celebration spin at platform top |

## Terrain Description — Section 01

| Element | Center (x, y, z) | Size | Top z | Notes |
|---------|-------------------|------|-------|-------|
| Starting platform | (0, -2.5, -0.25) | 5.0×1.0×0.25 box | 0 | Flat start |
| Central platform | (0, 0, -0.25) | 5.0×1.5×0.25 box | 0 | With hfield (±5m×±1.5m, h=0-0.277m) |
| **Ramp (15°)** | (0, 4.48, 0.41) | 5.0×2.5×0.25 box | ~0.66 | Tilted ~15° around x-axis |
| **High platform** | (0, 7.83, 1.04) | 5.0×1.0×0.25 box | **1.294** | Top of ramp |
| Boundary walls | x=±5.25 | 0.25m thick, 2.45m tall | — | Left/right/rear |

**Robot spawn (section011 CURRENT)**: y=-2.5, z=0.5 (START platform, competition-correct spawn). Target: 3 waypoints → celebration. Distance: ~10.3m.

Challenge: height field bumps (max 0.277m at y≈0) + 3 smiley zones + 15° upslope + 3 red packet zones + platform edge transition + celebration at top.

## Multi-Waypoint Navigation System

```python
# 3 waypoints guide the robot through Section 01 scoring zones
waypoints = [[0.0, 0.0], [0.0, 4.4], [0.0, 7.83]]  # smiley → red_packet → platform
waypoint_radius = 1.0   # auto-advance when within this distance
final_radius = 0.5      # platform target requires more precision

# After reaching WP2 (platform), celebration spin activates:
# CELEB_SPIN_RIGHT(180°) → CELEB_SPIN_LEFT(180°) → CELEB_HOLD(30 steps) → CELEB_DONE
# Episode truncates when CELEB_DONE is reached (success!)
```

## Celebration Spin State Machine

```
CELEB_IDLE(0) → CELEB_SPIN_RIGHT(1) → CELEB_SPIN_LEFT(2) → CELEB_HOLD(3) → CELEB_DONE(4)
```

| Parameter | Value |
|-----------|-------|
| Spin angle per phase | 180° (π radians) |
| Heading tolerance | 0.3 rad (≈17°) |
| Speed limit during spin | 0.3 m/s |
| Hold duration | 30 steps |

**Observation**: dim 53 encodes celebration progress:
- 0.0 = navigating, 0.25 = spin_right, 0.5 = spin_left, 0.75 = hold, 1.0 = done

## Terrain Traversal Strategy — Slopes

- **Height progress reward**: Reward z-axis gain (climbing) separately from Y-axis forward progress
- **Slope-aware orientation**: Don't penalize pitch on 15° slope — robot SHOULD lean forward
- **Platform edge**: Fall risk at ramp→platform transition; tilt termination at 65°

## Curriculum Stage

```
Stage 2A: Section 011 (slopes + multi-waypoint + celebration spin)
├── Environment: vbot_navigation_section011
├── Architecture: 3-waypoint navigation (smiley→red_packet→platform) + celebration spin
├── Spawn: y=-2.5 (START platform), target: 3 waypoints + celebration
├── Steps: 50M
├── Warm-start: Nav1 best checkpoint, optimizer reset, LR=1.5e-4
├── Goal: Navigate all waypoints, climb ramp, complete celebration spin on platform
```

## Current Reward Config (v3: Multi-Waypoint + Celebration Spin)

```python
position_tracking: 1.5
fine_position_tracking: 8.0   # exp(-d/0.5) gated by ever_reached
heading_tracking: 0.8
forward_velocity: 3.5
distance_progress: 2.0
alive_bonus: 0.05              # Anti-lazy: 0.05×3000=150 < arrival(160)
approach_scale: 50.0
arrival_bonus: 160.0
stop_scale: 6.0
zero_ang_bonus: 6.0
near_target_speed: -0.5
departure_penalty: -5.0
height_progress: 8.0           # z-delta climbing reward
traversal_bonus: 15.0          # milestone bonuses (×2)
smiley_bonus: 20.0             # 3×20=60 potential (pass-through)
red_packet_bonus: 10.0         # 3×10=30 potential (pass-through)
celebration_bonus: 30.0        # One-time: celebration spin completed
waypoint_bonus: 25.0           # 3×25=75 one-time per waypoint
waypoint_approach: 40.0        # step-delta toward current waypoint
waypoint_facing: 0.6           # face current waypoint
spin_progress: 3.0             # continuous: progress toward target heading
spin_hold: 5.0                 # reward stillness in HOLD phase
swing_contact_penalty: -0.15   # High-speed foot-ground contact
orientation: -0.03
lin_vel_z: -0.15
ang_vel_xy: -0.02
torques: -1e-5
dof_vel: -5e-5
dof_acc: -2.5e-7
action_rate: -0.01
termination: -75.0
```

## PPO Hyperparameters

| Parameter | Value |
|-----------|-------|
| learning_rate | 1.5e-4 (warm-start) |
| lr_scheduler | linear |
| rollouts | 24 |
| learning_epochs | 6 |
| mini_batches | 32 |
| entropy_loss_scale | 0.012 |
| ratio_clip | 0.15 |
| max_env_steps | 100M |
| discount_factor | 0.99 |
| policy_net | (256,128,64) |
| value_net | (512,256,128) |

## Key Files

| File | Purpose |
|------|---------|
| `starter_kit/navigation2/vbot/cfg.py` | Environment config + reward scales for section011 |
| `starter_kit/navigation2/vbot/vbot_section011_np.py` | Section 01 environment implementation |
| `starter_kit/navigation2/vbot/xmls/scene_section011.xml` | Section 01 MJCF scene |
| `starter_kit/navigation2/vbot/xmls/0126_C_section01.xml` | Section 01 collision model |
