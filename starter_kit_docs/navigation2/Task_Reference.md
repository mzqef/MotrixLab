# Navigation2 Task Reference — VBot Obstacle Course

> **This file contains task-specific concrete values** for Navigation2 (Stage 2 — obstacles, stairs, full course).
> Skills reference this file instead of hardcoding values.
> For abstract methodology, see `.github/copilot-instructions.md` and `.github/skills/`.

---

## Environment IDs

| Environment ID | Terrain | Package |
|----------------|---------|---------|
| `vbot_navigation_section01` | Slopes + high platform | `starter_kit/navigation2/` |
| `vbot_navigation_section02` | Stairs + bridge + spheres + cones | `starter_kit/navigation2/` |
| `vbot_navigation_section03` | Gold balls + steep slope + high step | `starter_kit/navigation2/` |
| `vbot_navigation_stairs` | Stairs + platforms | `starter_kit/navigation2/` |
| `vbot_navigation_long_course` | Full 30m course (all sections) | `starter_kit/navigation2/` |

## Competition Scoring (Stage 2)

```
Stage 2 Section 1 (20 pts):
├── Base traverse: ~5 pts
├── Smiley circles: 2×2 + 2×4 = 12 pts
├── Red packets: 3×2 = 6 pts
└── Celebration dance: 2 pts

Stage 2 Section 2 (60 pts):
├── Wave terrain: 8-12 pts
├── Stairs: 15-20 pts
├── Bridge/Riverbed: 10-15 pts
└── Red packets: 6-12 pts (scattered)

Stage 2 Section 3 (25 pts):
├── Rolling balls: 10-15 pts (dynamic avoidance)
├── Random terrain: 5 pts
└── Final celebration: 5 pts

Total Stage 2: 105 pts max
```

### Scoring Tactics

| Element | Points | Strategy |
|---------|--------|----------|
| Smileys (Section 1) | 4-12 pts | Stop inside circle, stay **stable** 1-2 seconds |
| Red packets | 2 pts each | Slight detour, touch/pass through |
| Celebration zones | 2-5 pts | Execute motion at end zone |
| Section 2 stairs | 15-20 pts | Prioritize stability over speed |
| Rolling balls (Section 3) | 10-15 pts | Reactive avoidance, prefer edges |

## Terrain Descriptions (Detailed)

### Section 01 — Slopes + High Platform

| Element | Center (x, y, z) | Size | Top z | Notes |
|---------|-------------------|------|-------|-------|
| Starting platform | (0, -2.5, -0.25) | 5.0×1.0×0.25 box | 0 | Flat start |
| Central platform | (0, 0, -0.25) | 5.0×1.5×0.25 box | 0 | With hfield (±5m×±1.5m, h=0-0.277m) |
| **Ramp (15°)** | (0, 4.48, 0.41) | 5.0×2.5×0.25 box | ~0.66 | Tilted ~15° around x-axis |
| **High platform** | (0, 7.83, 1.04) | 5.0×1.0×0.25 box | **1.294** | Top of ramp |
| Boundary walls | x=±5.25 | 0.25m thick, 2.45m tall | — | Left/right/rear |

**Robot spawn**: y≈-2.4, z=0 → Target: y≈10.2. Total ~12.6m. Challenge: 15° upslope + high platform climb.

### Section 02 — Stairs + Bridge + Spheres + Cones

**Two routes from Section 01 high platform (z=1.294):**

#### Left Route (harder)
| Element | Details |
|---------|---------|
| **Left stairs up** (10 steps) | x=-3.0, ΔZ≈0.15/step, z: 1.37→2.79 |
| **Arch bridge** | 23 segments, peak z≈2.86, width ~2.64m, with railings |
| Bridge support | 4 cylindrical pillars (R=0.4), 4 platform bases |
| **Left stairs down** (10 steps) | x=-3.0, z: 2.79→1.37 |

#### Right Route (easier)
| Element | Details |
|---------|---------|
| **Right stairs up** (10 steps) | x=2.0, ΔZ≈0.10/step, z: 1.32→2.29 |
| **5 spheres** | R=0.75, scattered at y=15.8-19.7, z=0.8-1.2 |
| 8 cones (STL mesh) | Scattered obstacles |
| **Right stairs down** (10 steps) | x=2.0, z: 2.29→1.32 |

**End platform**: (0, 24.33, z≈1.294). Robot spawn: (-2.5, 12.0, 1.8).

### Section 03 — Gold Balls + Steep Slope + High Step

| Element | Center (x, y, z) | Size | Notes |
|---------|-------------------|------|-------|
| Entry platform | (0, 26.33, 1.044) | 5.0×1.0×0.25 box | z=1.294, from S02 |
| **High step** | (0, 27.58, 0.544) | 5.0×0.25×**0.75** box | 0.75m tall obstacle |
| **Steep ramp (21.8°)** | (0, 27.62, 1.301) | Tilted 21.8° | After high step |
| Middle platform | (0, 29.33, 0.794) | 5.0×1.5×0.5 box | z=1.294, with hfield |
| **3 gold balls** | x={-3, 0, 3}, y=31.23 | R=0.75 each | Blocking, spacing 3m |
| **End platform** | (0, 32.33, 0.994) | 5.0×1.5×0.5 box | **z=1.494** (highest) |
| End wall | (0, 34.33, 2.564) | Blocking wall | Course end |

Robot spawn: (0, 26.0, 1.8).

### Full Course Layout (Y-axis)

```
Y →  -3.5    0    4.5   7.8   10.3  12.4  14.2  15~20  21.4  23.2  24.3  26.3  27.6  29.3  31.2  32.3  34.3
      |----Section 01----|----high----|--------Section 02 stairs+bridge+spheres---------|----Section 03 ramp+balls---|
      z=0   z=0~1.29     z=1.29      z=1.29→2.79    z=2.5~2.86     z=2.79→1.29       z=1.29→1.49
```

| Section | Y Range | Z Range | Core Challenge |
|---------|---------|---------|----------------|
| **01** | -3.5 ~ 8.8 | 0 → 1.294 | 15° slope + high platform |
| **02** | 8.8 ~ 24.3 | 1.294 → 2.794 → 1.294 | Stairs (left steep/right gentle) + arch bridge + spheres + cones |
| **03** | 24.3 ~ 34.3 | 1.294 → 1.494 | 0.75m high step + 21.8° slope + 3 gold balls |

## Terrain Traversal Strategies

### Wave Terrain

- **Adaptive stride**: Shorter steps on downslopes, longer on upslopes
- **COM control**: Keep body low during transitions
- **Velocity modulation**: Slow down at wave peaks
- **Reward addition**: Height variance penalty + forward progress bonus on waves

### Stairs

- **Key challenge**: Step height clearance, balance, edge detection
- **Higher knee lift**: Increase calf joint flexion during ascent
- **Slower velocity**: Stability over speed
- **Reward additions**: Knee lift bonus (when terrain gradient > 0.1), foot slip penalty

### Rolling Balls (Section 3)

- **Include ball positions in observation** (if visible)
- **Prefer edges** over center
- **Pause & proceed**: Wait for safe window
- **Recovery training**: Train to recover from impacts

### Celebration Zones

- Smileys require **stopping** (not just passing through)
- Need low gyro readings (`< 0.5 rad/s`)
- Red packets are instant touch bonuses

## Curriculum Stages (Navigation2)

```
Stage 2A: Section 01 (slopes)
├── Environment: vbot_navigation_section01
├── Steps: 30M

Stage 2B: Stairs
├── Environment: vbot_navigation_stairs
├── Steps: 40M
├── Warm-start: Stage 2A best, LR × 0.3

Stage 2C: Section 03 (balls + slopes)
├── Environment: vbot_navigation_section03
├── Steps: 30M
├── Warm-start: Stage 2B best, LR × 0.3

Final: Full Course
├── Environment: vbot_navigation_long_course
├── Steps: 50M
├── Warm-start: Stage 2C best
├── Goal: End-to-end 30m navigation
```

## Advanced Reward Techniques (navigation2-Specific)

### Checkpoint Training

```python
def checkpoint_curriculum(checkpoints, robot_pos, reached_set):
    """Incremental rewards for reaching waypoints along the course."""
    reward = 0.0
    checkpoint_rewards = [5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5]
    for i, (cp_pos, cp_radius) in enumerate(checkpoints):
        if i not in reached_set:
            dist = np.linalg.norm(robot_pos[:2] - cp_pos)
            if dist < cp_radius:
                reward += checkpoint_rewards[i]
                reached_set.add(i)
    return reward
```

### Knee Lift Bonus (Stairs)

```python
if terrain_gradient > 0.1:  # Ascending
    knee_lift = -joint_pos['calf']
    reward += 0.2 * max(0, knee_lift - 1.5)
```

### Ball Avoidance Observation Extension

```python
ball_positions = get_ball_positions()   # [N, 3] relative
ball_velocities = get_ball_velocities() # [N, 3]
obs = np.concatenate([obs, ball_positions.flatten(), ball_velocities.flatten()])
```

### Celebration Zone Reward

```python
def celebration_reward(robot_pos, zones, gyro):
    for zone_pos, zone_radius, zone_type in zones:
        if np.linalg.norm(robot_pos[:2] - zone_pos) < zone_radius:
            if zone_type == "smiley" and np.linalg.norm(gyro) < 0.5:
                return 2.0 if small_smiley else 4.0
            elif zone_type == "red_packet":
                return 2.0
    return 0.0
```

## Key Files

| File | Purpose |
|------|---------|
| `starter_kit/navigation2/vbot/cfg.py` | Environment configs (Section 01-03, stairs, long course) |
| `starter_kit/navigation2/vbot/vbot_section*_np.py` | Environment implementations |
| `starter_kit/navigation2/vbot/xmls/` | MJCF scene files |
| `starter_kit/navigation2/vbot/xmls/scene_section01.xml` | Section 01 scene |
| `starter_kit/navigation2/vbot/xmls/scene_section02.xml` | Section 02 scene |
| `starter_kit/navigation2/vbot/xmls/scene_section03.xml` | Section 03 scene |
| `starter_kit/navigation2/vbot/xmls/0126_C_section0*.xml` | Collision model XMLs |
