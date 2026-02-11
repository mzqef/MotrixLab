# MotrixLab Project — Tutorial for Navigation2 (Obstacle Course)

Welcome! This is a **step-by-step walkthrough** of the Navigation2 task in MotrixLab — training a quadruped robot (VBot) to traverse a 30-meter obstacle course with slopes, stairs, bridges, and rolling balls. This tutorial builds on the Navigation1 foundation but focuses on the unique challenges of multi-terrain, multi-section navigation.

> **Prerequisite**: Read `starter_kit_docs/navigation1/Tutorial.md` first. This tutorial assumes familiarity with the MotrixLab framework, registry pattern, NpEnv base class, and basic RL training workflow.

---

## 1. Navigation2 vs Navigation1: What's Different?

| Aspect | Navigation1 (Stage 1) | Navigation2 (Stage 2) |
|--------|----------------------|----------------------|
| **Terrain** | Flat circular platform (R=12.5m) | 30m linear course with 3D terrain |
| **Navigation** | Radial approach to center (0,0) | Forward Y-axis traversal with waypoints |
| **Elevation** | z=0 everywhere | z=0 → 1.294 → 2.794 → 1.294 → 1.494 |
| **Obstacles** | None | Spheres, cones, gold balls, walls |
| **Environments** | 1 (`vbot_navigation_section001`) | 5 (section01/02/03, stairs, long_course) |
| **Scoring** | 20 pts (binary: reach inner fence + center) | 105 pts (checkpoints, smileys, red packets, dance) |
| **Episode length** | 1000 steps (10s) | 4000–9000 steps (40–90s) |
| **Difficulty** | Easy — flat, no obstacles | Hard — slopes, stairs, bridges, dynamic obstacles |

**Key insight**: Navigation2 is not just "Navigation1 with obstacles." The multi-terrain nature requires fundamentally different reward engineering, curriculum design, and policy architecture considerations.

---

## 2. Course Layout & Terrain Map

### Full Course Overview (Y-axis, ~34m total)

```
Y →  -3.5    0    4.5   7.8   10.3  12.4  15~20  21.4  24.3  27.6  29.3  31.2  32.3  34.3
      |----Section 01----|----high----|--------Section 02 (stairs+bridge)----------|--Section 03--|
      z=0   z=0~1.29     z=1.29      z=1.29→2.79    z≈2.86     z=2.79→1.29       z=1.29→1.49
```

### Section 01 — Slopes + High Platform (20 pts)

| Element | Location | Z-height | Challenge |
|---------|----------|----------|-----------|
| Starting flat | y=-3.5 ~ 0 | z=0 | Easy locomotion warm-up |
| Height field | y=0 ~ ±1.5m | z=0–0.277 | Mild undulation |
| 15° ramp | y≈4.48 | z=0.41–0.66 | Uphill walking |
| High platform | y≈7.83 | z=1.294 | Step up + flat platform |

**Robot spawn**: (0, -2.4, 0.5). **Target**: (0, 7.8, 1.294). Distance ≈12.6m.

### Section 02 — Stairs + Bridge + Obstacles (60 pts)

Two routes from Section 01's high platform:

| Route | Element | Key Stats | Difficulty |
|-------|---------|-----------|------------|
| **Left** (x=-3) | 10-step stairs UP | ΔZ≈0.15/step, z: 1.37→2.79 | Steep |
| **Left** | Arch bridge | 23 segments, peak z≈2.86, width ~2.64m | Narrow |
| **Left** | 10-step stairs DOWN | z: 2.79→1.37 | Balance |
| **Right** (x=+2) | 10-step stairs UP | ΔZ≈0.10/step, z: 1.32→2.29 | Moderate |
| **Right** | 5 spheres | R=0.75, scattered | Obstacle avoidance |
| **Right** | 8 cones | Scattered | Obstacle avoidance |
| **Right** | 10-step stairs DOWN | z: 2.29→1.32 | Balance |

**Robot spawn**: (0, 9.5, 1.8). **Target**: (0, 24.0, 1.294). Distance ≈14.5m.

### Section 03 — Gold Balls + Steep Ramp (25 pts)

| Element | Location | Key Stats | Challenge |
|---------|----------|-----------|-----------|
| Entry platform | y≈26.3 | z=1.294 | Flat transition |
| 0.75m high step | y≈27.6 | 0.75m wall | Major obstacle |
| 21.8° steep ramp | y≈27.6 | Tilted 21.8° | Steep climb |
| 3 gold balls | y≈31.2 | R=0.75, x={-3,0,3} | Path blocked |
| Final platform | y≈32.3 | z=1.494 | Course end |

**Robot spawn**: (0, 26.0, 1.8). **Target**: (0, 32.3, 1.494). Distance ≈6.3m.

---

## 3. Environment Architecture: 5 Environments, 1 Robot

### Why Multiple Environments?

Each section trains a different locomotion skill. Training all skills simultaneously on the long course is extremely hard (sparse reward, 90-second episodes). Instead, the curriculum approach trains each skill in isolation, then combines:

| Environment | Purpose | Key Skill |
|-------------|---------|-----------|
| `vbot_navigation_section011` | Section 01 — slopes | Slope climbing, platform transitions |
| `vbot_navigation_section012` | Section 02 — stairs/bridge | Stair climbing/descending, narrow traversal |
| `vbot_navigation_section013` | Section 03 — balls/ramp | Steep ramp, obstacle avoidance |
| `vbot_navigation_stairs` | Stairs only (deprecated) | Legacy environment |
| `vbot_navigation_long_course` | Full 30m course | End-to-end navigation with waypoints |

### Shared Architecture

All 5 environments share the same observation (54-dim) and action (12-dim) spaces, so a policy trained on one environment can be warm-started on another. This is critical for curriculum transfer.

```python
# All environments use identical obs/action spaces
observation_space = Box(shape=(54,), ...)  # Same as Navigation1
action_space = Box(shape=(12,), ...)       # 4 legs × 3 joints
```

### Waypoint System (Long Course Only)

The long course environment uses a multi-waypoint navigation system:

```python
# 7 waypoints guide the robot through all 3 sections
WAYPOINTS = [
    (0.0, 6.0),      # WP0: Section 01 exit
    (-3.0, 12.0),    # WP1: Left staircase entrance
    (-3.0, 15.0),    # WP2: Bridge start
    (-3.0, 20.5),    # WP3: Bridge end
    (-3.0, 23.0),    # WP4: Left staircase 2 bottom
    (0.0, 24.5),     # WP5: Section 02 exit
    (0.0, 32.3),     # WP6: Final platform (finish)
]
WAYPOINT_THRESHOLD = 1.5  # meters to trigger next waypoint
```

The robot's observation always points to the **current** waypoint. When it gets within 1.5m, the system auto-switches to the next waypoint. This keeps the observation space fixed at 54 dimensions.

---

## 4. Configuration Files: Where to Edit What

### 4.1 Environment Config (`starter_kit/navigation2/vbot/cfg.py`)

This is where you define:
- **Spawn position** and randomization range
- **Target position** (via `Commands.pose_command_range`)
- **Episode length** (`max_episode_steps`)
- **Reward scales** (`RewardConfig.scales`)
- **Action scale**, noise, and normalization

Each section has its own config class inheriting from `VBotStairsEnvCfg`:

```python
@registry.envcfg("vbot_navigation_section011")
@dataclass
class VBotSection011EnvCfg(VBotStairsEnvCfg):
    model_file: str = ".../scene_section011.xml"
    max_episode_steps: int = 4000
    
    @dataclass
    class RewardConfig:
        scales: dict = field(default_factory=lambda: {
            "forward_velocity": 1.5,
            "alive_bonus": 1.0,
            "arrival_bonus": 50.0,
            # ... more scales
        })
```

**Tip**: Always override `RewardConfig` in the specific section class, not in the base class. Navigation1 had a config drift bug where base class edits were silently lost.

### 4.2 RL Hyperparameters (`starter_kit/navigation2/vbot/rl_cfgs.py`)

PPO hyperparameters per section:

```python
@rlcfg("vbot_navigation_section011")
@dataclass
class VBotSection011PPOConfig(PPOCfg):
    learning_rate: float = 3e-4
    rollouts: int = 24
    learning_epochs: int = 8
    max_env_steps: int = 100_000_000
```

**Important**: Navigation2 sections use longer training horizons (100M–300M steps) because the terrain is much harder than flat Navigation1.

### 4.3 MJCF Scene Files (`starter_kit/navigation2/vbot/xmls/`)

| File | Content |
|------|---------|
| `scene_section011.xml` | Section 01 terrain (ramp + platform) |
| `scene_section012.xml` | Section 02 terrain (stairs + bridge + spheres) |
| `scene_section013.xml` | Section 03 terrain (wall + ramp + gold balls) |
| `scene_world_full.xml` | All 3 sections combined |
| `0126_C_section0*.xml` | Collision model definitions |

To understand terrain geometry, use the `mjcf-xml-reasoning` skill or read the XML files directly. Key terrain parameters (positions, sizes, angles) are documented in `Task_Reference.md`.

### 4.4 Environment Implementations (`vbot_section0*_np.py`)

Each file implements `reset()`, `apply_action()`, and `update_state()` for its section. The reward function lives in `update_state()`.

---

## 5. Training Workflow for Navigation2

### 5.1 Pre-Training Checklist

Before any training run:

```powershell
# 1. Read the experiment report
Get-Content starter_kit_docs/navigation2/REPORT_NAV2.md

# 2. Verify runtime config
uv run python -c "
from starter_kit.navigation2.vbot import cfg as _
from motrix_envs.registry import make
env = make('vbot_navigation_section011', num_envs=1)
s = env._cfg.reward_config.scales
print('alive:', s.get('alive_bonus'), '| arrival:', s.get('arrival_bonus'), '| term:', s.get('termination'))
print('fwd_vel:', s.get('forward_velocity'), '| max_steps:', env._cfg.max_episode_steps)
"

# 3. Check AutoML history
Get-ChildItem starter_kit_log/automl_* -Directory | ForEach-Object {
    Write-Host "=== $($_.Name) ==="
}

# 4. List recent training runs
Get-ChildItem runs/vbot_navigation_section011/ -Directory -ErrorAction SilentlyContinue |
    Sort-Object Name -Descending | Select-Object -First 5
```

### 5.2 Section-by-Section Training (Curriculum)

**Stage 2A — Section 011 (slopes)**:
```powershell
# Smoke test (verify code runs)
uv run scripts/train.py --env vbot_navigation_section011 --max-env-steps 200000

# Visual debug
uv run scripts/train.py --env vbot_navigation_section011 --render

# Full training (after reward fixes)
uv run scripts/train.py --env vbot_navigation_section011
```

**Stage 2B — Section 012 (stairs), warm-started**:
```powershell
# Warm-start from Section 011's best checkpoint
uv run scripts/train.py --env vbot_navigation_section012 \
    --checkpoint runs/vbot_navigation_section011/<best_run>/checkpoints/best_agent.pt
```

**Stage 2C — Section 013 (balls + ramp), warm-started**:
```powershell
uv run scripts/train.py --env vbot_navigation_section013 \
    --checkpoint runs/vbot_navigation_section012/<best_run>/checkpoints/best_agent.pt
```

**Final — Long Course**:
```powershell
uv run scripts/train.py --env vbot_navigation_long_course \
    --checkpoint runs/vbot_navigation_section013/<best_run>/checkpoints/best_agent.pt
```

### 5.3 AutoML Batch Search (Preferred Over Manual train.py)

For reward weight and hyperparameter tuning, always use AutoML:

```powershell
uv run starter_kit_schedule/scripts/automl.py --mode stage --budget-hours 8 --hp-trials 15

# Monitor progress
Get-Content starter_kit_schedule/progress/automl_state.yaml

# Read results
Get-Content starter_kit_log/automl_*/report.md
```

### 5.4 Evaluation

```powershell
# Evaluate latest policy
uv run scripts/play.py --env vbot_navigation_section011

# Evaluate specific checkpoint
uv run scripts/play.py --env vbot_navigation_section011 \
    --policy runs/vbot_navigation_section011/<run>/checkpoints/best_agent.pt

# VLM visual analysis (capture frames + send to gpt-4.1)
uv run scripts/capture_vlm.py --env vbot_navigation_section011 --max-frames 25
```

### 5.5 Monitoring with TensorBoard

```powershell
uv run tensorboard --logdir runs/vbot_navigation_section011
```

Key signals to watch:

| Signal | Healthy | Unhealthy |
|--------|---------|-----------|
| Reward ↑ AND forward_progress ↑ | ✅ Learning | — |
| Reward ↑ AND forward_progress flat | — | ❌ Reward hacking (passive rewards) |
| Episode length → max | — | ⚠️ Lazy robot |
| Episode length ↓ rapidly | — | ⚠️ Sprint-crash or too-harsh termination |

---

## 6. Terrain-Specific Reward Engineering

### 6.1 Slopes (Section 011)

The key challenge is climbing a 15° ramp and stepping up to a 1.294m platform. Standard flat-ground rewards underweight vertical progress.

**Considerations**:
- `forward_velocity` alone doesn't account for uphill effort
- Height progress (z-axis) should be rewarded separately
- Stability penalties may need loosening on slopes (body tilts naturally)
- Platform edge transition is a fall risk — need termination penalty tuned carefully

### 6.2 Stairs (Section 012)

Stair climbing is the hardest terrain challenge. Each step requires precise foot placement and clearance.

**Considerations**:
- Knee lift bonus: reward higher calf joint flexion when terrain gradient > 0.1
- Foot slip penalty: penalize lateral foot movement during stance phase
- Slower target speed: stairs are about stability, not speed
- Bridge traversal: narrow (2.64m), needs lateral stability (tight hip abduction)

### 6.3 Obstacle Avoidance (Sections 012, 013)

Spheres (R=0.75m) and gold balls block paths. The policy needs to learn detour behavior.

**Considerations**:
- Obstacle positions could be added to observation (if observable)
- Contact penalty: penalize collisions with obstacle bodies
- Edge preference: reward paths along course boundaries
- Gold balls have 2.5m gaps — robot must navigate precisely between them

### 6.4 Long Course Waypoint Navigation

The long course requires chaining skills across all terrain types.

**Considerations**:
- Waypoint bonus structure: progressive bonuses (later waypoints = higher reward)
- Distance-to-current-waypoint as continuous signal
- Route selection: policy must discover left vs right path; reward shouldn't bias
- Time pressure: 90 seconds for 34m = need ~0.38 m/s average speed

---

## 7. Key Code Patterns for Navigation2

### 7.1 Checking Which Section Uses Which Scene

```python
# In cfg.py, each section points to its XML:
class VBotSection011EnvCfg(VBotStairsEnvCfg):
    model_file = ".../xmls/scene_section011.xml"

class VBotSection012EnvCfg(VBotStairsEnvCfg):
    model_file = ".../xmls/scene_section012.xml"

class VBotSection013EnvCfg(VBotStairsEnvCfg):
    model_file = ".../xmls/scene_section013.xml"

class VBotLongCourseEnvCfg(VBotStairsEnvCfg):
    model_file = ".../xmls/scene_world_full.xml"  # All combined
```

### 7.2 Config Verification One-Liner

```powershell
# Quick config check for any NAV2 environment
uv run python -c "
from starter_kit.navigation2.vbot import cfg as _
from motrix_envs.registry import make
for env_name in ['vbot_navigation_section011', 'vbot_navigation_section012', 'vbot_navigation_section013', 'vbot_navigation_long_course']:
    try:
        env = make(env_name, num_envs=1)
        s = env._cfg.reward_config.scales
        print(f'{env_name}: alive={s.get(\"alive_bonus\", \"?\")} arrival={s.get(\"arrival_bonus\", \"?\")} term={s.get(\"termination\", \"?\")} max_steps={env._cfg.max_episode_steps}')
    except Exception as e:
        print(f'{env_name}: ERROR {e}')
"
```

### 7.3 Reward Budget Audit Script

```powershell
# Run BEFORE every training launch
uv run python -c "
from starter_kit.navigation2.vbot import cfg as _
from motrix_envs.registry import make
for env_name in ['vbot_navigation_section011', 'vbot_navigation_section012', 'vbot_navigation_section013']:
    env = make(env_name, num_envs=1)
    cfg = env._cfg
    s = cfg.reward_config.scales
    max_steps = cfg.max_episode_steps
    alive = s.get('alive_bonus', 0) * max_steps
    arrival = s.get('arrival_bonus', 0)
    death = s.get('termination', 0)
    ratio = alive / max(arrival, 0.01)
    print(f'{env_name}:')
    print(f'  alive_budget={alive:.0f}  arrival={arrival:.0f}  ratio={ratio:.1f}:1  death={death}')
    if ratio > 5:
        print(f'  ⚠️ WARNING: Lazy robot likely! alive >> arrival')
    print()
"
```

---

## 8. Debugging Tips for Navigation2

### 8.1 Common Failure Modes

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| Robot stands still | alive_bonus dominates arrival (lazy robot) | Reduce alive_bonus, increase arrival, shorten episode |
| Robot sprints and crashes | forward_velocity too high + weak termination | Speed-distance coupling, stronger termination |
| Robot falls on slopes | Stability penalties too tight for tilted body | Loosen orientation penalty on slopes |
| Robot can't climb stairs | Insufficient knee lift | Add knee lift bonus when ascending |
| Robot gets stuck at platform edge | No height progress signal | Add z-axis progress reward |
| Robot oscillates near obstacles | Approach/retreat signal conflict | Use step-delta with asymmetric clip |
| Robot takes wrong route | Waypoint path biases one side | Check waypoint positions, ensure neutral reward |

### 8.2 VLM Visual Analysis

Always use VLM analysis before and after reward changes:

```powershell
# Before changing rewards — see current behavior
uv run scripts/capture_vlm.py --env vbot_navigation_section011 --max-frames 20

# After changing rewards — verify improvement
uv run scripts/capture_vlm.py --env vbot_navigation_section011 --max-frames 20 \
    --vlm-prompt "Compare this behavior to expectation: robot should climb the 15° slope"
```

### 8.3 Terrain Geometry Verification

If the robot seems to clip through terrain or float above it:

```powershell
# View the environment with random actions
uv run scripts/view.py --env vbot_navigation_section011

# Analyze the MJCF scene file
# Use mjcf-xml-reasoning skill or read directly
```

---

## 9. Competition Scoring Strategy

### 9.1 Point Breakdown

| Section | Points | Strategy Priority |
|---------|--------|-------------------|
| Section 1 | 20 pts | Medium — slopes are manageable, good warm-up |
| Section 2 | **60 pts** | **HIGH** — most points, hardest terrain |
| Section 3 | 25 pts | Medium — steep ramp + balls, moderate difficulty |

### 9.2 Scoring Elements

| Element | Points | How to Score |
|---------|--------|-------------|
| **Smiley circles** | 2-4 pts each | Stop inside, stay stable 1-2 seconds |
| **Red packets** | 2 pts each | Touch/pass through |
| **Celebration zones** | 2-5 pts | Execute motion at end zone |
| **Stair completion** | 15-20 pts | Traverse stairs successfully |
| **Bridge crossing** | 10-15 pts | Cross arch bridge |
| **Rolling ball avoidance** | 10-15 pts | Navigate past gold balls |

### 9.3 Risk vs Reward

> **⚠️ CRITICAL** (inherited from Navigation1): Stability > Speed. Falling at any point forfeits ALL points for that section's attempt. A conservative policy that never falls beats a fast one that falls once.

For the competition:
- **Section 1** (20 pts): Conservative approach — slow, stable slope climbing
- **Section 2** (60 pts): This section alone is worth 3× Section 1 — invest the most training time here
- **Section 3** (25 pts): Ball avoidance requires either observation extension or reactive behavior

---

## 10. Advanced Topics

### 10.1 Observation Extension for Dynamic Obstacles

The standard 54-dim observation doesn't include obstacle positions. For Section 03's gold balls, consider:

```python
# Extend observation with ball positions (relative to robot)
ball_positions = get_ball_positions()   # [N, 3] relative
ball_velocities = get_ball_velocities() # [N, 3]
extended_obs = np.concatenate([base_obs, ball_positions.flatten(), ball_velocities.flatten()])
```

**Tradeoff**: Changes observation dimension → policy cannot be directly warm-started from standard environments. May need adapter layer or train from scratch.

### 10.2 Terrain-Aware Stability Penalties

On slopes and stairs, the robot's body naturally tilts. Standard orientation penalties can over-punish correct behavior:

```python
# Standard: penalize any tilt
orientation_penalty = -0.05 * (roll² + pitch²)

# Terrain-aware: penalize deviation from terrain normal
terrain_normal = estimate_terrain_slope(robot_pos)
deviation_from_terrain = compute_angle(body_up, terrain_normal)
orientation_penalty = -0.05 * deviation_from_terrain²
```

### 10.3 Multi-Policy Switching

An advanced strategy: train separate policies for each terrain type, then switch at section boundaries:

```
Policy A (flat + slope)  →  Policy B (stairs + bridge)  →  Policy C (ramp + balls)
```

This avoids catastrophic forgetting but requires reliable terrain detection for switching.

---

## 11. File Reference

| File | Purpose |
|------|---------|
| `starter_kit/navigation2/vbot/cfg.py` | All environment configs + reward scales |
| `starter_kit/navigation2/vbot/rl_cfgs.py` | PPO hyperparameters per section |
| `starter_kit/navigation2/vbot/vbot_section011_np.py` | Section 01 environment implementation |
| `starter_kit/navigation2/vbot/vbot_section012_np.py` | Section 02 environment implementation |
| `starter_kit/navigation2/vbot/vbot_section013_np.py` | Section 03 environment implementation |
| `starter_kit/navigation2/vbot/vbot_long_course_np.py` | Long course environment (waypoint system) |
| `starter_kit/navigation2/vbot/xmls/` | MJCF scene files |
| `starter_kit_docs/navigation2/Task_Reference.md` | Terrain geometry, scoring, curriculum stages |
| `starter_kit_docs/navigation2/REPORT_NAV2.md` | Experiment history (append-only) |
| `starter_kit_docs/navigation2/Tutorial_RL_Reward_Engineering.md` | Reward engineering guide for NAV2 |
