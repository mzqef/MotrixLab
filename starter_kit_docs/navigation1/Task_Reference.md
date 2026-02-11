# Navigation1 Task Reference — VBot Section001 (Flat Ground)

> **This file contains task-specific concrete values** for Navigation1 (Stage 1 — flat ground).
> Skills reference this file instead of hardcoding values.
> For abstract methodology, see `.github/copilot-instructions.md` and `.github/skills/`.

---

## Environment IDs

| Environment ID | Terrain | Package |
|----------------|---------|---------|
| `vbot_navigation_section001` | Flat ground (circular platform R=12.5m) | `starter_kit/navigation1/` |

## Competition Scoring (Stage 1)

| Element | Points | Details |
|---------|--------|---------|
| Stage 1 total | **20 pts** | 10 dogs × 2 pts each |
| Per-dog scoring | 2 pts | Inner fence (+1pt) + center (+1pt) |
| Failure penalty | **Lose ALL dog pts** | ANY single fall or out-of-bounds = both +1s lost |

> **⚠️ CRITICAL:** Stability > Speed. A conservative policy that never falls beats a fast one that falls once. A policy that gets 9/10 dogs fast but 1 falls = 18 pts (not 20).

## VBot Robot Architecture

### 12-DOF Quadruped Structure

```
VBot Kinematic Tree:
base (floating, 6 DOF via freejoint)
├── FR_hip → FR_thigh → FR_calf (3 DOF)
├── FL_hip → FL_thigh → FL_calf (3 DOF)
├── RR_hip → RR_thigh → RR_calf (3 DOF)
└── RL_hip → RL_thigh → RL_calf (3 DOF)
```

### Joint Configuration

| Joint Group | Name Pattern | Range (rad) | Function |
|-------------|--------------|-------------|----------|
| Hip Abduction/Adduction | `*_hip_joint` | ±0.6 ~ ±1.0 | Lateral stability |
| Hip Flexion/Extension | `*_thigh_joint` | 0.5 ~ 1.2 | Forward stride |
| Knee Flexion | `*_calf_joint` | -2.5 ~ -1.2 | Ground clearance |

### Default Standing Pose (radians)

```python
default_joint_angles = {
    "FR_hip_joint": 0.0,   "FR_thigh_joint": 0.9,   "FR_calf_joint": -1.8,
    "FL_hip_joint": 0.0,   "FL_thigh_joint": 0.9,   "FL_calf_joint": -1.8,
    "RR_hip_joint": 0.0,   "RR_thigh_joint": 0.9,   "RR_calf_joint": -1.8,
    "RL_hip_joint": 0.0,   "RL_thigh_joint": 0.9,   "RL_calf_joint": -1.8,
}
```

### Actuator Configuration

| Parameter | Value |
|-----------|-------|
| Control Mode | Position Servo (PD) |
| kp (stiffness) | 80.0 N·m/rad |
| kv (damping) | 6.0 N·m·s/rad |
| action_scale | 0.25 |
| Torque Limits | Hip/Thigh: ±17 N·m, Calf: ±34 N·m |

### Observation Space (54 dimensions)

```python
obs = [
    base_linear_velocity,      # 3D, normalized by 2.0
    base_angular_velocity,     # 3D (gyro), normalized by 0.25
    projected_gravity,         # 3D, gravity in body frame
    joint_positions_relative,  # 12D, normalized by 1.0
    joint_velocities,          # 12D, normalized by 0.05
    last_actions,              # 12D
    velocity_commands,         # 3D (vx, vy, yaw_rate)
    position_error,            # 2D (to target)
    heading_error,             # 1D
    distance_normalized,       # 1D
    reached_flag,              # 1D
    stop_ready_flag,           # 1D
]
```

### Action Space (12 dimensions)

```python
target_position = default_angles + action * action_scale  # action ∈ [-1, 1]^12
torque = kp * (target - current_pos) - kv * current_vel
```

## Terrain Description (Section 001)

| Element | Type | Center (x, y, z) | Size | Top z |
|---------|------|-------------------|------|-------|
| Platform | Cylinder | (0, 0, -1.0) | R=12.5m, half-height=1.0m | **z = 0** |

- Flat circular platform, no obstacles, no slopes.
- Robot spawn: (0, -2.4, 0.5) → Target: (0, 10.2, ?) — distance ~12.6m.

## Reward Scales (Current — see REPORT for history)

> **Source**: `starter_kit/navigation1/vbot/cfg.py` → `RewardConfig.scales`

```python
# === Navigation core ===
"position_tracking": 1.5,       # exp(-d/5.0)
"fine_position_tracking": 8.0,  # sigma=0.5, range<2.5m
"heading_tracking": 0.8,
"forward_velocity": 1.5,
"distance_progress": 1.5,       # linear: clip(1 - d/d_max, -0.5, 1.0)
"alive_bonus": 0.15,            # Per-step survival (always active — Round5)

# === Approach/arrival ===
"approach_scale": 30.0,          # Step-delta distance improvement
"arrival_bonus": 100.0,          # One-time on reaching target
"inner_fence_bonus": 40.0,       # One-time at d<0.75m
"stop_scale": 5.0,               # Precision stopping (speed-gated)
"zero_ang_bonus": 10.0,
"near_target_speed": -2.0,       # Quadratic speed-distance coupling
"boundary_penalty": -3.0,        # Platform edge safety

# === Stability ===
"orientation": -0.05,
"lin_vel_z": -0.3,
"ang_vel_xy": -0.03,
"torques": -1e-5,
"dof_vel": -5e-5,
"dof_acc": -2.5e-7,
"action_rate": -0.01,

# === Terminal ===
"termination": -100.0,
```

## Reward Search Space (AutoML)

> **Source**: `starter_kit_schedule/scripts/automl.py` → `REWARD_SEARCH_SPACE`

| Parameter | Type | Range | Default | Category |
|-----------|------|-------|---------|----------|
| `position_tracking` | uniform | 0.5 – 5.0 | 1.5 | Navigation core |
| `fine_position_tracking` | uniform | 2.0 – 12.0 | 8.0 | Navigation core |
| `heading_tracking` | uniform | 0.1 – 2.0 | 1.0 | Navigation core |
| `forward_velocity` | uniform | 1.0 – 2.5 | 1.5 | Navigation core |
| `distance_progress` | uniform | 0.5 – 5.0 | 1.5 | Navigation |
| `approach_scale` | uniform | 15.0 – 50.0 | 30.0 | Navigation |
| `arrival_bonus` | uniform | 50.0 – 250.0 | 100.0 | Navigation |
| `alive_bonus` | uniform | 0.05 – 0.5 | 0.15 | Navigation |
| `stop_scale` | uniform | 2.0 – 10.0 | 5.0 | Navigation |
| `zero_ang_bonus` | uniform | 2.0 – 15.0 | 10.0 | Navigation |
| `inner_fence_bonus` | uniform | 10.0 – 80.0 | 40.0 | Navigation |
| `near_target_speed` | uniform | -5.0 – -0.5 | -2.0 | Speed coupling |
| `boundary_penalty` | uniform | -5.0 – -0.5 | -3.0 | Edge safety |
| `orientation` | uniform | -0.3 – -0.01 | -0.05 | Stability |
| `lin_vel_z` | uniform | -2.0 – -0.1 | -0.3 | Stability |
| `ang_vel_xy` | uniform | -0.3 – -0.01 | -0.03 | Stability |
| `torques` | log-uniform | -1e-3 – -1e-6 | -1e-5 | Efficiency |
| `dof_vel` | log-uniform | -1e-3 – -1e-5 | -5e-5 | Efficiency |
| `dof_acc` | log-uniform | -1e-5 – -1e-8 | -2.5e-7 | Efficiency |
| `action_rate` | uniform | -0.05 – -0.001 | -0.01 | Smoothness |
| `termination` | uniform | -150 – -50 | -100.0 | Terminal |

## PPO Hyperparameters (VBotSection001PPOConfig)

```python
@dataclass
class VBotSection001PPOConfig(PPOCfg):
    policy_hidden_layer_sizes: tuple = (256, 128, 64)
    value_hidden_layer_sizes: tuple = (256, 128, 64)
    rollouts: int = 32
    learning_epochs: int = 5
    mini_batches: int = 16
    discount_factor: float = 0.99
    lambda_param: float = 0.95
    learning_rate: float = 5e-4
    lr_scheduler_type: str | None = "linear"
    ratio_clip: float = 0.2
    value_clip: float = 0.2
    entropy_loss_scale: float = 0.01
    value_loss_scale: float = 2.0
    grad_norm_clip: float = 1.0
```

## Curriculum Stages (Navigation1)

```
Stage 1: Easy (2-5m spawn)
├── spawn_inner=2.0, spawn_outer=5.0
├── LR: 5e-4, linear anneal
├── Target: reached > 70%

Stage 2: Medium (5-8m spawn)
├── spawn_inner=5.0, spawn_outer=8.0
├── LR: 2.5e-4, warm-start from Stage 1
├── Target: reached > 60%

Stage 3: Competition (8-11m spawn)
├── spawn_inner=8.0, spawn_outer=11.0
├── LR: 1.25e-4, warm-start from Stage 2
├── Target: reached > 80% (≥16/20 pts)

Final: Full platform (0-11m spawn)
├── spawn_inner=0.0, spawn_outer=11.0
├── LR: 1e-4, warm-start from Stage 3
├── Target: reached > 90%
```

## Empirical AutoML Findings

### Round 1 (15 trials, 5M steps each)

| Finding | Value |
|---------|-------|
| Best learning rate | ~2.4e-04 |
| Rollouts sweet spot | 32 (16 and 48 also viable) |
| Termination penalty | -200 worked at 5M but caused issues at 50M |
| Learning rate < 5e-5 | Too slow to learn meaningfully in 5M |

### Round 2 (anti-laziness, 10M steps each)

| Finding | Value |
|---------|-------|
| arrival_bonus=87.70 | High values show late learning surge |
| alive_bonus=0.13 | Very low alive_bonus forces goal-seeking |
| fine_position_tracking=8.83 | High values help precise positioning |
| Late learning surge | Reward can jump 50%+ after step 4000-5000 |

## Known Reward Exploits & Fixes (navigation1-Specific)

| Exploit | Detection Signal | Root Cause | Fix |
|---------|-----------------|-----------|-----|
| **Lazy robot** | Distance ↑, reached% ↓, ep_len near max | `alive_bonus × max_steps >> arrival_bonus` | Conditional alive_bonus + time_decay + massive arrival_bonus |
| **Sprint-crash** | fwd_vel ↑, ep_len ↓, reached% ↓ after peak | `forward_velocity` scale too high (>1.0 in original) | Speed cap (0.6 m/s clip) + near_target_speed penalty |
| **Deceleration moat** | Robot hovers at ~1m, never reaches 0.3m | `near_target_speed` activation radius too large (2.0m) | Reduce activation to ≤0.5m or use speed-distance coupling |
| **Touch-and-die** | Brief target touches then crash | `alive_bonus=0` after reaching → no survival incentive | Keep alive_bonus always active; use success_truncation instead |
| **Fly-through stop** | Stop bonus collected while sprinting through zone | `stop_bonus` not gated on speed | Speed-gate: only reward when `speed_xy < 0.3` |
| **Dead gradient** | No learning at far distances | `exp(-d/0.5) ≈ 0` for d > 3m | Widen sigma to 5.0: `exp(-d/5.0)` |
| **Config drift** | Training suddenly worse despite "same" config | Dataclass inheritance: subclass doesn't override base | Always override config in env-specific class |

## LR Scheduler Findings (navigation1-Specific)

The KL-adaptive LR scheduler is **fundamentally unstable** for this task:

| kl_threshold | Peak LR | Peak Reached% | Outcome |
|-------------|---------|---------------|---------|
| 0.016 | 0.00180 | 67% | LR spikes → policy destroyed |
| 0.008 | 0.00025 | 32% | LR crushed → learning stalls |
| 0.012 | 0.00038 | 50% | Still declines after peak |

**Solution**: Use `lr_scheduler_type = "linear"` — smooth monotonic LR anneal.

## Key Files

| File | Purpose |
|------|---------|
| `starter_kit/navigation1/vbot/cfg.py` | Environment config + reward scales |
| `starter_kit/navigation1/vbot/vbot_section001_np.py` | Environment implementation + reward function |
| `starter_kit/navigation1/vbot/rl_cfgs.py` | PPO hyperparameters |
| `starter_kit/navigation1/vbot/xmls/` | MJCF scene files |
