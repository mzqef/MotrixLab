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

## Reward Scales (Current — Session 11 Speed-Optimized + Frozen Normalizer)

> **Source**: `starter_kit_schedule/configs/stage3_frozen_continue.json` (active) / `starter_kit/navigation1/vbot/cfg.py` (base defaults)

```python
# === Navigation core (speed-optimized) ===
"position_tracking": 1.73,       # exp(-d/5.0)
"fine_position_tracking": 12.0,  # sigma=0.5, range<2.5m, GATED by ever_reached (R11 fix)
"heading_tracking": 0.30,
"forward_velocity": 3.5,         # Doubled from 1.77 → 3.5 for speed
"distance_progress": 1.5,        # linear: clip(1 - d/d_max, -0.5, 1.0)
"alive_bonus": 0.08,             # Reduced from 0.15 to discourage standing

# === Approach/arrival (increased for speed-opt) ===
"approach_scale": 50.0,           # Step-delta distance improvement (was 40.46)
"arrival_bonus": 160.0,           # One-time on reaching target (was 130.19)
"inner_fence_bonus": 30.26,       # One-time at d<0.75m
"stop_scale": 5.97,               # Precision stopping (speed-gated)
"zero_ang_bonus": 9.27,
"near_target_speed": -0.4,        # Quadratic speed-distance coupling (was -0.71)
"boundary_penalty": -4.44,        # Platform edge safety

# === Stability (halved for dynamic gait) ===
"orientation": -0.025,
"lin_vel_z": -0.15,               # Was -0.3
"ang_vel_xy": -0.02,              # Was -0.03
"torques": -5e-6,
"dof_vel": -3e-5,
"dof_acc": -1.5e-7,
"action_rate": -0.003,            # Was -0.01

# === Terminal ===
"termination": -75.0,
"departure_penalty": -5.0,

# === Code-level reward fixes (in vbot_section001_np.py) ===
# R11: time_decay = 1.0 (constant, removed decay)
# R11: fine_position_tracking gated behind ever_reached
# Round7: 50-step stop_bonus budget cap
# Round5: alive_bonus always active (both branches)
```

## Best Checkpoint (Competition Submission)

| Rank | File | Location | Reached% | Episodes | Steps | Speed 8-12m |
|------|------|----------|----------|----------|-------|-------------|
| **🏆 1** | `stage3_continue_agent1600_reached100_4608.pt` | `starter_kit_schedule/checkpoints/` | **100.00%** | 4608 | 479 | 1.65 m/s |
| 2 | `stage3_frozen_agent8800_reached9998.pt` | `starter_kit_schedule/checkpoints/` | 99.95% | 12,288 | 479 | 1.65 m/s |

**Primary submission**: `stage3_continue_agent1600_reached100_4608.pt`  
**Expected competition score**: **20/20** (10 dogs × 2 pts, 0% fall rate)

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

## PPO Hyperparameters (Final — Speed-Opt + Frozen Normalizer)

```python
@dataclass
class VBotSection001PPOConfig(PPOCfg):
    policy_hidden_layer_sizes: tuple = (256, 128, 64)
    value_hidden_layer_sizes: tuple = (512, 256, 128)   # Wider value net
    rollouts: int = 24
    learning_epochs: int = 6
    mini_batches: int = 32
    discount_factor: float = 0.99
    lambda_param: float = 0.95
    learning_rate: float = 3e-5        # Reduced for continuation (was 5e-5)
    lr_scheduler_type: str | None = "linear"
    ratio_clip: float = 0.12           # Tight for fine-tuning (was 0.15)
    value_clip: float = 0.2
    entropy_loss_scale: float = 0.008
    value_loss_scale: float = 2.0
    grad_norm_clip: float = 1.0
    kl_threshold: float = 0.015
    freeze_preprocessor: bool = True   # Freeze RunningStandardScaler stats
```

## Curriculum Stages (Navigation1) — COMPLETED

```
Stage 1: Easy (2-5m spawn) ✅
├── spawn_inner=2.0, spawn_outer=5.0
├── LR: 4.34e-4, linear anneal, AutoML-tuned (T1-best)
├── Result: 66.58% reached (R16 seed=2026, agent_9600.pt)
├── Breakthroughs: R11 fix (remove time_decay + gate fine_tracking)

Stage 2: Medium (5-8m spawn) ✅
├── spawn_inner=5.0, spawn_outer=8.0
├── LR: 1.3e-4, warm-start from R16 agent_9600.pt
├── Result: 97.76% reached at step 1000

Stage 3: Competition (8-11m spawn) ✅
├── spawn_inner=8.0, spawn_outer=11.0
├── LR: 5e-5 → 3e-5, warm-start from Stage 2
├── Speed-optimized rewards, frozen normalizer
├── Result: 100.00% reached (4608/4608), 479 avg steps, 1.65 m/s

Final Competition Readiness: ✅
├── Expected score: 20/20
├── Best checkpoint: stage3_continue_agent1600_reached100_4608.pt
├── Backup: stage3_frozen_agent8800_reached9998.pt
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
