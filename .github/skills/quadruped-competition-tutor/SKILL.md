---
name: quadruped-competition-tutor
description: Comprehensive tutoring for the MotrixArena S1 quadruped robot navigation competition. Covers VBot robot design, reinforcement learning strategies, reward function engineering, terrain traversal techniques, and scoring optimization to achieve top rankings.
---

## Purpose

Guide competitors to **top rankings** in MotrixArena S1 quadruped navigation challenge:

- **Quadruped fundamentals** - VBot 12-DOF robot architecture, gait patterns, stability
- **RL for robotics** - PPO training, observation design, action spaces
- **Reward engineering** - Sigmoid distance, collision detection, checkpoint training
- **Terrain strategies** - Stairs, waves, rolling obstacles, celebration zones
- **Score optimization** - Maximize points through bonus zones and efficient traversal
- **Submission preparation** - Code, weights, video, technical documentation

## Competition Overview: MotrixArena S1

### Two-Phase Structure

| Phase | Terrain | Max Points | Time Limit |
|-------|---------|------------|------------|
| **Stage 1 (navigation1)** | Flat ground | 20 pts | Per-dog limit |
| **Stage 2 (navigation2)** | Obstacle course | 105 pts | Episode limits |

### Scoring Summary

```
Stage 1 (20 pts total):
└── 10 navigation dogs × 2 pts each

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
```

### Competition Environment IDs

| Environment ID | Terrain | Training Focus |
|----------------|---------|----------------|
| `vbot_navigation_section001` | Flat ground | Basic locomotion, Stage 1 (navigation1) |
| `vbot_navigation_section01` | Section 1 challenges | Bonus collection (navigation2) |
| `vbot_navigation_section02` | Section 2 terrain | Terrain traversal (navigation2) |
| `vbot_navigation_section03` | Section 3 | Rolling balls, finale (navigation2) |
| `vbot_navigation_stairs` | Stairs + platforms | Stair climbing (navigation2) |
| `vbot_navigation_long_course` | Full 30m course | End-to-end training (navigation2) |

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

**Default Standing Pose (radians):**
```python
default_joint_angles = {
    "FR_hip_joint": 0.0,   "FR_thigh_joint": 0.9,   "FR_calf_joint": -1.8,
    "FL_hip_joint": 0.0,   "FL_thigh_joint": 0.9,   "FL_calf_joint": -1.8,
    "RR_hip_joint": 0.0,   "RR_thigh_joint": 0.9,   "RR_calf_joint": -1.8,
    "RL_hip_joint": 0.0,   "RL_thigh_joint": 0.9,   "RL_calf_joint": -1.8,
}
```

### Actuator Configuration

| Parameter | Value | Purpose |
|-----------|-------|---------|
| Control Mode | Position Servo (PD) | Stable joint tracking |
| kp (stiffness) | 80.0 N·m/rad | Position gain |
| kv (damping) | 6.0 N·m·s/rad | Velocity damping |
| action_scale | 0.25 | Scale [-1,1] actions to radians |
| Torque Limits | Hip/Thigh: ±17 N·m, Calf: ±34 N·m | From XML forcerange |

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

Output: 12 joint position offsets in range [-1, 1], scaled by `action_scale`:
```python
target_position = default_angles + action * action_scale
torque = kp * (target - current_pos) - kv * current_vel
```

## Training Infrastructure

> **Campaign management, checkpoints, and monitoring:** See `training-campaign` skill.

### Commands

> **🔴 AutoML-First Policy:** NEVER use `train.py` for parameter search.
> See `.github/copilot-instructions.md` for the full policy.

```powershell
# === PRIMARY: AutoML pipeline (USE THIS for all parameter exploration) ===
uv run starter_kit_schedule/scripts/automl.py `
    --mode stage `
    --budget-hours 8 `
    --hp-trials 15

# === SMOKE TEST ONLY (<500K steps) ===
uv run scripts/train.py --env vbot_navigation_section001 --train-backend torch --max-env-steps 200000

# === VISUAL DEBUGGING ONLY ===
uv run scripts/train.py --env vbot_navigation_section001 --render

# === FINAL DEPLOYMENT (after AutoML found best config) ===
uv run scripts/train.py --env vbot_navigation_section001 --train-backend torch

# View trained policy
uv run scripts/play.py --env vbot_navigation_section001
```

### PPO Hyperparameters (VBotSection001PPOConfig)

> **Tuning and search:** See `hyperparameter-optimization` skill.

```python
@dataclass
class VBotSection001PPOConfig(PPOCfg):
    policy_hidden_layer_sizes: tuple = (256, 128, 64)
    value_hidden_layer_sizes: tuple = (256, 128, 64)
    
    rollouts: int = 24              # Steps before update
    learning_epochs: int = 8        # PPO epochs per update
    mini_batches: int = 4           # Batches per epoch
    discount_factor: float = 0.99   # Gamma
    lambda_param: float = 0.95      # GAE lambda
    
    learning_rate: float = 3e-4
    learning_rate_scheduler_kl_threshold: float = 0.008
    
    ratio_clip: float = 0.2         # PPO clip
    value_clip: float = 0.2
    entropy_loss_scale: float = 0.005  # Exploration bonus (was 0.0 — fixed)
    value_loss_scale: float = 2.0
    grad_norm_clip: float = 1.0
```

### Training Timeline Expectations

| Environment | Convergence | Reward Range | Notes |
|-------------|-------------|--------------|-------|
| Flat navigation | 500K steps | 20-40 | Fast, stable |
| Stairs | 2-5M steps | 15-30 | Medium difficulty |
| Section 001-013 | 5-10M steps | Variable | Curriculum helps |
| Long course | 10-20M steps | 50-80 | Requires staged training |

## Reward Function Engineering

> **Current implementation:** `starter_kit/navigation*/vbot/vbot_*_np.py` → `_compute_reward()`.
> **Reward weights:** `starter_kit/navigation*/vbot/cfg.py` → `RewardConfig.scales` dict.
> **Exploration methodology** (how to find/test/archive rewards): See `reward-penalty-engineering` skill.
> **Archived configurations:** See `starter_kit_schedule/reward_library/`.

### Core Reward Principles

1. **Dense rewards** - Provide continuous feedback, not just goal completion
2. **Balance exploration vs exploitation** - Entropy bonus in early training
3. **Avoid reward hacking** - Test that high rewards = desired behavior
4. **Smooth gradients** - Sigmoid/exponential better than step functions

### Navigation Task Reward Structure

> **⚠️ CRITICAL COMPETITION INSIGHT (Stage 1):** 10 dogs spawn randomly. Each navigates to inner fence (+1pt) then center (+1pt) = max 20pts. **ANY single fall or out-of-bounds = that dog loses ALL points (both +1s).** This means stability is MORE important than speed — a conservative policy that never falls beats a fast one that falls once.

```python
reward_scales = {
    # === Primary Navigation ===
    "position_tracking": 1.5,       # Exponential decay: exp(-d/5.0)
    "fine_position_tracking": 5.0,  # Extra reward when < 1.5m: exp(-d/0.3)
    "heading_tracking": 0.8,        # Face movement direction
    "forward_velocity": 1.5,        # Encourage speed toward goal
    "distance_progress": 2.0,       # Linear: clip(1 - d/initial_distance, -0.5, 1.0)
    
    # === Goal Completion (anti-laziness aware) ===
    "approach_scale": 8.0,          # Bonus for reducing distance
    "arrival_bonus": 50.0,          # One-time bonus on reaching target
    "stop_scale": 2.0,              # Bonus for stopping at target
    "zero_ang_bonus": 6.0,          # Bonus for standing still at target
    "alive_bonus": 0.5,             # CONDITIONAL: only when NOT ever_reached
    
    # === Stability Penalties ===
    "orientation": -0.05,           # Penalize body tilt
    "lin_vel_z": -0.3,              # Penalize vertical bounce
    "ang_vel_xy": -0.03,            # Penalize body rotation
    "torques": -1e-5,               # Energy efficiency
    "dof_vel": -5e-5,               # Smooth joint motion
    "dof_acc": -2.5e-7,             # Jerk reduction
    "action_rate": -0.01,           # Smooth policy
    
    # === Terminal States ===
    "termination": -100.0,          # Body-ground collision
}
```

> **Anti-laziness mechanisms active:**
> 1. `alive_bonus` is **conditional** — becomes 0 after reaching target (prevents "stand around" exploit)
> 2. Navigation rewards multiply by `time_decay = clip(1 - 0.5*steps/max_steps, 0.5, 1.0)` — creates urgency
> 3. `arrival_bonus=50` >> `alive_bonus(0.5) × max_steps(4000) / 2` — dominates per-step accumulation
> 4. `_update_truncate()`: 50 consecutive steps of reached + stopped → episode ends early
>
> See `reward-penalty-engineering` skill → "Lazy Robot Case Study" for full details.
```

### Advanced Reward Techniques

#### 1. Sigmoid Distance Reward

```python
def sigmoid_distance_reward(distance, distance_scale=5.0):
    """
    Smoother than linear, provides gradient even far from goal.
    R = 1 / (1 + exp(0.5 * distance / scale))
    """
    distance_ratio = distance / distance_scale
    return 1.0 / (1.0 + np.exp(0.5 * distance_ratio))
```

**Comparison:**
| Distance | Linear (1-d/5) | Sigmoid |
|----------|----------------|---------|
| 0m | 1.0 | 0.5 |
| 2m | 0.6 | 0.45 |
| 5m | 0.0 | 0.38 |
| 10m | -1.0 (clipped) | 0.27 |

#### 2. Swing Time Reward (Gait Quality)

```python
def swing_time_reward(foot_contacts, dt=0.01):
    """
    Encourage natural gait rhythm (0.3-0.7s swing, optimal 0.5s).
    Track time since last contact per foot.
    """
    for foot in ['FR', 'FL', 'RR', 'RL']:
        if foot_contacts[foot]:
            swing_time = time_since_contact[foot]
            if 0.3 <= swing_time <= 0.7:
                reward += gaussian(swing_time, mean=0.5, std=0.1) * 0.2
            time_since_contact[foot] = 0
        else:
            time_since_contact[foot] += dt
    return reward
```

#### 3. Collision Detection Reward

```python
def collision_penalty(contact_forces, foot_names=['FR', 'FL', 'RR', 'RL']):
    """
    Detect non-foot collisions using force sensor threshold.
    F > 5 * |Fz| suggests horizontal collision (not standing).
    """
    for geom, force in contact_forces.items():
        if geom not in foot_names:
            Fxy = np.linalg.norm(force[:2])
            Fz = abs(force[2])
            if Fxy > 5 * Fz:  # Significant horizontal force
                return -5.0  # Collision penalty
    return 0.0
```

#### 4. Checkpoint Training

```python
def checkpoint_curriculum(checkpoints, robot_pos, reached_set):
    """
    Incremental rewards for reaching waypoints.
    Stage 2 Section 2 has ~8 checkpoints along the course.
    """
    reward = 0.0
    checkpoint_rewards = [5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5]  # Increasing
    
    for i, (cp_pos, cp_radius) in enumerate(checkpoints):
        if i not in reached_set:
            dist = np.linalg.norm(robot_pos[:2] - cp_pos)
            if dist < cp_radius:
                reward += checkpoint_rewards[i]
                reached_set.add(i)
    return reward
```

#### 5. Anti-Stagnation Mechanism

```python
def stagnation_penalty(position_history, threshold=0.1, window=10):
    """
    Penalize if robot moves < 0.1m in 10 control cycles.
    Progressive penalty increases with stagnation duration.
    """
    if len(position_history) < window:
        return 0.0
    
    displacement = np.linalg.norm(position_history[-1] - position_history[-window])
    if displacement < threshold:
        stagnation_count += 1
        return -0.1 * min(stagnation_count, 10)  # Cap at -1.0
    else:
        stagnation_count = 0
        return 0.0
```

#### 6. Roll/Pitch Safety Boundary

```python
def orientation_safety(roll, pitch, limit_deg=60):
    """
    Terminate or heavily penalize dangerous tilts.
    """
    limit_rad = np.deg2rad(limit_deg)
    if abs(roll) > limit_rad or abs(pitch) > limit_rad:
        return -10.0  # Emergency penalty
    return 0.0
```

## Terrain Traversal Strategies

### Wave Terrain (Section 2A)

**Challenges:** Undulating surface, momentum maintenance, foot placement timing

**Strategies:**
1. **Initial stabilization** - Train standing on waves before walking
2. **Adaptive stride length** - Shorter steps on downslopes, longer on upslopes
3. **COM (Center of Mass) control** - Keep body low during transitions
4. **Velocity modulation** - Slow down approaching wave peaks

**Reward additions:**
```python
# Height variance penalty (encourage smooth traversal)
height_variance = np.var(foot_heights)
reward -= 0.1 * height_variance

# Forward progress bonus on waves
if terrain_type == "wave" and forward_vel > 0.3:
    reward += 0.5
```

### Stairs (Section 2B)

**Challenges:** Step height clearance, balance during ascent/descent, edge detection

**Strategies:**
1. **Slope adaptation** - Detect inclination, adjust body pitch
2. **Foot edge distance** - Train to avoid step edges (slip risk)
3. **Dynamics compensation** - Higher knee lift during ascent
4. **Slower velocity** - Prioritize stability over speed

**Reward additions:**
```python
# Knee lift bonus during stair ascent
if terrain_gradient > 0.1:  # Ascending
    knee_lift = -joint_pos['calf']  # More negative = more bent
    reward += 0.2 * max(0, knee_lift - 1.5)

# Penalize foot slip (large tangential forces on steps)
if stair_contact and tangential_force > threshold:
    reward -= 0.5
```

### Rolling Balls (Section 3)

**Challenges:** Dynamic obstacles, trajectory prediction, reactive avoidance

**Strategies:**
1. **Peripheral awareness** - Include ball positions in observation
2. **Conservative paths** - Prefer edges over center
3. **Pause & proceed** - Wait for safe window if needed
4. **Recovery training** - Train to recover from ball impacts

**Observation addition:**
```python
# Add ball observations (if visible)
ball_positions = get_ball_positions()  # [N, 3] relative positions
ball_velocities = get_ball_velocities()  # [N, 3]
obs = np.concatenate([obs, ball_positions.flatten(), ball_velocities.flatten()])
```

### Celebration Zones

**Bonus points for:**
- Smiley circles: Stand still inside for 1-2 seconds
- Red packets: Touch/pass through
- Final dance: Execute specific motion pattern

**Celebration reward:**
```python
def celebration_reward(robot_pos, celebration_zones, gyro):
    """
    +2 to +5 pts for successful celebrations
    """
    for zone_pos, zone_radius, zone_type in celebration_zones:
        if np.linalg.norm(robot_pos[:2] - zone_pos) < zone_radius:
            if zone_type == "smiley":
                # Need to be stable (low rotation rate)
                if np.linalg.norm(gyro) < 0.5:
                    return 2.0 if small_smiley else 4.0
            elif zone_type == "red_packet":
                return 2.0  # Instant bonus
    return 0.0
```

### Curriculum Strategy

> **Full curriculum plans, warm-start strategies, and promotion criteria:** See `curriculum-learning` skill.
> **Reward exploration methodology:** See `reward-penalty-engineering` skill.

### Stage Overview

```
Stage 1: Flat Ground → Stage 2A: Waves → Stage 2B: Stairs → Stage 2C: Balls → Full Course
     (500K)              (2M)              (3M)              (2M)           (5M)
```

Each stage warm-starts from the previous best checkpoint with reduced LR (0.3-0.5×).

### Checkpoint Loading

```python
# Load pretrained policy for curriculum
trainer = ppo.Trainer(
    env_name="vbot_navigation_stairs",
    sim_backend=None,
    cfg_override={
        "num_envs": 2048,
        "checkpoint_path": "runs/vbot_navigation_section001/checkpoints/agent.pt"
    }
)
trainer.train()
```

## Scoring Optimization Tactics

### Stage 1: Flat Ground (20 pts)

**Target:** 10/10 dogs, 2 pts each = 20 pts

> **⚠️ CRITICAL:** Each dog navigates to inner fence (+1pt) then center (+1pt). If the dog **falls at ANY point**, it loses ALL its points (both +1s). A single fall costs 2 points. Therefore:
> - A policy that gets 10/10 dogs to center slowly = **20 pts**
> - A policy that gets 9/10 dogs fast but 1 falls = **18 pts**
> - Stability > Speed for competition scoring

**Key metrics:**
- Success rate (100% target → must be near-perfect)
- Zero falls (termination penalty must be effective)
- Time to reach (secondary — tiebreaker only)

**Optimization:**
1. Train to 100% success rate on flat with randomized targets
2. Keep stability penalties active — never sacrifice stability for speed
3. Use `vbot_navigation_section001` with position tolerance 0.3m
4. Position_threshold for truncation ensures robot actually stops at goal

### Stage 2: Section 1 (20 pts)

**Break down:**
| Element | Points | Strategy |
|---------|--------|----------|
| Base traverse | 5 | Just walk forward |
| Small smileys (×2) | 4 | Stop in circle, stay stable 1s |
| Large smileys (×2) | 8 | Stop in circle, stay stable 1s |
| Red packets (×3) | 6 | Slight detour, touch each |

**Observation:** Smileys require **stopping**, not just passing through!

### Stage 2: Section 2 (60 pts)

**Checkpoint strategy:**
1. Place checkpoints every 2-3m along route
2. Assign increasing rewards (5→8 pts progression)
3. Track reached checkpoints in episode info

**Time management:**
- Section 2 has 60 second episode limit
- Average speed needed: 0.5 m/s for 30m course
- Budget extra time for stairs (slowest segment)

### Stage 2: Section 3 (25 pts)

**Dynamic obstacles:**
- Ball positions change each episode
- Need reactive policy, not memorized path
- Include ball observations in training

**Final celebration (5 pts):**
- Must execute specific motion at end zone
- Train celebration as separate skill, then combine

## Common Failure Modes & Fixes

| Failure | Symptom | Fix |
|---------|---------|-----|
| **Lazy robot (reward hacking)** | Distance UP, reached% DOWN, ep_len near max | Conditional alive_bonus, time_decay, increase arrival_bonus |
| **Falls on flat ground** | Body touches ground | Increase orientation penalty, add armature |
| **Stuck on stair edge** | No progress, foot scraping | Add foot clearance reward, knee lift bonus |
| **Overshoots target** | Oscillates around goal | Add velocity damping near target, fine tracking reward |
| **Wobbly gait** | Body rotation, unstable | Increase ang_vel_xy penalty, action smoothness |
| **Slow convergence** | Reward plateau | Check reward scaling, try curriculum |
| **Ball collisions** | Gets hit, falls | Add ball observations, train avoidance |
| **Celebration failure** | Moves through smileys | Add explicit stop reward, stability check |

## Submission Checklist

### Required Materials

1. **Code Repository**
   - All training code
   - Environment configurations
   - Reward function implementations
   
2. **Policy Weights**
   - Final checkpoint (`.pt` or `.safetensors`)
   - Include any preprocessing/normalization stats
   
3. **Demo Video (≤3 min)**
   - Show full course completion
   - Highlight challenging sections (stairs, balls)
   - Include timing overlay
   
4. **Technical Document (1-5 pages)**
   - Architecture description
   - Reward function design rationale
   - Training curriculum
   - Key innovations

### Pre-Submission Testing

```powershell
# Test on evaluation environment
uv run scripts/play.py --env vbot_navigation_section001

# Play long course
uv run scripts/play.py --env vbot_navigation_long_course
```

### Final Checks

- [ ] No hardcoded positions or memorized trajectories
- [ ] Handles position randomization
- [ ] Recovers from perturbations
- [ ] Completes course with no falls
- [ ] More bonus zones collected

## Quick Reference: Environment Configs

### Modify Reward Scales

```python
# In starter_kit/navigation1/vbot/cfg.py → RewardConfig class
@dataclass
class RewardConfig:
    scales: dict[str, float] = field(
        default_factory=lambda: {
            "position_tracking": 2.0,  # ← Adjust these
            "termination": -200.0,
            # ... add new rewards here
        }
    )
```

### Add New Reward Terms

```python
# In starter_kit/navigation1/vbot/vbot_section001_np.py → _compute_reward()
def _compute_reward(self, data, info, velocity_commands):
    reward = np.zeros(self._num_envs)
    
    # Existing rewards...
    
    # Add custom reward
    distance = np.linalg.norm(info["position_error"], axis=1)
    reward += self._cfg.reward_config.scales.get("custom_distance", 0.0) * \
              sigmoid_distance_reward(distance)
    
    return reward
```

### Extend Observation Space

```python
# In vbot_section001_np.py
def __init__(self, cfg, num_envs=1):
    # Change observation space size
    self._observation_space = gym.spaces.Box(
        low=-np.inf, high=np.inf, 
        shape=(54 + NEW_OBS_DIM,),  # ← Add dimensions
        dtype=np.float32
    )
```

## VBot XML Quick Reference

| Element | Location | Purpose |
|---------|----------|---------|
| Robot model | `xmls/vbot.xml` (included) | Joint limits, masses |
| Contact params | `<default class="foot">` | Friction, condim |
| Sensors | `<sensor>` | IMU, contact forces |
| Actuators | `<actuator>` | Joint servos |

## Summary: Path to Top Ranking

1. **Master flat ground first** (Stage 1 = easy 20 pts)
2. **Build curriculum** for each terrain type
3. **Engineer dense rewards** with Sigmoid distance, checkpoints
4. **Add safety penalties** for orientation, collision
5. **Include terrain-specific rewards** (knee lift for stairs, etc.)
6. **Test exhaustively** before submission
7. **Document innovations** in technical report
