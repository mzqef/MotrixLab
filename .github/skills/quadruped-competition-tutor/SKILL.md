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

| Phase | Terrain | Description |
|-------|---------|-------------|
| **Stage 1 (navigation1)** | Flat ground | Basic navigation — simpler terrain, lower points |
| **Stage 2 (navigation2)** | Obstacle course | Complex terrain — slopes, stairs, balls, higher points |

> **Detailed scoring breakdown, environment IDs, and point allocations** are in:
> - `starter_kit_docs/navigation1/Task_Reference.md` → "Competition Scoring" section
> - `starter_kit_docs/navigation2/Task_Reference.md` → "Competition Scoring" section

## VBot Robot Architecture

> **Full VBot architecture** (12-DOF kinematic tree, joint configurations, actuator parameters, observation/action spaces, default standing pose) is documented in:
> `starter_kit_docs/navigation1/Task_Reference.md` → "VBot Robot Architecture" section.

### Quick Reference

- **12 DOF**: 4 legs × 3 joints (hip, thigh, calf)
- **Control**: Position servo (PD) with `action_scale=0.25`
- **Observation**: 54D vector (velocities, gravity, joints, commands, errors)
- **Action**: 12D joint position offsets in [-1, 1]

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
uv run scripts/train.py --env <env-name> --train-backend torch --max-env-steps 200000

# === VISUAL DEBUGGING ONLY ===
uv run scripts/train.py --env <env-name> --render

# === FINAL DEPLOYMENT (after AutoML found best config) ===
uv run scripts/train.py --env <env-name> --train-backend torch

# View trained policy
uv run scripts/play.py --env <env-name>
```

### PPO Hyperparameters

> **PPO configuration and tuning:** See `hyperparameter-optimization` skill.
> **Task-specific PPO config values:** See `starter_kit_docs/{task-name}/Task_Reference.md` → "PPO Hyperparameters" section.

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

> **Detailed terrain descriptions, obstacle coordinates, and task-specific traversal strategies** are in:
> `starter_kit_docs/{task-name}/Task_Reference.md` → "Terrain Description" and "Traversal Strategies" sections.

### Abstract Terrain Strategy Patterns

These patterns apply to any terrain type:

| Terrain Type | Key Challenge | Strategy Pattern |
|-------------|---------------|------------------|
| **Undulating surface** | Momentum, foot placement | Lower COM, shorter strides on downslopes |
| **Stairs** | Step clearance, balance | Higher knee lift, slower velocity, slope detection |
| **Dynamic obstacles** | Reactive avoidance | Include obstacle positions in observation space |
| **Narrow passages** | Precision navigation | Reduce stride width, add boundary penalty |
| **Bonus zones** | Stop & stay | Explicit stop reward + stability check |

### Curriculum Strategy

> **Full curriculum plans, warm-start strategies, and promotion criteria:** See `curriculum-learning` skill.
> **Reward exploration methodology:** See `reward-penalty-engineering` skill.

General principle: Train on the simplest terrain first, then progressively add difficulty. Each stage warm-starts from the previous best checkpoint with reduced LR (0.3-0.5×).

## Scoring Optimization Tactics

> **Detailed scoring breakdowns** with point allocations per section, stage, and element type:
> - `starter_kit_docs/navigation1/Task_Reference.md` → "Competition Scoring" section
> - `starter_kit_docs/navigation2/Task_Reference.md` → "Competition Scoring" section

### Abstract Scoring Optimization Principles

| Principle | Rationale |
|-----------|----------|
| **Stability > Speed** | Falls lose ALL points for that attempt; slow-and-steady wins |
| **Easy points first** | Master the simplest stage before attempting harder ones |
| **100% success rate** | Near-perfect success rate is more valuable than speed |
| **Bonus zone collection** | Plan paths to pass through bonus zones when feasible |
| **Time management** | Budget extra time for difficult segments |
| **Position tolerance** | Ensure robot actually stops at goal (not just passes through) |

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
uv run scripts/play.py --env <env-name>

# Play long course (if applicable)
uv run scripts/play.py --env <full-course-env>
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
# In starter_kit/{task}/vbot/cfg.py → RewardConfig class
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
# In starter_kit/{task}/vbot/vbot_*_np.py → _compute_reward()
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
# In vbot_*_np.py
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
