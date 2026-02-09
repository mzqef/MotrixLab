

# MotrixLab Project — Deep-Dive Tutorial for PhD Students

Welcome! This is a **hand-holding, step-by-step, file-by-file walkthrough** of the MotrixLab codebase, written for new PhD students and anyone who feels lost in RL for robotics. We’ll explain every major part, how the pieces fit, and give you practical, explicit tips for learning, debugging, and extending the system. **No question is too basic!**

---

## 1. Project Structure: What’s in Each Folder?

MotrixLab is a modular RL framework for robotics, focused on the MotrixArena S1 quadruped navigation competition. The codebase is organized as a workspace with several key folders:

- **motrix_envs/**: Simulation environment definitions (robot tasks, physics, rewards, observation/action spaces). This is the core for defining new tasks. 
  - `src/motrix_envs/np/env.py`: The main base class for all NumPy-based environments. 
  - `src/motrix_envs/registry.py`: Handles environment registration and creation.
  - `src/motrix_envs/base.py`: The abstract base class for all environments.
  - `src/motrix_envs/np/renderer.py`: Handles visualization.
  - `src/motrix_envs/np/reward.py`: Utility reward shaping functions.
- **motrix_rl/**: RL training integration (mainly PPO, JAX/Torch). Contains RL config registries and training logic.
  - `src/motrix_rl/cfgs.py`: RL config dataclasses and registration.
  - `src/motrix_rl/registry.py`: RL config registry and decorator logic.
  - `src/motrix_rl/skrl/cfg.py`: PPO config dataclass for SKRL.
- **scripts/**: Entry points for training, evaluation, and visualization. These are the main scripts you run.
  - `train.py`: Main training entry point.
  - `play.py`: Policy evaluation and demo.
  - `view.py`: Visualizes the environment with random actions.
- **starter_kit/**: Competition-specific environments and configs. Stage 1 (flat) and Stage 2 (obstacles/stairs) are here.
  - `navigation1/vbot/cfg.py`: VBot env configs and reward scales (Stage 1).
  - `navigation1/vbot/vbot_section001_np.py`: VBot environment implementation (Stage 1).
  - `navigation1/vbot/rl_cfgs.py`: RL hyperparameters for Stage 1.
  - `navigation2/vbot/cfg.py`, `vbot_section011_np.py`, etc.: Stage 2 (obstacles/stairs).
- **starter_kit_schedule/**: Training campaign pipeline, reward library, AutoML scripts, and progress tracking.
  - `scripts/automl.py`: AutoML HP search engine and curriculum pipeline.
  - `progress/`: Tracks experiment progress.
  - `templates/`: YAML templates for curriculum, reward configs, search spaces.
- **starter_kit_docs/**: Guides, scoring rules, reward design docs, and competition explanations.
- **runs/**: Stores training outputs (checkpoints, logs, TensorBoard data).
- **README.md, CLAUDE.md**: Project instructions, operational notes, and AI agent guidance.

**Tip:** The workspace is a Python 3.10 project managed by [UV](https://github.com/astral-sh/uv) for dependencies. All configs use Python dataclasses (no YAML/JSON). If you’re lost, start with the README and this tutorial!

---

## 2. Simulation Environments: How Are Tasks Defined?

### 2.1 Environment Class Pattern (The Heart of MotrixLab)

All environments subclass `NpEnv` (see `motrix_envs/src/motrix_envs/np/env.py`). Each environment must implement three methods:

- `reset(data)`: Initializes or randomizes the state, computes initial observations.
- `apply_action(actions, state)`: Converts actions to torques, sets actuator controls.
- `update_state(state)`: Computes observations, reward, and termination; updates state.

The main step loop is: `apply_action → physics_step → update_state → reset done envs`.

#### Example: VBot Navigation Environment

See `starter_kit/navigation1/vbot/vbot_section001_np.py` for a real implementation. This file:
- Sets up the robot, target, and spawn logic.
- Defines the action/observation space (12 actuators, 54-dim obs).
- Handles reward calculation and termination.

### 2.2 Environment Registration (How Does the System Find My Env?)

MotrixLab uses a dual-registry decorator pattern:

1. `@registry.envcfg("env-name")` on a `@dataclass(EnvCfg)` — registers the config class.
2. `@registry.env("env-name", "np")` on a `NpEnv` subclass — registers the environment implementation.

Config must be registered before the env class. Importing the module triggers registration (see `__init__.py` files with `# noqa: F401` imports).

#### Example: Registering a Custom Environment

```python
from motrix_envs import registry
from dataclasses import dataclass

@registry.envcfg("my-custom-env")
@dataclass
class MyEnvCfg(EnvCfg):
    ...

@registry.env("my-custom-env", "np")
class MyCustomEnv(NpEnv):
    ...
```

### 2.3 Competition Environments (Where’s the Real Action?)

Competition environments are in `starter_kit/navigation1/vbot/` (Stage 1, flat) and `starter_kit/navigation2/vbot/` (Stage 2, obstacles/stairs). Each section has its own config and environment class.

**Example:**
- `starter_kit/navigation1/vbot/cfg.py`: Contains `VBotSection001EnvCfg` and reward scales.
- `starter_kit/navigation1/vbot/vbot_section001_np.py`: Implements the environment logic for Stage 1.

### 2.4 Creating and Using Environments (How Do I Actually Make One?)

Environments are created via the registry:
```python
from motrix_envs.registry import make
env = make("vbot_navigation_section001", num_envs=2048)
```

You can override config parameters at creation:
```python
env = make("vbot_navigation_section001", num_envs=128, env_cfg_override={"spawn_inner_radius": 6.0})
```

---

## 3. RL Training Integration: How Does Training Work?

### 3.1 RL Config Registration (How Are Hyperparameters Set?)

RL configs are registered with `@rlcfg("env-name")` or `@rlcfg("env-name", backend="jax")` decorators. These are dataclasses that specify PPO hyperparameters, network sizes, and training settings.

**Example:**
- `motrix_rl/src/motrix_rl/cfgs.py` and `starter_kit/navigation1/vbot/rl_cfgs.py` contain RL configs for each environment.

### 3.2 Training, Evaluation, and Visualization Scripts (What Do I Run?)

- `scripts/train.py`: Main entry for training. Selects backend (JAX/Torch), loads env, runs PPO.
- `scripts/play.py`: Evaluates trained policies (auto-finds latest checkpoint if not specified).
- `scripts/view.py`: Visualizes the environment with random actions (no training).

#### Example: Training a Policy
```bash
uv run scripts/train.py --env vbot_navigation_section001
```

#### Example: Evaluating a Policy
```bash
uv run scripts/play.py --env vbot_navigation_section001
```

#### Example: Visualizing the Environment
```bash
uv run scripts/view.py --env vbot_navigation_section001
```

### 3.3 Training Workflow (Step-by-Step)

1. **Install dependencies:**
   ```bash
   uv sync --all-packages --all-extras
   ```
2. **Train:**
   ```bash
   uv run scripts/train.py --env vbot_navigation_section001
   ```
3. **Monitor:**
   ```bash
   uv run tensorboard --logdir runs/vbot_navigation_section001
   ```
4. **Evaluate:**
   ```bash
   uv run scripts/play.py --env vbot_navigation_section001
   ```
5. **Visualize:**
   ```bash
   uv run scripts/view.py --env vbot_navigation_section001
   ```

**Tip:** Training outputs (checkpoints, logs) are saved in `runs/<env-name>/`.

---

## 4. Reward Engineering and Curriculum Learning: How Do I Make the Robot Smarter?

### 4.1 Reward Function Design (How Are Rewards Built?)

Reward functions are defined in config dataclasses (e.g., `RewardConfig`) in `starter_kit/navigation1/vbot/cfg.py`. The `scales` dictionary sets the weights for each reward/penalty term.

**Key reward components:**
- Position tracking, heading tracking, forward velocity, approach/arrival bonuses, stop bonuses, boundary/termination penalties, stability penalties, and more.

**Engineering tips:**
- Use empirically validated scales (see reward-penalty-engineering skill and `starter_kit_docs/MotrixArena_S1_四足机器人奖励函数设计与越障任务优化.md`).
- Avoid reward hacking by balancing alive_bonus, arrival_bonus, and termination penalties.
- Tune reward shaping for each curriculum stage.

#### Example: RewardConfig in `cfg.py`
```python
@dataclass
class RewardConfig:
    scales: dict[str, float] = field(default_factory=lambda: {
        "position_tracking": 1.5,
        "fine_position_tracking": 8.0,
        ...
    })
```

### 4.1b Round5 Reward Fixes — Structural Bug Fixes (Feb 9 Session 4)

Four critical structural bugs were identified through VLM visual analysis and code inspection, then fixed in `vbot_section001_np.py`:

#### Fix 1: `alive_bonus` Always Active (removes "touch and die" cycle)

**Bug**: `alive_bonus` was zeroed after `ever_reached=True`, so the robot had no incentive to stay alive after touching the target → learned to crash immediately for faster episode resets.

```python
# BEFORE (buggy): alive_bonus zeroed after reaching target
ever_reached = info.get("ever_reached", ...)
alive_bonus = np.where(ever_reached, 0.0, 1.0)  # 0 after touch → crash incentive

# AFTER (fixed): alive_bonus always active
alive_bonus = np.ones(self._num_envs, dtype=np.float32)  # Always reward survival
# success_truncation (50 steps stopped) handles episode termination naturally
```

#### Fix 2: Speed-Distance Coupling (replaces narrow `near_target_speed`)

**Bug**: `near_target_speed` only activated at d<0.5m — too narrow. Robot could sprint at full speed until the last 0.5m with no penalty.

```python
# BEFORE: penalty only when d < 0.5m (too narrow)
penalty = where(d < 0.5 and not reached, speed * scale, 0)

# AFTER: smooth distance-proportional speed limit
desired_speed = np.clip(distance_to_target * 0.5, 0.05, 0.6)  # 0.6 at d>=1.2m, 0.05 at d=0.1m
speed_excess = np.maximum(speed_xy - desired_speed, 0.0)
penalty = scale * speed_excess ** 2  # Quadratic penalty, smooth gradient
```

| Distance | Desired Speed | Robot at 0.5m/s → Penalty |
|----------|--------------|---------------------------|
| 5.0m     | 0.60 m/s     | 0 (free to sprint)        |
| 1.0m     | 0.50 m/s     | 0 (free to walk)          |
| 0.5m     | 0.25 m/s     | -0.125                    |
| 0.2m     | 0.10 m/s     | -0.320                    |
| 0.1m     | 0.05 m/s     | -0.405                    |

#### Fix 3: Speed-Gated `stop_bonus` (prevents fly-through reward)

**Bug**: `stop_bonus` triggered whenever `reached_all` (d<0.5m), even if robot was sprinting through at 0.6 m/s.

```python
# BEFORE: stop_bonus for any robot in the zone, regardless of speed
stop_bonus = np.where(reached_all, stop_base + zero_ang_bonus, 0.0)

# AFTER: only reward genuinely slow robots
genuinely_slow = np.logical_and(reached_all, speed_xy < 0.3)
stop_bonus = np.where(genuinely_slow, stop_base + zero_ang_bonus, 0.0)
```

#### Fix 4: Symmetric Approach Retreat Penalty

**Bug**: Below 1.5m, approach reward was clamped to `max(0)` — no penalty for retreating from target, enabling "hovering" behavior.

```python
# BEFORE: no retreat penalty below 1.5m
approach_reward = np.clip(raw_approach, -0.3, 1.0)
approach_reward = np.where(d < 1.5, np.maximum(approach_reward, 0.0), approach_reward)  # free retreat!

# AFTER: symmetric penalty everywhere
approach_reward = np.clip(raw_approach, -0.5, 1.0)  # retreat always penalized
```

### 4.1c Round6 Fixes — Reward Budget Root Cause (Feb 10 Session 5)

After Round5 fixes, `reached%` remained near 0%. A formal **reward budget analysis** revealed the real root cause: the robot was *rationally choosing to stand still* because passive rewards dominated active rewards.

#### The Reward Budget Methodology

**Always compute the total reward for the desired behavior (walking to target) vs the degenerate behavior (standing still) over the full episode.** If the degenerate behavior earns more total reward, the policy will exploit it — no amount of HP tuning can fix a broken incentive structure.

```python
# Standing still at d=3.5m for max_episode_steps=4000:
per_step = position_tracking(0.75) + heading(0.50) + alive(0.15)  # = 1.40/step
standing_total = 1.40 * 4000 * 0.75(time_decay) = 4,185

# Walking to target in ~583 steps + 50 stopped:
walk_total = approach + forward + arrival + stop ≈ 2,031

# STANDING WINS by 2,154!
```

**Root cause**: `max_episode_steps=4000` allowed standing to accumulate 4185 passive reward, which dwarfed the 2031 reward for successfully navigating + reaching. With `max_episode_steps=1000`: standing=1046 vs walking=2031 → walking wins by 985.

#### Round6 Four Critical Fixes

| # | Fix | Before → After | Impact |
|---|-----|----------------|--------|
| 1 | `max_episode_steps` | 4000 → **1000** | Standing reward budget cut by 75% |
| 2 | `forward_velocity` | 0.8 → **1.5** | Phase5 halved the movement incentive; restored |
| 3 | Approach retreat penalty | `clip(-0.5, 1.0)` → **`clip(0.0, 1.0)`** | Step-delta retreat was punishing exploration |
| 4 | `termination` | -200 → **-100** | Too harsh; caused risk aversion |

**Result**: Round6 v4 achieved **27.7% reached** at 12M steps — up from 0% with Round5.

#### TensorBoard Component Analysis (diagnostic technique)

Breaking down reward by component revealed the actual incentive structure:

| Category | Per-Episode Total | Notes |
|----------|-------------------|-------|
| **Passive (standing)**: position + heading + alive | 960 | Dominates |
| **Active (walking)**: forward + approach + distance | 134 | Small |
| **Penalties (movement cost)**: stability penalties | -430 | Cancels active |

**Key insight**: When `forward_velocity` was reduced from 1.5 to 0.8 in Phase5, the active reward signal (134/episode) was completely cancelled by movement penalties (-430/episode). The robot had **negative incentive to walk**.

### 4.2 Curriculum Learning (How Do I Train in Stages?)

Curriculum learning is managed in `starter_kit_schedule/` and via the AutoML pipeline. The idea is to train in stages, starting with easier spawn ranges and progressing to harder ones, using warm-starts and promotion criteria.

**Example curriculum:**
- Stage A: spawn_inner_radius=3.0, spawn_outer_radius=8.0 (medium difficulty)
- Stage B: spawn_inner_radius=6.0, spawn_outer_radius=10.0 (competition-like)
- Stage C: spawn_inner_radius=9.0, spawn_outer_radius=10.0 (final fine-tuning)

Promotion criteria are based on reached_fraction, inner_fence, and mean distance metrics.

#### Reward Budget Audit Checklist

Before launching any training run, verify:
1. **Standing reward** = (position_tracking + heading + alive) × max_episode_steps × time_decay
2. **Walking reward** = standing_reward_partial + approach + forward + arrival + stop bonuses
3. **Walking > Standing?** If not, shorten episode length or increase movement incentives
4. **Death cost** = termination / alive_budget — should be 10-50% (too low = reckless, too high = conservative)
5. **Penalty budget** = sum of stability penalties per episode — must not cancel movement rewards

#### Example: Running Curriculum AutoML
```powershell
uv run starter_kit_schedule/scripts/automl.py --mode stage --budget-hours 12 --hp-trials 8
```

---

## 5. RL Hyperparameters and AutoML Optimization: How Do I Tune for Best Results?

### 5.1 RL Hyperparameters (What Should I Change?)

RL hyperparameters (learning rate, rollouts, epochs, mini_batches, etc.) are set in RL config dataclasses. For competition, use the values found by HP optimization (see AutoML logs and `starter_kit/navigation1/vbot/rl_cfgs.py`).

#### Example: RL Config in `rl_cfgs.py`
```python
@rlcfg("vbot_navigation_section001")
@dataclass
class VBotSection001PPOConfig(PPOCfg):
    learning_rate: float = 5e-4
    rollouts: int = 32
    ...
```

### 5.2 AutoML Pipeline (How Do I Search for the Best Settings?)

The AutoML pipeline (`starter_kit_schedule/scripts/automl.py`) performs joint hyperparameter and reward search using Bayesian/random/grid optimization. It runs curriculum stages, tracks progress, and writes reports.

**Usage:**
```powershell
uv run starter_kit_schedule/scripts/automl.py --mode stage --budget-hours 12 --hp-trials 8
```

Results are logged in `starter_kit_log/{automl_id}/` and progress is tracked in `starter_kit_schedule/progress/automl_state.yaml`.

### 5.3 AutoML Scoring Alignment with Competition (Round5/6)

The AutoML `compute_score()` function was updated to align with competition scoring:

```python
# OLD: reward-dominated (40% reward, 30% reached, 20% distance, 10% speed)
# NEW: competition-aligned (binary scoring — reach inner fence +1, reach center +1)
score = (
    0.60 * success_rate +           # Most important: did it reach?
    0.25 * (1 - termination_rate) +  # Stay alive (don't fall)
    0.10 * min(reward / 10, 1.0) +  # Reward as tiebreaker
    0.05 * (1 - min(ep_len/1000, 1))  # Speed bonus (Round6: /1000 not /4000)
)
```

### 5.4 AutoML Search Space Tightening (Round6)

The search spaces were tightened around the proven Round6 v4 config to avoid wasting trials on bad regions:

| Parameter | Old Range | New Range | Rationale |
|-----------|-----------|-----------|----------|
| learning_rate | [1e-5, 1e-3] | **[2e-4, 8e-4]** | Narrowed around proven 5e-4 |
| forward_velocity | [0.3, 1.2] | **[1.0, 2.5]** | Must be ≥1.0 for walking incentive |
| approach_scale | [2, 10] | **[15, 50]** | Step-delta needs high scale |
| termination | [-300, -100] | **[-150, -50]** | -200 too harsh, -100 worked |
| network sizes | incl [128,64] | **removed** | Too small for 54-dim obs |
| rollouts | incl 16 | **removed** | Too few for stable updates |

---

## 6. Competition Rules and Scoring: How Do I Win?

### 6.1 Stage 1 (Flat Navigation)
- 10 robots start from the outer blue fence (R≈10m), must reach the inner fence (R≈0.75m) and then the center (0,0).
- +1 point for reaching the inner fence, +1 for reaching the center, per robot (max 20 points).
- Falling or going out of bounds forfeits all points for that robot.

### 6.2 Stage 2 (Obstacle Course)
- Multiple sections: waves, stairs, bridges, obstacles.
- Points for reaching checkpoints, special zones, and performing celebration actions.
- Falling or going out of bounds forfeits all points for that section.

**See `starter_kit_docs/MotrixArena_S1_计分规则讲解.md` for full details.**

---

## 7. Practical Workflow and Troubleshooting: What Should I Do Day-to-Day?

### 7.1 Typical Workflow
1. Install dependencies: `uv sync --all-packages --all-extras`
2. Edit reward scales and curriculum in `starter_kit/navigation1/vbot/cfg.py`
3. Edit RL hyperparameters in `starter_kit/navigation1/vbot/rl_cfgs.py`
4. Train: `uv run scripts/train.py --env vbot_navigation_section001 --render`
5. Monitor with TensorBoard: `uv run tensorboard --logdir runs/vbot_navigation_section001`
6. Evaluate: `uv run scripts/play.py --env vbot_navigation_section001`
7. Run AutoML for optimization: `uv run starter_kit_schedule/scripts/automl.py --mode stage --budget-hours 12 --hp-trials 8`

### 7.2 Debugging Tips
- If training is unstable, check reward budget ratios (see reward docs).
- If reached_fraction collapses, suspect reward hacking (policy prefers staying alive over reaching goal).
- Use `scripts/view.py` to visualize environment and debug spawn/termination logic.
- **Use `scripts/capture_vlm.py` for automated VLM visual analysis** — captures frames of the trained policy and sends them to a VLM for behavior diagnosis, gait quality assessment, and reward engineering suggestions:
  ```powershell
  uv run scripts/capture_vlm.py --env vbot_navigation_section001 --max-frames 20
  ```
- Check logs in `runs/` and AutoML reports in `starter_kit_log/` for experiment history.
- If you get errors about missing registration, make sure the right `__init__.py` files are importing your env modules!

### 7.3 Known Reward Hacking Patterns

| Pattern | Symptom | Root Cause | Fix |
|---------|---------|------------|-----|
| **Lazy Robot** | Reward↑ while reached%↓ | alive_bonus dominates arrival_bonus (90:1 ratio) | Reduce alive_bonus, increase arrival_bonus |
| **Standing Dominance** | 0% reached despite correct rewards | max_episode_steps too long → passive rewards exceed walking rewards | Shorten episode (4000→1000 steps). **Always audit reward budget!** |
| **Sprint-Crash** | Episode length collapses, fwd_vel→max | Per-episode rewards favor many short episodes | Speed cap, speed-distance coupling penalty |
| **Touch and Die** | Robot touches target then crashes | alive_bonus zeroed after reaching → no survival incentive | Keep alive_bonus always active |
| **Deceleration Moat** | Robot hovers at ~1m, never reaches | near_target_speed penalty zone too wide (2m) | Reduce to 0.5m or use distance-proportional coupling |
| **Fly-Through** | stop_bonus earned at high speed | No speed gate on stop_bonus | Gate stop_bonus on speed < 0.3 m/s |
| **Conservative Hovering** | Robot approaches to ~0.5m then stalls | termination too harsh (>-250) | Reduce to -150 with speed-distance coupling |
| **Negative Walking Incentive** | Robot stands despite movement reward | Stability penalties (-430/ep) cancel movement rewards (+134/ep) | Increase forward_velocity scale to overcome penalty budget |

---

## 8. File-by-File Codebase Navigation: What Does Each File Do?

### 8.1 motrix_envs
- `src/motrix_envs/np/env.py`: The main base class for all NumPy-based environments. Handles the simulation loop, state management, and step logic. **Read this to understand the environment lifecycle!**
- `src/motrix_envs/registry.py`: Handles environment registration, decorators, and the `make()` function. **If your env isn’t showing up, check here.**
- `src/motrix_envs/base.py`: Abstract base class for all environments. Defines the config dataclass (`EnvCfg`).
- `src/motrix_envs/np/renderer.py`: Handles visualization and rendering of environments.
- `src/motrix_envs/np/reward.py`: Utility reward shaping functions (tolerance, sigmoid, etc.).

### 8.2 motrix_rl
- `src/motrix_rl/cfgs.py`: RL config dataclasses and registration for built-in envs.
- `src/motrix_rl/registry.py`: RL config registry and decorator logic. **If your RL config isn’t being picked up, check here.**
- `src/motrix_rl/skrl/cfg.py`: PPO config dataclass for SKRL.

### 8.3 starter_kit/navigation1/vbot
- `cfg.py`: VBot env configs and reward scales (Stage 1). **This is where you tune reward weights!**
- `vbot_section001_np.py`: VBot environment implementation (Stage 1). **Read this to see how the robot is spawned, how rewards are computed, and how termination is handled.**
- `rl_cfgs.py`: RL hyperparameters for Stage 1. **Tweak these for PPO tuning.**

### 8.4 scripts
- `train.py`: Main training entry point. Handles env import, backend selection, and launches PPO training.
- `play.py`: Policy evaluation and demo. Auto-finds latest/best checkpoint if not specified.
- `view.py`: Visualizes the environment with random actions. Great for debugging spawn/termination logic.

### 8.5 starter_kit_schedule
- `scripts/automl.py`: AutoML HP search engine and curriculum pipeline. **This is the main entry for curriculum learning and hyperparameter search.**
- `progress/`: Tracks experiment progress.
- `templates/`: YAML templates for curriculum, reward configs, search spaces.

---

## 9. Advanced Topics and Further Reading

- **Reward/Penalty Engineering:** See `starter_kit_docs/MotrixArena_S1_四足机器人奖励函数设计与越障任务优化.md` for detailed reward design strategies.
- **Curriculum Learning:** See `.github/skills/curriculum-learning/SKILL.md` for multi-stage training best practices.
- **Hyperparameter Optimization:** See `.github/skills/hyperparameter-optimization/SKILL.md` for search strategies.
- **Competition Guides:** See `starter_kit_docs/` for scoring, rules, and technical explanations.

---

## 10. Frequently Asked Questions (FAQ)

**Q: My environment isn’t showing up in the registry!**
A: Make sure your config class is decorated with `@registry.envcfg` and your env class with `@registry.env`. Also, make sure the module is imported somewhere (see `__init__.py`).

**Q: My reward function isn’t working as expected!**
A: Double-check the reward scales in your config, and use the reward shaping utilities in `motrix_envs/np/reward.py` for smooth shaping.

**Q: Training is unstable or reward collapses.**
A: Check for reward hacking (policy finds loopholes in your reward). Try adjusting alive_bonus, arrival_bonus, and termination penalties. Use curriculum learning to ease the task.

**Q: How do I add a new robot or task?**
A: Copy an existing env config and env class, register them with new names, and add your logic. Don’t forget to import your module in the right `__init__.py`!

**Q: Where do I find the best hyperparameters?**
A: Use the AutoML pipeline (`starter_kit_schedule/scripts/automl.py`) and check the logs in `starter_kit_log/`.

---

## 11. Final Advice

MotrixLab is a powerful, modular RL framework for robotics. Take your time to read the configs, experiment with reward scales, and use the scripts to run and debug your experiments. The best way to learn is to try small changes and observe their effects in training and evaluation. Good luck, and don’t hesitate to ask for help or clarification!