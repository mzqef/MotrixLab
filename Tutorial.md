

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

### 4.2 Curriculum Learning (How Do I Train in Stages?)

Curriculum learning is managed in `starter_kit_schedule/` and via the AutoML pipeline. The idea is to train in stages, starting with easier spawn ranges and progressing to harder ones, using warm-starts and promotion criteria.

**Example curriculum:**
- Stage A: spawn_inner_radius=3.0, spawn_outer_radius=8.0 (medium difficulty)
- Stage B: spawn_inner_radius=6.0, spawn_outer_radius=10.0 (competition-like)
- Stage C: spawn_inner_radius=9.0, spawn_outer_radius=10.0 (final fine-tuning)

Promotion criteria are based on reached_fraction, inner_fence, and mean distance metrics.

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
- Check logs in `runs/` and AutoML reports in `starter_kit_log/` for experiment history.
- If you get errors about missing registration, make sure the right `__init__.py` files are importing your env modules!

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