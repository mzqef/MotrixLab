# Copilot Instructions — MotrixLab

## Project Overview

MotrixLab is an RL framework for robotics built on the **MotrixSim** physics engine. The active focus is the **MotrixArena S1** quadruped navigation competition (VBot robot traversing obstacle courses). Two workspace packages cooperate:

- **`motrix_envs`** — simulation environment definitions (observation, action, reward). Framework-agnostic, NumPy backend only (`"np"`).
- **`motrix_rl`** — RL training integration. Currently only **SKRL PPO** with JAX or PyTorch backends.

Competition work lives in `starter_kit/navigation1/` (flat) and `starter_kit/navigation2/` (obstacles/stairs). These are **not** inside the `motrix_envs` package — they register environments via the same `@registry.envcfg` / `@registry.env` decorators but are imported through their respective `__init__.py` files.

## Operational Guardrails

> **CRITICAL for AI agents:** The pipeline code is **tested and working**. When asked to launch training or automl:
> 1. Do NOT re-read pipeline files (`automl.py`, `train_one.py`, `evaluate.py`, `run.py`) to "check" them.
> 2. Do NOT re-verify imports, dependencies, or JSON serialization — known issues are **fixed**.
> 3. Go directly to the command and run it.
>
> **Known fixed issues:**
> - numpy int64/float64 JSON serialization → fixed in `automl.py` with `_NumpyEncoder` + native types in `sample_from_space()`
> - Import order (`@rlcfg` requires env registration first) → fixed in `train_one.py` (imports `vbot` before `motrix_rl`)
> - Zero reward function → fully implemented in `vbot_section001_np.py`
> - Dual environment confusion → only `vbot_navigation_section001` registered for nav1

## Essential Commands

```powershell
# Install (Python 3.10 required, uses UV)
uv sync --all-packages --all-extras

# Train (single run)
uv run scripts/train.py --env <env-name>                 # auto-selects JAX/Torch
uv run scripts/train.py --env <env-name> --render         # with live visualization
uv run scripts/train.py --env <env-name> --train-backend torch

# AutoML / HP Search (preferred for optimization)
uv run starter_kit_schedule/scripts/automl.py --mode stage --budget-hours 12 --hp-trials 8

# Monitor AutoML state
Get-Content starter_kit_schedule/progress/automl_state.yaml

# Evaluate / Play (auto-finds latest best checkpoint)
uv run scripts/play.py --env <env-name>
uv run scripts/play.py --env <env-name> --policy runs/<env>/.../<checkpoint>.pickle

# View environment (no training)
uv run scripts/view.py --env <env-name>

# TensorBoard
uv run tensorboard --logdir runs/<env-name>

# Lint
uv run ruff check .
```

## Architecture & Registry Pattern

Environments and RL configs use a **dual-registry decorator pattern**:

1. **`@registry.envcfg("env-name")`** on a `@dataclass(EnvCfg)` — registers the config class.
2. **`@registry.env("env-name", "np")`** on a `NpEnv` subclass — registers the environment implementation.
3. **`@rlcfg("env-name")` / `@rlcfg("env-name", backend="jax")`** on a `PPOCfg` dataclass — registers RL hyperparameters. `backend=None` means both JAX and Torch.

Config must be registered before the env class. Environment modules must be imported (even if unused) to trigger registration — hence `__init__.py` files with `# noqa: F401` imports.

```
motrix_envs.registry.make("env-name", num_envs=2048)  # creates env instance
motrix_rl.registry.default_rl_cfg("env-name", "ppo", "jax")  # gets RL config
```

## Environment Implementation Pattern

Every environment subclasses `NpEnv` and implements three methods:

| Method | Purpose |
|--------|---------|
| `reset(data) → (obs, info)` | Initialize/randomize state, compute initial observations |
| `apply_action(actions, state) → state` | Convert actions to torques, set `actuator_ctrls` |
| `update_state(state) → state` | Compute obs, reward, terminated; update `state.obs/reward/terminated` |

The step loop is: `apply_action → physics_step (sim_substeps iterations) → update_state → reset done envs`.

Key state object: `NpEnvState(data, obs, reward, terminated, truncated, info)` where `data` is `mtx.SceneData`.

## VBot Navigation Conventions (Competition)

- **12 actuators** (4 legs × 3 joints: hip, thigh, calf). Action space: `[-1, 1]^12`, scaled by `action_scale`.
- **PD control**: `torque = kp*(target - current) - kv*velocity`, with per-joint torque limits.
- **Observation**: 54-dim vector — linear velocity, gyro, projected gravity, joint pos/vel, last actions, velocity commands, position/heading error, distance, reached flag.
- **Reward scales** defined in `RewardConfig.scales` dict inside `cfg.py`. Modify weights here to tune behavior.
- **Termination**: body contact sensor (`base_contact > 0.01`) or episode timeout.
- **Configs inherit**: `VBotEnvCfg → VBotStairsEnvCfg → VBotSection0*EnvCfg`. Each section overrides `model_file`, `InitState`, `Commands`, and timing.

## Code Style & Tooling

- **Python 3.10 only** (`==3.10.*`). UV workspace with two member packages.
- **Ruff**: line length 120, `I/E/F/W` rules, `F401/F403` suppressed in `__init__.py`.
- **Dataclass configs** — all configuration uses `@dataclass`, not YAML/JSON. Nested dataclasses for sub-configs (`NoiseConfig`, `ControlConfig`, `RewardConfig`, etc.).
- **No test suite** currently exists. Validate changes by running `uv run scripts/train.py --env <env> --render` for a few hundred steps.
- Comments are frequently in **Chinese** (Simplified). Maintain the existing language when editing comments.

## Key Directories

| Path | Purpose |
|------|---------|
| `scripts/` | Entry points: `train.py`, `play.py`, `view.py` |
| `motrix_envs/src/motrix_envs/np/env.py` | `NpEnv` base class — core simulation loop |
| `motrix_envs/src/motrix_envs/registry.py` | Environment registry (`make`, `envcfg`, `env`) |
| `motrix_rl/src/motrix_rl/cfgs.py` | All RL training configs (PPO hyperparameters) |
| `motrix_rl/src/motrix_rl/registry.py` | RL config registry (`rlcfg`, `default_rl_cfg`) |
| `starter_kit/navigation1/vbot/cfg.py` | VBot env configs with reward scales (Stage 1) |
| `starter_kit/navigation1/vbot/vbot_section001_np.py` | VBot environment implementation (competition) |
| `starter_kit/navigation2/vbot/cfg.py` | VBot env configs (Stage 2, obstacles/stairs) |
| `starter_kit/navigation2/vbot/vbot_section*_np.py` | VBot environment implementations (Stage 2) |
| `starter_kit/navigation2/vbot/xmls/` | MuJoCo MJCF scene files |
| `starter_kit_schedule/` | Training campaign pipeline, reward library, hyperparameter search |
| `starter_kit_schedule/scripts/automl.py` | AutoML HP search engine |
| `starter_kit_log/{automl_id}/` | Self-contained automl run: configs/, experiments/, index.yaml, report.md |
| `starter_kit_docs/` | Competition guides and scoring rules |
| `runs/` | Training outputs (checkpoints, TensorBoard logs) |
| `.github/skills/` | AI agent skill files for specialized tasks |
