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
> - Dual environment confusion → only `vbot_navigation_section001` registered for navigation1

## 🔴 Experiment Reports (MANDATORY check before reward changes)

> **ALWAYS read `REPORT_NAV*.md` files at `/starter_kit_docs/{<task-name>}` before modifying rewards.**

> These reports track all experiments, discoveries, and current configuration state.

**What to look for in reports:**
- Current curriculum stage (spawn_inner/outer_radius values in cfg.py)
- Which experiments used which stage config
- Promotion criteria results (reached%, ep_len stability)
- Active TODO items in "Next Steps" section
- Lessons learned that affect curriculum design
> **After completing any experiment or making significant changes**, append results to the relevant REPORT_NAV*.md file. Never overwrite existing content — the history is a chronological record.

## 🔴 AutoML-First Policy (MANDATORY)

> **NEVER use `train.py` for parameter exploration or reward hypothesis testing as long as `automl.py` works.**
> The AutoML pipeline (`automl.py`) exists precisely for this purpose and MUST be used.

## 🔴 VLM-First Visual Analysis Policy (MANDATORY)

> **ALWAYS use `capture_vlm.py` + Copilot CLI subagent for visual debugging, behavior analysis, and reward/penalty design feedback.**
> Before hand-tuning reward scales or diagnosing policy bugs by reading code alone, **capture frames and send them to the VLM**. Visual evidence beats guesswork.

### When to use the Copilot CLI subagent (skill: `subagent-copilot-cli`)

| Task | Command | Rationale |
|------|---------|-----------|
| **Policy behavior diagnosis** | `capture_vlm.py` | Automated frame capture + VLM analysis identifies gait bugs, falls, circling |
| **Reward/penalty design feedback** | `capture_vlm.py --vlm-prompt "..."` | VLM sees *actual robot behavior* and suggests reward fixes |
| **Before/after reward change comparison** | Capture before & after, send both to VLM | Visual diff of policy behavior after reward modification |
| **Failure mode analysis** | `capture_vlm.py --vlm-prompt "Focus on why robot falls"` | VLM reads leg poses, body tilt, terrain interaction |
| **Gait quality assessment** | `capture_vlm.py --vlm-prompt "Analyze gait symmetry"` | Quadruped-specific visual analysis |
| **Scene/terrain understanding** | `copilot --model gpt-4.1 ... -p "Read XML scene"` | Analyze MJCF scene files for obstacle layout |
| **Reward structure code review** | `copilot --model gpt-4.1 ... -p "Analyze cfg.py"` | Deep-dive reward function design |
| **Training curve interpretation** | `copilot --model gpt-4.1 ... -p "Read plot"` | Analyze TensorBoard export images |

### Mandatory VLM checkpoints

1. **After every reward/penalty change** → Run `capture_vlm.py` to verify the visual effect
2. **When policy shows unexpected behavior** → Run `capture_vlm.py` before editing reward code
3. **Before declaring a training run successful** → Run `capture_vlm.py` to confirm visual quality
4. **When designing new reward components** → Use VLM analysis of current behavior to identify gaps

### When to use each tool

| Task | Tool | Rationale |
|------|------|-----------|
| **Reward weight search** | `automl.py` | Batch comparison with structured reports |
| **HP tuning (lr, entropy, etc.)** | `automl.py` | Joint HP+reward search, Bayesian optimization |
| **Reward hypothesis testing** | `automl.py` | Run N trials in parallel, compare side-by-side |
| **Curriculum stage promotion** | `automl.py` | Run full stage with best HP from previous stage |
| **Smoke test (<500K steps)** | `train.py` | Quick sanity check that code runs without errors |
| **Visual debugging (--render)** | `train.py` | Watch the robot's behavior in real-time |
| **Policy evaluation / playback** | `play.py` | Evaluate a trained checkpoint |
| **VLM visual analysis** | `capture_vlm.py` | Play policy, capture frames, send to gpt-4.1 for bug detection |
| **Reward design feedback** | `capture_vlm.py` | VLM sees behavior, suggests reward/penalty changes |
| **Code/config analysis** | Copilot CLI subagent | Deep analysis of reward structure, HP config, scene XML |

### What NOT to do

❌ **Do NOT** iterate manually with `train.py`, changing one reward weight, running, reading TensorBoard, killing, changing another weight, running again. This is **manual one-at-a-time search** — slow, error-prone, and wasteful.

❌ **Do NOT** run `train.py` with `--max-env-steps 5000000` multiple times to "compare". Use `automl.py --hp-trials N` instead.

❌ **Do NOT** hand-tune reward scales by repeatedly editing `cfg.py` and running `train.py`. Update `REWARD_SEARCH_SPACE` in `automl.py` and let the search find the best values.

❌ **Do NOT** diagnose policy behavior bugs by reading code alone. Run `capture_vlm.py` to get visual evidence first.

❌ **Do NOT** design new reward/penalty components without first running `capture_vlm.py` on the current policy to see what behavior needs fixing.

### What TO do

✅ When testing a new reward component: Add it to `REWARD_SEARCH_SPACE` in `automl.py`, run batch search.

✅ When comparing reward weight variants: Define the search range in `automl.py`, run `--hp-trials 8+`.

✅ When results come back: Read the AutoML report in `starter_kit_log/automl_*/report.md` for structured comparison.

✅ **After any training run**: Run `capture_vlm.py` to get VLM visual feedback on the learned behavior.

✅ **Before designing new rewards**: Run `capture_vlm.py` on current best policy to see what behavior to fix.

✅ **For reward/penalty engineering**: Use `capture_vlm.py --vlm-prompt "Suggest reward changes"` to get data-driven suggestions from visual evidence.

### AutoML command (use this, not train.py)

```powershell
# Standard batch search (HP + reward weights)
uv run starter_kit_schedule/scripts/automl.py --mode stage --budget-hours 8 --hp-trials 15

# Monitor
Get-Content starter_kit_schedule/progress/automl_state.yaml

# Results
Get-Content starter_kit_log/automl_*/report.md
```

### Exception: `train.py` is acceptable ONLY for:

1. **Smoke tests** — `--max-env-steps 200000` to verify code changes compile and run
2. **Visual debugging** — `--render` to watch robot behavior in real-time
3. **Final deployment runs** — After AutoML has found the best config, train the winning config to full steps
4. **Warm-start curriculum promotion** — Loading a checkpoint and running to completion with known-good config

## 🔴 Pre-Training Checklist (MANDATORY)

> **ALWAYS review existing experiment history before starting new training or making changes.**

```powershell
# 1. Read experiment reports for this task
Get-Content starter_kit_docs/<task-name>/REPORT_*.md

# 2. List AutoML runs and their outcomes
Get-ChildItem starter_kit_log/automl_* -Directory | ForEach-Object {
    $state = Join-Path $_.FullName "state.yaml"
    if (Test-Path $state) { Write-Host "=== $($_.Name) ==="; Get-Content $state | Select-Object -First 20 }
}

# 3. Check current automl progress
if (Test-Path starter_kit_schedule/progress/automl_state.yaml) {
    Get-Content starter_kit_schedule/progress/automl_state.yaml
}

# 4. List recent training runs
Get-ChildItem runs/<env-name>/ -Directory | Sort-Object Name -Descending | Select-Object -First 10

# 5. Review reward library
Get-ChildItem starter_kit_schedule/reward_library/ -Recurse -Filter "*.yaml" -ErrorAction SilentlyContinue
```

**What to look for:**
- Best reward/composite score achieved so far
- Which HP configurations worked best
- Known failure modes already diagnosed
- Active TODO items in the report's "Next Steps" section
- Current curriculum stage and promotion criteria

## 🔴 Reward Engineering Methodology (MANDATORY)

### Reward Budget Audit Principle

> **Before committing to any reward configuration**, compute the maximum cumulative reward for:
> 1. The **desired behavior** (e.g., walk to target, stop, collect bonuses)
> 2. The **degenerate behavior** (e.g., stand still, sprint-crash-reset, hover safely)
>
> If degenerate reward ≥ desired reward, the agent **WILL** find and exploit the degenerate strategy given enough training time. Fix the budget before training.

**Template:**
```
STANDING STILL for max_steps:
  per_step_passive_reward × max_steps × time_decay ≈ ???

COMPLETING TASK in ~N steps:
  one_time_bonuses + active_per_step × N + passive_per_step × N ≈ ???

If STANDING > COMPLETING → broken incentive. Fix before training.
```

### One Variable Per Experiment

> **Never change more than one reward dimension per experiment cycle.** If you change both the termination penalty AND add a new gait reward, you cannot attribute outcomes. Weight adjustments should use **multiplicative steps** (×0.5, ×2, ×10), not additive.

### Exploration Cycle

Every reward/penalty change follows this cycle:

```
DIAGNOSE → HYPOTHESIZE → IMPLEMENT → TEST → EVALUATE → ARCHIVE
    ↑                                                      |
    └──────────────────────────────────────────────────────┘
```

1. **Diagnose**: Watch the policy (`play.py`, `capture_vlm.py`). Identify the specific behavioral gap — not "reward is too low" but "robot hovers at 1m and never approaches."
2. **Hypothesize**: Formulate a testable hypothesis: "If I add signal X with weight W, the robot should do Y, but might also Z."
3. **Implement**: Minimal change, one variable at a time. Adjust weights before adding code.
4. **Test**: Use `automl.py --hp-trials 8+` for batch comparison. NEVER iterate manually with `train.py`.
5. **Evaluate**: Did target behavior improve? Side effects? Consistent across seeds? Reward curve healthy?
6. **Archive**: Record result in `starter_kit_schedule/reward_library/` — both successes AND failures.

### Discovery Strategies

When generating reward hypotheses, use these abstract strategies:

| Strategy | Description |
|----------|-------------|
| **Inversion** | Directly penalize the undesired behavior (bouncing → penalize vertical velocity) |
| **Gradient Shaping** | If the robot is stuck, the reward surface is flat — add a signal that creates local gradient |
| **Proxy Decomposition** | Break the competition score into sub-goals, create a signal for each |
| **Biomimetic Analogy** | What would a real animal "want" in this situation? |
| **Ablation Discovery** | Remove one existing reward component — if nothing changes, it was irrelevant |
| **Competition-Score Alignment** | Compare training reward to competition scoring — gaps indicate missing signals |
| **Library Browse** | Check `starter_kit_schedule/reward_library/` before inventing new components |

### Anti-Patterns

| Anti-Pattern | Why It Fails | Instead |
|--------------|-------------|---------|
| Manual `train.py` iteration | Slow one-at-a-time search; no structured comparison | `automl.py --hp-trials 8+` |
| Changing 3+ rewards at once | Cannot attribute outcomes | One variable per cycle |
| Only watching reward curves | High reward ≠ good behavior | Always watch policy visually + check reached%, distance, ep_len |
| Discarding failed experiments | Wastes future effort | Archive in `reward_library/rejected/` |
| Tuning weights without diagnosis | Blind search | Diagnose behavior first (VLM) |
| Unconditional per-step bonuses | Robot learns to survive, not navigate | Make per-step bonuses conditional or time-decayed |
| Goal reward too small vs per-step | Per-step accumulation dominates one-time bonuses | Goal reward must dominate: `bonus > per_step × typical_episode_len / 3` |

## 🔴 Warm-Start & Curriculum Best Practices (MANDATORY)

| Practice | Guidance |
|----------|----------|
| **LR on transfer** | Reduce to 0.3–0.5× of original to prevent catastrophic forgetting |
| **Optimizer state** | Reset optimizer when warm-starting (momentum/variance from old task is stale) |
| **Never warm-start from degraded runs** | Optimizer state carries instability — use fresh training |
| **Test promotion criteria** | Run 1M steps before committing to full stage |
| **Checkpoint frequently** | Every 500 iters for early stages, 1000 for later |
| **Anti-laziness at long horizons** | Conditional per-step bonuses + time_decay are essential for >10M step training |

## Hardware Performance (measured)

| Metric | Value |
|--------|-------|
| Backend | PyTorch (JAX NOT available) |
| GPU | NVIDIA (torch_gpu=True) |
| Training speed | ~7,500-12,500 steps/sec |
| 200K steps | ~16s |
| 2M steps | ~4 min |
| 5M steps (HP trial) | ~7-8 min |
| 50M steps (full run) | ~70 min |
| 100M steps | ~2.2 hours |

## Essential Commands

```powershell
# Install (Python 3.10 required, uses UV)
uv sync --all-packages --all-extras

# === PRIMARY: AutoML Pipeline (USE THIS for all parameter exploration) ===
uv run starter_kit_schedule/scripts/automl.py --mode stage --budget-hours 8 --hp-trials 15

# Monitor AutoML state
Get-Content starter_kit_schedule/progress/automl_state.yaml

# Read AutoML results
Get-Content starter_kit_log/automl_*/report.md

# === SECONDARY: Single run (smoke tests, visual debug, final deployment ONLY) ===
uv run scripts/train.py --env <env-name>                 # auto-selects JAX/Torch
uv run scripts/train.py --env <env-name> --render         # with live visualization
uv run scripts/train.py --env <env-name> --train-backend torch

# Evaluate / Play (auto-finds latest best checkpoint)
uv run scripts/play.py --env <env-name>
uv run scripts/play.py --env <env-name> --policy runs/<env>/.../<checkpoint>.pickle

# VLM Visual Analysis (play policy, capture frames, analyze with gpt-4.1)
uv run scripts/capture_vlm.py --env <env-name>
uv run scripts/capture_vlm.py --env <env-name> --max-frames 25 --vlm-prompt "Focus on gait bugs"

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
- **No test suite** currently exists. Validate changes by running `uv run scripts/train.py --env <env> --render` for a few hundred steps, then verify with `uv run scripts/capture_vlm.py --env <env>` for VLM-based visual analysis.
- **Visual debugging**: Always use `capture_vlm.py` + Copilot CLI subagent (`subagent-copilot-cli` skill) for behavior analysis, failure diagnosis, and reward/penalty design feedback. See the skill file at `.github/skills/subagent-copilot-cli/SKILL.md`.
- Comments are frequently in **Chinese** (Simplified). Maintain the existing language when editing comments.

## TUTORIAL, REPORT and LOG REQUIREMENTS

- For each task, in `starter_kit_docs/{<task-name>}/`, keep periodically updated:
  - `Tutorial.md`, `Tutorial_RL_Reward_Engineering.md` — guides to the task
  - `REPORT_*.md` — chronological experiment reports (append-only, never overwrite)
  - `Task_Reference.md` — task-specific reference: environment IDs, reward scales, search spaces, terrain descriptions, robot config, competition scoring. **Skills reference this file for concrete values instead of hardcoding them.**

## AI Agent Skills Reference

| Skill | File | Purpose | When to Use |
|-------|------|---------|-------------|
| **training-pipeline** | `.github/skills/training-pipeline/SKILL.md` | Index skill and entry point for all RL training tasks. Quick Start commands, known fixed issues, and routing table to other skills. | Start here for any training-related task. Review before launching any experiment. |
| **curriculum-learning** | `.github/skills/curriculum-learning/SKILL.md` | Abstract multi-stage curriculum training methodology. Generic stage progression templates, warm-start strategies, promotion criteria. Task-specific stages in Task_Reference.md. | Designing multi-stage training plans, defining promotion thresholds, configuring warm-start LR multipliers, transferring checkpoints between environments. |
| **hyperparameter-optimization** | `.github/skills/hyperparameter-optimization/SKILL.md` | Abstract PPO hyperparameter AND reward weight search methodology. Search space schema, presets (quick/full/fine-tune), optimization strategies. Task-specific ranges in Task_Reference.md. | Tuning learning rate, network size, PPO dynamics, or reward weights. |
| **reward-penalty-engineering** | `.github/skills/reward-penalty-engineering/SKILL.md` | Process-oriented methodology for systematic reward discovery. Diagnose → Hypothesize → Implement → Test → Evaluate → Archive cycle. Abstract "Lazy Robot" case study. | Identifying behavioral gaps, formulating reward hypotheses, running controlled experiments, archiving findings in the reward library. |
| **training-campaign** | `.github/skills/training-campaign/SKILL.md` | Execute and monitor long-running RL training campaigns. Generic checkpoint management, experiment logging, progress monitoring, AutoML pipeline architecture. | Starting/resuming training runs, monitoring progress, managing checkpoints, understanding the AutoML subprocess pipeline. |
| **quadruped-competition-tutor** | `.github/skills/quadruped-competition-tutor/SKILL.md` | Abstract competition strategy guide. Generic reward engineering principles, terrain traversal pattern templates, scoring optimization tactics, submission checklist. Task-specific details in Task_Reference.md. | Understanding competition structure, abstract reward strategies, terrain pattern recognition, submission preparation. |
| **mjcf-xml-reasoning** | `.github/skills/mjcf-xml-reasoning/SKILL.md` | Master guide for reading and reasoning about MuJoCo MJCF XML model files. All XML elements, geometry types, joint configurations, contact physics, height field sizing. Terrain geometry details in Task_Reference.md. | Understanding robot/scene XML definitions, debugging physics issues, analyzing terrain layouts, modifying scene files. |
| **subagent-copilot-cli** | `.github/skills/subagent-copilot-cli/SKILL.md` | Delegate visual analysis tasks to GitHub Copilot CLI. Automated VLM policy analysis via `capture_vlm.py`, screenshot analysis, training curve interpretation, PDF reading. Generic env-name patterns. | Visual debugging of trained policies, gait quality assessment, failure mode diagnosis, reward curve interpretation. |

## Key Directories

| Path | Purpose |
|------|---------|
| `scripts/` | Entry points: `train.py`, `play.py`, `view.py`, `capture_vlm.py` |
| `motrix_envs/src/motrix_envs/np/env.py` | `NpEnv` base class — core simulation loop |
| `motrix_envs/src/motrix_envs/registry.py` | Environment registry (`make`, `envcfg`, `env`) |
| `motrix_rl/src/motrix_rl/cfgs.py` | All RL training configs (PPO hyperparameters) |
| `motrix_rl/src/motrix_rl/registry.py` | RL config registry (`rlcfg`, `default_rl_cfg`) |
| `starter_kit/{task}/vbot/cfg.py` | VBot env configs with reward scales |
| `starter_kit/{task}/vbot/vbot_*_np.py` | VBot environment implementations |
| `starter_kit/{task}/vbot/xmls/` | MuJoCo MJCF scene files |
| `starter_kit_schedule/` | Training campaign pipeline, reward library, hyperparameter search |
| `starter_kit_schedule/scripts/automl.py` | AutoML HP search engine |
| `starter_kit_log/{automl_id}/` | Self-contained automl run: configs/, experiments/, index.yaml, report.md |
| `starter_kit_docs/` | Competition guides and scoring rules |
| `starter_kit_docs/{task}/Task_Reference.md` | Task-specific env IDs, reward scales, terrain data, curriculum stages |
| `runs/` | Training outputs (checkpoints, TensorBoard logs) |
| `starter_kit_log/vlm_captures/` | VLM frame captures and analysis reports |
| `.github/skills/` | AI agent skill files for specialized tasks |
| `.github/skills/subagent-copilot-cli/` | **Copilot CLI subagent skill — visual debugging, reward design, analysis** |
