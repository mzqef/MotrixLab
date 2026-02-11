---
name: hyperparameter-optimization
description: Unified PPO hyperparameter and reward/penalty weight search for VBot navigation. Grid, random, and Bayesian optimization across learning rate, network architecture, training dynamics, and reward scales.
---

## Purpose

**Unified hyperparameter tuning** for PPO training — reward/penalty weights ARE hyperparameters and are searched in the same loop:

- PPO dynamics (learning rate, entropy, clipping, epochs)
- Network architecture (layer sizes)
- **Reward/penalty weights** (navigation rewards, stability penalties, termination)
- Grid/random/Bayesian search strategies
- Parameter importance analysis
- Best config export and comparison

> **🔴 AutoML-First Policy (MANDATORY):**
> **NEVER** use `train.py` for parameter search. The AutoML pipeline (`automl.py`) handles unified HP+reward search.
> See `.github/copilot-instructions.md` for the full policy.
>
> `train.py` is ONLY for: smoke tests (<500K steps), `--render` visual debug, or final deployment runs.

> **Core Design Principle:** Reward weights and PPO parameters interact. Searching them separately is suboptimal — a high termination penalty needs a different learning rate than a low one. The AutoML pipeline searches both jointly in a single trial loop.

> **IMPORTANT — Operational Guardrails:**
> - The AutoML pipeline (`automl.py`) handles unified HP+reward search automatically. Do NOT re-implement search logic.
> - `sample_from_space()` already returns native Python types (int, float). The numpy serialization bug is **fixed**.
> - To run search, use the command in Quick Start below. Do not re-read or re-inspect `automl.py`.

> **Related Skills:**
> - `training-pipeline` — Hub with Quick Start (start here)
> - `reward-penalty-engineering` — Exploration methodology for **what** to search (diagnosis → hypothesis → test cycle)
> - `training-campaign` — Execute search experiments
> - `curriculum-learning` — Stage-specific tuning

## Quick Start

```powershell
# Launch AutoML unified search (HP + reward weights)
uv run starter_kit_schedule/scripts/automl.py `
    --mode stage `
    --budget-hours 12 `
    --hp-trials 8

# Monitor search progress
Get-Content starter_kit_schedule/progress/automl_state.yaml
```

## When to Use

| Task | Use This |
|------|----------|
| Tune learning rate | ✅ |
| Search network architectures | ✅ |
| Optimize PPO dynamics | ✅ |
| Compare search methods | ✅ |
| **Search reward/penalty weights** | ✅ (built-in — every trial samples reward scales too) |
| Diagnose bad behavior to decide **what** to add/remove | ❌ → Use `reward-penalty-engineering` |

## Unified Search Space

Each trial samples **both** PPO parameters and reward weights. This is the complete search space:

### PPO Parameters

| Parameter | Type | Range | Default | Impact |
|-----------|------|-------|---------|--------|
| `learning_rate` | log-uniform | 1e-5 – 1e-3 | 3e-4 | **Critical** — convergence speed |
| `discount_factor` | choice | 0.95–0.999 | 0.99 | Credit assignment horizon |
| `lambda_param` | choice | 0.9–0.99 | 0.95 | GAE bias-variance |
| `ratio_clip` | choice | 0.1–0.3 | 0.2 | Policy update magnitude |
| `entropy_loss_scale` | log-uniform | 1e-4 – 1e-2 | 1e-3 | Exploration strength |
| `learning_epochs` | choice | 4, 5, 6, 8 | 5 | Updates per rollout |
| `mini_batches` | choice | 16, 32, 64 | 32 | Batch size |
| `rollouts` | choice | 16, 24, 32, 48 | 24 | Steps before update |

### Network Architecture Options

| Config | Layers | Parameters | Use Case |
|--------|--------|------------|----------|
| Small | [128, 64] | ~20K | Fast iteration |
| Medium | [256, 128, 64] | ~80K | **Default** |
| Large | [512, 256, 128] | ~200K | Complex policies |
| Deep | [256, 256, 256] | ~150K | Feature extraction |

### Reward / Penalty Weights

Reward weights come from `starter_kit/{task}/vbot/cfg.py` → `RewardConfig.scales`. Each trial samples a complete reward weight vector jointly with PPO params.

> **Concrete reward weight ranges, default values, and known warnings** (sprint-crash, deceleration moat, lazy robot) are task-specific.
> See: `starter_kit_docs/{task-name}/Task_Reference.md` → "Reward Search Space" section.

General schema for reward weight search entries:

```yaml
reward_weights:
  <reward_name>:
    type: "uniform" | "loguniform" | "choice"
    low: <min_value>     # for uniform/loguniform
    high: <max_value>    # for uniform/loguniform
    values: [...]        # for choice
    prior: <default>     # center of search
    category: "navigation" | "stability" | "efficiency" | "terminal"
```

**Key principle:** `arrival_bonus` (or equivalent goal reward) must be large enough to dominate per-step bonuses like `alive_bonus × typical_episode_length`. See `.github/copilot-instructions.md` → "Reward Budget Audit Principle".

## Search Space Schema

```yaml
# starter_kit_schedule/templates/ppo_search_space.yaml
search_space_id: "unified_navigation_v1"

# Fixed parameters
fixed:
  seed: 42
  num_envs: 2048
  grad_norm_clip: 1.0
  ratio_clip: 0.2
  value_clip: 0.2

# Searchable PPO parameters
ppo_params:
  learning_rate:
    type: "loguniform"
    low: 1e-5
    high: 1e-3
    prior: 3e-4
    
  policy_hidden_layer_sizes:
    type: "categorical"
    choices:
      - [128, 64]
      - [256, 128, 64]
      - [512, 256, 128]
      - [256, 256, 256]
    prior: [256, 128, 64]
    
  rollouts:
    type: "choice"
    values: [16, 24, 32, 48]
    prior: 24
    
  learning_epochs:
    type: "choice"
    values: [4, 5, 6, 8]
    prior: 5
    
  entropy_loss_scale:
    type: "loguniform"
    low: 1e-4
    high: 1e-2
    prior: 1e-3

# Searchable reward/penalty weights (sampled jointly with PPO params)
reward_weights:
  position_tracking:
    type: "uniform"
    low: 0.5
    high: 5.0
    prior: 2.0
    
  heading_tracking:
    type: "uniform"
    low: 0.1
    high: 2.0
    prior: 1.0
    
  approach_scale:
    type: "uniform"
    low: 2.0
    high: 10.0
    prior: 4.0
    
  arrival_bonus:
    type: "uniform"
    low: 5.0
    high: 25.0
    prior: 10.0
    
  orientation:
    type: "uniform"
    low: -0.3
    high: -0.01
    prior: -0.05
    
  lin_vel_z:
    type: "uniform"
    low: -2.0
    high: -0.1
    prior: -0.5
    
  action_rate:
    type: "uniform"
    low: -0.05
    high: -0.001
    prior: -0.01
    
  termination:
    type: "choice"
    values: [-500, -300, -200, -100, -50]
    prior: -200

# Constraints (filter bad combinations)
constraints:
  - "learning_rate > 5e-4 and learning_epochs > 8"  # Skip: unstable
  - "entropy_loss_scale > 0.05 and ratio_clip < 0.15"  # Skip: no learning
  - "termination < -400 and learning_rate > 5e-4"  # Skip: harsh penalty + fast LR
```

## Search Methods

| Method | Trials | Best For | Command Flag |
|--------|--------|----------|--------------|
| Grid | All combos | Small spaces (<20) | `--search-method grid` |
| Random | 30-100 | Large spaces | `--search-method random` |
| Bayesian | 20-50 | Expensive evals | `--search-method bayesian` |

### Grid Search
- Exhaustive coverage
- Good for discrete choices (layer sizes, termination penalty)
- Exponential cost with parameters — only practical for 2-3 params

### Random Search
- Better than grid for >3 parameters
- Samples PPO and reward weights jointly from distributions
- Can set `--hp-trials 50`

### Bayesian Optimization
- Builds surrogate model over the joint (PPO + reward weights) space
- Perturbs best-so-far config in both PPO and reward dimensions
- Best for expensive evaluations

## How It Works

Each AutoML trial:

```
1. Sample PPO params   (lr, entropy, architecture, rollouts, ...)
2. Sample reward weights (position_tracking, termination, ...)
3. Write unified config JSON → starter_kit_log/automl/<id>/configs/
4. Launch train_one.py subprocess with both HP + reward overrides
5. Evaluate via TensorBoard logs
6. Score = f(reward_mean, success_rate, episode_length, termination_rate)
7. Update Bayesian model → suggest next trial
```

After all trials complete, the best (HP + reward weights) pair proceeds to full training.

## Commands

```powershell
# === PREFERRED: AutoML unified search ===
uv run starter_kit_schedule/scripts/automl.py `
    --mode stage `
    --budget-hours 12 `
    --hp-trials 8

# === SINGLE TRAINING RUN (manual HP selection) ===
uv run scripts/train.py --env <env-name>

# === ANALYZE RESULTS ===
uv run starter_kit_schedule/scripts/analyze.py `
    --metric episode_reward_mean `
    --sort descending
```

## Analysis Output

```yaml
# starter_kit_log/analysis/hyperparameter_importance.yaml
importance_ranking:
  # PPO params
  - parameter: learning_rate
    importance: 0.30
    best_value: 2.1e-4
  - parameter: entropy_loss_scale
    importance: 0.12
    best_value: 8.5e-4
  - parameter: policy_hidden_layer_sizes
    importance: 0.10
    best_value: [256, 128, 64]
  # Reward weights
  - parameter: termination
    importance: 0.18
    best_value: -200
  - parameter: approach_scale
    importance: 0.15
    best_value: 6.2
  - parameter: position_tracking
    importance: 0.08
    best_value: 2.5

top_5_configs:
  - config_id: config_023
    learning_rate: 2.1e-4
    entropy_loss_scale: 8.5e-4
    termination: -200
    approach_scale: 6.2
    episode_reward_mean: 38.2
    
  - config_id: config_017
    learning_rate: 1.8e-4
    termination: -300
    episode_reward_mean: 36.8
```

## Presets

### Quick Validation (reward weights only)
```yaml
# 6 configs — fix PPO defaults, explore reward scales
ppo_params: {}  # all defaults
reward_weights:
  termination: { type: "choice", values: [-500, -200, -50] }
  approach_scale: { type: "choice", values: [4, 8] }
```

### Quick Validation (PPO only)
```yaml
# 6 configs — fix reward defaults, explore PPO
ppo_params:
  learning_rate: { type: "choice", values: [1e-4, 3e-4, 1e-3] }
  rollouts: { type: "choice", values: [24, 48] }
reward_weights: {}  # all defaults
```

### Full Unified Search
```yaml
# ~200 configs (use random sampling)
# All PPO + network + reward weight parameters
```

### Fine-Tuning Search
```yaml
# Narrow ranges around best known config
ppo_params:
  learning_rate: { type: "loguniform", low: 1e-4, high: 5e-4 }
reward_weights:
  approach_scale: { type: "uniform", low: 4.0, high: 8.0 }
  termination: { type: "choice", values: [-300, -200, -100] }
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| All configs perform similarly | Widen search ranges (especially reward weights) |
| Best config unstable | Add constraint: high LR + many epochs |
| Search takes too long | Reduce `--hp-trials`, shorter training |
| Bayesian stuck | Increase initial random trials |
| Reward weights dominate | Fix reward weights, search PPO only first |
| PPO sensitive to reward scale | Search them jointly (default behavior) |

## Best Practices

1. **Search PPO + rewards jointly** — They interact; separate search is suboptimal
2. **Start with random search** — Better coverage than grid for the combined space
3. **Use priors** — Center search on known good values from `cfg.py` defaults
4. **Add constraints** — Filter unstable combinations early
5. **Shorter training for search** — 10M steps for comparison (5M too short to detect laziness)
6. **Run multiple seeds** — At least 3 per config for reliability
7. **Log all configs** — Stored automatically in `starter_kit_log/experiments/`
8. **Diagnose first, then search** — Use `reward-penalty-engineering` skill to identify WHAT to search before tuning weights
9. **Review experiment history before launching** — See `training-pipeline` Step 0
10. **Watch for lazy robot at long horizons** — 5M trials may look great but fail at 50M. Check that reached% is increasing, not just reward.

## Empirical Findings

> **Task-specific empirical findings** (best LR, rollout sweet spots, sprint-crash thresholds, deceleration moat, anti-laziness results, LR scheduler findings) are documented in:
> `starter_kit_docs/{task-name}/Task_Reference.md` → "Empirical AutoML Findings" section.
>
> **Always check this reference before launching new searches** to avoid repeating experiments.

General lessons that apply across tasks:

| Insight | Details |
|---------|--------|
| Joint PPO+reward search | Better than separate tuning — reward weights interact with LR/entropy |
| Trial length matters | 5M steps too short to detect reward hacking; use ≥10M for serious candidates |
| Manual iteration wastes time | N manual `train.py` experiments should be 1 AutoML batch |
| Per-step bonuses are dangerous | Unconditional per-step rewards dominate goal bonuses at long horizons |
