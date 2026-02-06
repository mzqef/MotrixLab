---
name: hyperparameter-optimization
description: PPO hyperparameter search for VBot navigation. Grid, random, and Bayesian optimization across learning rate, network architecture, and training dynamics.
---

## Purpose

**Systematic hyperparameter tuning** for PPO training:

- Grid/random/Bayesian search strategies
- Parameter importance analysis
- Constraint-based config filtering
- Best config export and comparison

> **IMPORTANT — Operational Guardrails:**
> - The AutoML pipeline (`automl.py`) handles HP search automatically. Do NOT re-implement search logic.
> - `sample_from_space()` already returns native Python types (int, float). The numpy serialization bug is **fixed**.
> - To run HP search, use the command in Quick Start below. Do not re-read or re-inspect `automl.py`.

> **Related Skills:**
> - `training-pipeline` — Hub with Quick Start (start here)
> - `reward-penalty-engineering` — Exploration methodology for what to search
> - `training-campaign` — Execute search experiments
> - `curriculum-learning` — Stage-specific HP tuning

## Quick Start

```powershell
# Launch AutoML HP search (preferred method)
uv run starter_kit_schedule/scripts/automl.py `
    --mode stage `
    --budget-hours 12 `
    --hp-trials 8 `
    --reward-generations 3

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
| Automated reward weight search | ✅ (use `reward-penalty-engineering` for what to search) |

## PPO Parameter Reference

| Parameter | Type | Range | Default | Impact |
|-----------|------|-------|---------|--------|
| `learning_rate` | log-uniform | 1e-5 - 1e-2 | 3e-4 | **Critical** - convergence speed |
| `discount_factor` | choice | 0.95-0.999 | 0.99 | Credit assignment horizon |
| `lambda_param` | choice | 0.9-0.99 | 0.95 | GAE bias-variance |
| `ratio_clip` | choice | 0.1-0.3 | 0.2 | Policy update magnitude |
| `entropy_loss_scale` | log-uniform | 1e-4 - 1e-1 | 1e-3 | Exploration strength |
| `learning_epochs` | choice | 2-10 | 5 | Updates per rollout |
| `mini_batches` | choice | 8-128 | 32 | Batch size |
| `rollouts` | choice | 16-64 | 24 | Steps before update |

## Network Architecture Options

| Config | Layers | Parameters | Use Case |
|--------|--------|------------|----------|
| Small | [128, 64] | ~20K | Fast iteration |
| Medium | [256, 128, 64] | ~80K | **Default** |
| Large | [512, 256, 128] | ~200K | Complex policies |
| Deep | [256, 256, 256] | ~150K | Feature extraction |

## Search Space Schema

```yaml
# starter_kit_schedule/configs/ppo_search_space.yaml
search_space_id: "ppo_navigation_v1"

# Fixed parameters
fixed:
  seed: 42
  num_envs: 2048
  grad_norm_clip: 1.0
  ratio_clip: 0.2
  value_clip: 0.2

# Searchable parameters
searchable:
  learning_rate:
    type: "loguniform"
    low: 1e-5
    high: 1e-2
    prior: 3e-4
    
  policy_hidden_layer_sizes:
    type: "categorical"
    choices:
      - [128, 64]
      - [256, 128, 64]
      - [512, 256, 128]
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
    high: 1e-1
    prior: 1e-3

# Constraints (filter bad combinations)
constraints:
  - "learning_rate > 5e-4 and learning_epochs > 8"  # Skip: unstable
  - "entropy_loss_scale > 0.05 and ratio_clip < 0.15"  # Skip: no learning
```

## Search Methods

| Method | Trials | Best For | Command Flag |
|--------|--------|----------|--------------|
| Grid | All combos | Small spaces (<20) | `--search-method grid` |
| Random | 30-100 | Large spaces | `--search-method random` |
| Bayesian | 20-50 | Expensive evals | `--search-method bayesian` |

### Grid Search
- Exhaustive coverage
- Good for discrete choices (layer sizes, epochs)
- Exponential cost with parameters

### Random Search
- Better than grid for >3 parameters
- Samples from distributions
- Can set `--max-trials 50`

### Bayesian Optimization
- Builds surrogate model
- Focuses on promising regions
- Best for expensive evaluations

## Commands

```powershell
# === PREFERRED: AutoML pipeline (handles HP search automatically) ===
uv run starter_kit_schedule/scripts/automl.py `
    --mode stage `
    --budget-hours 12 `
    --hp-trials 8 `
    --reward-generations 3

# === SINGLE TRAINING RUN (manual HP selection) ===
uv run scripts/train.py --env vbot_navigation_section001

# === ANALYZE RESULTS ===
uv run starter_kit_schedule/scripts/analyze.py `
    --metric episode_reward_mean `
    --sort descending
```

## Analysis Output

```yaml
# starter_kit_log/analysis/hyperparameter_importance.yaml
importance_ranking:
  - parameter: learning_rate
    importance: 0.45
    best_value: 2.1e-4
    
  - parameter: entropy_loss_scale
    importance: 0.22
    best_value: 8.5e-4
    
  - parameter: policy_hidden_layer_sizes
    importance: 0.15
    best_value: [256, 128, 64]

top_5_configs:
  - config_id: config_023
    learning_rate: 2.1e-4
    entropy_loss_scale: 8.5e-4
    episode_reward_mean: 38.2
    
  - config_id: config_017
    learning_rate: 1.8e-4
    episode_reward_mean: 36.8
```

## Presets

### Quick Validation
```yaml
# 6 configs, key parameters only
searchable:
  learning_rate: { type: "choice", values: [1e-4, 3e-4, 1e-3] }
  rollouts: { type: "choice", values: [24, 48] }
```

### Full Search
```yaml
# ~200 configs (use random sampling)
# All PPO + network parameters
```

### Fine-Tuning Search
```yaml
# Narrow ranges around best known config
learning_rate: { type: "loguniform", low: 1e-4, high: 5e-4 }
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| All configs perform similarly | Widen search ranges |
| Best config unstable | Add constraint: high LR + many epochs |
| Search takes too long | Reduce max_trials, shorter training |
| Bayesian stuck | Increase initial random trials |

## Best Practices

1. **Start with random search** - Better coverage than grid for >3 params
2. **Use priors** - Center search on known good values
3. **Add constraints** - Filter unstable combinations early
4. **Shorter training for search** - 5-10M steps sufficient for comparison
5. **Run multiple seeds** - At least 3 per config for reliability
6. **Log all configs** - Store in `starter_kit_log/experiments/`
```
