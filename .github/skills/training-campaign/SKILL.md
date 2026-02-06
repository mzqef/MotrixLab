---
name: training-campaign
description: Execute and monitor long-running RL training campaigns. Progress tracking, checkpoint management, experiment logging, and resume capabilities.
---

## Purpose

**Long-running training management** for VBot navigation:

- Execute multi-day training campaigns
- Checkpoint registry and resume
- Structured experiment logging
- Progress monitoring and alerts

> **IMPORTANT — Operational Guardrails:**
> - The AutoML pipeline is **tested and working**. Do NOT re-read `automl.py`, `train_one.py`, or `evaluate.py` before launching.
> - When asked to start/resume training, use the commands below directly.
> - The pipeline handles import ordering, JSON serialization, and subprocess management internally.

> **Related Skills:**
> - `training-pipeline` — Hub with Quick Start commands (start here)
> - `curriculum-learning` — Define curriculum plans
> - `hyperparameter-optimization` — Search configurations
> - `reward-penalty-engineering` — Reward exploration methodology

## When to Use

| Task | Use This |
|------|----------|
| Start training campaign | ✅ |
| Resume interrupted run | ✅ |
| Monitor progress | ✅ |
| Checkpoint management | ✅ |
| Design rewards | ❌ Use `reward-penalty-engineering` |

## Commands

### Start Training

```powershell
# === PREFERRED: AutoML pipeline (handles everything) ===
uv run starter_kit_schedule/scripts/automl.py `
    --mode stage `
    --budget-hours 12 `
    --hp-trials 8

# === SIMPLE: Single training run ===
uv run scripts/train.py --env vbot_navigation_section001

# === WITH RENDERING (for visual debugging) ===
uv run scripts/train.py --env vbot_navigation_section001 --render

# === PYTORCH BACKEND (Windows recommended) ===
uv run scripts/train.py --env vbot_navigation_section001 --train-backend torch
```

### Monitor Progress

```powershell
# Check AutoML state
Get-Content starter_kit_schedule/progress/automl_state.yaml

# TensorBoard (opens web dashboard)
uv run tensorboard --logdir runs/vbot_navigation_section001

# List checkpoints
Get-ChildItem runs/vbot_navigation_section001/ -Recurse -Filter "*.pt"
```

### Evaluate

```powershell
# Play latest checkpoint
uv run scripts/play.py --env vbot_navigation_section001

# Play specific checkpoint
uv run scripts/play.py --env vbot_navigation_section001 `
    --policy runs/vbot_navigation_section001/<run_dir>/checkpoints/agent.pt
```

## Directory Structure

```
starter_kit_schedule/
├── templates/                 # All YAML templates & config references
│   ├── automl_config.yaml     # AutoML configuration template
│   ├── config_template.yaml   # Individual training config
│   ├── curriculum_plan_template.yaml
│   ├── plan_template.yaml
│   ├── reward_config_template.yaml
│   └── search_space_template.yaml
├── progress/
│   └── automl_state.yaml      # AutoML search state (primary tracking file)
├── checkpoints/
│   └── registry.yaml          # All checkpoints index
└── reward_library/            # Archived reward/penalty components

starter_kit_log/
└── <automl_id>/               # Self-contained per-run folder
    ├── configs/               # HP + reward configs per trial
    ├── experiments/           # Per-experiment summaries
    ├── index.yaml             # Run-level index
    └── state.yaml             # AutoML state snapshot

runs/                          # Training outputs
└── vbot_navigation_section001/
    └── <timestamp>_PPO/
        ├── checkpoints/       # Policy checkpoints
        ├── events.out.tfevents.*  # TensorBoard logs
        └── experiment_meta.json   # HP config snapshot
```

## AutoML Pipeline Architecture

The AutoML pipeline runs as a single process that spawns subprocesses:

```
run.py (entry point, sets --env vbot_navigation_section001)
  └── automl.py (HP search engine)
       ├── sample_from_space() → HP config (native Python types)
       ├── _train_and_eval() → spawns subprocess:
       │    └── train_one.py (imports vbot FIRST, then motrix_rl)
       │         └── Trainer(env_name, cfg_override=rl_overrides).train()
       ├── evaluate.py → reads TensorBoard event files
       │    └── Returns: final_reward, max_reward, distance_to_target
       └── Saves state to: starter_kit_schedule/progress/automl_state.yaml
```

## Expected Training Times

| Hardware | 50M Steps | 100M Steps |
|----------|-----------|------------|
| RTX 3090 | ~4 hours | ~8 hours |
| RTX 4090 | ~2.5 hours | ~5 hours |
| A100 | ~1.5 hours | ~3 hours |

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Training stuck | Check GPU memory, reduce `num_envs` |
| OOM error | Reduce `num_envs` or `mini_batches` |
| Resume fails | Check `current_run.yaml` for last checkpoint |
| Metrics missing | Check `metrics.jsonl` write permissions |

## Best Practices

1. **Checkpoint every 500-1000 iters** - Training can be interrupted
2. **Use separate log directories** - One per experiment
3. **Monitor GPU memory** - Set alerts at 90% usage
5. **Version control configs** - Store templates in `templates/`
5. **Back up best checkpoints** - Before advancing stages
6. **Use `--resume` liberally** - Don't restart from scratch
```
