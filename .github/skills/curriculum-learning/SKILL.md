---
name: curriculum-learning
description: Multi-stage curriculum training for VBot quadruped navigation. Stage progression with warm-starts and promotion criteria.
---

## Purpose

**Curriculum-based training** for progressive skill acquisition:

- Stage progression with warm-start
- Per-stage reward overrides
- Promotion criteria (reward threshold, success rate)
- Checkpoint transfer between stages

> **🔴 AutoML-First Policy (MANDATORY):**
> **NEVER** use `train.py` for parameter search. Use `automl.py` for each curriculum stage.
> See `.github/copilot-instructions.md` for the full policy.
>
> **Related Skills:**
> - `training-pipeline` — Hub with Quick Start commands (start here)
> - `reward-penalty-engineering` — Exploration methodology for stage rewards
> - `training-campaign` — Execute and monitor curriculum runs
> - `hyperparameter-optimization` — Tune PPO and reward weights per stage

## When to Use

| Task | Use This |
|------|----------|
| Design multi-stage curriculum | ✅ |
| Define stage progression criteria | ✅ |
| Configure warm-start transfers | ✅ |
| Single-stage training | ❌ Use `training-campaign` |

> **Before designing or running any curriculum**, review existing experiment history following `training-pipeline` skill → Step 0.

## 📋 Experiment Reports (MANDATORY check before any work)

> **ALWAYS read `REPORT_NAV*.md` files at the workspace root before starting curriculum work.**
> These reports track all experiments, discoveries, and current configuration state.

```powershell
# Check what reports exist and their current state
Get-ChildItem REPORT_NAV*.md | Select-Object Name, Length, LastWriteTime

# Read the latest nav1 report (flat ground curriculum)
Get-Content REPORT_NAV1.md | Select-Object -Last 100
```

**What to look for in reports:**
- Current curriculum stage (spawn_inner/outer_radius values in cfg.py)
- Which experiments used which stage config
- Promotion criteria results (reached%, ep_len stability)
- Active TODO items in "Next Steps" section
- Lessons learned that affect curriculum design

## Registered Environments

| Environment ID | Terrain | Package |
|----------------|---------|---------|
| `vbot_navigation_section001` | Flat ground (Stage 1) | `starter_kit/navigation1/` |
| `vbot_navigation_section01` | Section 01 | `starter_kit/navigation2/` |
| `vbot_navigation_section02` | Section 02 | `starter_kit/navigation2/` |
| `vbot_navigation_section03` | Section 03 | `starter_kit/navigation2/` |
| `vbot_navigation_stairs` | Stairs + platforms | `starter_kit/navigation2/` |
| `vbot_navigation_long_course` | Full 30m course | `starter_kit/navigation2/` |

## Curriculum Pipeline

### Recommended Progression

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           CURRICULUM STAGES                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  STAGE 1: Flat Ground (20 pts)                                              │
│  └── Environment: vbot_navigation_section001                                │
│  └── Steps: 50M (use AutoML pipeline)                                       │
│  └── Goal: Basic locomotion + navigation                                    │
│                          ↓                                                   │
│                   [Best Checkpoint]                                          │
│                          ↓                                                   │
│  STAGE 2A: Section 01                                                       │
│  └── Environment: vbot_navigation_section01                                 │
│  └── Steps: 30M                                                             │
│  └── Warm-start: Stage 1 best, LR × 0.5                                     │
│                          ↓                                                   │
│  STAGE 2B: Stairs                                                           │
│  └── Environment: vbot_navigation_stairs                                    │
│  └── Steps: 40M                                                             │
│  └── Warm-start: Stage 2A best, LR × 0.3                                    │
│                          ↓                                                   │
│  STAGE 2C: Section 03                                                       │
│  └── Environment: vbot_navigation_section03                                 │
│  └── Steps: 30M                                                             │
│                          ↓                                                   │
│  FINAL: Full Course                                                         │
│  └── Environment: vbot_navigation_long_course                               │
│  └── Steps: 50M                                                             │
│  └── Goal: End-to-end 30m navigation                                        │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Curriculum Plan Schema

```yaml
# starter_kit_schedule/templates/curriculum_full.yaml
plan_id: "curriculum_vbot_20260206"
name: "VBot Full Curriculum"

curriculum:
  - stage_id: "stage1_flat"
    environment: "vbot_navigation_section001"
    max_env_steps: 50_000_000
    checkpoint_interval: 500
    warm_start: null  # Start fresh
    
    reward_overrides:
      position_tracking: 2.0
      heading_tracking: 1.0
      termination: -200.0
      orientation: -0.05
    
    promotion_criteria:
      metric: "episode_reward_mean"
      threshold: 30.0
      min_steps: 20_000_000
      success_rate: 0.95
    
  - stage_id: "stage2a_waves"
    environment: "vbot_navigation_section012"
    max_env_steps: 30_000_000
    
    warm_start:
      from_stage: "stage1_flat"
      strategy: "best_checkpoint"
      reset_optimizer: true
      learning_rate_multiplier: 0.5
    
    reward_overrides:
      position_tracking: 1.5
      height_variance_penalty: -0.1
    
    promotion_criteria:
      metric: "episode_reward_mean"
      threshold: 25.0
      success_rate: 0.85
```

## Stage Configuration Reference

| Stage | Environment | Steps | LR Mult | Key Rewards |
|-------|-------------|-------|---------|-------------|
| 1: Flat | `vbot_navigation_section001` | 50M | 1.0 | position, heading |
| 2A: Section01 | `vbot_navigation_section01` | 30M | 0.5 | terrain adaptation |
| 2B: Stairs | `vbot_navigation_stairs` | 40M | 0.3 | knee_lift, edge_avoid |
| 2C: Section03 | `vbot_navigation_section03` | 30M | 0.3 | obstacle avoidance |
| Final | `vbot_navigation_long_course` | 50M | 1.0 | all combined |

## Warm-Start Strategies

| Strategy | When to Use | Config |
|----------|-------------|--------|
| **Best Checkpoint** | Default for curriculum | `reset_optimizer: true, lr_mult: 0.5` |
| **Final Checkpoint** | Keep momentum | `reset_optimizer: false` |
| **Frozen Layers** | New observation space | `freeze_layers: ["encoder"]` |

## Promotion Criteria

```yaml
promotion_criteria:
  metric: "episode_reward_mean"  # Primary metric
  threshold: 30.0                # Minimum to pass
  min_steps: 20_000_000          # Train at least this much
  success_rate: 0.95             # Episode success rate
  patience: 5_000_000            # Steps without improvement before abort
```

## Commands

```powershell
# === PRIMARY: AutoML pipeline for EACH stage (USE THIS) ===
uv run starter_kit_schedule/scripts/automl.py `
    --mode stage `
    --budget-hours 8 `
    --hp-trials 15

# === MONITOR AutoML ===
Get-Content starter_kit_schedule/progress/automl_state.yaml

# === READ AutoML RESULTS ===
Get-Content starter_kit_log/automl_*/report.md

# === FINAL DEPLOYMENT (after AutoML found best config for this stage) ===
uv run scripts/train.py --env vbot_navigation_section001 --train-backend torch

# === EVALUATE a checkpoint ===
uv run scripts/play.py --env vbot_navigation_section001

# === TENSORBOARD ===
uv run tensorboard --logdir runs/vbot_navigation_section001
```

## Progress State Schema

```yaml
# starter_kit_schedule/progress/curriculum_state.yaml
curriculum_id: "curriculum_vbot_20260206"
current_stage: "stage2_section01"
status: "running"

stages_completed:
  - stage_id: "stage1_flat"
    completed_at: "2026-02-06T18:00:00Z"
    best_checkpoint: "runs/vbot_navigation_section001/<run>/checkpoints/best_agent.pt"
    final_metrics:
      episode_reward_mean: 35.2
      success_rate: 0.97

stages_pending:
  - "stage2_stairs"
  - "stage2_section03"
  - "final_long_course"
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Stage transfer fails | Lower LR mult (0.3×), longer adaptation |
| Promotion never triggers | Reduce threshold, check success_rate calc |
| Performance drops after warm-start | Reset optimizer, use smaller LR |

## Best Practices

1. **Master Stage 1 first** - Solid flat-ground performance is foundation
2. **Lower LR on transfer** - 0.3-0.5× prevents catastrophic forgetting
3. **Test promotion criteria** - Run 1M steps before committing to full stage
4. **Checkpoint frequently** - Every 500 iters for Stage 1, 1000 for later
5. **Log per-stage metrics** - Compare performance across stages
6. **Enable successful truncation** - `_update_truncate()` ends episodes early when robot reaches+stops for 50 steps. This speeds up training by not wasting steps after the goal is achieved.
7. **Anti-laziness must be active** - When running long curriculum stages (50M+), conditional alive_bonus and time_decay are essential. See `reward-penalty-engineering` Lazy Robot case study.
