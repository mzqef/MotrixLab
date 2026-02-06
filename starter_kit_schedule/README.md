# Curriculum Learning Pipeline for VBot Navigation

Multi-stage curriculum training system for the MotrixArena S1 quadruped navigation competition.

## ⚡ Quick Start

```powershell
# 1. Initialize curriculum campaign
uv run python starter_kit_schedule/scripts/init_campaign.py `
    --name "VBot Stage1 Curriculum" `
    --template curriculum_plan_template.yaml

# 2. Start training
uv run python starter_kit_schedule/scripts/run_search.py

# 3. Monitor progress
uv run python starter_kit_schedule/scripts/status.py --watch

# 4. Analyze results
uv run python starter_kit_schedule/scripts/analyze.py --top 5
```

## 📋 Curriculum Stages

```
┌───────────────────────────────────────────────────────────────────────────────┐
│                        CURRICULUM PROGRESSION                                  │
├───────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  STAGE 1: Flat Ground    ─────►  STAGE 2A: Waves    ─────►  STAGE 2B: Stairs │
│  └─ Basic locomotion            └─ Terrain adapt           └─ Climbing        │
│  └─ Goal navigation             └─ Height variance         └─ Foot clearance  │
│  └─ 50M steps                   └─ 30M steps               └─ 40M steps       │
│                                                                               │
│                              STAGE 2C: Obstacles  ◄────────────┘              │
│                              └─ Ball avoidance                                │
│                              └─ 30M steps                                     │
│                                    │                                          │
│                                    ▼                                          │
│                              FINAL: Full Course                               │
│                              └─ All terrain types                             │
│                              └─ 50M steps                                     │
│                                                                               │
└───────────────────────────────────────────────────────────────────────────────┘
```

## 🎯 Reward Engineering

### Key Rewards Table

| Component | Weight | Stage | Description |
|-----------|--------|-------|-------------|
| `position_tracking` | 2.0 | All | Primary goal-seeking |
| `fine_position_tracking` | 2.0 | All | Dense reward when close |
| `heading_tracking` | 1.0 | All | Face direction of travel |
| `orientation` | -0.05 | All | Penalize body tilt |
| `lin_vel_z` | -0.5 | All | Penalize bouncing |
| `termination` | -200 | All | Body collision |
| `knee_lift_bonus` | 0.2 | Stairs | Leg clearance |
| `ball_collision` | -5.0 | Obstacles | Dynamic obstacle |

### Tuning Tips

- **Robot doesn't move**: Increase `position_tracking` weight (2.0 → 3.0)
- **Robot falls often**: Increase `orientation` penalty (-0.05 → -0.1)
- **Robot bounces**: Increase `lin_vel_z` penalty (-0.5 → -1.0)
- **Robot is jerky**: Increase `action_rate` penalty (-0.01 → -0.02)
- **Reward is too sparse**: Add checkpoint bonuses along path

## 📁 Directory Structure

```
starter_kit_schedule/
├── plans/                     # Curriculum plan definitions
│   ├── active_plan.yaml       # Current active training plan
│   └── archive/               # Completed plans
│
├── configs/                   # Hyperparameter configurations
│   └── generated/             # Auto-generated configs from search
│
├── progress/                  # Execution tracking
│   ├── current_run.yaml       # Currently running experiment
│   ├── queue.yaml             # Pending experiments
│   └── completed.yaml         # Finished experiments
│
├── checkpoints/               # Checkpoint registry for warm-starts
│
├── scripts/                   # Pipeline scripts
│   ├── init_campaign.py       # Initialize new curriculum campaign
│   ├── run_search.py          # Execute training runs
│   ├── status.py              # Monitor progress
│   └── analyze.py             # Analyze and compare results
│
└── templates/                 # Configuration templates
    ├── curriculum_plan_template.yaml   # Multi-stage curriculum
    ├── reward_config_template.yaml     # Reward engineering config
    ├── search_space_template.yaml      # Hyperparameter search space
    └── config_template.yaml            # Basic config template

starter_kit_log/
├── experiments/               # Individual experiment logs
│   └── EXP_YYYYMMDD_HHMMSS/   # Per-experiment data
├── campaigns/                 # Campaign-level summaries
│   └── campaign_YYYYMMDD/     # Per-campaign data
└── analysis/                  # Comparison reports
    ├── rankings/              # Sorted by metrics
    ├── hyperparameter_importance/
    └── visualizations/
```

## ⚙️ Templates

### `curriculum_plan_template.yaml`
Multi-stage curriculum with stage-specific reward overrides and promotion criteria.

### `reward_config_template.yaml`
Comprehensive reward engineering configuration with all components documented.

### `search_space_template.yaml`
Hyperparameter search space focused on reward weights and PPO dynamics.

## 🔍 Search Presets

| Preset | Trials | Focus | Use When |
|--------|--------|-------|----------|
| `quick_test` | 10 | Key params only | Sanity check |
| `reward_focus` | 50 | Reward weights | Tuning reward balance |
| `ppo_focus` | 50 | PPO params | Tuning learning dynamics |
| `full_search` | 200 | Everything | Final optimization |

## 📊 Monitoring Commands

```powershell
# Watch training progress in real-time
uv run python starter_kit_schedule/scripts/status.py --watch

# Check specific campaign
uv run python starter_kit_schedule/scripts/status.py --campaign campaign_20250101

# Show top 5 experiments by reward
uv run python starter_kit_schedule/scripts/analyze.py --top 5 --sort reward

# Export best config for deployment
uv run python starter_kit_schedule/scripts/analyze.py --export-best configs/best.yaml
```

## 🔗 Integration with subagent-copilot-cli

For image analysis and training curve visualization:

```powershell
# Analyze reward curve screenshot
gh copilot explain "Analyze reward curve at starter_kit_log/experiments/EXP_001/reward_curve.png"

# Compare simulation frames
gh copilot explain "Compare robot behavior at frame_100.png vs frame_500.png"
```

## 📝 Notes

- Always start with Stage 1 flat ground training
- Use warm-start with reduced LR (0.5×) when advancing stages
- Monitor termination rate - should decrease as training progresses
- Save checkpoints frequently (every 500 updates recommended)
- Back up best checkpoints before advancing to next stage
