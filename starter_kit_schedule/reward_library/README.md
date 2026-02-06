# Reward & Penalty Library

Archive of all reward/penalty components and configurations tested during VBot navigation training.

## Structure

```
reward_library/
├── README.md           # This file — index and conventions
├── components/         # Individual reward/penalty definitions
│   └── <name>.yaml     # One YAML per reward idea (tested or untested)
├── configs/            # Complete reward configurations (tested combos)
│   └── <name>.yaml     # Full reward_scales dict + performance notes
└── rejected/           # Ideas that didn't work (with notes on why)
    └── <name>.yaml     # Same format as components, status: rejected
```

## Conventions

### Naming

- Component files: `<category>_<descriptive_name>.yaml` (e.g., `stability_body_height.yaml`)
- Config files: `<environment>_<descriptor>.yaml` (e.g., `flat_speed_optimized.yaml`)
- Use lowercase with underscores, no spaces

### Categories

| Category | Description |
|----------|-------------|
| `navigation` | Goal-seeking signals (distance, heading, checkpoints) |
| `stability` | Body orientation, height, balance |
| `efficiency` | Energy, smoothness, jerk |
| `terrain` | Terrain-specific (stairs, waves, obstacles) |
| `gait` | Foot timing, contact patterns, stride |
| `safety` | Collision avoidance, termination |
| `competition` | Competition-score-aligned signals |

### Status Values

| Status | Meaning |
|--------|---------|
| `untested` | Idea documented but not yet run |
| `tested` | Run at least once with results recorded |
| `promising` | Showed improvement, worth iterating on |
| `adopted` | Integrated into active training config |
| `rejected` | Tested and failed — moved to `rejected/` |

## Usage

```powershell
# Browse components
Get-ChildItem starter_kit_schedule/reward_library/components/ -Name

# Search for terrain-specific rewards
Select-String -Path starter_kit_schedule/reward_library/components/*.yaml -Pattern "category: terrain"

# Count tested vs untested
Select-String -Path starter_kit_schedule/reward_library/components/*.yaml -Pattern "status:" | Group-Object Line
```

## Related

- **Component reference & scale ranges:** `starter_kit_schedule/templates/reward_config_template.yaml`
- **Exploration methodology:** `.github/skills/reward-penalty-engineering/SKILL.md`
- **Reward code examples:** `.github/skills/quadruped-competition-tutor/SKILL.md`
