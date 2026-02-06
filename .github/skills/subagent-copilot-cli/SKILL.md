---
name: subagent-copilot-cli
description: Delegate analysis tasks to GitHub Copilot CLI as a parallel subagent for MotrixLab RL project. Handles screenshot analysis, image file inspection, simulation frame interpretation, reward curve analysis, and general research conversations.
---

## Purpose

Use the Copilot CLI subagent for **analysis** tasks in MotrixLab RL workflows:

- **Screenshot analysis** - Capture and analyze simulation renders, environment states
- **Image file inspection** - Read training plots, reward curves, TensorBoard exports
- **PDF document reading** - Parse navigation1.pdf/navigation2.pdf instructions
- **Parallel research agent** - Offload complex analysis while you coordinate
- **Code inspection** - Analyze reward structures, environment configs, policy architectures
- **Visual debugging** - Interpret failure modes from rendered frames

> **NOT for:** Training execution, `train.py`/`view.py`/`play.py` commands, or TensorBoard launching.  
> Use this skill for **understanding and analysis**, not execution.

## Model Requirement

**IMPORTANT:** Always use the free `gpt-4.1` model:

```powershell
copilot --model gpt-4.1 ...
```

| Model | Cost | Use Case |
|-------|------|----------|
| `gpt-4.1` | Free | **Always use this** |
| `gpt-5`, `claude-opus-4.5` | Premium | Avoid unless explicitly requested |

## Core Invocation Pattern

### Basic Non-Interactive (Most Common)

```powershell
$result = copilot --model gpt-4.1 --allow-all -p "<prompt>" -s
```

| Flag | Purpose |
|------|---------|
| `--model gpt-4.1` | Use free model |
| `--allow-all` | Grant all permissions (file access, tools) |
| `-p "<prompt>"` | Execute this prompt non-interactively |
| `-s` | Silent mode (output only agent response) |

### With Project Context

```powershell
# Add MotrixLab directories for context
copilot --model gpt-4.1 --allow-all --add-dir d:\MotrixLab\starter_kit\navigation1 -p "<prompt>" -s

# Multiple directories
copilot --model gpt-4.1 --allow-all --add-dir d:\MotrixLab\starter_kit --add-dir d:\MotrixLab\runs -p "<prompt>" -s
```

### Interactive Mode (For Multi-Turn Analysis)

```powershell
# Start interactive session for complex analysis
copilot --model gpt-4.1 --allow-all -i "Let's analyze the VBot navigation reward structure together"
```

## Screenshot & Simulation Frame Analysis

### Capture Simulation Screenshot

```powershell
# After view.py or train.py --render is running, capture window
# Use external screenshot tool, then analyze:
$analysis = copilot --model gpt-4.1 --allow-all -p "Look at d:\MotrixLab\screenshots\vbot_render.png. Describe: 1) Robot pose and stance 2) Terrain features visible 3) Distance from goal marker 4) Any collision or instability signs" -s
```

### Analyze Rendered Frames Directory

```powershell
# Analyze multiple frames from a training run
$frames = copilot --model gpt-4.1 --allow-all --add-dir d:\MotrixLab\renders\episode_001 -p "Examine all PNG frames in this directory. Create a timeline of robot behavior: stance changes, terrain traversal progress, any falls or recovery attempts." -s
```

### Compare Before/After States

```powershell
# Capture pre and post action states
$comparison = copilot --model gpt-4.1 --allow-all -p "Compare d:\MotrixLab\screenshots\before_stairs.png and d:\MotrixLab\screenshots\after_stairs.png. Did the VBot successfully ascend? What gait pattern is visible?" -s
```

### Failure Mode Analysis

```powershell
# When robot falls or fails navigation
$diagnosis = copilot --model gpt-4.1 --allow-all -p "Look at d:\MotrixLab\screenshots\failure_frame.png. What caused the VBot to fail? Check: 1) Leg configuration 2) Body orientation 3) Terrain interaction 4) Likely cause of termination" -s
```

## PDF Instruction Document Analysis

### Read Navigation Instructions

```powershell
# Analyze competition instructions
$nav1_info = copilot --model gpt-4.1 --allow-all -p "Read d:\MotrixLab\starter_kit\navigation1\navigation1.pdf. Summarize: 1) Task objectives 2) Terrain description 3) Scoring criteria 4) Time limits 5) Robot constraints" -s

$nav2_info = copilot --model gpt-4.1 --allow-all -p "Read d:\MotrixLab\starter_kit\navigation2\navigation2.pdf. What are the key differences from navigation1? List new challenges." -s
```

### Compare Navigation Tasks

```powershell
# Side-by-side comparison
$comparison = copilot --model gpt-4.1 --allow-all -p "Read both navigation1.pdf and navigation2.pdf from d:\MotrixLab\starter_kit\. Create a comparison table of: terrain complexity, distance, time limits, scoring weights." -s
```

## Training Metrics & Reward Analysis

### Analyze TensorBoard Exports

```powershell
# If you exported TensorBoard plots as images
$rewards = copilot --model gpt-4.1 --allow-all -p "Look at d:\MotrixLab\analysis\reward_curve.png. Describe the training progress: 1) Convergence pattern 2) Reward scale 3) Any plateaus or instabilities 4) Estimated episodes to convergence" -s
```

### Interpret Learning Curves

```powershell
# Multiple metric plots
$metrics = copilot --model gpt-4.1 --allow-all --add-dir d:\MotrixLab\analysis\tensorboard_exports -p "Examine all training plots. For each metric (episode_reward, policy_loss, value_loss, entropy): describe trend, identify anomalies, suggest hyperparameter adjustments." -s
```

### Compare Training Runs

```powershell
# Compare two different hyperparameter settings
$comparison = copilot --model gpt-4.1 --allow-all -p "Compare reward curves: d:\MotrixLab\analysis\run_lr001.png vs d:\MotrixLab\analysis\run_lr0001.png. Which learning rate produces faster convergence? More stable training?" -s
```

## Code & Configuration Analysis

### Analyze Reward Structure

```powershell
# Deep dive into reward function design
$rewards = copilot --model gpt-4.1 --allow-all -p "Read d:\MotrixLab\starter_kit\navigation1\vbot\cfg.py. Analyze the RewardConfig class: 1) List all reward components with weights 2) Identify potential reward hacking risks 3) Suggest improvements for navigation task" -s
```

### Analyze Policy Architecture

```powershell
# Understand the neural network structure
$arch = copilot --model gpt-4.1 --allow-all --add-dir d:\MotrixLab\motrix_rl\src\motrix_rl -p "Find the PPO policy network definition. Describe: layer sizes, activation functions, observation preprocessing, action output format." -s
```

### Inspect XML Scene Files

```powershell
# Understand terrain and physics setup
$scene = copilot --model gpt-4.1 --allow-all -p "Read d:\MotrixLab\starter_kit\navigation1\vbot\xmls\scene_section001.xml. Describe: 1) Terrain geometry 2) Obstacle placements 3) Goal marker positions 4) Physics parameters" -s
```

## Parallel Analysis Conversations

### Research Assistant Pattern

```powershell
# Ask subagent to research while you work on something else
$research = copilot --model gpt-4.1 --allow-all --add-dir d:\MotrixLab -p "Research question: What curriculum learning strategies would help VBot learn stair climbing? Consider the reward structure in cfg.py and suggest a 3-stage curriculum." -s
Write-Host "Research findings: $research"
```

### Hypothesis Testing

```powershell
# Have subagent analyze your hypothesis
$test = copilot --model gpt-4.1 --allow-all -p "Hypothesis: The VBot fails on stairs because heading_tracking reward conflicts with position_tracking when approaching stairs at an angle. Analyze cfg.py reward weights and confirm or refute this." -s
```

### Literature-Informed Analysis

```powershell
# Connect project to RL best practices
$lit = copilot --model gpt-4.1 --allow-all -p "Compare the PPO hyperparameters in d:\MotrixLab\motrix_rl\src\motrix_rl\skrl\cfg.py against recommended settings from Schulman et al. (2017) and SKRL documentation. What should be adjusted for locomotion tasks?" -s
```

## Information Exchange Pattern

The subagent is **stateless** - each invocation is independent. To exchange information:

### Capture Output to Variable

```powershell
# Get analysis result and use it
$reward_analysis = copilot --model gpt-4.1 --allow-all -p "What is the termination penalty in navigation1 cfg.py?" -s

if ($reward_analysis -match "-200") {
    Write-Host "High termination penalty detected - robot will be conservative"
}
```

### Chain Multiple Analyses

```powershell
# First: identify the problem
$problem = copilot --model gpt-4.1 --allow-all -p "Look at d:\MotrixLab\screenshots\failure.png. What type of failure is this?" -s

# Second: suggest fix based on problem
$solution = copilot --model gpt-4.1 --allow-all -p "The VBot failure type is: $problem. What reward modifications in cfg.py would prevent this?" -s
```

### Save Analysis Reports

```powershell
# Generate and save detailed analysis
copilot --model gpt-4.1 --allow-all -p "Analyze d:\MotrixLab\starter_kit\navigation1\vbot\cfg.py thoroughly. Output a markdown report covering: environment overview, reward breakdown, training recommendations." -s > d:\MotrixLab\analysis\nav1_report.md
```

## Visual Debugging Workflows

### Episode Failure Investigation

```powershell
$errorDir = "d:\MotrixLab\debug\episode_failure_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
New-Item -ItemType Directory -Path $errorDir -Force

# After capturing failure frames to $errorDir...
$diagnosis = copilot --model gpt-4.1 --allow-all --add-dir $errorDir -p "Examine all frames in this directory showing a failed episode. Create a failure timeline: 1) Initial state 2) Critical moment before failure 3) Failure frame 4) Root cause analysis" -s

Write-Host "Failure diagnosis: $diagnosis"
```

### Gait Analysis

```powershell
# Analyze locomotion quality
$gait = copilot --model gpt-4.1 --allow-all --add-dir d:\MotrixLab\renders\locomotion_test -p "Examine the sequence of VBot frames. Analyze gait quality: 1) Foot contact pattern 2) Stride symmetry 3) Body stability 4) Compare to typical quadruped trotting gait" -s
```

## Analysis Question Templates

| Context | Prompt Template |
|---------|-----------------|
| **Screenshot state** | "What is the robot's current pose? Is it stable or about to fall?" |
| **Failure diagnosis** | "What caused this failure? Check leg positions, terrain contact, body angle." |
| **Reward analysis** | "Is this reward structure balanced? Any risk of reward hacking?" |
| **Config comparison** | "What differs between these configs? Which is harder?" |
| **Progress check** | "Based on this reward curve, is training converging? How many more steps needed?" |
| **Gait quality** | "Is this a healthy quadruped gait? What locomotion issues are visible?" |

## MotrixLab Project Context

### Key Files for Analysis

| File | Analysis Purpose |
|------|------------------|
| `starter_kit/navigation1/vbot/cfg.py` | Reward structure, env params |
| `starter_kit/navigation1/vbot/vbot_section001_np.py` | Reward function implementation |
| `starter_kit/navigation2/vbot/cfg.py` | Navigation2 env params |
| `starter_kit/navigation1/vbot/xmls/*.xml` | Scene geometry, physics |
| `starter_kit/navigation2/vbot/xmls/*.xml` | Obstacle course scenes |
| `motrix_rl/src/motrix_rl/cfgs.py` | PPO hyperparameters |
| `runs/*/checkpoints/` | Trained policy analysis |

### VBot Navigation Environments

| Environment | Terrain | Package |
|-------------|---------|---------|
| `vbot_navigation_section001` | Flat ground (Stage 1) | navigation1 |
| `vbot_navigation_section01` | Section 01 | navigation2 |
| `vbot_navigation_section02` | Section 02 | navigation2 |
| `vbot_navigation_section03` | Section 03 | navigation2 |
| `vbot_navigation_stairs` | Stairs + platforms | navigation2 |
| `vbot_navigation_long_course` | Full 30m course | navigation2 |

## Best Practices

1. **One analysis per invocation** - Keep prompts focused for reliable results
2. **Provide full paths** - Always use absolute paths to files
3. **Capture to variables** - Store analysis output with `$result = copilot ...`
4. **Use `-s` flag** - Silent mode gives clean output for parsing
5. **Add context directories** - Use `--add-dir` when subagent needs multiple files
6. **Chain small tasks** - Multiple focused analyses beat one complex prompt
7. **Save reports** - Redirect output to markdown files for persistence

## Limitations

| Limitation | Workaround |
|------------|------------|
| Stateless between calls | Chain invocations, pass context in prompts |
| Cannot run training | Use for analysis only, execute commands yourself |
| Large images slow | Crop or resize before analysis |
| PDF parsing imperfect | Ask specific questions, verify key details |
| Model constrained | Must use `gpt-4.1` for free tier |

````
