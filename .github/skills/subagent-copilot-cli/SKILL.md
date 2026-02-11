---
name: subagent-copilot-cli
description: Delegate analysis tasks to GitHub Copilot CLI as a parallel subagent for MotrixLab RL project. Handles automated policy playback frame capture, VLM-based visual behavior analysis, screenshot analysis, image file inspection, simulation frame interpretation, reward curve analysis, and general research conversations.
---

## Purpose

Use the Copilot CLI subagent for **analysis** tasks in MotrixLab RL workflows:

- **Automated VLM Policy Analysis** — Play a trained policy, auto-capture frames, send to VLM for behavior diagnosis
- **Screenshot analysis** — Capture and analyze simulation renders, environment states
- **Image file inspection** — Read training plots, reward curves, TensorBoard exports
- **PDF document reading** — Parse competition instruction PDFs
- **Parallel research agent** — Offload complex analysis while you coordinate
- **Code inspection** — Analyze reward structures, environment configs, policy architectures
- **Visual debugging** — Interpret failure modes from rendered frames

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

---

## 🔴 Automated VLM Policy Analysis Pipeline (PRIMARY WORKFLOW)

The `capture_vlm.py` script automates the full pipeline: **play policy → capture frames → send to VLM → get visual analysis report**.

### Quick Start

```powershell
# Play best policy, capture 20 frames, analyze with gpt-4.1
uv run scripts/capture_vlm.py --env <env-name>
```

### Full Options

```powershell
# Specify policy, capture settings, and custom VLM focus
uv run scripts/capture_vlm.py --env <env-name> \
    --policy runs/<env-name>/.../best_agent.pt \
    --capture-every 30 --max-frames 30 \
    --vlm-prompt "Focus on leg coordination and whether the robot reaches the target"

# Capture only (no VLM), analyze later manually
uv run scripts/capture_vlm.py --env <env-name> --no-vlm

# Use a different VLM model
uv run scripts/capture_vlm.py --env <env-name> --vlm-model gpt-4.1
```

### Script Parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--env` | (required) | Environment name |
| `--policy` | auto-discover | Policy checkpoint path |
| `--train-backend` | `torch` | Inference backend (torch/jax) |
| `--num-envs` | `1` | Parallel envs for playback |
| `--capture-every` | `15` | Capture frame every N sim steps |
| `--max-frames` | `20` | Total frames to capture |
| `--warmup-steps` | `30` | Steps before capture starts |
| `--capture-delay` | `0.15` | Seconds delay before screenshot |
| `--no-vlm` | `false` | Skip VLM, only save frames |
| `--vlm-model` | `gpt-4.1` | Copilot CLI model |
| `--vlm-prompt` | (default) | Custom analysis focus |
| `--vlm-batch-size` | `10` | Frames per VLM call |
| `--output-dir` | auto | Frame output directory |

### What the Pipeline Does

1. **Loads the trained policy** (auto-discovers best checkpoint or uses `--policy`)
2. **Runs the policy** in the simulation with rendering enabled
3. **Captures screenshots** at regular intervals using PIL ImageGrab
4. **Saves frames** to `starter_kit_log/vlm_captures/{env}/{timestamp}/`
5. **Sends frames** to GitHub Copilot CLI (`gpt-4.1`) with a structured analysis prompt
6. **Generates a report** at `vlm_analysis.md` covering:
   - Robot pose & stability
   - Gait quality & leg coordination
   - Navigation progress toward target
   - Detected failure modes & bugs
   - Reward engineering suggestions
   - Training recommendations

### Output Structure

```
starter_kit_log/vlm_captures/<env-name>/<timestamp>/
├── frame_00045.png          # Captured simulation frames
├── frame_00060.png
├── frame_00075.png
├── ...
├── capture_metadata.txt     # Run configuration
└── vlm_analysis.md          # VLM analysis report
```

### Usage Patterns

#### After Training — Quick Visual Check

```powershell
# Train, then immediately check visual quality
uv run scripts/train.py --env <env-name> --train-backend torch
uv run scripts/capture_vlm.py --env <env-name> --max-frames 15
```

#### Comparing Two Policies

```powershell
# Capture frames from policy A
uv run scripts/capture_vlm.py --env <env-name> \
    --policy runs/.../checkpoint_A.pt --output-dir analysis/policy_a

# Capture frames from policy B
uv run scripts/capture_vlm.py --env <env-name> \
    --policy runs/.../checkpoint_B.pt --output-dir analysis/policy_b

# Compare with VLM
copilot --model gpt-4.1 --allow-all \
    --add-dir analysis/policy_a --add-dir analysis/policy_b \
    -p "Compare robot behavior between policy_a/ and policy_b/ frames. Which policy has better gait, navigation, and stability?" -s
```

#### Capture Only + Manual VLM Later

```powershell
# Just capture
uv run scripts/capture_vlm.py --env <env-name> --no-vlm

# Analyze specific frames later
copilot --model gpt-4.1 --allow-all \
    --add-dir starter_kit_log/vlm_captures/<env-name>/latest \
    -p "Examine frame_00060.png and frame_00075.png — the robot seems to stumble. What's happening?" -s
```

#### Focus on Specific Bugs

```powershell
# VLM focus on leg issues
uv run scripts/capture_vlm.py --env <env-name> \
    --vlm-prompt "The robot's rear legs seem to drag. Focus on rear leg joint angles and contact patterns."

# VLM focus on navigation failures
uv run scripts/capture_vlm.py --env <env-name> \
    --vlm-prompt "The robot circles instead of going straight to target. Analyze heading and path curvature."
```

---

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
copilot --model gpt-4.1 --allow-all --add-dir d:\MotrixLab\starter_kit\<task> -p "<prompt>" -s

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

### Analyze VLM Capture Output (from capture_vlm.py)

```powershell
# Re-analyze previously captured frames with a new prompt
$captureDir = "starter_kit_log/vlm_captures/<env-name>/<timestamp>"
copilot --model gpt-4.1 --allow-all --add-dir $captureDir -p "Re-examine these policy evaluation frames. This time focus specifically on: 1) Whether the robot reaches the target platform 2) Any reward hacking behavior 3) Energy efficiency of the gait" -s
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
$task_info = copilot --model gpt-4.1 --allow-all -p "Read d:\MotrixLab\starter_kit\<task>\<task>.pdf. Summarize: 1) Task objectives 2) Terrain description 3) Scoring criteria 4) Time limits 5) Robot constraints" -s
```

### Compare Navigation Tasks

```powershell
# Side-by-side comparison
$comparison = copilot --model gpt-4.1 --allow-all -p "Read all PDF files from d:\MotrixLab\starter_kit\. Create a comparison table of: terrain complexity, distance, time limits, scoring weights." -s
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
$rewards = copilot --model gpt-4.1 --allow-all -p "Read d:\MotrixLab\starter_kit\<task>\vbot\cfg.py. Analyze the RewardConfig class: 1) List all reward components with weights 2) Identify potential reward hacking risks 3) Suggest improvements for navigation task" -s
```

### Analyze Policy Architecture

```powershell
# Understand the neural network structure
$arch = copilot --model gpt-4.1 --allow-all --add-dir d:\MotrixLab\motrix_rl\src\motrix_rl -p "Find the PPO policy network definition. Describe: layer sizes, activation functions, observation preprocessing, action output format." -s
```

### Inspect XML Scene Files

```powershell
# Understand terrain and physics setup
$scene = copilot --model gpt-4.1 --allow-all -p "Read d:\MotrixLab\starter_kit\<task>\vbot\xmls\<scene>.xml. Describe: 1) Terrain geometry 2) Obstacle placements 3) Goal marker positions 4) Physics parameters" -s
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
$reward_analysis = copilot --model gpt-4.1 --allow-all -p "What is the termination penalty in <task> cfg.py?" -s

if ($reward_analysis -match "termination") {
    Write-Host "Termination penalty detected - robot will be conservative"
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
copilot --model gpt-4.1 --allow-all -p "Analyze d:\MotrixLab\starter_kit\<task>\vbot\cfg.py thoroughly. Output a markdown report covering: environment overview, reward breakdown, training recommendations." -s > d:\MotrixLab\analysis\task_report.md
```

## Visual Debugging Workflows

### Automated Visual Debug (Preferred)

```powershell
# One-command visual debugging: play policy, capture frames, get VLM diagnosis
uv run scripts/capture_vlm.py --env <env-name> \
    --max-frames 25 --capture-every 10 \
    --vlm-prompt "This policy was trained for 5M steps but the robot seems to fall. Diagnose the issue."

# Read the analysis report
Get-Content starter_kit_log/vlm_captures/<env-name>/*/vlm_analysis.md
```

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
| `scripts/capture_vlm.py` | **VLM frame capture + analysis pipeline** |
| `starter_kit/{task}/vbot/cfg.py` | Reward structure, env params |
| `starter_kit/{task}/vbot/vbot_*_np.py` | Reward function implementation |
| `starter_kit/{task}/vbot/xmls/*.xml` | Scene geometry, physics |
| `motrix_rl/src/motrix_rl/cfgs.py` | PPO hyperparameters |
| `runs/*/checkpoints/` | Trained policy analysis |
| `starter_kit_log/vlm_captures/` | VLM capture outputs + analysis reports |
| `starter_kit_docs/{task}/Task_Reference.md` | Task-specific env IDs, reward scales, terrain data |

### VBot Navigation Environments

> **Full list of environment IDs, terrains, and packages** is documented in:
> - `starter_kit_docs/navigation1/Task_Reference.md` → "Environment IDs" section
> - `starter_kit_docs/navigation2/Task_Reference.md` → "Environment IDs" section

## Best Practices

1. **Use `capture_vlm.py` first** — For policy visual analysis, always prefer the automated pipeline over manual screenshots
2. **One analysis per invocation** — Keep prompts focused for reliable results
3. **Provide full paths** — Always use absolute paths to files
4. **Capture to variables** — Store analysis output with `$result = copilot ...`
5. **Use `-s` flag** — Silent mode gives clean output for parsing
6. **Add context directories** — Use `--add-dir` when subagent needs multiple files
7. **Chain small tasks** — Multiple focused analyses beat one complex prompt
8. **Save reports** — Redirect output to markdown files for persistence
9. **Batch size for VLM** — Keep `--vlm-batch-size` ≤ 10 to avoid context overflow
10. **Custom prompts** — Use `--vlm-prompt` to focus VLM on specific suspected bugs

## Limitations

| Limitation | Workaround |
|------------|------------|
| Stateless between calls | Chain invocations, pass context in prompts |
| Cannot run training | Use for analysis only, execute commands yourself |
| Large images slow | Crop or resize before analysis |
| PDF parsing imperfect | Ask specific questions, verify key details |
| Model constrained | Must use `gpt-4.1` for free tier |
| Screenshot captures full screen | Position MotrixSim window prominently before capture |
| No in-engine camera capture | Scene XMLs lack `<camera>` definitions; uses PIL ImageGrab instead |
| VLM batch size limit | Keep ≤ 10 frames per batch to avoid token limits |

````
