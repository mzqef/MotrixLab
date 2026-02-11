---
name: reward-penalty-engineering
description: Methodology for exploring, testing, and archiving reward/penalty functions for VBot quadruped navigation. A process-oriented guide for systematic reward discovery.
---

## Purpose

This skill teaches the **methodology of reward/penalty exploration** — how to discover, test, evaluate, and archive reward signals. It is a process guide, not a recipe book.

- **How to identify** what reward/penalty to try next
- **How to formulate** a hypothesis and test it
- **How to evaluate** whether a reward change helped
- **How to archive** findings in the reward library for reuse

> **IMPORTANT:**
> - The reward function lives in `starter_kit/navigation*/vbot/vbot_*_np.py` → `_compute_reward()`.
> - Reward weights are in `starter_kit/navigation*/vbot/cfg.py` → `RewardConfig.scales` dict.
> - Current reward component details, default values, and search ranges are documented in `starter_kit_docs/{task-name}/Task_Reference.md`.
> - Anti-laziness mechanisms (conditional alive_bonus, time_decay, successful truncation) are active. Do NOT remove them.
> - Do NOT re-implement the reward function from scratch. Always check the existing implementation and documentation before making changes. Modify weights or add new terms incrementally with clear hypotheses, testing, and evaluation and archiving.

> This skill does NOT contain reward component examples or scale tables.
> Those live in their respective locations:
>
> | What | Where |
> |------|-------|
> | Component reference & scale ranges | `starter_kit_schedule/templates/reward_config_template.yaml` |
> | Archived reward/penalty instances | `starter_kit_schedule/reward_library/` |
> | Terrain strategies & reward code | `quadruped-competition-tutor` skill |
> | Stage-specific reward overrides | `curriculum-learning` skill |
> | Reward weight search spaces | `hyperparameter-optimization` skill |
> | Visual reward debugging | `subagent-copilot-cli` skill |


---

## When to Use This Skill

| Situation | Use This |
|-----------|----------|
| "I need a new reward idea" | ✅ Follow the Discovery Process |
| "This reward isn't working, what now?" | ✅ Follow Diagnostic Methodology |
| "I want to compare two reward designs" | ✅ Follow Experiment Protocol |
| "I found a good reward, where to save it?" | ✅ Follow Archiving Process |
| "What are the reward scale ranges?" | ❌ Read `reward_config_template.yaml` |
| "What reward code exists for stairs?" | ❌ Read `quadruped-competition-tutor` |
| "How do I tune reward weights automatically?" | ❌ Read `hyperparameter-optimization` |

---

## The Exploration Cycle

Reward engineering is iterative. Every change follows this cycle:

```
    ┌──────────────┐
    │   DIAGNOSE   │ ← What behavior is wrong?
    └──────┬───────┘
           ▼
    ┌──────────────┐
    │  HYPOTHESIZE  │ ← What reward signal could fix it?
    └──────┬───────┘
           ▼
    ┌──────────────┐
    │   IMPLEMENT   │ ← Minimal change, one variable at a time
    └──────┬───────┘
           ▼
    ┌──────────────┐
    │     TEST      │ ← Short run (1-2M steps), multiple seeds
    └──────┬───────┘
           ▼
    ┌──────────────┐
    │   EVALUATE    │ ← Did the hypothesis hold?
    └──────┬───────┘
           ▼
    ┌──────────────┐
    │   ARCHIVE     │ ← Record result in reward library
    └──────┬───────┘
           │
           ▼
      Next cycle
```

**Rule:** Never change more than one reward dimension per cycle. If you change both the termination penalty AND add a new gait reward, you cannot attribute outcomes.

---

## Phase 1: Diagnose

### Behavioral Signals

Before touching rewards, identify **what behavior** is wrong. Not "the reward is too low" but a concrete observable:

| Observable | Likely Reward Gap |
|------------|-------------------|
| Robot doesn't move | Missing or weak positive incentive |
| Robot moves but falls | Missing or weak stability penalty |
| Robot oscillates near goal | Reward gradient too steep near target |
| Robot takes bizarre paths | Reward hacking — high reward from unintended behavior |
| Robot crouches/crawls | Missing height maintenance signal |
| Robot ignores obstacles | Missing proximity/collision signal |
| Robot is fast but jerky | Missing smoothness penalty |
| Robot is stable but slow | Positive incentive too weak relative to penalties |
| Reward curve plateaus | Reward provides no gradient in current state region |
| **Robot stands still near target** | **alive_bonus accumulation > goal reward — see Lazy Robot Case Study below** |
| **Distance increases during training** | **Reward hacking via per-step bonus. Check alive_bonus × avg_ep_len vs arrival_bonus** |
| **Episode length near max, reached% drops** | **Robot exploiting per-step rewards instead of completing task** |

### Diagnostic Commands

```powershell
# 1. Watch the policy — ALWAYS start here before looking at numbers
uv run scripts/play.py --env <env-name>

# 2. Train with rendering to see behavior in real time
uv run scripts/train.py --env <env-name> --render

# 3. TensorBoard for reward curves
uv run tensorboard --logdir runs/<env-name>
```

### Visual Diagnosis

Use `subagent-copilot-cli` to analyze simulation frames and training curves:

```powershell
# Describe what you see, ask what reward signal is missing
copilot --model gpt-4.1 --allow-all -p "Watch this simulation frame. The robot is <describe behavior>. What reward signal might cause this?" -s
```

> **Key insight:** A reward signal is "missing" if the agent has no gradient pointing toward the desired behavior in its current state. The fix may be a new reward, a penalty, or reshaping an existing one.

---

## Phase 2: Hypothesize

### Formulating a Good Hypothesis

A testable reward hypothesis has three parts:

1. **Behavior target:** What the robot should do differently
2. **Signal mechanism:** What mathematical signal encodes that behavior
3. **Expected side effect:** What might go wrong

**Template:**

> "If I add/modify `<signal>` with weight `<w>`, the robot should `<desired behavior>`, but might also `<risk>`."

### Discovery Strategies

When you don't know what to try, use these strategies to generate candidates:

#### Strategy 1: Inversion

Take the undesired behavior and directly penalize it.

> Robot bouncing → penalize vertical velocity
> Robot spinning → penalize angular velocity
> Robot retreating → penalize backward displacement

#### Strategy 2: Shaping the Gradient

If the robot is stuck, the reward surface is flat in its current region. Add a signal that creates local gradient:

> Robot stuck far from goal → Add distance-based shaping (sigmoid, exponential)
> Robot stuck near goal → Add fine-grained proximity bonus
> Robot stuck on terrain edge → Add progress checkpoints

#### Strategy 3: Proxy Decomposition

Break the competition score into component sub-goals and create a signal for each:

> Final score = traversal + bonus zones + time bonus
> → Create separate signals for: forward progress, zone proximity, speed

#### Strategy 4: Biomimetic Analogy

What would a real quadruped "want" in this situation?

> Stairs → lift knees higher
> Uneven ground → keep center of mass low
> Obstacles → slow down, increase awareness

#### Strategy 5: Ablation Discovery

Temporarily remove one existing reward and see what degrades:

```powershell
# Remove component to see its effect
python scripts/train.py --env <env> --seed 42 --cfg-override "reward_config.scales.<component>=0.0"
```

If removing a component doesn't change behavior, it was irrelevant. If behavior collapses, it was critical.

#### Strategy 6: Competition-Score Alignment

Compare training reward to competition scoring rules. Gaps indicate missing signals:

> Competition awards points for stopping in smiley zones
> → but training reward only rewards forward velocity
> → mismatch: need a "stop in zone" signal

> **Refer to** `quadruped-competition-tutor` skill for competition scoring rules.

#### Strategy 7: Browse the Library

Check previously tried components in the reward library before inventing new ones:

```powershell
# Browse archived reward components
Get-ChildItem starter_kit_schedule/reward_library/components/ | Select-Object Name
# Read a specific component's notes
Get-Content starter_kit_schedule/reward_library/components/<name>.yaml
```

---

## Phase 3: Implement

### Principles

1. **One variable at a time** — Change a single reward component per experiment
2. **Minimal change** — Prefer adjusting a weight before adding new code
3. **Use existing infrastructure** — Check `reward_config_template.yaml` for components that can be enabled/disabled before writing new code

### Where to Make Changes

| Change Type | Location |
|-------------|----------|
| Adjust existing weight | `starter_kit/{task}/vbot/cfg.py` → `RewardConfig.scales` dict |
| Add new reward term | `starter_kit/{task}/vbot/vbot_*_np.py` → `_compute_reward()` |
| Configure component | `starter_kit_schedule/templates/reward_config_template.yaml` |

### Change Magnitude Guidelines

When adjusting weights, use **multiplicative steps** not additive:

- **Small adjustment:** ×0.5 or ×2 (halve or double)
- **Medium adjustment:** ×0.1 or ×10
- **Large adjustment:** ×0.01 or ×100

For new components, start with a weight that produces reward magnitude comparable to existing dominant terms (check `reward_breakdown` logs).

---

## Phase 4: Test

### 🔴 AutoML-First Testing (MANDATORY)

> **NEVER** iterate manually with `train.py`, changing one reward weight, running, reading TensorBoard, killing, repeating.
> This is **manual one-at-a-time search** — slow, error-prone, and wasteful.
> **ALWAYS** use `automl.py` for batch reward hypothesis testing.

**The correct workflow:**
1. Add your reward hypothesis as a search range in `REWARD_SEARCH_SPACE` (in `automl.py`)
2. Run `automl.py --hp-trials 8+` to test multiple configurations in one batch
3. Read the structured comparison in `starter_kit_log/automl_*/report.md`
4. Archive results in the reward library

**Example: Testing near_target_speed activation radius**
```python
# In automl.py REWARD_SEARCH_SPACE:
"near_target_speed": {"type": "uniform", "low": -2.0, "high": -0.1},
"near_target_activation": {"type": "choice", "values": [0.3, 0.5, 1.0, 2.0]},
```
Then run: `uv run starter_kit_schedule/scripts/automl.py --mode stage --hp-trials 8`

**Why AutoML is better:**
- Tests N configurations in parallel with identical conditions
- Produces structured comparison table (reward, reached%, distance, ep_len)
- Bayesian suggestion improves with each trial
- No manual TensorBoard reading or experiment-killing needed

### Exception: `train.py` is acceptable for testing ONLY when:
- **Smoke test** — `--max-env-steps 200000` to verify new reward code compiles
- **Visual debugging** — `--render` to watch behavior qualitatively
- The reward requires NEW CODE in `_compute_reward()` (test compilation first, then use automl)

### Experiment Protocol

```yaml
# Record this BEFORE running the experiment
experiment:
  id: "RPE_YYYYMMDD_NNN"         # Auto-incrementing
  hypothesis: "<one sentence>"
  change: "<what exactly changed>"
  baseline: "<what to compare against>"
  environment: "<env id>"
  tool: "automl.py"               # MUST be automl.py, not train.py
  hp_trials: 8                    # Minimum 8 for meaningful comparison
  steps_per_trial: 5_000_000      # Each trial
  metrics_to_watch:
    - episode_reward_mean
    - reached_fraction
    - <behavior-specific metric>
```

### Running Tests

```powershell
# CORRECT: Batch search with AutoML (use this!)
uv run starter_kit_schedule/scripts/automl.py `
    --mode stage `
    --budget-hours 4 `
    --hp-trials 8

# Read results
Get-Content starter_kit_log/automl_*/report.md

# WRONG: Manual iteration (do NOT do this!)
# uv run scripts/train.py --env vbot_navigation_section001  # ← NEVER for parameter search
```

### What Counts as "Enough" Testing

| Purpose | Steps | Seeds |
|---------|-------|-------|
| Quick sanity check | 500K | 1 |
| Hypothesis validation | 2M | 3 |
| Serious candidate | 5M | 3 |
| Pre-deployment | 10M+ | 5 |

---

## Phase 5: Evaluate

### Comparison Checklist

After a test run, answer these questions:

1. **Did the target behavior improve?** (Watch the policy, not just numbers)
2. **Did any other behavior degrade?** (Side effects)
3. **Is the improvement consistent across seeds?** (Not lucky variance)
4. **Is the reward curve healthy?** (Monotonic improvement, no collapse)
5. **Did training become slower?** (Some signals slow convergence)

### Decision Matrix

| Target improved? | Side effects? | Consistent? | Verdict |
|------------------|---------------|-------------|---------|
| Yes | None | Yes | **ADOPT** — Archive and integrate |
| Yes | Minor | Yes | **ITERATE** — Tune weight to reduce side effect |
| Yes | Major | Yes | **RETHINK** — Signal is right, mechanism needs redesign |
| Yes | Any | No | **INCONCLUSIVE** — More seeds or longer run |
| No | — | — | **REJECT** — Archive with notes, try different approach |

### Quantitative Evaluation

Use `starter_kit_schedule/scripts/analyze.py` for systematic comparison:

```powershell
# Compare experiments by reward metric
uv run starter_kit_schedule/scripts/analyze.py `
    --metric episode_reward_mean `
    --group-by reward_config

# Visual comparison via subagent
copilot --model gpt-4.1 --allow-all -p "Compare reward curves in runs/<exp_a>/ vs runs/<exp_b>/. Which converged faster? Which is more stable?" -s
```

---

## Phase 6: Archive

### Why Archive Everything

Every experiment result — positive or negative — is valuable. Negative results prevent re-trying failed ideas. Positive results enable reuse and combination.

### Reward Library Location

All reward/penalty instances are archived in:

```
starter_kit_schedule/reward_library/
├── README.md                      # Library index and conventions
├── components/                    # Individual reward/penalty definitions
│   └── <component_name>.yaml      # One per reward idea
├── configs/                       # Complete reward configurations
│   └── <config_name>.yaml         # Tested combinations that work
└── rejected/                      # Ideas that didn't work
    └── <idea_name>.yaml           # With notes on why
```

### Component Archive Format

```yaml
# starter_kit_schedule/reward_library/components/<name>.yaml
name: "<descriptive name>"
type: "reward" | "penalty"
category: "navigation" | "stability" | "efficiency" | "terrain" | "gait" | "safety"
status: "tested" | "promising" | "rejected" | "untested"

description: "<what this signal does>"
hypothesis: "<why it should help>"
mechanism: "<mathematical formulation or pseudocode>"

weight_range:
  tested: [<min>, <max>]
  recommended: <value>

terrain_applicability:
  - flat
  - stairs
  - waves
  - obstacles

experiments:
  - id: "RPE_20260206_001"
    result: "improved" | "degraded" | "neutral"
    notes: "<brief finding>"

implementation:
  file: "<path to _compute_reward if custom code needed>"
  config_key: "<key in reward_scales dict>"
  requires_code_change: true | false
```

### Config Archive Format

```yaml
# starter_kit_schedule/reward_library/configs/<name>.yaml
name: "<config name>"
description: "<what this config is optimized for>"
environment: "<env id>"
date_tested: "YYYY-MM-DD"

reward_scales:
  position_tracking: 2.0
  heading_tracking: 1.0
  # ... all weights
  termination: -200.0

performance:
  episode_reward_mean: <value>
  success_rate: <value>
  notes: "<qualitative assessment>"

based_on: "<parent config name, if iterative>"
```

### Archiving Commands

```powershell
# After an experiment, archive the finding
# (manual for now — create the YAML by hand or instruct Copilot)

# List what's in the library
Get-ChildItem starter_kit_schedule/reward_library/components/ -Name
Get-ChildItem starter_kit_schedule/reward_library/configs/ -Name
Get-ChildItem starter_kit_schedule/reward_library/rejected/ -Name
```

---

## Anti-Patterns to Avoid

| Anti-Pattern | Why It Fails | Instead |
|--------------|-------------|---------|
| **Manual train.py iteration** | **Slow one-at-a-time search; no structured comparison** | **Use automl.py --hp-trials 8+ for batch search** |
| Changing 3+ rewards at once | Cannot attribute outcomes | One variable per cycle |
| Copying rewards from papers | Context differs | Use as hypothesis, test locally |
| Only watching reward curves | High reward ≠ good behavior | Always watch policy visually |
| Discarding failed experiments | Wastes future effort | Archive in `rejected/` |
| Tuning weights without diagnosis | Blind search | Diagnose behavior first |
| Adding rewards without removing | Reward bloat, slow training | Ablate unneeded components |
| Assuming terrain-agnostic rewards | Different terrains need different signals | Test per-terrain |
| **Unconditional per-step bonuses** | **Robot learns to survive, not navigate. alive_bonus × max_steps >> goal reward** | **Make per-step bonuses conditional (e.g., only before reaching goal)** |
| **Goal reward too small vs per-step** | **arrival_bonus=15 vs alive_bonus×3800≈1900** | **arrival_bonus must dominate: set to ≥ alive × typical_episode_len / 3** |
| **Trusting reward curves alone** | **"Reward=7.6" looked great, but robot was lazy** | **Always verify with distance, reached%, episode length metrics** |

---

## Optimal Reward Calculation Principle

For each reward/penalty design, first read the workspace `REPORT_NAV*.md` files to locate or add representative test cases (for example: robot standing still, robot randomly walking, robot sprinting to the target and staying there safely). For each test case estimate the expected cumulative reward magnitudes (order-of-magnitude) before running long experiments — this helps set sensible search ranges and avoids reward hacking and reward-budget imbalances that lead to exploits.

---

## 🔴 Case Study: The Lazy Robot (Reward Hacking)

This case study documents a critical reward hacking failure discovered during long training, and the three-pronged fix. The specific numbers below are illustrative — see `starter_kit_docs/{task-name}/Task_Reference.md` for exact values.

### Timeline

1. **Short AutoML trials**: Best trial achieved good reward, moderate reached%. Looked promising.
2. **Long full training**: Early metrics excellent — distance decreasing, high reach rate.
3. **At ~30 min mark**: Robot became lazy. Distance went UP, reached% dropped, episode length near max.
4. **Root cause**: `alive_bonus × ~max_steps` dwarfed `arrival_bonus`. The robot discovered standing still in a safe spot was the optimal strategy.

### Detection Signals

| Metric | Healthy | Lazy Robot |
|--------|---------|------------|
| Distance to target | Decreasing → 0 | Decreasing then **increasing** |
| Reached % | Increasing | Increasing then **dropping to ~0** |
| Episode length | Moderate (500-2000) | **Near max_steps** |
| Reward | Increasing | **Still looks good** (alive_bonus accumulates!) |
| Arrival bonus (TensorBoard) | Increasing | **Dropping toward 0** |

### The Fix: Anti-Laziness Trifecta

#### 1. Conditional alive_bonus
```python
# BEFORE (exploitable):
alive_bonus = scales["alive_bonus"]  # 0.5 every step, unconditionally

# AFTER (fixed):
alive_bonus = np.where(ever_reached, 0.0, scales["alive_bonus"])
# Once robot reaches target, alive bonus stops → no "stand around" exploit
```

#### 2. Time decay on navigation rewards
```python
time_decay = np.clip(1.0 - 0.5 * env_steps / max_episode_steps, 0.5, 1.0)
# Step 0: time_decay = 1.0 (full reward)
# Step 2000 (half episode): time_decay = 0.75
# Step 4000 (max): time_decay = 0.5
# Creates URGENCY — reach the goal early for maximum reward
# NOTE: Penalties are NOT multiplied by time_decay (stay full strength)
```

#### 3. Massive arrival_bonus
```python
# BEFORE: arrival_bonus too small (trivially beaten by alive_bonus accumulation)
# AFTER: arrival_bonus large enough to dominate
# Rule of thumb: arrival_bonus > alive_bonus × max_steps / 4
```

### Lesson

**Always audit the reward budget.** Compute: what is the maximum reward achievable by the desired behavior vs. by the degenerate behavior? If they're close, the agent WILL find the exploit given enough training time.

---

## Summary: Exploration Philosophy

1. **Diagnose before prescribing** — Watch the policy, identify the behavioral gap
2. **One variable per experiment** — Isolate cause and effect
3. **Test short, test often** — 2M step runs across 3 seeds reveal more than one 20M run
4. **Archive everything** — Positive results reuse, negative results prevent repetition
5. **Methodology over recipes** — The process of finding good rewards matters more than any specific reward
6. **Library over memory** — Store tried components in `reward_library/`, not in your head
7. **Competition ≠ training** — Verify that training rewards actually improve competition score
