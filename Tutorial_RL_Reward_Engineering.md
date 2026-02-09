# Tutorial: RL Reward Engineering for Quadruped Robot Navigation

**A Case Study from MotrixArena S1 Competition**

> This tutorial distills hard-won lessons from 8+ experiments optimizing a quadruped robot (VBot) to navigate from the edge of a circular platform to its center. Every section is backed by real experimental data, real failures, and real fixes.

---

## Table of Contents

1. [The Task](#1-the-task)
2. [The RL Training Loop (How It Works)](#2-the-rl-training-loop)
3. [Lesson 1: The Reward Budget Audit](#3-lesson-1-the-reward-budget-audit)
4. [Lesson 2: Reward Hacking — The Lazy Robot](#4-lesson-2-reward-hacking-the-lazy-robot)
5. [Lesson 3: Reward Hacking — The Sprint-and-Crash Robot](#5-lesson-3-reward-hacking-the-sprint-and-crash-robot)
6. [Lesson 4: The Learning Rate Scheduler Trap](#6-lesson-4-the-learning-rate-scheduler-trap)
7. [Lesson 5: Curriculum Learning](#7-lesson-5-curriculum-learning)
8. [Lesson 6: Config Persistence is a Real Engineering Problem](#8-lesson-6-config-persistence)
9. [Lesson 7: Observation Normalization Mismatches](#9-lesson-7-observation-normalization)
10. [The Complete Experiment Timeline](#10-the-complete-experiment-timeline)
11. [Design Principles (Summary)](#11-design-principles)
12. [Lesson 8: Use AutoML Batch Search, Not Manual train.py](#12-lesson-8-automl-batch-search)
13. [Appendix: Key Code Patterns](#appendix-key-code-patterns)

---

## 1. The Task

**Goal**: Train a 12-joint quadruped robot to walk from the outer ring of a circular platform (radius ~12.5m) to the center (0,0), stop, and hold position.

**Environment**:
- 2048 parallel environments (vectorized NumPy simulation)
- **Observation**: 54-dimensional vector (linear velocity, gyro, projected gravity, joint positions/velocities, last actions, velocity commands, position error, heading error, distance, reached flag)
- **Action**: 12-dimensional (4 legs × 3 joints), mapped to PD torque targets via `torque = kp*(target - current) - kv*velocity`
- **Success**: Robot arrives within 0.3m of center AND stops (speed < 0.15 m/s) for 50 consecutive steps (0.5 seconds)
- **Episode**: Max 4000 steps (40 seconds) or until robot falls (body contact with ground)

**Competition scoring**: 10 robots × 2 points each = 20 max. Robots spawn at ~10m from center.

---

## 2. The RL Training Loop

Understanding **PPO (Proximal Policy Optimization)** at a high level is essential. The loop works like this:

```
for each iteration:
    1. ROLLOUT: Run all 2048 envs for 32 steps, collecting (obs, action, reward, done)
    2. COMPUTE ADVANTAGES: Use GAE (λ=0.95, γ=0.99) to estimate advantage
    3. UPDATE POLICY: For 5 epochs, sample mini-batches (16 total), 
       update policy/value networks via clipped PPO loss
    4. SCHEDULER: Adjust learning rate (KL-adaptive or linear anneal)
    5. LOG: Write reward, reached%, distance, etc. to TensorBoard
    
    Total env steps per iteration = 2048 envs × 32 rollout steps = 65,536
    At 100M total steps: ~1,525 iterations
    TensorBoard logs every 1000 iterations = every 65.5M env steps
```

**Key hyperparameters** and what they control:

| Parameter | Value | What It Does |
|-----------|-------|--------------|
| `num_envs` | 2048 | More parallel envs = more diverse experience per iteration |
| `rollouts` | 32 | Steps per env before update. Longer = more context for GAE, but staler data |
| `learning_epochs` | 5 | How many passes over the collected data. More = risk overfitting to stale rollout |
| `mini_batches` | 16 | Divides the rollout data into chunks. Fewer = larger effective batch |
| `learning_rate` | 5e-4 | Step size. Too high = instability. Too low = no progress |
| `discount_factor` (γ) | 0.99 | How much to weight future rewards. 0.99 = long-horizon planning |
| `entropy_loss_scale` | 0.01 | Encourages exploration by penalizing deterministic policies |

---

## 3. Lesson 1: The Reward Budget Audit

> **Core Principle**: Before training, compute the **maximum reward** a policy can earn from each behavior. If degenerate behavior earns more than desired behavior, the policy **will** find the exploit.

### The Audit Process

For any reward function, compute:

```
For each reward component:
    MAX_VALUE = scale × max_achievable_per_step × steps_it_can_be_earned

Compare: 
    Total_DESIRED_behavior_reward  vs  Total_DEGENERATE_behavior_reward
```

### Our Initial Audit (BEFORE fixing)

| Reward Source | Scale | Per-Step Max | Steps Active | **Episode Total** |
|--------------|-------|-------------|-------------|-------------------|
| `alive_bonus` | 1.5 | 1.5 | 3000 (all) | **4,500** |
| `position_tracking` | 1.5 | ~1.0 | 3000 | ~4,500 |
| `arrival_bonus` | 50 | 50 (one-time) | 1 | **50** |
| `inner_fence_bonus` | 25 | 25 (one-time) | 1 | **25** |
| `termination` | -10 | -10 (one-time) | 1 | **-10** |

**Degenerate behavior (stand still, survive)**: alive_bonus = **4,500 points**  
**Desired behavior (reach center)**: arrival = 50 + inner_fence = 25 = **75 points**  
**Ratio**: 4500 : 75 = **60:1 in favor of laziness**  

The policy has **zero rational incentive** to attempt the risky approach to center when simply standing still earns 60× more.

### The Fix: Budget Rebalancing

```python
# Phase5 rewards (after fix)
alive_bonus: 0.15      # 0.15 × 4000 = 600 (effective ~450 with time_decay)
arrival_bonus: 100.0    # One-time, must DOMINATE alive budget
inner_fence_bonus: 40.0 # Intermediate milestone
termination: -150.0     # Death costs 33% of alive budget — expensive!
```

**New ratio**: Goal completion (~390) vs effective alive (~450) ≈ **1:1** (balanced)  
**Death penalty**: -150 / 450 = 33% of alive budget lost per fall

### The Anti-Laziness Trifecta

From extensive experimentation, three values must satisfy:

```
1. alive_bonus × max_steps  <  2 × arrival_bonus          # Can't out-earn goal by surviving
2. |termination|           >  0.25 × alive_bonus × max_steps  # Dying is costly
3. arrival_bonus           >  50% of alive_budget              # Goal dominates
```

---

## 4. Lesson 2: Reward Hacking — The Lazy Robot

### What It Looks Like

<img src="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg'/>" alt="placeholder"/>

| Metric | Step 5000 | Step 9000 | Signal |
|--------|-----------|-----------|--------|
| **Total Reward** | 4.93 | 6.63 | ↑ Going up |
| **Reached%** | 55.4% | 6.4% | ↓ Collapsing |
| Inner Fence% | 70.6% | 78.8% | ↑ Stable |
| Distance | 0.98m | 1.10m | → Plateau |

**The red flag**: Reward going up while task performance goes down = **reward hacking**.

### Why It Happens

The policy discovered that navigating to ~0.75m from center (inner fence zone) and then **stopping** earns:
- Full `alive_bonus` (1.5/step × remaining steps)
- Strong `position_tracking` (distance small, exp(-d/5) ≈ 0.85)
- No termination penalty (not falling)

Versus actually reaching center (0.3m):
- Risk of overshooting → termination
- `near_target_speed` penalty if moving too fast
- Only 50 bonus for arrival (negligible vs 4500 alive)

The policy rationally chose the low-risk, high-reward degenerate strategy.

### How to Diagnose

1. **Plot reward AND reached% together** — if they diverge, you have a hack
2. **Check `alive_bonus` per-component trend** — if it's flat/increasing while other components decrease, the robot is surviving but not performing
3. **Check episode length** — if it stays at max (3000 steps), the robot isn't falling but also isn't completing the task

### How to Fix

See Section 3 — the reward budget audit and rebalancing.

---

## 5. Lesson 3: Reward Hacking — The Sprint-and-Crash Robot

### A Different Exploit

After fixing the Lazy Robot via budget rebalancing, a **new** exploit emerged:

| Component | Step 1000 | Step 5000 | Step 12000 | Trend |
|-----------|-----------|-----------|------------|-------|
| `forward_velocity` | 0.11 | 0.55 | **0.83** | ↑↑ Sprinting |
| `stop_bonus` | 0.02 | 2.10 | **4.45** | ↑↑ Brief touches |
| Episode length | 390 | 1366 | **331** | ↑↓ Crashed |

### The Exploit

1. **Sprint** at maximum speed toward center
2. **Briefly touch** the 0.3m zone (collect `arrival_bonus` + accumulate `stop_bonus`)
3. **Crash/fall** almost immediately (momentum from sprint → tumble)
4. **Reset** → repeat → more episodes per unit time = more reward

The policy optimized for **episodes per hour**, not per-episode quality. Each sprint-and-crash gives arrival_bonus=100 + brief stop_bonus, then quickly resets for another attempt.

### Why It's Different from Lazy Robot

| | Lazy Robot | Sprint-and-Crash |
|---|-----------|-----------------|
| **Exploits** | Per-step rewards (alive_bonus) | Per-episode rewards (arrival + reset) |
| **Episode length** | Maximum (3000) | Very short (300-500) |
| **Robot behavior** | Stands still near target | Runs fast, falls at target |
| **forward_velocity** | Near zero | Maximum |
| **Fix** | Reduce alive_bonus | Reduce forward_velocity scale |

### How to Fix

```python
# Reduce speed incentive
forward_velocity: 0.8 → 0.3 → 0.2  # Iteratively reduced

# Add speed penalty near target
near_target_speed: -1.5  # Penalize speed when distance < threshold

# CRITICAL: Tune the activation radius!
# Too large (2.0m) → "deceleration moat" — robot hovers at 1m, never reaches 0.3m
# Too small (0.3m) → no effect, robot still sprints
# Sweet spot: 0.5m — robot can slow down at last moment
near_target_activation_radius: 0.5  # Only penalize speed within 0.5m of target
```

### The Deceleration Moat Problem

A subtle failure: `near_target_speed` penalty with activation radius 2.0m created a zone where the robot was **punished for moving**. Result: robots learned to hover at ~1m from target, getting high `position_tracking` reward but never crossing the 0.3m threshold.

```
             2m                    0.3m
Robot ----→  |===PENALTY ZONE===|  |TARGET|
             ↑                     ↑
    Robot stops here           Never gets here
    (near_target_speed          (too penalized
     punishment too strong)      to approach)
```

**Fix**: Reduce activation radius from 2.0m to 0.5m. Robot can approach freely until the last 0.5m, then decelerates smoothly.

---

## 6. Lesson 4: The Learning Rate Scheduler Trap

### KL-Adaptive LR Scheduler

SKRL's default scheduler adjusts LR based on KL divergence between old and new policy:

```python
# Pseudocode
if kl > kl_threshold:
    lr = lr * (1 - damping)  # Reduce if policy changes too much
else:
    lr = lr * (1 + damping)  # Increase if policy is too conservative
```

### The Problem: It's Unstable for This Task

We tested three thresholds, all failed:

| kl_threshold | What Happened | LR Trajectory |
|-------------|---------------|---------------|
| **0.016** (loose) | LR spiked to 0.0018 (7× initial) → massive policy update → performance collapsed from 67% to 18% | 0.00025 → 0.0018 → 0.00025 |
| **0.008** (tight) | KL always exceeded threshold → LR crushed to floor → learning stalled at 12% | 0.00025 → 0.000167 (stuck) |
| **0.012** (middle) | Neither stable nor unstable → oscillated, peaked at 50% then declined to 26% | 0.0005 → 0.00038 → 0.00028 |

### Why It Fails

Early training produces **naturally high KL divergence** — the policy is changing rapidly from random to purposeful behavior. A KL-adaptive scheduler interprets this as "too much change" and aggressively damps LR, starving the policy of the large gradients it needs to learn basic locomotion.

Conversely, if the threshold is too loose, the scheduler allows LR to climb, creating a positive feedback loop: higher LR → larger updates → lower KL briefly → even higher LR → overshoot → policy collapse.

### The Fix: Linear Anneal

Simple, predictable, stable:

```python
# Linear LR schedule
lr(t) = lr_initial × max(1 - t/T, 0.01)
# Where T = total_updates, t = current update step
# Anneal from 5e-4 to 5e-6 over the full training run
```

**Implementation** (added to the PPO trainer):

```python
if lr_scheduler_type == "linear":
    total_updates = max_env_steps / (rollouts * num_envs) * learning_epochs
    scheduler = LambdaLR(optimizer, 
        lr_lambda=lambda epoch: max(1.0 - epoch / total_updates, 0.01))
elif lr_scheduler_type == "kl_adaptive":
    scheduler = KLAdaptiveRL(...)
else:
    scheduler = None  # Fixed LR
```

**Result**: LR decreased smoothly from 0.000495 to 0.000382 over 12K steps. No spikes, no crashes. The same peak→decline pattern still appeared, proving the LR scheduler was never the root cause of performance degradation — it was the **reward function** (sprint-crash exploit, Section 5).

### Takeaway

> Don't add complex adaptive mechanisms when a simple schedule works. Linear anneal gives you one fewer thing to debug.

---

## 7. Lesson 5: Curriculum Learning

### The Problem with Training at Competition Difficulty

Competition requires spawn at 9-10m from center. But training from scratch at this distance:
- 90%+ of initial random actions lead to falling (episode length ~300-500 steps)
- Robot rarely reaches target even by accident → arrival_bonus almost never fires
- Very sparse reward signal → slow/no learning

### The Curriculum Approach

Train in stages of increasing difficulty, starting easy and promoting when performance plateaus:

```
Stage 1: Easy (2-5m spawn)
├── Goal: Learn locomotion + heading + basic approach + stopping
├── Advantage: ~20% of random walks accidentally reach 0.3m center
├── Config: spawn_inner=2.0, spawn_outer=5.0, lr=5e-4
├── Promotion criteria: reached > 70%, stable episode length > 1500
│
Stage 2: Medium (5-8m spawn)
├── Goal: Extend learned skills to medium distance
├── Config: spawn_inner=5.0, spawn_outer=8.0, lr=2.5e-4
├── Warm-start: Load Stage 1 best checkpoint, RESET optimizer
├── Promotion criteria: reached > 60%
│
Stage 3: Competition (8-11m spawn)
├── Goal: Full competition performance
├── Config: spawn_inner=8.0, spawn_outer=11.0, lr=1.25e-4
├── Warm-start: Load Stage 2 best checkpoint, RESET optimizer
├── Target: reached > 80% (= 16/20 competition points)
```

### Annular Spawning

Critical implementation detail: robots spawn uniformly in an annular ring, not a circle:

```python
def _random_point_in_annulus(n, inner_r, outer_r):
    """Uniform random points in annulus [inner_r, outer_r]"""
    theta = np.random.uniform(0, 2*np.pi, n)
    # r = sqrt(U*(R2²-R1²) + R1²) for uniform area distribution
    r = np.sqrt(np.random.uniform(0, 1, n) * (outer_r**2 - inner_r**2) + inner_r**2)
    return np.column_stack([r * np.cos(theta), r * np.sin(theta)])
```

**Why not `r = uniform(inner_r, outer_r)`?** That would over-sample near `inner_r` — the area of an annular slice is proportional to `r`, so you need the sqrt transformation for uniform area sampling.

### Warm-Start Gotcha: Poisoned Optimizer State

**Critical finding**: If a training run degraded (reward collapsed), warm-starting from its checkpoint **does not recover**, even with the bug fixed. Why?

The Adam optimizer maintains per-parameter **momentum** (1st moment) and **variance** (2nd moment) estimates. After a degradation:
- Momentum points in the "wrong" direction (toward the degenerate policy)
- Variance estimates are inflated from the instability
- The new gradients are small in comparison → optimizer barely moves

**Fix**: When warm-starting, **reset the optimizer state to fresh** (or train from scratch with fixed configs). The policy weights are worth keeping; the optimizer state is not.

---

## 8. Lesson 6: Config Persistence is a Real Engineering Problem

### What Happened

During Session 1, we applied "Phase5" reward changes via code edits to `cfg.py`. We verified the changes appeared correct. We launched training. In hindsight, **the changes were never saved** (editor revert, Python import cache, or file system issue).

All 4 experiments in Session 1 ran with the **original pre-Phase5 rewards**. The REPORT documented Phase5 values that never actually ran.

Discovery: At the start of Session 2, a runtime verification check printed:
```
alive_bonus: 0.5       ← NOT 0.15 as reported!
arrival_bonus: 50.0    ← NOT 100.0!
```

### Why It's Hard to Catch

Python dataclass configs are **compiled at import time**. If you edit `cfg.py` but a cached `.pyc` file or in-memory module is used, your changes don't take effect. Additionally, `VBotSection001EnvCfg` **inherited** from `VBotEnvCfg` — changes to the base class `RewardConfig` could be overridden, lost, or stale.

### Prevention

1. **Always verify runtime config before training**:
```python
uv run python -c "
from starter_kit.navigation1.vbot import cfg as _
from motrix_envs.registry import make
env = make('vbot_navigation_section001', num_envs=1)
s = env._cfg.reward_config.scales
print('alive_bonus:', s['alive_bonus'])  # Should be 0.15
print('arrival_bonus:', s['arrival_bonus'])  # Should be 100.0
"
```

2. **Override configs in the subclass, not the base class**:
```python
# BAD: editing VBotEnvCfg.RewardConfig (base class)
# Changes can be lost when subclass overrides are added later

# GOOD: dedicated override in VBotSection001EnvCfg
@dataclass
class VBotSection001EnvCfg(VBotStairsEnvCfg):
    @dataclass
    class RewardConfig:  # ← OWN copy, not inherited
        scales: dict[str, float] = field(default_factory=lambda: {...})
    reward_config: RewardConfig = field(default_factory=RewardConfig)
```

3. **Log the actual config in TensorBoard** (if possible) or at training start.

---

## 9. Lesson 7: Observation Normalization Mismatches

### The Bug

Training environment:
```python
position_error_normalized = position_error / 12.0
distance_normalized = np.clip(distance_to_target / 12.0, 0, 1)
```

Evaluation script (`play_10_robots_1_target.py`):
```python
position_error_normalized = position_error / 5.0    # ← WRONG!
distance_normalized = np.clip(distance_to_target / 5.0, 0, 1)  # ← WRONG!
```

### The Impact

The policy received position signals **2.4× amplified** during evaluation vs training. A robot 5m from center saw `d_norm = 5/5 = 1.0` in evaluation but was trained with `d_norm = 5/12 = 0.42` — completely different input.

### Why It's Hard to Catch

- No error messages — the observation shape is correct, just the values are wrong
- The policy still *kinda* works (it sees "target is far away" either way), just poorly
- During training, everything looks fine because normalization matches

### Prevention

- Define normalization constants in **one place** (the config), import everywhere
- Add assertions: `assert obs.max() < 5.0, "Observation normalization check failed"`
- Test that evaluation script produces the same observations as training env for the same state

---

## 10. The Complete Experiment Timeline

This shows the iterative nature of RL experimentation — each run reveals new information:

| # | Config Change | Steps | Peak Reached% | Outcome | Discovery |
|---|--------------|-------|---------------|---------|-----------|
| 1 | Phase5 rewards, kl=0.016 | 15K | **67%** | Collapsed to 18% | KL threshold too loose → LR spike |
| 2 | Warm-start from Exp1, kl=0.008 | 5K | 27% | Stagnant | Poisoned optimizer state |
| 3 | Fresh, kl=0.008 | 10K | 32% | Plateau at 12% | KL threshold too tight → LR crushed |
| 4 | kl=0.012, spawn=full | 5K | 50% | Declined to 26% | No KL threshold works |
| 5 | **Linear LR** scheduler | 12K | **59%** | Sprint-crash exploit | forward_velocity=1.5 too high |
| 6 | Config drift discovered | 7K | 0.05% | Failed | Phase5 rewards never persisted |
| 7 | Phase5 + near_target d<2m | 12K | 20% | Deceleration moat | near_target_speed radius=2m too large |
| 8 | near_target_speed d<0.5m | 19K | **52%** | Sprint-crash returned | Best peak — but unstable after 12K |
| 9 | forward_velocity=0.2 | 4K | 0% | Robot lazy | Velocity reward too weak |
| 10 | forward_velocity=0.5 | 6K | 8.6% | Still too weak | Need ≥0.8 for navigation drive |
| 11 | Speed cap + term=-250 | 8K | 7% | Too conservative | Termination too harsh |
| 12 | term=-200, speed cap=0.6 | — | — | Interrupted | Should have used AutoML |

**Key insight**: Progress is not linear. Each experiment's failure teaches you something specific. The biggest breakthroughs come from **diagnosing the per-component reward breakdown**, not from tuning a single number.

**Critical lesson from Exp7-12**: Manual one-at-a-time search via `train.py` is wasteful. These 6 experiments should have been a single `automl.py --hp-trials 8` batch search. See [Lesson 8](#12-lesson-8-automl-batch-search).

### How to Read TensorBoard for RL

The most important signals:

| Signal | Healthy | Unhealthy |
|--------|---------|-----------|
| Reward ↑ AND reached% ↑ | ✅ Learning | — |
| Reward ↑ AND reached% ↓ | — | ❌ Reward hacking |
| Reward → AND reached% → | — | ⚠️ Plateau (need config change) |
| Episode length ↓ rapidly | — | ⚠️ Dying more (termination too cheap?) or sprint-crash |
| Episode length ↑ to max | — | ⚠️ Lazy robot (not completing task, just surviving) |
| LR ↓ rapidly to floor | — | ❌ KL scheduler crushing learning |
| LR ↑ above 3× initial | — | ❌ KL scheduler runaway |

---

## 11. Design Principles (Summary)

### Reward Engineering

1. **Audit the budget first.** Compute max reward for desired vs degenerate behavior. If degenerate wins, fix the scales before training.

2. **The anti-laziness trifecta:** `alive_bonus ≈ 0.1–0.3`, `arrival_bonus ≥ 80`, `termination ≤ -100`.

3. **One-time bonuses must dominate per-step accumulations.** If `alive_bonus × max_steps > 5 × arrival_bonus`, the policy will always choose survival over completion.

4. **Penalties create "moats" if the activation radius is too large.** `near_target_speed` within 2m prevented approach; within 0.5m worked perfectly.

5. **Different exploits need different fixes.** Lazy Robot (per-step) ≠ Sprint-Crash (per-episode). Diagnosing which exploit is active requires per-component reward breakdown.

### Training Protocol

6. **Linear LR anneal > KL-adaptive** for tasks with rapidly-changing KL dynamics (early RL training).

7. **Curriculum: easy → hard.** Start at 2-5m, promote when performance plateaus, extend to competition distance.

8. **Don't warm-start from degraded runs.** Reset the optimizer state. Only the policy weights are worth keeping.

9. **Verify config at runtime, not in the editor.** Python import caching and inheritance can silently use stale values.

### Experimental Methodology

10. **Change one variable per experiment.** Changing kl_threshold, spawn radius, and LR simultaneously makes attribution impossible.

11. **Always plot reward AND task metric.** If they diverge, you have a reward hack, not a training problem.

12. **Per-component reward breakdown is your best diagnostic tool.** Look at `Reward Instant/forward_velocity`, `Reward Instant/stop_bonus`, etc. in TensorBoard.

13. **Short runs reveal failure modes quickly.** 5-10K TensorBoard steps (~2-5 hours) is enough to see if reward shaping is working. Don't wait for 10M steps to discover a broken budget.

14. **Use AutoML batch search, not manual train.py iteration.** When comparing N reward configurations, use `automl.py --hp-trials N`. Manual one-at-a-time search is slow, error-prone, and produces no structured comparison. See [Lesson 8](#12-lesson-8-automl-batch-search).

---

## 12. Lesson 8: Use AutoML Batch Search, Not Manual train.py

### The Anti-Pattern: Manual One-at-a-Time Search

In Session 3, we made a fundamental methodological error. We iterated manually with `train.py`:

```
Edit cfg.py (change forward_velocity from 0.8 to 0.2)
→ Run train.py → Wait 5 min → Read TensorBoard → Kill
→ Edit cfg.py (change to 0.5) → Run train.py → Wait → Kill
→ Edit cfg.py (change to 0.8, add speed cap) → Run → Kill
→ ... repeat 6 times, 30+ minutes total
```

This produced 6 experiments (Exp7-12), each testing a single variable change, with no structured comparison and no statistical significance.

### Why It's Wrong

| Problem | Impact |
|---------|--------|
| **No parallel comparison** | Each experiment runs independently, no shared baseline |
| **No Bayesian suggestion** | Human guesses which parameter to try next |
| **No structured report** | Results scattered across TensorBoard logs and ad-hoc notes |
| **Sequential execution** | 6 × 5 min = 30 min. AutoML could run 8 trials in ~40 min with comparison table |
| **Manual attribution** | "Was it the speed cap or the termination that helped?" — impossible to tell |
| **No reproducibility** | Config changes made by hand, may not be recorded properly |

### The Correct Workflow: AutoML Batch Search

Instead of manually editing `cfg.py` and running `train.py`, define the search space and let AutoML find the best combination:

```python
# In automl.py REWARD_SEARCH_SPACE, define ranges for all parameters to explore:
REWARD_SEARCH_SPACE = {
    "forward_velocity": {"type": "uniform", "low": 0.2, "high": 1.0},
    "near_target_speed": {"type": "uniform", "low": -2.0, "high": -0.1},
    "termination": {"type": "choice", "values": [-300, -200, -150, -100]},
    # ... other parameters
}
```

Then run:
```powershell
uv run starter_kit_schedule/scripts/automl.py --mode stage --budget-hours 4 --hp-trials 8
```

This will:
1. Sample 8 different configurations from the search space
2. Run each for the configured number of steps
3. Read TensorBoard metrics automatically
4. Produce a structured comparison report in `starter_kit_log/automl_*/report.md`
5. Use Bayesian optimization to suggest better configs in subsequent trials

### When Each Tool Is Appropriate

| Task | Tool | Why |
|------|------|-----|
| **Parameter exploration** | `automl.py` | Batch comparison, structured reports |
| **Reward hypothesis testing** | `automl.py` | N trials with side-by-side comparison |
| **Smoke test** | `train.py --max-env-steps 200000` | Quick check that code compiles |
| **Visual debugging** | `train.py --render` | Watch robot behavior in real-time |
| **Final deployment** | `train.py` | Train winning config to full steps |

### Lesson

> **If you're about to run `train.py` more than twice with different configs — STOP. Use `automl.py` instead.**
> The AutoML pipeline exists precisely for parameter search. Manual iteration is the RL equivalent of debugging by adding print statements throughout your code instead of using a debugger.

---

## Appendix: Key Code Patterns

### A. Reward Budget Audit Script

```python
# Run this BEFORE launching training
uv run python -c "
from starter_kit.navigation1.vbot import cfg as _
from motrix_envs.registry import make
env = make('vbot_navigation_section001', num_envs=1)
cfg = env._cfg
s = cfg.reward_config.scales
max_steps = cfg.max_episode_steps
alive = s['alive_bonus'] * max_steps
arrival = s['arrival_bonus']
death = s['termination']
print(f'Alive budget: {s[\"alive_bonus\"]} x {max_steps} = {alive}')
print(f'Arrival bonus: {arrival}')
print(f'Death penalty: {death}')
print(f'Alive/Arrival ratio: {alive/arrival:.1f}:1')
print(f'Death/Alive ratio: {abs(death)/alive:.1%}')
print()
if alive > 5 * arrival:
    print('WARNING: Lazy Robot exploit likely! alive >> arrival')
if abs(death) < 0.1 * alive:
    print('WARNING: Death is too cheap! |termination| < 10% alive budget')
"
```

### B. TensorBoard Metrics Check Script

```python
# Check training progress from command line
uv run python -c "
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
ea = EventAccumulator('runs/vbot_navigation_section001/YOUR_RUN_DIR')
ea.Reload()
for tag in ['Reward / Instantaneous reward (mean)', 
            'metrics / reached_fraction (mean)',
            'metrics / distance_to_target (mean)',
            'Learning / Learning rate']:
    events = ea.Scalars(tag)
    if events:
        last = events[-1]
        print(f'{tag.split(\"/\")[-1].strip()}: step={last.step}, val={last.value:.4f}')
"
```

### C. Annular Spawn Implementation

```python
def _random_point_in_annulus(self, n, inner_r, outer_r):
    """Uniform random points in annulus [inner_r, outer_r]"""
    theta = np.random.uniform(0, 2 * np.pi, n)
    # sqrt(U*(R2²-R1²)+R1²) for uniform area distribution
    r = np.sqrt(
        np.random.uniform(0, 1, n) * (outer_r**2 - inner_r**2) + inner_r**2
    )
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.column_stack([x, y])
```

### D. Linear LR Scheduler Setup

```python
# In PPO trainer initialization
if cfg.lr_scheduler_type == "linear":
    _total_updates = int(cfg.max_env_steps / (cfg.rollouts * cfg.num_envs))
    scheduler_class = torch.optim.lr_scheduler.LambdaLR
    scheduler_kwargs = {
        "lr_lambda": lambda epoch: max(1.0 - epoch / max(_total_updates, 1), 0.01)
    }
```

### E. Config Verification One-Liner

```powershell
# Quick config check — run before every training launch
uv run python -c "from starter_kit.navigation1.vbot import cfg as _; from motrix_envs.registry import make; env = make('vbot_navigation_section001', num_envs=1); s = env._cfg.reward_config.scales; print('alive:', s['alive_bonus'], '| arrival:', s['arrival_bonus'], '| term:', s['termination'], '| fwd_vel:', s['forward_velocity'], '| spawn:', env._cfg.spawn_inner_radius, '-', env._cfg.spawn_outer_radius)"
```

---

## Further Reading

- **REPORT_NAV1.md** — Full experiment log with every data point, attempt, and outcome
- **`.github/copilot-instructions.md`** — AutoML-first policy and essential commands
- **`.github/skills/reward-penalty-engineering/SKILL.md`** — Systematic process for reward discovery
- **`.github/skills/curriculum-learning/SKILL.md`** — Stage progression with warm-starts
- **`.github/skills/hyperparameter-optimization/SKILL.md`** — AutoML grid/random/Bayesian search
- **`starter_kit_schedule/scripts/automl.py`** — AutoML batch search engine (use instead of train.py)
- **MotrixArena S1 Competition Guide** — `starter_kit_docs/MotrixArena_S1_仿真强化学习挑战赛_赛事全攻略.md`
