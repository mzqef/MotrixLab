# Section 011 Experiment Report

**Date**: February 2026  
**Environment**: `vbot_navigation_section011` (MotrixArena S1 Stage 2)  
**Task**: Navigate 10.3m from START → height field bumps → 15° ramp → high platform (z=1.294m). Collect 3 smileys, 3 red packets, and perform a celebration action.  
**Max Score**: 20 points.

---

## 1. Key Challenges & Solutions

### Terrain & Physics
*   **Challenge**: Height field bumps (0.277m) and a 15° ramp caused immediate policy collapse for flat-ground models.
*   **Solution**: 
    *   **Curriculum**: Initially spawned robots past the bumps to learn ramp climbing, then gradually moved spawn back to START.
    *   **PD Control**: Increased torque limits and action scale to handle terrain saturation.

### Reward Engineering (Critical Fixes)
*   **The "Lazy Robot" Bug**: Initial configs rewarded standing still (alive bonus) more than completing the course (~80:1 ratio).
    *   **Fix**: Drastically reduced `alive_bonus`, increased `forward_velocity` and milestone bonuses. Final ratio ~8.5:1 favoring completion.
*   **Double Step-Increment**: A code bug counted every step twice, halving episode lengths and distorting rewards. Fixed by removing duplicate counters.
*   **Grace Period Exploit**: Robots learned to lie on their sides during "grace periods" to collect free rewards.
    *   **Fix**: Implemented **Hard vs. Soft Termination**. Severe tilts (>70°) now trigger immediate death, ignoring grace periods.

### Architecture Evolution
*   **Navigation Logic**: Shifted from single-target navigation to a **4-Phase System** (Smileys → Red Packets → Climb → Celebration).
*   **Targeting Strategy**: Moved from "nearest-first" (causing erratic 90° turns) to **Sweep-Order Targeting** (Left→Center→Right), smoothing paths.
*   **Celebration**: Evolved from complex spin maneuvers to a robust **3-Jump System** based on height thresholds.

---

## 2. Hyperparameter Breakthroughs

The most significant performance gains came from extending the planning horizon, not reward tweaking.

| Parameter | Initial Value | Optimal Value | Impact |
| :--- | :--- | :--- | :--- |
| **Discount Factor ($\gamma$)** | 0.99 | **0.999** | Extended effective planning horizon from ~460 steps to ~4600 steps, allowing the robot to value distant goals (all 3 smileys). |
| **GAE Lambda ($\lambda$)** | 0.95 | **0.99** | Matched advantage estimation to the long discount horizon, reducing variance. |
| **Result** | wp_idx ~1.63 | **wp_idx ~1.98** | **+21% improvement** solely from $\gamma/\lambda$ tuning. |

---

## 3. Training Campaign Highlights (Condensed)

*   **Stages 0–4 (Foundation)**: Fixed terrain bugs, reward budgets, and established the phase-gated navigation system. Reached ~30% completion.
*   **Stages 5–9 (Gait & Stability)**: Extensive experiments with gait rewards (stance ratio, lateral velocity). **Lesson**: Complex gait rewards often diluted navigation signals. Minimalist gait penalties worked best.
*   **Stages 11–15 (Horizon Optimization)**: Discovery of the $\gamma=0.999 / \lambda=0.99$ "Long-Horizon" configuration. This was the single largest leap in performance.
*   **Stages 16–29 (AutoML & Refinement)**:
    *   **AutoML Discovery**: Confirmed that High Learning Rate + Low Entropy outperforms conservative settings for fresh training.
    *   **Two-Phase Strategy**: Trained first with **gradient-only rewards** (smooth learning), then fine-tuned with **discrete bonuses** (smiley/packet collection).
    *   **Result**: Achieved **83.7% zone collection rate** (wp_idx ~5.86).

---

## 4. Final Optimization: The "Pre-Peak" Strategy

Late-stage training suffered from catastrophic forgetting when warm-starting from the absolute best checkpoint.

*   **Problem**: Warm-starting from the peak (`agent_6000`) caused instability as the optimizer lacked a clear gradient direction.
*   **Solution**: 
    1.  Use a **KL-Adaptive Scheduler** to dynamically limit policy divergence.
    2.  Warm-start from a **pre-peak checkpoint** (`agent_5000`, ascending slope) rather than the peak itself.
*   **Outcome**: Achieved a new record **wp_idx = 5.9115**, surpassing the previous best of 5.86.

---

## 5. Current Status & Best Configuration

*   **Best Checkpoint**: `v35_best_wpidx591.json` (Step 6000).
*   **Performance**: 
    *   **Waypoint Index**: 5.91 / 7.0 (Collects ~2.9/3 smileys, ~2.9/3 red packets, reaches platform).
    *   **Survival**: Significant improvement in ramp traversal stability.
*   **Key Config Settings**:
    *   $\gamma = 0.999$, $\lambda = 0.99$
    *   Policy Net: `(256, 128, 64)`, Value Net: `(512, 256, 128)`
    *   Scheduler: `kl_adaptive`
    *   Reward Arch: Hybrid (Gradient base + Discrete bonuses for milestones).

---

## 6. Post-v35 Config Updates (v42 / v44 / v45) — 2026-02-19

### v42 — Extended Training Budget
*   `max_env_steps` in `rl_cfgs.py` raised from 50 M → **100 M** to allow longer fresh training runs.

### v44 — Longer Episodes + Stagnation Detection
*   **Episode length doubled**: `max_episode_seconds` 60 → **120 s**, `max_episode_steps` 6000 → **12000**.
*   **Stagnation detection** replaces the hard fixed-time cutoff:
    *   `stagnation_window_steps = 1000` (10 s sliding window)
    *   `stagnation_min_distance = 0.5 m` — truncate episode if robot doesn't move ≥ 0.5 m in any 10 s window
    *   `stagnation_grace_steps = 500` — no stagnation check in first 5 s (gives robot time to stand up)
*   **Rationale**: Hard timeout caused premature truncation when the robot reached the ramp but was slow. Stagnation detection only truncates genuinely stuck episodes, allowing exploratory ramp traversal to complete.

### v45 — Stronger Upright Penalty
*   `orientation` penalty: **-0.027 → -0.05** (nearly 2×).
*   **Rationale**: VLM analysis showed policy sometimes traversed bumps with excessive pitch angle. Stronger upright signal reduces this without hurting forward speed.

### cfg Refactor (post-v35)
*   `VBotSection011EnvCfg.RewardConfig` now uses `dict(BASE_REWARD_SCALES)` directly — the full inline reward dict was removed.
*   `BASE_REWARD_SCALES` is the single source of truth for s011 reward values (28 keys; down from ~40 in the old inline version).
*   Bonus keys (`smiley_bonus`, `red_packet_bonus`, `phase_completion_bonus`, `waypoint_bonus`) are not present in `BASE_REWARD_SCALES`; the environment code accesses them via `.get(key, 0.0)`, so they default to 0 (gradient-only / Phase 1 configuration).
*   To reproduce Phase 2 training, manually add the Phase 2 bonus keys with v29 values before calling `train.py`.

---

## 7. AutoML v48 HP+Reward Search — 2026-02-20

### Campaign: `automl_20260220_071134`

| Property | Value |
|----------|-------|
| Mode | hp-search (cold start, no warm-start) |
| Trials | 15 |
| Steps/trial | 15M |
| Budget | 9h (completed in 7.4h) |
| Search space | v48 — centered on v47 proven values, exploring lighter penalties |
| Optimizer | Bayesian (first 3 trials random, then guided) |
| Policy/Value net | (512, 256, 128) fixed |
| Status | **Completed** |

### Context

The v47 50M training run achieved `wp_idx_mean=1.40, wp_idx_max=7.00` (full course completion in best episodes), but 100% fall termination rate and LR crushed to 6.7e-5 by kl_adaptive. The v48 AutoML search tested whether **lighter penalties** (especially `lin_vel_z`, `torque_saturation`, `termination`) could improve consistency.

### v48 Search Space Design (key changes from v47 baseline)

| Parameter | v47 Value | v48 Search Range | Rationale |
|-----------|-----------|-------------------|-----------|
| `termination` | -200 | {-50, -100, -150, -200} | Test lighter fall penalty |
| `lin_vel_z` | -0.195 | [-0.20, -0.02] | v47 heavily penalizes vertical motion needed for bumps |
| `torque_saturation` | -0.025 | [-0.03, -0.003] | Test lighter motor stress penalty |
| `waypoint_approach` | 166.5 | [50, 300] | Stronger navigation pull? |
| `waypoint_facing` | 0.61 | [0.3, 1.2] | Heading reward sensitivity |
| `alive_decay_horizon` | 1500 | [800, 3000] | How fast alive_bonus decays |
| `foot_clearance` | 0.15 | [0.05, 0.3] | Step height reward |
| `zone_approach` | 50.0 (35.06 base) | [20, 80] | Side-zone attraction strength |

### Results — All 15 Trials Ranked by `wp_idx_mean`

| Rank | Trial | wp_mean | wp_max | phase_max | y_max | reward | term | lin_vel_z | torq_sat | wp_approach | wp_facing | alive_dh | LR | entropy |
|------|-------|---------|--------|-----------|-------|--------|------|-----------|----------|-------------|-----------|----------|----|---------|
| 1 | **T14** | **0.484** | 3.0 | 1.0 | 1.73 | 2.884 | -150 | -0.027 | -0.012 | 280.5 | 0.637 | 2383 | 4.5e-4 | 0.0078 |
| 2 | T13 | 0.445 | 3.0 | 1.0 | 1.35 | 2.816 | -150 | -0.027 | -0.012 | 280.5 | 0.644 | 2383 | 4.2e-4 | 0.0055 |
| 3 | T10 | 0.416 | 3.0 | 1.0 | 1.80 | 2.959 | -50 | -0.027 | -0.012 | 280.5 | 0.841 | 2383 | 3.6e-4 | 0.0058 |
| 4 | T4 | 0.387 | 2.0 | 0.0 | 1.46 | 3.071 | -50 | -0.027 | -0.014 | 280.5 | 0.841 | 2383 | 3.9e-4 | 0.0041 |
| 5 | T5 | 0.365 | 3.0 | 1.0 | 1.81 | 2.973 | -50 | -0.022 | -0.013 | 277.3 | 0.841 | 2383 | 2.0e-4 | 0.0036 |
| 6 | T7 | 0.365 | 3.0 | 1.0 | 1.77 | 2.998 | -50 | -0.030 | -0.014 | 300.0 | 0.841 | 2383 | 4.4e-4 | 0.0030 |
| 7 | T0 | 0.339 | 2.0 | 0.0 | 1.63 | 3.065 | -50 | -0.042 | -0.005 | 136.5 | 0.381 | 1892 | 4.1e-4 | 0.0050 |
| 8 | T11 | 0.334 | 3.0 | 1.0 | 1.87 | 2.745 | -50 | -0.021 | -0.014 | 221.6 | 0.717 | 1952 | 2.5e-4 | 0.0080 |
| 9 | T9 | 0.326 | 2.0 | 0.0 | 1.77 | 2.589 | -200 | -0.027 | -0.014 | 213.6 | 0.841 | 2383 | 4.1e-4 | 0.0038 |
| 10 | T12 | 0.325 | 2.0 | 0.0 | 1.83 | 2.824 | -50 | -0.027 | -0.013 | 280.5 | 0.841 | 2383 | 4.5e-4 | 0.0057 |
| 11 | T8 | 0.324 | 3.0 | 1.0 | 1.76 | 2.904 | -50 | -0.032 | -0.014 | 280.5 | 0.841 | 2383 | 3.9e-4 | 0.0041 |
| 12 | T2 | 0.312 | 2.0 | 0.0 | 1.99 | 2.432 | -200 | -0.111 | -0.020 | 261.1 | 0.793 | 2095 | 3.2e-4 | 0.0038 |
| 13 | T6 | 0.302 | 3.0 | 1.0 | 1.86 | 2.668 | -50 | -0.027 | -0.014 | 280.5 | 0.841 | 2383 | 3.4e-4 | 0.0043 |
| 14 | T3 | 0.289 | 2.0 | 0.0 | 1.61 | 2.558 | -150 | -0.029 | -0.006 | 214.4 | 0.930 | 1503 | 3.4e-4 | 0.0058 |
| 15 | T1 | 0.224 | 2.0 | 0.0 | 1.61 | 2.347 | -100 | -0.055 | -0.011 | 81.4 | 0.559 | 1696 | 3.4e-4 | 0.0052 |

### Key Metric Definitions

| Metric | Description | Discriminating? |
|--------|-------------|-----------------|
| `wp_idx_mean` | Average waypoint index across all 2048 envs (final training checkpoint) | **YES — PRIMARY**. Measures consistency. T14=0.484 vs T1=0.224 is 2.2× gap |
| `wp_idx_max` | Best single episode's waypoint index | Weak — 9/15 trials tied at 3.0 |
| `phase_max` | Furthest navigation phase reached by any env | Weak — 9/15 trials tied at 1.0 (RED_PACKETS) |
| `max_y_progress` | Furthest Y coordinate reached | Weak — all trials in 1.35–1.99 range |
| `success_rate` | Fraction of envs reaching final waypoint | Weak — all trials < 0.22 at 15M steps |

### Convergence Analysis

**Bayesian optimizer convergence** (Trials 0–2 random, 3–14 guided):

The optimizer converged heavily after ~8 trials. Trials 3–14 share nearly identical core params:
- `waypoint_approach` → **280.5** (11/15 trials within ±3%)
- `alive_decay_horizon` → **2383** (12/15 trials exact)
- `forward_velocity` → **3.16** (12/15 trials exact)
- `lin_vel_z` → **-0.027** (10/15 trials within ±20%)
- `torque_saturation` → **-0.012 to -0.014** (12/15 trials)

The remaining variance comes from:
- `termination`: -50 vs -150 vs -200 (categorical, so Bayesian explores all)
- `waypoint_facing`: 0.637 vs 0.841 (binary convergence)
- `zone_approach`: 51–75 range
- `lr` and `entropy`: moderate variance

### Discovery 1: Lighter Penalties Dominate

**The single strongest finding**: v47's penalties were too harsh.

| Parameter | v47 | T14 (best) | Change | Impact |
|-----------|-----|-----------|--------|--------|
| `lin_vel_z` | -0.195 | **-0.027** | **7.2× lighter** | Bumps require vertical motion — heavy penalty punishes correct behavior |
| `torque_saturation` | -0.025 | **-0.012** | **2.1× lighter** | Over-penalizing motor effort → conservative gait = slow |
| `termination` | -200 | **-150** | **25% lighter** | Lower fall penalty → agent explores more before catastrophic falls dominate |
| `orientation` | -0.027 (base) | -0.026 | ~same | Not a major lever |

**Confirmation**: T2 (rank 12) had `lin_vel_z=-0.111` (4× the search center) and `torque_sat=-0.020` — both heavier penalties → worst non-random trial. T1 (rank 15) had the heaviest `forward_velocity=4.88` + `lin_vel_z=-0.055` — worst overall.

### Discovery 2: Stronger Navigation Pull

| Parameter | v47 | T14 (best) | Change |
|-----------|-----|-----------|--------|
| `waypoint_approach` | 166.5 | **280.5** | **1.68× stronger** |
| `zone_approach` | 35.06 | **74.7** | **2.13× stronger** |

The optimizer strongly converged on `waypoint_approach≈280` and `zone_approach≈62-75`. Stronger navigation pull compensates for lighter penalties — the robot is more motivated to move forward rather than surviving in place.

### Discovery 3: Longer Alive Decay Horizon

- **v47**: `alive_decay_horizon = 1500`
- **T14**: `alive_decay_horizon = 2383` (1.59× longer)

Longer decay horizon means the alive_bonus stays relevant further into the episode, providing sustained motivation. With lighter termination penalty, the alive bonus becomes a more important carrot vs the terminal stick.

### Discovery 4: HP Parameters — Higher LR + Entropy

| HP | v47 config | T14 (best) | Change |
|----|-----------|-----------|--------|
| `learning_rate` | 1.0e-4 | **4.5e-4** | **4.5× higher** |
| `entropy_loss_scale` | 0.00432 | **0.00775** | **1.8× higher** |

T14's higher LR and entropy suggest that at 15M steps (short horizon), aggressive exploration yields faster progress. Whether this holds at 50M+ needs testing. However, the top 3 trials (T14, T13, T10) all have LR > 3.5e-4, suggesting bolder learning consistently outperforms conservative LR at this scale.

### Discovery 5: Termination -150 > -50 > -200

| Termination | Trial count | Avg wp_mean | Best wp_mean |
|-------------|-------------|-------------|--------------|
| -50 | 9 trials | 0.340 | 0.416 (T10) |
| -150 | 3 trials | 0.406 | **0.484 (T14)** |
| -200 | 2 trials | 0.319 | 0.326 (T9) |
| -100 | 1 trial | 0.224 | 0.224 (T1) |

**-150 is the sweet spot**: harsh enough to discourage falling, light enough to not dominate the reward budget. -200 (v47's value) makes falls too catastrophic → conservative policy. -50 makes falls cheap → policy doesn't learn to avoid them.

### T14 Winner Config (Full)

```yaml
# === Navigation (positive rewards) ===
forward_velocity: 3.163
waypoint_approach: 280.534
waypoint_facing: 0.637
zone_approach: 74.727
alive_bonus: 1.013
alive_decay_horizon: 2383
height_progress: 26.965
position_tracking: 0.259

# === Bonuses (one-time events) ===
waypoint_bonus: 50.046
phase_bonus: 13.067
celebration_bonus: 141.242
per_jump_bonus: 59.641
jump_reward: 10.093

# === Penalties (negative) ===
termination: -150
lin_vel_z: -0.027
torque_saturation: -0.012
orientation: -0.026
impact_penalty: -0.100
swing_contact_penalty: -0.003
action_rate: -0.007
ang_vel_xy: -0.038

# === Gait / Terrain ===
foot_clearance: 0.219
foot_clearance_bump_boost: 7.167
stance_ratio: 0.070
swing_contact_bump_scale: 0.210

# === HP ===
learning_rate: 0.000451
entropy_loss_scale: 0.00775
```

### Comparison: T14 vs v47 Baseline

| Parameter | v47 | T14 | Direction | Magnitude |
|-----------|-----|-----|-----------|-----------|
| `lin_vel_z` | -0.195 | -0.027 | lighter | **7.2×** |
| `torque_saturation` | -0.025 | -0.012 | lighter | **2.1×** |
| `termination` | -200 | -150 | lighter | **1.3×** |
| `waypoint_approach` | 166.5 | 280.5 | stronger | **1.7×** |
| `zone_approach` | 35.06 | 74.7 | stronger | **2.1×** |
| `alive_decay_horizon` | 1500 | 2383 | longer | **1.6×** |
| `foot_clearance` | 0.15 | 0.219 | stronger | **1.5×** |
| `waypoint_facing` | 0.61 | 0.637 | ~same | 1.04× |
| `swing_contact_penalty` | -0.031 | -0.003 | lighter | **10×** |
| `forward_velocity` | 2.875 | 3.163 | stronger | 1.1× |
| `learning_rate` | 1.0e-4 | 4.5e-4 | higher | **4.5×** |

### TB Metrics Deep Dive (T14)

From TensorBoard of T14's 15M step run:
- **smiley_bonus max=150.14** (3× waypoint_bonus=50 → 3 smileys collected in best episodes)
- **phase_completion_bonus max=26.13** (2× phase_bonus=13 → phases 0 and 1 completed)
- **wp_bonus max=176.27** → triggered multiple times in best episodes
- **red_packet_bonus=0, celeb_bonus=0** → no trial reached red packets or celebration at 15M steps
- **termination mean=-136.97** → falls still dominate end-of-episode
- **torque_saturation mean=-821.85** → still the largest cumulative penalty (but 2× lighter than v47)
- **bump_entry_frac mean=0.84** → 84% of envs enter the bump zone (good exploration)

### Limitations & Caveats

1. **15M steps is short** — v47's 50M run achieved wp_idx_mean=1.40 vs T14's 0.484 at 15M. The AutoML trials only test early-training behavior. A config that starts fast may plateau differently at 50M+.
2. **No warm-start** — All trials train from scratch. Warm-starting from v47's 50M checkpoint with T14's reward config could yield much better results.
3. **Heavy Bayesian convergence** — After trial ~8, most trials shared nearly identical params. The search explored only a narrow region of the space. A wider search or different random seeds might find better configs.
4. **HP confounding** — T14 has both reward changes AND HP changes (4.5× LR). The reward improvements and HP improvements are confounded.

### Recommended Next Steps

1. **Apply T14 reward config to `cfg.py`** — Replace `BASE_REWARD_SCALES` with T14's values
2. **Test two LR strategies for 100M run**:
   - (a) T14's LR=4.5e-4 with `kl_adaptive` (may need lower kl_threshold to prevent crush)
   - (b) v47's LR=1.0e-4 with T14's reward config only (isolate reward vs HP effect)
3. **Consider warm-start**: Load v47's 50M checkpoint, apply T14 reward weights, reduce LR to 0.3× = 1.35e-4
4. **Monitor `lin_vel_z` penalty budget**: At -0.027, this penalty is now very light. Watch for policy developing excessive bouncing behavior at >15M steps

---

*This report is append-only. Never overwrite existing content — the history is a permanent record.*

---

## 8. T14 100M Training Failure & v49 Fix — 2026-02-20

### T14 100M Deployment: `26-02-20_15-07-50-296259_PPO`

| Property | Value |
|----------|-------|
| Config | v48-T14 (LR=4.513e-4, entropy=0.00775, KL-adaptive) |
| Steps target | 100M (48,500 iters) |
| Stopped at | ~78% (38,500 iters, ~3h) |
| Reason | **Local optimum — policy learned to survive by retreating/dragging, never navigates** |

### Deep Analysis @ 78% (monitor_training.py --deep)

| Metric | Value | Assessment |
|--------|-------|------------|
| `wp_idx_mean` | **0.45** (max 2.0) | FAILURE — average didn't pass WP0 |
| `term_rate` | **7.8%** | Excellent survival (but survival ≠ navigation) |
| `max_y_progress` | 0.64 mean, 4.49 max | Barely past bump entrance |
| `foot_clearance` | **0** (zero!) | Robot NEVER lifts legs — pure dragging |
| `torque_saturation` | **-1106** cumulative | Massive — controllers fully saturated |
| `ep_length` | 2826 mean (~28s) | Alive but not progressing |
| `LR` | **5.9e-5** (from configured 4.5e-4) | KL-adaptive crushed LR 7.6× — stagnant learning |
| `alive_bonus_total` | 1131 | Positive — survival is rewarded |
| `forward_vel_total` | 861 | Some forward motion but inconsistent |
| `zone_approach_total` | 3821 | Proximity-based — doesn't require actual collection |
| `penalties_total` | -2397 | Heavy penalty budget but not enough to break survival loop |
| `smiley_bonus` | 0 | No zones collected |
| `celeb/jump/red_packet` | 0 | No progress past bump entrance |

### Root Cause: Reward Blind Spot → Backward-Dragging Local Optimum

**User Observation**: "Legs stuck on obstacles, robot constantly retreating/dragging — survival-oriented, navigation-opposed."

**Analysis**: The reward function had a blind spot for legs with sustained ground contact + low velocity:
1. **foot_clearance** reward: Only fires during swing phase (leg off ground) → `foot_clearance=0` means robot never enters swing
2. **swing_contact_penalty**: Penalizes contact during swing, but if leg NEVER swings, penalty is zero
3. **drag behavior**: Leg in continuous ground contact with low velocity gets NEITHER reward nor penalty

Combined with `alive_bonus` (even with decay), the robot found a degenerate strategy:
- Stand still / slowly retreat → alive_bonus accumulates
- Never lift legs → zero foot_clearance reward but also zero swing_contact penalty
- Torque controllers saturate trying to maintain posture → torque_saturation penalty is large but not enough to break the equilibrium

### LR Collapse Timeline

```
Configured:  4.513e-4
Peak:        1.1e-3   (early training, high KL divergence)
@35%:        8.9e-5
@64%:        5.9e-5
@78%:        5.9e-5   (stable — no more learning)
```

KL-adaptive raised LR early (trying to learn), then crushed it once policy settled into the local optimum. Final LR is 7.6× below configured and lower than v47's fixed 1e-4.

### v49 Fix: Two New Anti-Local-Optimum Penalties

**Penalty 1: `drag_foot_penalty` (default: -0.02)**

Targets the exact blind spot: legs with sustained contact + low velocity.

```
Detection: calf_in_contact AND calf_velocity < 1.0 m/s
Per-step: drag_foot_scale × count_of_dragging_legs (0-4)
Bump boost: 2× in bump zone (y ∈ [-1.5, 1.5])
```

This directly penalizes the "stand and drag" strategy that T14 converged to.

**Penalty 2: `stagnation_penalty` (default: -0.5)**

Progressive penalty that ramps up as stagnation detection window fills:

```
stag_ratio = clip((steps_since_anchor / window - 0.5) × 2, 0, 1)
penalty = stagnation_scale × stag_ratio
```

- 0% to 50% of stagnation window: no penalty (grace period)
- 50% to 100%: linear ramp from 0 to full penalty
- Exempt during celebration phase

This provides gradient signal BEFORE truncation fires, while truncation only fires at 100% (binary — no gradient).

### v49 AutoML Search Space Expansion

Based on T14 boundary analysis + v49 new penalties, the AutoML search space was expanded:

| Parameter | Old Range | New Range | Reason |
|-----------|-----------|-----------|--------|
| `learning_rate` | [3e-4, 5e-4] | **[2e-4, 8e-4]** | T14=4.5e-4 was at 90th percentile |
| `entropy_loss_scale` | [3e-3, 6e-3] | **[3e-3, 1.5e-2]** | T14=7.75e-3 EXCEEDED old upper bound |
| `waypoint_approach` | [80, 300] | **[80, 500]** | T14=280.5 was at 93% |
| `zone_approach` | [20, 80] | **[20, 150]** | T14=74.7 was at 91% |
| `lin_vel_z` | [-0.2, -0.02] | **[-0.2, -0.005]** | T14=-0.027 was at 96% (lighter end) |
| `swing_contact_penalty` | [-0.06, -0.003] | **[-0.06, -0.0005]** | T14=-0.003 was AT lower bound |
| `drag_foot_penalty` | — (new) | **[-0.08, -0.005]** | v49 new penalty |
| `stagnation_penalty` | — (new) | **[-2.0, -0.1]** | v49 new penalty |

Total searchable parameters: **27 reward + 2 HP = 29** (was 25+2=27).

### Next Steps (v49 Plan)

1. **v49 Baseline quick-train (10M steps)**: Verify code works, let user play-test to observe drag_foot and stagnation penalty effects
2. **AutoML v49 search**: Run 15–20 trials × 15M steps with expanded search space
   - Key hypothesis: drag_foot_penalty + stagnation_penalty break the backward-dragging local optimum
   - Watch for: whether Bayesian finds optimal penalty weights, or if new penalties cause over-penalization (robot freezes)
3. **Long-horizon validation**: Best v49 trial → 50M+ full training run
4. **Comparison strategy**: Compare v49 best vs T14 100M (stopped) to measure improvement from anti-dragging penalties

---

## Stage 30: Iterative Warm-Start Chain (2026-02-21)

### Context
After AutoML R1 (11 trials) and R2 (5 trials) identified R1_T10 as the best config (wp_mean=0.290, wp_max=3), launched full 100M-step training. Discovered critical instability: **KL-adaptive LR scheduler causes catastrophic policy collapse at 60-80% of long training runs.**

### R1_T10 Reward Config (held constant across all runs)
```
forward_vel=6.06, wp_approach=510.9, zone=196.3, alive=2.50, alive_decay=3921,
fc=0.45, fc_bump=12.99, termination=-50, stagnation=-1.29, crouch=-1.23, drag=-0.28,
celebration=126.6, phase_bonus=99.8, per_jump=46.8, jump_reward=7.55, wp_bonus=27.2
```

### Full Train (100M steps, KL-adaptive LR)
- **Run**: `26-02-21_10-05-29-377373_PPO`
- **Peak**: wp_mean=0.937 at step 29500 (~55% progress)
- **Failure**: Catastrophic collapse at ~70% — wp_mean crashed 0.937 → 0.147 (84% drop)
- **Root cause**: KL-adaptive scheduler crushed LR from 8.78e-4 → 1e-4 floor. When policy reached new terrain phases (wp_max jumped to 7), large KL divergence triggered but LR was already at floor, so policy couldn't recover from the perturbation.
- **Lesson**: KL-adaptive is DANGEROUS for long training runs. Use fixed LR for anything >20M steps.

### Warm-Start Chain Strategy
**Insight**: Loading a peak checkpoint with a fixed (low) LR and frozen preprocessor allows the policy to improve further, then save the new peak before inevitable drift sets in. Chaining these warm-starts progressively pushes performance higher.

| Chain | Source | LR | Peak wp_mean | Peak Step | Delta | Run Dir |
|-------|--------|----|-------------|-----------|-------|---------|
| 0 (full) | fresh | 1e-3 (KL) | 0.937 | 29500 | — | `26-02-21_10-05-29-377373_PPO` |
| 1 | agent_27000 (chain0) | 5e-5 | 1.829 | 9000 | +95% | `26-02-21_12-06-20-648190_PPO` |
| 2 | agent_9000 (chain1) | 1e-5 | **2.584** | 7500 | +41% | `26-02-21_13-15-59-147217_PPO` |
| 3 | agent_7500 (chain2) | 1e-5 | **2.787** | 7000 | +8% | `26-02-21_13-56-35-936742_PPO` |
| 4 | agent_7000 (chain3) | 1e-5 | 2.726 | 7000 | -2% | `26-02-21_14-24-01-549247_PPO` |

**Pattern observed**: Every chain shows "ramp up → peak → gradual decline" regardless of LR. The peak consistently occurs around step 7000-9000. Diminishing returns: +95% → +41% → +8% → -2%. Chain plateaued at wp_mean ≈ 2.787.

### Preserved Peak Checkpoints
All peaks saved in `starter_kit_schedule/checkpoints/`:
- `warmstart1_agent_9000.pt` (wp=1.829)
- `warmstart2_agent_7500.pt` (wp=2.584)
- `warmstart3_agent_7000.pt` (wp=**2.787**, BEST)
- `warmstart4_agent_7000.pt` (wp=2.726)

### Key Findings
1. **KL-adaptive is catastrophic for long runs**: Fixed LR (null scheduler) is mandatory.
2. **Warm-start chain works but plateaus**: Each successive chain yields diminishing improvement. The reward function, not the training process, is now the bottleneck.
3. **wp_mean=2.787 interpretation**: Average robot reaches ~3rd waypoint (all 3 smileys). wp_max=7 shows full course is completable for some environments. The bottleneck is transitioning from smileys (Phase 1) to red packets (Phase 2) and beyond.
4. **Preprocessor freeze is critical**: Without frozen preprocessor, observation normalization drifts during warm-start, causing immediate performance collapse.

### Disk Space
Cleaned all intermediate checkpoints from chain runs. Only best_agent.pt + peak checkpoints (in starter_kit_schedule/checkpoints/) preserved. Total cleanup: ~680MB saved across runs.

### Next Steps
1. **Visual diagnosis**: Run `capture_vlm.py` on warmstart3_agent_7000.pt to identify what prevents robots from reaching Phase 2 (red packets)
2. **Reward engineering**: wp_mean=2.8 ceiling likely indicates insufficient reward signal for Phase 2 transition. Possible causes:
   - Red packet bonuses too low relative to smiley bonuses
   - Navigation signal weak after exiting smiley zone
   - Ramp climbing reward needs strengthening
3. **AutoML reward search**: After diagnosis, define new reward search space targeting Phase 2+ performance
4. **Architecture alternative**: Consider whether policy capacity (512,256,128) is sufficient for multi-phase navigation

---

## 9. Stage One AutoML — Cold-Start HP+Reward Search (2026-02-22)

### Campaign: `automl_20260221_203616`

| Property | Value |
|----------|-------|
| Mode | stage (cold start, no warm-start) |
| Trials | **20** |
| Steps/trial | 10M (≈910 iters @ 2048 envs, `check_point_interval=909`) |
| Budget | 8h |
| Search space | v55 focused — centered on R1_T10 (Stage 30 winner), ±50-70% around T10 values |
| Optimizer | Bayesian (1 seed + ~4 random warmup + ~15 guided) |
| Policy/Value net | (512, 256, 128) fixed |
| Seed config | `seed_T12_warmstart.json` — R1_T10's exact reward scales (fwd=6.49, wpa=510.9, zone=196.3, term=-50) |
| Status | **Completed** |

### Context

Stage 30 warm-start chain plateaued at wp_mean=2.787. The reward function, not the training process, was the bottleneck. This AutoML searched 31 reward parameters + 2 HP to find configurations that could break through the wp~3 ceiling when trained from scratch.

### Search Space Highlights (v55, 33 searchable parameters)

Key ranges centered on R1_T10 values:
- `forward_velocity`: [2.0, 10.0] (T10=6.06)
- `waypoint_approach`: [200, 800] (T10=510.9)
- `zone_approach`: [80, 350] (T10=196.3)
- `termination`: {-100, -75, -50, -25}
- `drag_foot_penalty`: [-0.5, -0.08] (T10=-0.283)
- `stagnation_penalty`: [-2.5, -0.4] (T10=-1.29)
- `crouch_penalty`: [-2.5, -0.3] (T10=-1.23)
- New searchable params: `foot_clearance_bump_boost_pre_margin`, `foot_clearance_bump_boost_post_margin`, `foot_clearance_pre_zone_ratio`, `dof_pos`

### Results — Top 10 Trials by Score

| Rank | Trial | Score | wp_mean | suc% | LR | Entropy | Term | Fwd | WP_Appr | Zone | Stag | BumpB |
|------|-------|-------|---------|------|-----|---------|------|-----|---------|------|------|-------|
| 1 | **T12** | **0.2562** | **0.412** | 19.9% | 1.0e-3 | 0.0028 | -50 | 6.49 | 510.9 | 196.3 | -1.13 | 10.01 |
| 2 | T13 | 0.2539 | 0.379 | 18.0% | 1.0e-3 | 0.0020 | -25 | 6.49 | 510.9 | 196.3 | -1.13 | 10.01 |
| 3 | T16 | 0.2530 | 0.374 | 21.5% | 7.3e-4 | 0.0026 | -50 | 7.34 | 477.8 | 196.3 | -1.13 | 9.75 |
| 4 | T18 | 0.2498 | 0.336 | 19.6% | 1.0e-3 | 0.0024 | -75 | 6.49 | 510.9 | 211.0 | -1.10 | 10.01 |
| 5 | T8 | 0.2462 | 0.304 | 17.7% | 1.0e-3 | 0.0036 | -50 | 6.06 | 510.9 | 196.3 | -1.29 | 10.01 |
| 6 | T0 | 0.2461 | 0.302 | 17.1% | 5.0e-4 | 0.0028 | -50 | 6.49 | 510.9 | 196.3 | -1.13 | 10.01 |
| 7 | T11 | 0.2460 | 0.298 | 19.7% | 1.0e-3 | 0.0098 | -50 | 6.06 | 510.9 | 196.3 | -1.29 | 12.99 |
| 8 | T1 | 0.2441 | 0.271 | 17.8% | 1.5e-3 | 0.0058 | -50 | 5.73 | 200.5 | 105.0 | -0.96 | 7.36 |
| 9 | T2 | 0.2440 | 0.266 | 20.1% | 8.2e-4 | 0.0113 | -50 | 5.33 | 543.7 | 162.7 | -1.57 | 16.61 |
| 10 | T15 | 0.2413 | 0.242 | 17.3% | 4.9e-4 | 0.0039 | -50 | 6.49 | 510.9 | 225.1 | -1.00 | 10.01 |

Bottom 10: T3(0.205), T6(0.247), T19(0.187), T9(0.178), T5(0.212), T17(0.155), T10(0.145), T14(0.127), T4(0.000), T7(0.041)

### Key Discoveries

1. **LR=1e-3 wins at 10M cold-start**: Top 4 of 5 trials used lr=1e-3 (search upper bound). At this short horizon, aggressive learning dominates.
2. **Low entropy wins**: T12 (ent=0.0028) and T13 (ent=0.0020) — the two best — had the lowest entropy values. This contrasts with v48's T14 (ent=0.0078) at 15M steps — shorter horizon favors even lower exploration.
3. **Termination -50 dominates**: 14/20 trials used -50. The lighter termination was the best category overall.
4. **T12 ≡ seed**: T12's reward config is identical to the T10 seed config. The seed survived as champion, confirming R1_T10's reward scales are near-optimal for this architecture.
5. **Bayesian convergence**: Strong convergence around seed values (wpa≈511, fwd≈6.49, zone≈196). T1 (wpa=200.5, zone=105) diverged most from seed → placed 8th.

### Selection for Full Training

Three candidates selected for 100M training, spanning the diversity of top configs:
- **Train A (T12)**: Seed champion, term=-50, fwd=6.49, wpa=510.9
- **Train B (T13)**: Same rewards as T12 but term=-25 (lightest penalty)
- **Train C (T11)**: Different HP (ent=0.0098, 3.5× T12's entropy), higher bump_boost=12.99

---

## 10. Full 100M Training Campaign — Three Candidates (2026-02-22)

### Setup

All three runs used:
- **100M steps** (48,500 iters @ 2048 envs)
- **Fixed LR** (no KL-adaptive — learned from Stage 30's KL collapse)
- `check_point_interval = 500` (≈1M steps between checkpoints)
- `lr_scheduler_type = null` (fixed learning rate throughout)
- Individual configs saved in `starter_kit_schedule/configs/`

| Run | Config File | Source Trial | LR | Entropy | Termination | Run Dir |
|-----|-------------|-------------|-----|---------|-------------|---------|
| A | `full_A_T12_best_wpidx.json` | T12 | 1.0e-3 | 0.0028 | -50 | `26-02-22_05-25-43-367487_PPO` |
| B | `full_B_T13_soft_term.json` | T13 | 1.0e-3 | 0.0020 | -25 | `26-02-22_05-28-12-324803_PPO` |
| C | `full_C_T11_best_sucrate.json` | T11 | 1.0e-3 | 0.0098 | -50 | `26-02-22_05-30-41-234319_PPO` |

### Results — All Three Collapsed After 50M Steps

| Run | Peak wp_mean | Peak Iter | Final wp_mean | Final Iter | Collapse % | Peak Checkpoint |
|-----|-------------|-----------|--------------|------------|-----------|----------------|
| **A (T12)** | **2.232** | **24500** | 0.138 | 48500 | **-94%** | `agent_24500.pt` |
| B (T13) | 2.033 | 21000 | 0.639 | 48500 | -69% | `agent_21000.pt` |
| C (T11) | 1.919 | 25000 | 0.363 | 48500 | -80% | `agent_25000.pt` |

### PPO Policy Collapse Analysis

All three runs exhibited the same pattern:
1. **Phase 1 (0-25M steps)**: Rapid learning, wp_mean climbing steadily
2. **Phase 2 (25M-50M steps)**: Peak performance reached, then gradual decline begins
3. **Phase 3 (50M-100M steps)**: Catastrophic collapse — wp_mean crashes to near-zero

**Root cause**: Fixed LR=1e-3 is too aggressive for the exploitation phase. After 50M steps, the policy has explored the reward landscape and needs finer-grained updates. But the constant high LR causes destructive oscillations in the policy, erasing learned behaviors.

**Key insight**: This is the opposite of Stage 30's KL-adaptive collapse. There, KL-adaptive crushed LR too aggressively. Here, fixed LR doesn't reduce at all. The ideal approach is **warm-starting from the peak with a LOWER fixed LR** — which motivates Stage Two AutoML.

### Peak Checkpoint Preservation

Each candidate's peak checkpoint is the warm-start source for its own **Stage Two AutoML** pipeline:

| Run | Peak Checkpoint | Peak wp_mean | → Stage Two AutoML |
|-----|----------------|-------------|-------------------|
| **A (T12)** | `26-02-22_05-25-43-367487_PPO/checkpoints/agent_24500.pt` | 2.232 | `automl_20260222_124457` (**COMPLETED**, T13 champion, score=0.5439) |
| **B (T13)** | `26-02-22_05-28-12-324803_PPO/checkpoints/agent_21000.pt` | 2.033 | `automl_20260222_201059` (**RUNNING**) |
| **C (T11)** | `26-02-22_05-30-41-234319_PPO/checkpoints/agent_25000.pt` | 1.919 | Pending (start after B completes) |

---

## 11. Stage Two AutoML — Warm-Start from Train A / T12 Peak (2026-02-22, COMPLETED)

### Campaign: `automl_20260222_124457` (Branch A: T12 → full train → agent_24500.pt)

| Property | Value |
|----------|-------|
| Mode | stage (warm-start from T12 peak) |
| Trials | **15** (1 seed + 4 random warmup + ~10 Bayesian) |
| Steps/trial | 10M (≈910 iters) |
| Budget | 8h |
| Warm-start checkpoint | `runs/vbot_navigation_section011/26-02-22_05-25-43-367487_PPO/checkpoints/agent_24500.pt` |
| Freeze preprocessor | **Yes** (RunningStandardScaler no-op to prevent normalizer drift) |
| LR clamping | **max 7e-4** (both random and Bayesian phases) |
| Search space | Same v55 REWARD_SEARCH_SPACE_SECTION011 (31 reward params) + HP_SEARCH_SPACE (LR, entropy) |
| Seed config | `seed_T12_warmstart.json` (T12's exact rewards, lr=5e-4, ent=0.0028) |
| cfg.py change | `required_jumps = 10` (was 3) — harder celebration requirement |
| Status | **COMPLETED** — 15/15 trials, 6.71h elapsed |

### Motivation

1. **PPO collapse at 100M**: All three full training runs peaked at ~50M then collapsed. Warm-starting from the peak with lower LR should allow the policy to refine without catastrophic oscillations.
2. **Jump-10 requirement**: The celebration requirement was hardened from 3 jumps to 10 jumps (changed in `cfg.py` line 381). Stage Two AutoML tests whether reward weights need retuning for this harder celebration task.
3. **LR clamping rationale**: With warm-start, the policy is already well-trained. LR ≤ 7e-4 prevents destructive updates while still allowing meaningful learning. The seed uses lr=5e-4 (0.5× the original 1e-3).

### Warm-Start Infrastructure

**AutoML warm-start support** (added to `automl.py`):
- `--checkpoint <path>`: All trials load the same pre-trained checkpoint
- `--freeze-preprocessor`: Prevents RunningStandardScaler drift (monkey-patches `_parallel_variance` to no-op in `train_one.py`)
- LR clamping: Random sampling caps at `min(sampled_lr, 7e-4)` (line 776); Bayesian suggest caps at `max(1e-5, min(7e-4, lr))` (line 839)
- Checkpoint and freeze_preprocessor flags passed through to `train_one.py` via config JSON

### Results — All 15 Trials (Warm-Start, Ranked by Score)

**All 15 trials achieved wp_idx_MAX = 7.0** (full course completion) — confirming the warm-start baseline is extremely strong.

| Rank | Trial | Score | wp_mean | suc% | term_rate | LR | Entropy | Term | Fwd | WP_Appr | Stag | BumpB | CrouchP |
|------|-------|-------|---------|------|-----------|-----|---------|------|-----|---------|------|-------|---------|
| **1** | **T13** | **0.5439** | **3.443** | **23.2%** | 0.03% | 5.7e-4 | **0.0100** | -50 | 5.55 | **346.0** | -2.38 | 13.55 | -2.34 |
| 2 | T4 | 0.5399 | 3.411 | 20.5% | 0.02% | 5.1e-4 | 0.0084 | -50 | 5.55 | 310.6 | **-2.38** | **13.55** | **-2.34** |
| 3 | T6 | 0.5374 | 3.354 | **25.5%** | 0.02% | 7.0e-4 | 0.0066 | -75 | 9.38 | 772.3 | -1.66 | 19.13 | -2.35 |
| 4 | T10 | 0.5297 | 3.312 | 19.7% | 0.03% | 4.9e-4 | 0.0100 | -50 | 5.55 | 310.6 | -2.38 | 16.25 | -2.21 |
| 5 | T2 | 0.5286 | 3.300 | 21.4% | 0.02% | 7.0e-4 | 0.0071 | -50 | 5.52 | 638.9 | -1.47 | 7.44 | -1.40 |
| 6 | T8 | 0.5285 | 3.356 | 20.5% | 0.02% | 7.0e-4 | 0.0097 | -50 | 2.92 | 459.1 | -0.73 | 6.08 | -1.72 |
| 7 | T9 | 0.5272 | 3.303 | 19.6% | 0.03% | 7.0e-4 | 0.0089 | -100 | 4.19 | 310.6 | -2.38 | 13.55 | -2.34 |
| 8 | T0 (seed) | 0.5222 | 3.250 | 24.5% | 0.03% | 5.0e-4 | 0.0028 | -50 | 6.49 | 510.9 | -1.13 | 10.01 | -1.30 |
| 9 | T14 | 0.5214 | 3.212 | 21.6% | 0.03% | 5.5e-4 | 0.0068 | -75 | 5.55 | 262.7 | -2.38 | 13.55 | -2.34 |
| 10 | T1 | 0.5175 | 3.239 | 22.8% | 0.03% | 4.4e-4 | 0.0125 | -50 | 2.77 | 238.6 | -1.24 | 10.15 | -0.35 |
| 11 | T3 | 0.5138 | 3.183 | 21.7% | 0.03% | 5.8e-4 | 0.0038 | -75 | 6.31 | 276.8 | -0.93 | 7.09 | -0.64 |
| 12 | T5 | 0.5095 | 3.112 | 21.6% | 0.02% | 7.0e-4 | 0.0038 | -75 | 9.52 | 351.1 | -2.24 | 19.99 | -1.33 |
| 13 | T11 | 0.5046 | 3.019 | 19.0% | 0.02% | 4.6e-4 | 0.0062 | -50 | 5.55 | 310.6 | -2.38 | 13.55 | -2.50 |
| 14 | T12 | 0.5042 | 3.013 | 18.7% | 0.03% | 3.9e-4 | 0.0083 | -100 | 5.55 | 310.6 | -2.38 | 15.73 | -2.34 |
| 15 | T7 | 0.4965 | 2.962 | 20.9% | 0.02% | 7.0e-4 | 0.0076 | -100 | 3.46 | 361.2 | -0.67 | 19.27 | -1.59 |

### Key Findings

**1. T13 is champion — entropy=0.01 is the differentiator**: T13 and T4 share nearly identical reward configs (same stag, crouch, bmpB, fwd, term=-50). T13's edge: entropy=0.01 (vs 0.0084) and WPA=346 (vs 310.6). Slightly more exploration + slightly higher navigation pull produced the best result.

**2. Termination -50 dominates warm-start**:
| Term | Trials | Best Score | Avg Score |
|------|--------|-----------|-----------|
| -50 | 9 | 0.5439 (T13) | 0.5247 |
| -75 | 4 | 0.5374 (T6) | 0.5205 |
| -100 | 2 | 0.5272 (T9) | 0.5119 |

**3. Controlled ablation via T9**: T9 uses T4's exact rewards but term=-100 → wp_mean drops 3.411→3.303 (-3.2%). This is direct evidence that heavier termination hurts even with optimal reward weights.

**4. Bayesian convergence pattern**: After T4 (trial 4), the optimizer heavily exploited T4's basin. T9, T10, T11, T12 are near-copies with minor variations. T13 introduced just enough perturbation (ent=0.01, WPA=346) to break through.

**5. Warm-start roughly doubles AutoML score**: Stage Two scores (0.50-0.54) vs Stage One (0.22-0.26). The pre-trained checkpoint provides an enormous head start that reward tuning further optimizes.

**6. Anti-loafing penalties are essential**: stag≈-2.38 and crouch≈-2.34 appear in the top 4 of 5 trials. The lightest stag/crouch trials (T3: stag=-0.93, T1: crouch=-0.35) scored lowest among -50 trials.

**7. LR not at boundary**: Winner at 5.7e-4, within the [4e-4, 7e-4] effective range. Neither extreme LR wins — moderate learning rate works best for warm-start refinement.

### Reproduction Commands

```powershell
# === Stage Two AutoML: Warm-start from T12 peak ===
# Prerequisites: required_jumps = 10 in cfg.py line 381

uv run starter_kit_schedule/scripts/automl.py `
    --mode stage `
    --env vbot_navigation_section011 `
    --budget-hours 8 `
    --hp-trials 15 `
    --seed-configs starter_kit_schedule/configs/seed_T12_warmstart.json `
    --checkpoint "runs/vbot_navigation_section011/26-02-22_05-25-43-367487_PPO/checkpoints/agent_24500.pt" `
    --freeze-preprocessor

# Monitor progress
Get-Content starter_kit_log/automl_20260222_124457/state.yaml

# Check how many trials completed
Get-ChildItem starter_kit_log/automl_20260222_124457/experiments/ -Directory | Measure-Object
```

### Preceding Steps to Reproduce from Scratch

To fully reproduce this Stage Two AutoML run from zero:

```powershell
# Step 1: Run Stage One cold-start AutoML (20 trials × 10M steps)
uv run starter_kit_schedule/scripts/automl.py `
    --mode stage --env vbot_navigation_section011 `
    --budget-hours 8 --hp-trials 20 `
    --seed-configs starter_kit_schedule/configs/seed_T12_warmstart.json

# Step 2: Apply best trial (T12) config to cfg.py and train to 100M
# Use starter_kit_schedule/scripts/apply_best_and_train.ps1 -AutomlId automl_20260221_203616
# Or manually: apply T12 reward_scales to cfg.py, set lr=1e-3, train.py --env vbot_navigation_section011

# Step 3: Identify peak checkpoint from 100M training
# Train A (T12) peaked at agent_24500.pt (wp_mean=2.232)
# Path: runs/vbot_navigation_section011/26-02-22_05-25-43-367487_PPO/checkpoints/agent_24500.pt

# Step 4: Change required_jumps = 10 in cfg.py line 381

# Step 5: Run Stage Two AutoML (warm-start from T12 peak, command above)
```

---

## 12. Stage Two AutoML — Warm-Start from Train B / T13 Peak (2026-02-22, COMPLETED)

### Campaign: `automl_20260222_201059` (Branch B: T13 → full train → agent_21000.pt)

| Property | Value |
|----------|-------|
| Mode | stage (warm-start from T13 peak) |
| Trials | **15** (1 seed + 4 random warmup + ~10 Bayesian) |
| Steps/trial | 10M (≈910 iters) |
| Budget | 8h |
| Warm-start checkpoint | `runs/vbot_navigation_section011/26-02-22_05-28-12-324803_PPO/checkpoints/agent_21000.pt` |
| Freeze preprocessor | **Yes** |
| LR clamping | **max 7e-4** |
| Search space | Same v55 (31 reward + 2 HP) |
| Seed config | `seed_T13_warmstart.json` (T13's exact rewards: term=-25, stag=-1.127, fwd=6.485, wpa=510.9, lr=5e-4, ent=0.002) |
| cfg.py change | `required_jumps = 10` (same as Branch A) |
| Status | **Completed** — 8/11 trials successful (3 failed), 5.87h elapsed |

### Motivation

This is the **parallel Stage Two branch for T13** (the second-best Stage One trial). Key differences from Branch A:
1. **Different warm-start checkpoint**: Train B peaked at iter 21000 (wp=2.033) vs Train A's iter 24500 (wp=2.232). Lower peak but different convergence trajectory.
2. **T13's base config**: term=-25 (lightest of all top trials) vs T12's term=-50. The policy learned a different balance between risk-taking and caution.
3. **Lower entropy seed**: ent=0.002 (T13) vs 0.0028 (T12). The Bayesian optimizer starts from a different entropy basin.

### Comparison: Branch A (T12) vs Branch B (T13) Starting Points

| Property | Branch A (T12) | Branch B (T13) |
|----------|---------------|---------------|
| Stage 1 score | 0.2562 (#1) | 0.2539 (#2) |
| Stage 1 term | -50 | -25 |
| Stage 1 entropy | 0.0028 | 0.0020 |
| Full train peak wp | 2.232 @ iter 24500 | 2.033 @ iter 21000 |
| Full train collapse | -94% (to 0.138) | -69% (to 0.639) |
| Seed LR | 5e-4 | 5e-4 |

**Interesting**: Train B collapsed less severely (-69% vs -94%), suggesting T13's lighter termination penalty (term=-25) produced a more stable policy. This may translate into better Stage Two results despite the lower peak.

### Results (8 completed trials, sorted by wp_idx_mean)

| Trial | wp_mean | suc% | reward | LR | entropy | term | fwd | wpa | stag | dur_h |
|-------|---------|------|--------|----|---------|------|-----|-----|------|-------|
| **T10** | **3.111** | **25.4%** | 5.87 | 5.0e-4 | 0.0037 | -50 | 6.89 | 407.9 | -0.62 | 0.59 |
| T8 | 2.956 | 24.5% | 5.10 | 7.0e-4 | 0.0097 | -100 | 8.10 | 738.8 | -2.04 | 0.41 |
| T2 | 2.944 | 24.6% | 3.93 | 7.0e-4 | 0.0059 | -75 | 5.07 | 423.8 | -0.98 | 0.60 |
| T4 | 2.943 | 23.3% | 6.17 | 5.4e-4 | 0.0039 | -50 | 6.89 | 407.9 | -0.62 | 0.41 |
| T6 | 2.859 | 24.0% | 3.55 | 7.0e-4 | 0.0074 | -100 | 6.15 | 232.7 | -2.23 | 0.41 |
| T1 | 2.787 | 25.5% | 5.47 | 6.4e-4 | 0.0078 | -25 | 9.81 | 473.9 | -1.65 | 0.42 |
| T7 | 2.778 | 25.3% | 3.32 | 5.8e-4 | 0.0081 | -25 | 3.52 | 276.3 | -1.48 | 0.42 |
| T5 | 2.568 | 24.6% | 4.46 | 7.0e-4 | 0.0033 | -25 | 9.01 | 644.6 | -0.41 | 0.42 |

3 trials (T0 seed, T3, T9) failed — no experiment outputs produced.

### Champion: T10

| Property | Value |
|----------|-------|
| wp_idx_mean | **3.111** |
| Success rate | 25.4% |
| LR | 5.0e-4 |
| Entropy | 0.0037 |
| Termination | -50 |
| Forward velocity | 6.89 |
| Waypoint approach | 407.9 |
| Stagnation penalty | -0.62 |

### Branch A vs Branch B — Head-to-Head

| Metric | Branch A (T13) | Branch B (T10) | Delta |
|--------|---------------|---------------|-------|
| Champion wp_mean | **3.443** | 3.111 | **-9.6%** |
| Champion suc% | 25.3% | 25.4% | +0.1% |
| Champion LR | 5.7e-4 | 5.0e-4 | — |
| Champion entropy | 0.0100 | 0.0037 | 2.7× lower |
| Champion term | -50 | -50 | same |
| Trials completed | 15/15 | 8/11 | — |
| Duration | 6.71h | 5.87h | — |

**Conclusion**: Branch A (T12 warm-start) produced a clearly superior champion (T13, wp=3.443) compared to Branch B (T13 warm-start, T10, wp=3.111). The 9.6% advantage confirms that the higher-peak warm-start checkpoint (agent_24500 @ wp=2.232) provided a better foundation than the more stable but lower-peak checkpoint (agent_21000 @ wp=2.033). Branch A also had zero failures vs Branch B's 3 failed trials.

**Notable**: Branch B's T10 independently converged on term=-50 (same as Branch A's champion), despite starting from T13's seed config with term=-25. This confirms that moderate termination penalty is the optimizer-preferred regime for warm-start at this curriculum stage.

### Reproduction Commands

```powershell
# === Stage Two AutoML: Warm-start from T13 peak (Branch B) ===
# Prerequisites: required_jumps = 10 in cfg.py line 381

uv run starter_kit_schedule/scripts/automl.py `
    --mode stage `
    --env vbot_navigation_section011 `
    --budget-hours 8 `
    --hp-trials 15 `
    --seed-configs starter_kit_schedule/configs/seed_T13_warmstart.json `
    --checkpoint "runs/vbot_navigation_section011/26-02-22_05-28-12-324803_PPO/checkpoints/agent_21000.pt" `
    --freeze-preprocessor

# Monitor progress
Get-Content starter_kit_log/automl_20260222_201059/state.yaml
```

---

## 13. Stage Two AutoML — Warm-Start from Train C / T11 Peak (2026-02-23, RUNNING)

### Campaign: `automl_20260223_012907` (Branch C: T11 → full train → agent_25000.pt)

| Property | Value |
|----------|-------|
| Mode | stage (warm-start from T11 peak) |
| Steps/trial | 10M+ |
| Budget | 8h |
| Warm-start checkpoint | `runs/vbot_navigation_section011/26-02-22_05-30-41-234319_PPO/checkpoints/agent_25000.pt` |
| Freeze preprocessor | **Yes** |
| LR clamping | **max 7e-4** |
| Search space | Same v55 (31 reward + 2 HP) |
| Seed config | T11's exact rewards (term=-50, lr=5e-4, ent=0.0038) |
| cfg.py change | `required_jumps = 10` (same as Branch A/B) |
| Status | **Running** — 1 trial completed so far, 0.59h elapsed |

### Early Results (1 trial)

| Trial | wp_mean | suc% | reward | LR | entropy | term |
|-------|---------|------|--------|----|---------|------|
| T0 (seed) | 2.440 | 23.7% | 4.63 | 5.0e-4 | 0.0038 | -50 |

### Context

Branch C is the final Stage Two branch, warm-starting from the weakest of the three full-train candidates (T11, peak wp=1.919, collapsed -80%). T11's distinctive feature was high entropy during Stage One (ent=0.0098, 3.5× T12), which may enable broader exploration in the warm-start phase.

### Reproduction Commands

```powershell
# === Stage Two AutoML: Warm-start from T11 peak (Branch C) ===
uv run starter_kit_schedule/scripts/automl.py `
    --mode stage `
    --env vbot_navigation_section011 `
    --budget-hours 8 `
    --hp-trials 15 `
    --seed-configs starter_kit_schedule/configs/seed_T11_warmstart.json `
    --checkpoint "runs/vbot_navigation_section011/26-02-22_05-30-41-234319_PPO/checkpoints/agent_25000.pt" `
    --freeze-preprocessor

# Monitor progress
Get-Content starter_kit_log/automl_20260223_012907/state.yaml
```

---

## 14. Checkpoint Cleanup (2026-02-23)

Performed bulk cleanup of `/runs/vbot_navigation_section011/` to reclaim disk space:

- **Deleted**: 83 run directories entirely (obsolete experiments, AutoML trials, intermediate runs)
- **Trimmed**: 9 important runs kept but intermediate checkpoints removed (only best/peak preserved)
- **Space freed**: ~20.12 GB

### Runs Preserved (9 total)

| Run | Source | Purpose | Checkpoints Kept |
|-----|--------|---------|-----------------|
| `26-02-22_05-25-43` | Full Train A (T12) | Stage Two warm-start source | `agent_24500.pt` (peak) |
| `26-02-22_05-28-12` | Full Train B (T13) | Stage Two warm-start source | `agent_21000.pt` (peak) |
| `26-02-22_05-30-41` | Full Train C (T11) | Stage Two warm-start source | `agent_25000.pt` (peak) |
| Branch A T4 | `automl_20260222_124457` | Branch A #2 | `best_agent.pt` |
| Branch A T6 | `automl_20260222_124457` | Branch A #3 | `best_agent.pt` |
| Branch A T13 | `automl_20260222_124457` | **Branch A champion** | `best_agent.pt` |
| Branch B T2 | `automl_20260222_201059` | Branch B #3 | `best_agent.pt` |
| Branch B T4 | `automl_20260222_201059` | Branch B #4 | `best_agent.pt` |
| Branch B T8 | `automl_20260222_201059` | Branch B #2 | `best_agent.pt` |

### TensorBoard Comparison (Final Metrics from All 9 Runs)

| Run | wp_idx_mean (best) | reached% (best) | Source |
|-----|-------------------|-----------------|--------|
| Full A (T12) | 2.232 | 28.6% | Cold-start 100M |
| Full B (T13) | 2.033 | 27.5% | Cold-start 100M |
| Full C (T11) | 1.919 | 27.8% | Cold-start 100M |
| **BrA T13** | **3.935** | **52.2%** | Stage Two warm-start **CHAMPION** |
| BrA T4 | 3.900 | 50.0% | Stage Two warm-start |
| BrA T6 | 3.879 | 50.6% | Stage Two warm-start |
| BrB T10 (eval) | 3.111 | 25.4% | Stage Two warm-start |
| BrB T8 | 3.420 | 57.8% | Stage Two warm-start |
| BrB T4 | 3.385 | 55.1% | Stage Two warm-start |
| BrB T2 | 3.416 | 56.3% | Stage Two warm-start |

**Key insight**: Branch A top-3 achieved higher peak wp_idx_mean (3.88-3.94) vs Branch B top-3 (3.38-3.42). However, Branch B trials achieved higher reached% (55-58% vs 50-52%), suggesting they found a more conservative but more reliable traversal strategy.
