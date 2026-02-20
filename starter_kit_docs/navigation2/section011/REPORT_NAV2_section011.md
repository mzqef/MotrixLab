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
