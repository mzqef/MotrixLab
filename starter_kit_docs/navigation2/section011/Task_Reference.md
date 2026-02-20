# Section 011 Task Reference: Slopes, Zones & Celebration

> **Scope:** Concrete values for Stage 2A (Slopes, Height Field, Ramp, High Platform).
> **Environment ID:** `vbot_navigation_section011` (Active v49: v48-T14 base + drag_foot_penalty + stagnation_penalty; KL-adaptive scheduler, LR=4.5e-4, entropy=0.00775, net=(512,256,128); episode 120 s/12000 steps with stagnation detection).

---

## 1. Competition Scoring (20 pts Total)

| Zone Type | Count | Points | Location (XY Center) | Detection Rule |
| :--- | :---: | :---: | :--- | :--- |
| **Smiley** | 3 | 4 each (12) | `(-3, 0)`, `(0, 0)`, `(3, 0)` | **Center-Contact**: Touch exact center (Radius **0.2m**) on height field ($y \approx 0$). |
| **Red Packet** | 3 | 2 each (6) | `(-3, 4.4)`, `(0, 4.4)`, `(3, 4.4)` | **Center-Contact**: Touch exact center (Radius **0.2m**) on 15° ramp ($y \approx 4.4$). |
| **Celebration** | 1 | 2 | High Platform $(0, 7.83)$ | Perform **3 jumps** ($z > 1.55$ then land $z < 1.50$) on top platform ($z > 1.0$). |

*   **Spawn:** Random on START platform ($y \in [-3.5, -1.5]$). Current fixed spawn: $y=-2.5$.
*   **Rule Change:** Old 1.2m boundary detection replaced by strict 0.2m center-contact.

---

## 2. Terrain Specifications

| Element | Dimensions / Specs | Top Z | Notes |
| :--- | :--- | :--- | :--- |
| **Start Platform** | $5.0 \times 1.0$ box | 0.0 | Flat start. |
| **Height Field** | $\pm 5m \times \pm 1.5m$ | 0.277 (max) | Bumpy terrain at $y \approx 0$. |
| **Ramp** | $5.0 \times 2.5$ box | ~0.66 | Tilted **15°** around x-axis. |
| **High Platform** | $5.0 \times 1.0$ box | **1.294** | Target for celebration. |
| **Walls** | $x = \pm 5.25$ | 2.45 tall | Boundary limits. |

**Challenge Path:** Start $\to$ Bumps/Smileys $\to$ 15° Ramp/Red Packets $\to$ Platform Edge $\to$ Celebration Jumps.

---

## 3. Phase-Based Collection Logic (v15+)

Robots progress through 4 phases. Target selection is **nearest uncollected zone** in the current phase.

```python
PHASE_SMILEYS = 0       # Collect smileys (Any order)
PHASE_RED_PACKETS = 1   # Gate: >= 1 Smiley collected (Relaxed in Stage 1B)
PHASE_CLIMB = 2         # Gate: All 3 Red Packets collected
PHASE_CELEBRATION = 3   # Gate: Reached high platform
```

*   **Stage 1B Critical Fix:** Phase 0$\to$1 transition requires only **1 smiley** (was all 3), enabling learning convergence.
*   **Bonuses:** 25.0 pts awarded upon completing Smileys and Red Packets phases.

---

## 4. Celebration State Machine (v27: Multi-Jump)

Triggered when robot is on High Platform ($z > 1.0$).

| Parameter | Value | Description |
| :--- | :--- | :--- |
| **Jump Threshold** | $z > 1.55$ | Detects airborne state. |
| **Land Threshold** | $z < 1.50$ | Detects landing (counts 1 jump). |
| **Required Jumps** | 3 | Total jumps needed for completion. |
| **Rewards** | Jump: 10.0 (step)Per Jump: 25.0Completion: 80.0 | Continuous height reward + discrete bonuses. |
| **Observation** | `celeb_progress` | Encodes progress: 0.0 $\to$ 0.33 $\to$ 0.67 $\to$ 1.0. |

---

## 5. Observation Space (69-Dim v20)

Key inputs for policy:
*   **Proprioception:** Linear/Angular velocity, Projected Gravity, Joint Pos/Vel (12 each), Foot Contact (4).
*   **Navigation:** Position Error (2), Heading Error (1), Base Height (1).
*   **Task State:** `celeb_progress` (1).
*   **Advanced Sensors (v20):**
    *   `trunk_acc_norm` (3): Impact detection.
    *   `torques_normalized` (12): **Raw PD demand** (can exceed $\pm 1.0$), allowing policy to learn saturation avoidance.

---

## 6. Termination & Safety

**Hard Terminations (Immediate, No Grace):**
*   Tilt $> 70^\circ$, Out-of-Bounds, Joint Velocity Overflow, Joint Accel $> 80$ rad/s², NaNs.

**Soft Terminations (100-step Grace Period):**
*   Base Contact $> 0.01$ (allows stabilization after spawn/bumps).
*   Medium Tilt ($50^\circ - 70^\circ$).

**Penalty:** On termination, **60%** of accumulated bonus is deducted (soft clear to encourage risk-taking).

---

## 7. Reward Architecture Highlights (v48-T14)

*   **cfg refactor (post-v35):** `VBotSection011EnvCfg.RewardConfig` now uses `dict(BASE_REWARD_SCALES)` directly — no inline override. Bonus keys (`smiley_bonus`, `red_packet_bonus`, etc.) accessed via `.get(key, 0.0)` default to zero.
*   **Alive Bonus:** Segmented by posture.
    *   Tilt $< 26^\circ$ ($g_z > 0.9$): **100%** bonus.
    *   Tilt $26^\circ-45^\circ$: **50%** bonus.
    *   Tilt $> 45^\circ$: **0%** bonus.
*   **Key Gradients:** Strong `waypoint_approach` (**280.5**, was 166.5 in v47), `zone_approach` (**74.7**, was 35.06), and `height_progress` (27.0).
*   **Key Penalty Changes (v48-T14 vs v47):** `lin_vel_z` **-0.027** (was -0.195, 7.2× lighter), `torque_saturation` **-0.012** (was -0.025, 2.1× lighter), `termination` **-150** (was -200), `swing_contact_penalty` **-0.003** (was -0.031, 10× lighter).
*   **v49 Anti-Local-Optimum Penalties:**
    *   `drag_foot_penalty` **-0.02**: Per-dragging-leg penalty. Detects calf contact + velocity < 1.0 m/s. Bump zone 2× boost.
    *   `stagnation_penalty` **-0.5**: Linear ramp from 50% to 100% of stagnation window. Provides gradient signal before truncation.
*   **Active Config (v48-T14):**
    *   Discount ($\gamma$): **0.999**, GAE ($\lambda$): **0.99**
    *   LR: **4.513e-4** + KL-Adaptive scheduler (v48-T14: 4.5× higher than v47's 1e-4)
    *   Entropy: **0.00775** (1.8× higher than v47's 0.00432)
    *   Policy/Value net: **(512, 256, 128)** both (v47 had policy 256,128,64)
    *   Episode: **120 s / 12000 steps** with stagnation detection (v44).
    *   `max_env_steps` (rl_cfgs): **100 M**.
    *   Source: `automl_20260220_071134` trial T14 (wp_idx_mean=0.484 @15M, best of 15 trials)

---

## 8. Control System (PD)

*   **Architecture:** Software PD over raw torque actuators.
*   **Parameters:** $K_p=100$, $K_d=8$, Action Scale=$0.5$ rad.
*   **Saturation:** Max theoretical torque (50 Nm) exceeds limits (17/34 Nm). Policy observes **unclipped** torque demand to learn soft control.

---

## 9. Current Reward Scales (v49 — Active in cfg.py)

| Category | Parameter | Value | vs v47 | Notes |
|----------|-----------|-------|--------|-------|
| **Navigation** | `forward_velocity` | 3.163 | +10% | |
| | `waypoint_approach` | **280.534** | **1.68×** | Strongest navigation pull — key change |
| | `waypoint_facing` | 0.637 | ~same | |
| | `zone_approach` | **74.727** | **2.13×** | Side-zone attraction |
| | `position_tracking` | 0.259 | -33% | Lighter |
| **Alive** | `alive_bonus` | 1.013 | -30% | |
| | `alive_decay_horizon` | **2383** | **+59%** | Longer sustained motivation |
| **Terrain** | `height_progress` | 26.965 | ~same | |
| | `foot_clearance` | **0.219** | **+46%** | Stronger step-height reward |
| | `foot_clearance_bump_boost` | 7.167 | -10% | |
| **Bonuses** | `waypoint_bonus` | 50.046 | ~same | |
| | `phase_bonus` | 13.067 | -48% | |
| | `celebration_bonus` | **141.242** | **+77%** | Stronger end-game pull |
| | `per_jump_bonus` | **59.641** | **+139%** | |
| | `jump_reward` | 10.093 | ~same | |
| **Penalties** | `lin_vel_z` | **-0.027** | **7.2× lighter** | KEY: bumps need vertical motion |
| | `torque_saturation` | **-0.012** | **2.1× lighter** | KEY: less motor stress penalty |
| | `termination` | **-150** | **1.3× lighter** | Sweet spot (-50 too lenient, -200 too harsh) |
| | `swing_contact_penalty` | **-0.003** | **10× lighter** | KEY: less bump traversal penalty |
| | `orientation` | -0.026 | ~same | |
| | `impact_penalty` | -0.100 | +25% | Slightly heavier |
| | `action_rate` | -0.007 | ~same | |
| | `ang_vel_xy` | -0.038 | -16% | |
| | **`drag_foot_penalty`** | **-0.02** | **v49 NEW** | Per-dragging-leg (contact+low_vel), bump×2 |
| | **`stagnation_penalty`** | **-0.5** | **v49 NEW** | Linear ramp 50%→100% stagnation window |
| **Gait** | `stance_ratio` | 0.070 | +70% | |
| | `swing_contact_bump_scale` | 0.210 | -41% | |
| **Unchanged** | `height_approach` | 5.0 | same | Not in search |
| | `height_oscillation` | -2.0 | same | Not in search |
| | `torques` | -5e-6 | same | Not in search |
| | `dof_vel` | -3e-5 | same | Not in search |
| | `dof_acc` | -1.5e-7 | same | Not in search |
| | `score_clear_factor` | 0.0 | same | Not in search |
| | `slope_orientation` | 0.0 | same | Disabled |

---

## 10. AutoML Search Space Status & Future Exploration Opportunities

### v48 Search (completed 2026-02-20): What Was Searched

25 reward parameters + 2 HP parameters (LR, entropy) searched over 15 trials × 15M steps.

**v48 Outcome:** T14 winner deployed to 100M training → **FAILED** at 78% (local optimum: backward-dragging behavior, foot_clearance=0, LR crushed to 5.9e-5). See REPORT Section 8.

### v49 Search Space Expansion (2026-02-20): 27 reward + 2 HP = 29 parameters

Based on T14 boundary analysis + 2 new anti-local-optimum penalties:

| Change | Old | New | Reason |
|--------|-----|-----|--------|
| `learning_rate` | [3e-4, 5e-4] | **[2e-4, 8e-4]** | T14=4.5e-4 at 90th pctile |
| `entropy_loss_scale` | [3e-3, 6e-3] | **[3e-3, 1.5e-2]** | T14=7.75e-3 EXCEEDED upper |
| `waypoint_approach` | [80, 300] | **[80, 500]** | T14=280.5 at 93% |
| `zone_approach` | [20, 80] | **[20, 150]** | T14=74.7 at 91% |
| `lin_vel_z` | [-0.2, -0.02] | **[-0.2, -0.005]** | T14=-0.027 at 96% |
| `swing_contact_penalty` | [-0.06, -0.003] | **[-0.06, -0.0005]** | T14=-0.003 AT bound |
| `drag_foot_penalty` | — (new) | **[-0.08, -0.005]** | v49 anti-drag penalty |
| `stagnation_penalty` | — (new) | **[-2.0, -0.1]** | v49 anti-stagnation penalty |

### What Was NOT Searched (Fixed Parameters)

| Parameter | Fixed Value | Why Fixed | Explore? |
|-----------|-------------|-----------|----------|
| `height_approach` | 5.0 | v35 proven | LOW — secondary signal |
| `height_oscillation` | -2.0 | v35 proven | LOW — rarely triggers |
| `torques` | -5e-6 | v35 proven | LOW — negligible penalty magnitude |
| `dof_vel` | -3e-5 | v35 proven | LOW — negligible |
| `dof_acc` | -1.5e-7 | v35 proven | LOW — negligible |
| `slope_orientation` | 0.0 | Disabled | MAYBE — could help ramp phase |
| `score_clear_factor` | 0.0 | Disabled | LOW |
| `discount_factor` | 0.999 | Curriculum-proven (Stage 13) | LOW — already optimal |
| `lambda_param` | 0.99 | Curriculum-proven (Stage 15) | LOW — already optimal |
| `rollouts` | 24 | v23b-T7 proven | MAYBE — try 32/48 |
| `learning_epochs` | 6 | v23b-T7 proven | MAYBE — try 4/8 |
| `mini_batches` | 16 | v23b-T7 proven | LOW |
| `grad_norm_clip` | 1.0 | Standard | MAYBE — try 0.5 |
| `ratio_clip` | 0.2 | Standard PPO | LOW |
| `value_clip` | 0.2 | Standard PPO | LOW |

### Known Search Space Boundary Issues

| Parameter | v48 Range | T14 Value | At Boundary? | Action |
|-----------|-----------|-----------|-------------|--------|
| `entropy_loss_scale` | [3e-3, 6e-3] | **0.00775** | **YES — EXCEEDED upper bound** | Widen to [3e-3, 1.5e-2] |
| `waypoint_approach` | [80, 300] | 280.5 | Near upper bound (93%) | Widen to [80, 500] |
| `learning_rate` | [3e-4, 5e-4] | 4.5e-4 | 90th percentile | Widen to [2e-4, 8e-4] |
| `celebration_bonus` | [40, 200] | 141.2 | Interior (71%) | OK |
| `per_jump_bonus` | [10, 80] | 59.6 | Interior (71%) | OK |
| `zone_approach` | [20, 80] | 74.7 | Near upper bound (91%) | Widen to [20, 150] |
| `lin_vel_z` | [-0.2, -0.02] | -0.027 | Near lighter bound (96%) | Widen to [-0.2, -0.005] |
| `swing_contact_penalty` | [-0.06, -0.003] | -0.003 | **AT lower bound** | Widen to [-0.06, -0.0005] |
| `termination` | {-200,-150,-100,-50} | -150 | Interior | Consider {-175,-150,-125,-100} narrower |

### Recommended Future Search Directions (Priority Order)

**Search A — v49 Anti-Local-Optimum + Boundary Expansion (HIGH priority — NEXT):**
Combines boundary widening (6 params) + 2 new anti-dragging penalties. v48-T14 converged to backward-dragging local optimum at 100M (foot_clearance=0, LR crushed). The v49 penalties directly target this failure mode. Run 15–20 trials × 15M steps with expanded 29-parameter search space. **This is the immediate next AutoML run.**

**Search B — Long-Horizon Validation (HIGH priority):**
T14 was validated at 15M steps only. At 50M+ steps, behavior can diverge. Run best v49 trial to 50M+. v48-T14 proved this concern valid: 15M config collapsed at 78M. Long-horizon validation is MANDATORY before declaring any config optimal.

**Search C — Reward Components Not Yet Searched (MEDIUM priority):**
- `slope_orientation` currently disabled (0.0) — could help ramp traversal stability. Search [0.0, 0.1].
- `score_clear_factor` currently 0.0 — search [0.0, 0.5] to test whether partial score deduction on fall improves risk management.
- `stagnation_min_distance` and `stagnation_window_steps` are env-level, not searchable via reward — but could be explored manually.

**Search D — PPO Dynamics (LOW priority):**
- `rollouts`: 24 is fixed but untested at 32/48 with v48's reward landscape.
- `learning_epochs`: 6 is fixed — try 4 (less overfitting per batch) or 8.
- These interact with LR/entropy, so joint search is preferred.

**Search E — Warm-Start HP Search (MEDIUM priority):**
All 15 trials in v48 trained from scratch. A separate search using warm-start from the v47 50M checkpoint would test a different optimization landscape — the reward weights that work for cold-start may differ from those optimal for fine-tuning.

### Bayesian Convergence Warning

The v48 search converged heavily after ~8/15 trials — most guided trials shared nearly identical core params. For future searches:
- Use more random seeds (e.g., `--random-trials 5` instead of 3)
- Or use a different optimizer (random search with 20+ trials) to ensure broader exploration
- Or split into focused sub-searches (e.g., penalties-only, bonuses-only, HP-only)