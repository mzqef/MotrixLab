# Section 011: Slopes, Sweep-Order Zones & Jump Celebration

## 1. Overview
*   **Environment**: `vbot_navigation_section011`
*   **Terrain**: Start (flat) → Height Field (bumps, z=0–0.277m) → 15° Ramp → High Platform (z=1.294m).
*   **Distance**: ~10.3m.
*   **Goal**: Collect 12 items (smileys + red packets) and perform a jump celebration. Total 20 pts.
*   **Key Architecture (v48-T14)**: 54-dim observations, sweep-order zone targeting, torque saturation awareness, impact avoidance. Symmetric (512,256,128) policy+value nets. AutoML-optimized lighter penalties + stronger navigation pull.

## 2. Terrain & Challenges
| Segment | Y-Range | Z-Height | Key Challenge |
| :--- | :--- | :--- | :--- |
| **Start** | -3.5 to -1.5 | 0 | Safe spawn area |
| **Height Field** | -1.5 to 1.5 | 0 – 0.277 | Uneven bumps, 3 smiley zones |
| **15° Ramp** | ~2.0 to 7.0 | 0 → 1.294 | Uphill climbing, 3 red packets |
| **Platform** | ~7.0 to 8.8 | 1.294 | Edge transition, jump celebration |

*   **Spawn**: Randomized around `(0, -2.5, 0.5)`.

## 3. Phase-Based Collection (Sweep Ordering)
Replaces "nearest-first" logic to prevent dangerous 90° turns on slopes. Uses a diagonal zigzag pattern (~35° turns).

| Phase | Target | Logic |
| :--- | :--- | :--- |
| **0: Smileys** | 3 Zones | **Sweep L→C→R** or **R→C→L** based on spawn X-coordinate. |
| **1: Red Packets** | 3 Zones | **Reverse sweep** of previous direction (continuous zigzag). |
| **2: Climb** | Platform | Navigate to center `(0, 7.83)`. |
| **3: Celebration** | Jump | Execute jump in place. |

*   **Completion**: Requires `z > 1.0m` on platform and collecting all prior zones.

## 4. Jump Celebration (v16+)
Triggered automatically upon reaching the high platform.
*   **Sequence**: `IDLE` → `JUMP` → `DONE`.
*   **Success Condition**: Robot Z-height exceeds `1.55m`.
*   **Rewards**: Continuous elevation bonus (`8.0`) + One-time completion bonus (`100.0`).

## 5. Reward Engineering (v48-T14 Active)
### AutoML-Optimized Reward Balance

v48-T14 (AutoML `automl_20260220_071134`, trial T14) discovered the optimal reward configuration through 15-trial Bayesian search. The winning pattern: **lighter penalties + stronger navigation pull**.

**Key changes from v47:**
- `lin_vel_z`: -0.195 → **-0.027** (7.2× lighter — bumps need vertical motion)
- `torque_saturation`: -0.025 → **-0.012** (2.1× lighter)
- `swing_contact_penalty`: -0.031 → **-0.003** (10× lighter)
- `waypoint_approach`: 166.5 → **280.5** (1.68× stronger)
- `zone_approach`: 35.06 → **74.7** (2.13× stronger)
- `waypoint_facing`: 0.061 → **0.61** (10× boost)
- `celebration_bonus`: 80 → **141.2** (1.77×)
- `per_jump_bonus`: 25 → **59.6** (2.4×)

Full reward scales table: [Task_Reference.md](Task_Reference.md) Section 9.

### Proprioception Boosts (Bump Area)
*   **Foot Clearance**: `foot_clearance=0.15`, `bump_boost=8.0` within bump zone (y ∈ [-1.5, 1.5]).
*   **Swing Contact**: Reduced penalty for swing contacts within bump zone (`swing_contact_bump_scale=0.356`).

### Active Rewards
*   **Height Progress**: Strong reward for vertical gain (`12.0` × Δz).
*   *Disabled*: Height approach, oscillation, and slope orientation rewards.

## 6. Training & Configuration (v48-T14 Active)
*   **Config Files**:
    *   Logic/Spawns: `vbot_section011_np.py`, `cfg.py`
    *   Hyperparams: `rl_cfgs.py` (v48-T14: LR=4.513e-4, entropy=0.00775, KL-adaptive, γ=0.999, λ=0.99)
    *   Geometry: `scene_section011.xml`
    *   Network: (512, 256, 128) both policy and value (symmetric since v47)
*   **Key Commands**:
    ```bash
    # Train (100M steps, T14 config)
    uv run scripts/train.py --env vbot_navigation_section011
    
    # AutoML HP+reward search
    uv run starter_kit_schedule/scripts/automl.py --mode stage --budget-hours 8 --hp-trials 15
    
    # Evaluate
    uv run scripts/play.py --env vbot_navigation_section011
    
    # VLM visual analysis
    uv run scripts/capture_vlm.py --env vbot_navigation_section011
    ```
*   **Critical Metrics**: `wp_idx_mean`, `wp_idx_max`, `phase_max`, `reached%`, `ep_len`.

## 7. Code Cleanup (v20)
*   **Removed**: Dead contact geometry initialization (`_init_termination_contact`).
*   **Pruned**: 14 zero-weight reward keys (e.g., `spin_progress`, `body_balance`, `heading_tracking`) to simplify configuration.