# Section 011 — Final Training Pipeline (Stages 7–12)

## Goal
Reproduce the historical A_T4 lineage with the current strict environment, walk+sit celebration, and full reward function.

## Pipeline Overview

```
Stage 7 (Cold-Start AutoML, NO yaw)
  → Stage 8 (Full 100M Train, NO yaw)
    → Stage 9 (Warm-Start AutoML, NO yaw)
      → Stage 10 (Full Train, NO yaw)
        → Stage 11 (AutoML, SOFT term + 2π yaw)
          → Stage 12 (Final Full Train, SOFT term + 2π yaw)
```

---

### Stage 7: Cold-Start AutoML (Hard Termination, NO Random Yaw)
**Goal**: Find the top 3 reward/HP configurations from scratch.
**Method**: Run `automl.py` in `stage` mode, cold-start (no checkpoint).
**Parameters**: 20 trials × 15M steps.
**Conditions**: Hard termination (`hard_tilt_deg: 70`, `soft_tilt_deg: 50`, `enable_base_contact_term: True`), walk+sit celebration, random spawn position, **NO random yaw** (`reset_yaw_scale: 0.0`).
**Seed**: `seed_T12_warmstart.json` (T12 best historical cold-start config).
**Output**: Top 3 configurations.

### Stage 8: Full 100M Training (Hard Termination, NO Random Yaw)
**Goal**: Train the top 3 configurations from Stage 7 to convergence.
**Method**: Run `train.py` using the exact configs from Stage 7's top 3.
**Parameters**: Up to 100M steps.
**Conditions**: Hard termination, walk+sit celebration, random spawn position, **NO random yaw** (`reset_yaw_scale: 0.0`).
**Output**: The single best peak checkpoint (Top 1).

### Stage 9: Warm-Start AutoML (Hard Termination, NO Random Yaw)
**Goal**: Refine reward weights starting from the Stage 8 peak checkpoint.
**Method**: Run `automl.py` in `stage` mode, warm-start from Stage 8 best checkpoint.
**Parameters**: 15 trials × 10M steps.
**Conditions**: Hard termination, walk+sit celebration, random spawn position, **NO random yaw** (`reset_yaw_scale: 0.0`).
**Output**: Top 3 refined configurations.

### Stage 10: Full Warm-Start Training (Hard Termination, NO Random Yaw)
**Goal**: Train the top 3 from Stage 9 to convergence with warm-start.
**Method**: Run `train.py` warm-starting from Stage 8 best checkpoint.
**Parameters**: Up to 100M steps.
**Conditions**: Hard termination, walk+sit celebration, random spawn position, **NO random yaw** (`reset_yaw_scale: 0.0`).
**Output**: The single best peak checkpoint (Top 1).

### Stage 11: AutoML with Soft Termination + 2π Yaw
**Goal**: Adapt the policy to handle random initial headings and softer termination.
**Method**: Run `automl.py` in `stage` mode, warm-start from Stage 10 best checkpoint.
**Parameters**: 15 trials × 10M steps.
**Conditions**: **SOFT termination** (`hard_tilt_deg: 70`, `soft_tilt_deg: 0` [disabled], `enable_base_contact_term: False`), walk+sit celebration, random spawn position, **full 2π yaw** (`reset_yaw_scale: 1.0`).
**Output**: Top 3 configurations adapted for competition conditions.

### Stage 12: Final Full Training (Soft Termination + 2π Yaw)
**Goal**: Produce the final submission-ready policy.
**Method**: Run `train.py` warm-starting from Stage 10 best checkpoint, using Stage 11's best config.
**Parameters**: Up to 100M steps.
**Conditions**: **SOFT termination**, walk+sit celebration, random spawn position, **full 2π yaw** (`reset_yaw_scale: 1.0`).
**Output**: **Final submission checkpoint.**

---

## Key Design Decisions

1. **NO random yaw in Stages 7–10**: Learning to walk, navigate bumps, and climb stairs is hard enough without also needing to find the target direction. The robot always faces +Y initially.
2. **Random 2π yaw only in Stages 11–12**: Once the robot has a strong navigation policy, we add heading randomization with soft termination so it learns to recover from any initial orientation.
3. **Hard → Soft termination progression**: Hard termination teaches robust balance early. Soft termination in the final stages avoids premature episode termination during competition.
