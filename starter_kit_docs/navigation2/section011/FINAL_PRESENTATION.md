# Section011 — Final Presentation & Reproduction Recipe

> **Historical Best: wp_idx = 5.9115** (v35, warm-start chain)
> **Active Config: v49** — v48-T14 base + drag_foot_penalty + stagnation_penalty (anti-local-optimum). T14 100M failed at 78% (backward-dragging local optimum). v49 targets the root cause.
> See [Task_Reference.md](Task_Reference.md) Section 9 for full reward scales table.

---

## 1. Task Overview

**Environment**: `vbot_navigation_section011` — VBot quadruped robot navigating Section01 of the MotrixArena S1 competition.

**Course**: START platform → height-field bumps → 15° ramp → high platform (z=1.294m)

**Objectives** (20 points max):
- Collect 3 smiley zones (+4 pts each = 12 pts) on the height field
- Collect 3 red-packet zones (+2 pts each = 6 pts) on the ramp
- Reach the "2026" celebration platform and perform 3 jumps (+2 pts)

**Metric**: `wp_idx` — composite waypoint index encoding zone collection + phase progression (max ≈ 7.0)

---

## 2. Result: wp_idx Progression Over 35 Versions

```
wp_idx
  6.0 ┤                                                    ★ v35 = 5.91
  5.5 ┤                                              ●v29 = 5.86
  5.0 ┤                                         ●v28 = 5.05
  4.5 ┤
  4.0 ┤                    ●v16-S3
  3.5 ┤
  3.0 ┤               ●v15-S2
  2.5 ┤          ●v15-1C
  2.0 ┤     ●v15-1B  ●v16-S15
  1.5 ┤
  1.0 ┤●v15-S0
  0.5 ┤         ●v23b-T7 (gradient restart)
  0.0 ┤─────────────────────────────────────────────────────────
      v15   v16   v23b   v25   v27   v28   v29   v32   v35
```

---

## 3. The Three-Phase Training Recipe

The winning checkpoint was produced by a **3-phase approach**, each phase building on the previous:

### Phase 1: Architecture Discovery via AutoML (v23b-T7)

**Goal**: Find the optimal network architecture + PPO hyperparameters using gradient-only rewards (all discrete bonuses zeroed).

**Method**: AutoML HP search (`automl.py --hp-trials 15`), fresh training from scratch, 5M steps per trial.

**Winner config** (trial T7 of `automl_20260216_135420`):

| Parameter | Value |
|-----------|-------|
| Policy network | (256, 128, 64) |
| Value network | (512, 256, 128) |
| Learning rate | 4.24e-4 |
| Entropy | 4.11e-3 |
| Rollouts | 24 |
| Learning epochs | 6 |
| Mini-batches | 16 |
| Discount (γ) | 0.999 |
| Lambda (λ) | 0.99 |

**Result**: wp_idx ≈ 0.58 at 5M steps (gradient-only, no bonuses)

**Why this works**: The gradient-only architecture forces the robot to learn locomotion from continuous distance signals alone. This produces a robust walker that doesn't exploit discrete bonus hacking.

```powershell
# Phase 1: Fresh AutoML search (≈2 hours)
uv run starter_kit_schedule/scripts/automl.py --mode hp-search \
    --env vbot_navigation_section011 --budget-hours 4 --hp-trials 15
```

### Phase 2: Bonus Re-enablement (v28 → v29)

**Goal**: Add discrete competition bonuses back to the already-trained gradient walker. The robust locomotion from Phase 1 prevents bonus exploitation.

**v28** (re-enabled bonuses):
```python
# Added back to reward scales:
"smiley_bonus": 10.0,       # per-smiley one-time
"red_packet_bonus": 7.0,    # per-red-packet one-time  
"waypoint_bonus": 30.0,     # reaching high platform
"phase_completion_bonus": 10.0,
"jump_reward": 5.0,
"per_jump_bonus": 10.0,
"celebration_bonus": 40.0,
```
Result: wp_idx = 5.05

**v29** (boosted later-phase bonuses — final reward config):
```python
# Boosted values that became the permanent config:
"waypoint_bonus": 50.0,          # +67%
"smiley_bonus": 20.0,            # +100%
"red_packet_bonus": 20.0,        # +186%
"phase_completion_bonus": 25.0,  # +150%
"jump_reward": 10.0,             # +100%
"per_jump_bonus": 25.0,          # +150%
"celebration_bonus": 80.0,       # +100%
```
Result: wp_idx = 5.8643 (agent_6000.pt)

**Warm-start chain for Phase 2**:
```
v23b-T7 fresh → v25 (ordered targeting) → v27 (multi-jump) → v28 → v29
```

Each step: `uv run scripts/train.py --env vbot_navigation_section011 --train-backend torch --checkpoint <prev_best_agent.pt>`

LR reduced from 4.24e-4 → 1.0e-4 for warm-start stability.

### Phase 3: KL-Adaptive Fine-Tuning from Pre-Peak (v35)

**Goal**: Squeeze extra performance from the v29 policy using KL-adaptive LR scheduling + warm-start from a pre-peak checkpoint.

**Key insight**: SKRL warm-start loads network weights only — NOT optimizer state (Adam momentum/variance). This causes a 3000-5000 iter recovery period. Two discoveries mitigated this:

1. **KL-adaptive scheduler** (`lr_scheduler_type = "kl_adaptive"`): Automatically reduces LR when policy divergence exceeds a KL threshold, preventing catastrophic forgetting.

2. **Pre-peak warm-start**: Loading from `agent_5000.pt` (wp_idx=5.41, ascending slope) instead of `agent_6000.pt` (wp_idx=5.86, peak) gives the optimizer a clear uphill gradient to follow.

```powershell
# Phase 3: Fine-tune from v29 pre-peak checkpoint
# First, set lr_scheduler_type = "kl_adaptive" in rl_cfgs.py
uv run scripts/train.py --env vbot_navigation_section011 --train-backend torch \
    --checkpoint runs/vbot_navigation_section011/<v29_run>/checkpoints/agent_5000.pt
```

**Result**: wp_idx = **5.9115** at step 6000 (agent_6000.pt) — ALL-TIME BEST (warm-start chain)

### Phase 4: AutoML v48 — Fresh-Start HP+Reward Search (v46→v47→v48-T14)

**Goal**: Escape warm-start chain path-dependency. Train from scratch with symmetric networks + optimized reward balance.

**Architecture change (v47)**: Policy network expanded from (256,128,64) → **(512,256,128)** — now symmetric with value network. This was motivated by the hypothesis that the smaller policy was a bottleneck for the more complex penalty-lightened reward landscape.

**AutoML search** (`automl_20260220_071134`): 15 Bayesian trials × 15M steps, joint HP + reward weight search over 25 parameters.

**Winner: T14** (best wp_idx_mean=0.484 @ 15M cold-start):

| Key Parameter | v47 | T14 | Change |
|--------------|-----|-----|--------|
| lin_vel_z | -0.195 | **-0.027** | 7.2× lighter |
| torque_saturation | -0.025 | **-0.012** | 2.1× lighter |
| termination | -200 | **-150** | 25% lighter |
| waypoint_approach | 166.5 | **280.5** | 1.68× stronger |
| zone_approach | 35.06 | **74.7** | 2.13× stronger |
| swing_contact_penalty | -0.031 | **-0.003** | 10× lighter |
| Learning rate | 1e-4 | **4.5e-4** | 4.5× faster |
| Entropy | 0.0043 | **0.0078** | 1.8× more exploration |

**Key discovery**: Lighter penalties + stronger navigation pull is the dominant winning pattern. 9/15 trials reached phase RED_PACKETS (wp_max≥3.0). T14's 0.484 mean at 15M cold-start is expected to significantly exceed v47's 1.40 at 50M with continued training.

**100M deployment**: Training v48-T14 config from scratch to 100M steps (current).

See [Task_Reference.md](Task_Reference.md) Section 10 for future AutoML exploration plan (boundary expansion, long-horizon validation).

### Phase 5: Anti-Local-Optimum Penalties (v49)

**Problem**: T14 100M training converged to backward-dragging local optimum at ~78% (38.5k/48.5k iters). Deep analysis revealed:
- `foot_clearance = 0` — robot NEVER lifts legs (pure dragging gait)
- `torque_saturation = -1106` — controllers fully saturated fighting terrain
- `wp_idx_mean = 0.45` — average didn't even pass WP0
- LR crushed from 4.5e-4 → 5.9e-5 by KL-adaptive (7.6× collapse)

**Root cause**: Reward blind spot — legs with sustained ground contact + low velocity get neither foot_clearance reward (not in swing) nor swing_contact penalty (low velocity). Combined with alive_bonus, standing/retreating is a positive-reward strategy.

**v49 Fix — Two new penalties:**

| Penalty | Scale | Mechanism |
|---------|-------|-----------|
| `drag_foot_penalty` | -0.02 | Per-dragging-leg: calf_contact AND velocity < 1.0 m/s. Bump zone ×2. |
| `stagnation_penalty` | -0.5 | Linear ramp from 50%→100% of stagnation window. Provides gradient before truncation. |

**AutoML v49 search space expanded**: 6 boundary-widened params + 2 new penalties → 29 total searchable parameters. Key expansions: entropy [3e-3, 1.5e-2], waypoint_approach [80, 500], zone_approach [20, 150], swing_contact_penalty [-0.06, -0.0005].

See [Task_Reference.md](Task_Reference.md) Section 10 for the full v49 expansion table and next AutoML plan.

---

## 4. Complete Reproduction Commands

### Prerequisites

```powershell
# Python 3.10, UV package manager
uv sync --all-packages --all-extras
```

### Step-by-Step Reproduction

```powershell
# ============================================================
# STEP 1: Fresh training with v23b architecture (gradient-only)
# ============================================================
# Reward config: All bonuses set to 0.0 in cfg.py
# RL config: LR=4.24e-4, no scheduler
# This takes ~70 min for 50M steps

uv run scripts/train.py --env vbot_navigation_section011 --train-backend torch
# → Pick best_agent.pt or agent with highest wp_idx

# ============================================================
# STEP 2: Re-enable bonuses (v28 config) and warm-start
# ============================================================
# Reward config: Enable smiley/red_packet/waypoint/celebration bonuses
# RL config: LR=1e-4 (0.25× reduction for warm-start)

uv run scripts/train.py --env vbot_navigation_section011 --train-backend torch \
    --checkpoint <step1_best_agent.pt>
# → Expect wp_idx ≈ 5.0+

# ============================================================
# STEP 3: Boost later-phase bonuses (v29 config) and warm-start
# ============================================================
# Reward config: Boost to final values (smiley=20, red_packet=20, etc.)

uv run scripts/train.py --env vbot_navigation_section011 --train-backend torch \
    --checkpoint <step2_best_agent.pt>
# → Expect wp_idx ≈ 5.8+ at agent_6000.pt

# ============================================================
# STEP 4: KL-adaptive fine-tune from PRE-PEAK checkpoint
# ============================================================
# RL config: lr_scheduler_type = "kl_adaptive"
# IMPORTANT: Use agent_5000.pt (pre-peak), NOT agent_6000.pt (peak)

uv run scripts/train.py --env vbot_navigation_section011 --train-backend torch \
    --checkpoint <step3_agent_5000.pt>
# → Expect wp_idx ≈ 5.9+ at agent_6000.pt

# ============================================================
# STEP 5: Deploy best checkpoint
# ============================================================
Copy-Item <step4_run>/checkpoints/agent_6000.pt `
    starter_kit_schedule/checkpoints/vbot_navigation_section011/best_agent.pt

# ============================================================
# STEP 6: Evaluate
# ============================================================
uv run starter_kit_schedule/scripts/eval_checkpoint.py \
    --rank <step4_run_dir>
```

---

## 5. Final Configuration Files

### PPO Hyperparameters (`starter_kit/navigation2/vbot/rl_cfgs.py`) — v48-T14 Active

```python
@rlcfg("vbot_navigation_section011")
@dataclass
class VBotSection011PPOConfig(PPOCfg):
    seed: int = 42
    num_envs: int = 2048
    play_num_envs: int = 16
    max_env_steps: int = 100_000_000
    check_point_interval: int = 500

    learning_rate: float = 4.513e-4     # v48-T14 (was 1e-4)
    lr_scheduler_type: str | None = "kl_adaptive"
    rollouts: int = 24
    learning_epochs: int = 6
    mini_batches: int = 16
    discount_factor: float = 0.999
    lambda_param: float = 0.99
    grad_norm_clip: float = 1.0
    entropy_loss_scale: float = 0.00775  # v48-T14 (was 4.11e-3)

    ratio_clip: float = 0.2
    value_clip: float = 0.2
    clip_predicted_values: bool = True

    share_policy_value_features: bool = False
    policy_hidden_layer_sizes: tuple[int, ...] = (512, 256, 128)  # v47: symmetric with value
    value_hidden_layer_sizes: tuple[int, ...] = (512, 256, 128)
```

### Reward Scales (`starter_kit/navigation2/vbot/cfg.py → BASE_REWARD_SCALES`) — v48-T14 Active

> **Current state (v48-T14):** All bonuses integrated into `BASE_REWARD_SCALES`. No separate phase-2 override needed.
> Full table with vs-v47 comparison: [Task_Reference.md](Task_Reference.md) Section 9.

```python
BASE_REWARD_SCALES = {
    # === Continuous Navigation Rewards ===
    "forward_velocity": 3.163,         # (was 2.875)
    "waypoint_approach": 280.534,      # (was 166.5) — 1.68× stronger pull
    "waypoint_facing": 0.610,          # (was 0.061) — 10× boost
    "position_tracking": 0.384,
    "alive_bonus": 1.446,
    "alive_decay_horizon": 2383,       # (was 1500)

    # === Zone Navigation ===
    "zone_approach": 74.727,           # (was 35.06) — 2.13× stronger

    # === Terrain Adaptation ===
    "height_progress": 28.30,
    "height_approach": 5.0,
    "height_oscillation": -2.0,

    # === Foot Clearance ===
    "foot_clearance": 0.150,           # (was 0.053) — 2.8× boost
    "foot_clearance_bump_boost": 8.0,  # (was 4.39)

    # === One-Time Bonuses ===
    "smiley_bonus": 18.254,
    "red_packet_bonus": 14.757,
    "waypoint_bonus": 56.069,
    "phase_completion_bonus": 14.785,

    # === Celebration (3-Jump) ===
    "jump_reward": 8.636,
    "per_jump_bonus": 59.641,          # (was 25) — 2.4×
    "celebration_bonus": 141.242,      # (was 80) — 1.77×

    # === Gait Quality ===
    "stance_ratio": 0.041,

    # === Penalties (lighter = T14 winning pattern) ===
    "swing_contact_penalty": -0.003,   # (was -0.031) — 10× lighter
    "swing_contact_bump_scale": 0.356,
    "impact_penalty": -0.080,
    "torque_saturation": -0.012,       # (was -0.025) — 2.1× lighter
    "orientation": -0.05,
    "lin_vel_z": -0.027,               # (was -0.195) — 7.2× lighter
    "ang_vel_xy": -0.045,
    "torques": -5e-6,
    "dof_vel": -3e-5,
    "dof_acc": -1.5e-7,
    "action_rate": -0.008,
    "termination": -150.0,             # (was -200) — 25% lighter
    "score_clear_factor": 0.0,
    "slope_orientation": 0.04,
}
```

### Environment Config Highlights

| Parameter | Value | Notes |
|-----------|-------|-------|
| `action_scale` (flat) | 0.25 | Dynamic zones override: 0.40 bump/slope |
| `max_episode_seconds` | 120.0 | v44: doubled (was 60.0); stagnation detection replaces fixed cutoff |
| `max_episode_steps` | 12000 | v44: doubled (was 6000) |
| `grace_period_steps` | 100 | Soft termination grace |
| `stagnation_window_steps` | 1000 | v44: 10 s window for stagnation check |
| `stagnation_min_distance` | 0.5 m | v44: must travel ≥0.5 m per window |
| `stagnation_grace_steps` | 500 | v44: no stagnation check in first 5 s |
| `celebration_jump_threshold` | 1.55 | Jump detection height |
| `required_jumps` | 3 | Jumps needed for celebration |
| `celebration_landing_z` | 1.50 | Landing detection height |
| Spawn position | (0, -2.5, 0.35) ± (2.0, 0.5) | X: ±2 m (5 m wide platform), Y: ±0.5 m |
| Target waypoints | (0,0) → (0,4.4) → (0,7.83) | Smiley → Red Packet → Celebration |

---

## 6. All-Time Top-20 Leaderboard

| Rank | Run (timestamp) | Version | Checkpoint | wp_idx |
|------|----------------|---------|------------|--------|
| **1** | **20-42-31** | **v35** | **agent_6000** | **5.9115** |
| 2 | 10-45-19 | v29 | agent_6000 | 5.8643 |
| 3 | 20-42-31 | v35 | agent_5500 | 5.6765 |
| 4 | 14-36-02 | v32 | agent_6000 | 5.4923 |
| 5 | 10-45-19 | v29 | agent_5000 | 5.4084 |
| 6 | 20-42-31 | v35 | agent_5000 | 5.3736 |
| 7 | 14-36-02 | v32 | agent_5500 | 5.2667 |
| 8 | 08-20-06 | — | agent_10000 | 5.2065 |
| 9 | 03-50-32 | — | agent_4000 | 5.0512 |
| 10 | 20-42-31 | v35 | agent_4500 | 4.9799 |
| 11 | 14-36-02 | v32 | agent_5000 | 4.9870 |
| 12 | 08-20-06 | — | agent_4000 | 4.9550 |
| 13 | 14-36-02 | v32 | agent_12000 | 4.8393 |
| 14 | 08-20-06 | — | agent_9000 | 4.8386 |
| 15 | 01-45-06 | — | agent_5000 | 4.7600 |
| 16 | 01-45-06 | — | agent_4000 | 4.7485 |
| 17 | 10-45-19 | v29 | agent_4000 | 4.7329 |
| 18 | 20-42-31 | v35 | agent_4000 | 4.5025 |
| 19 | 03-50-32 | — | agent_5000 | 4.2443 |
| 20 | 20-42-31 | v35 | agent_3500 | 3.9638 |

---

## 7. Key Discoveries & Lessons Learned

### Discovery 1: Gradient-Only Pre-Training (v23b)

Training with **all discrete bonuses zeroed** forces the robot to learn locomotion from continuous distance gradients alone. This prevents:
- Bonus exploitation (standing near zones without moving)
- Sparse reward starvation (robot never reaches first bonus → no learning signal)
- Value function corruption (large discrete bonuses create noisy targets)

Once the robot can walk reliably, bonuses are re-enabled on a well-trained base policy.

### Discovery 2: KL-Adaptive LR Scheduler (v32/v35)

SKRL's warm-start only loads network weights, **not optimizer state** (Adam momentum + variance). The stale optimizer causes destructive updates:
- v29: peaks at 5.86, then crashes to 2.85 by step 7000
- v32 (KL-adaptive): peaks at 5.49, retains 4.84 at step 12000

`lr_scheduler_type = "kl_adaptive"` dynamically reduces LR when KL divergence between old/new policy exceeds a threshold, preventing catastrophic forgetting.

### Discovery 3: Pre-Peak Warm-Start (v35)

Loading from the **ascending slope** of the training curve beats loading from the **peak**:

| Warm-start source | KL scheduler | Peak wp_idx |
|-------------------|-------------|-------------|
| agent_6000 (peak, 5.86) | No | 3.22 (v34) |
| agent_6000 (peak, 5.86) | Yes | 5.49 (v32) |
| agent_5000 (ascending, 5.41) | Yes | **5.91 (v35)** |

**Why**: At the peak, gradients are flat — the optimizer doesn't know which direction to go. On the ascending slope, there's a clear uphill direction that aligns with the freshly-reset Adam state.

### Discovery 4: Curriculum-Discovered HP (Stages 13-15)

`γ=0.999` (discount factor) and `λ=0.99` (GAE lambda) were discovered through systematic curriculum stages:
- Stage 13: γ=0.99→0.999 improved wp_idx by +0.24 (longer planning horizon)
- Stage 15: λ=0.95→0.99 improved wp_idx by +0.11 (higher GAE, ~460-step credit)

These values persisted as optimal through all subsequent versions.

### Discovery 5: Asymmetric → Symmetric Networks (v47)

Originally Policy=(256,128,64) and Value=(512,256,128) — asymmetric, with larger value network. v47 changed to **symmetric (512,256,128) for both**, based on the hypothesis that the smaller policy was a bottleneck for complex reward landscapes. This became the standard architecture for all subsequent training.

### Discovery 6: Lighter Penalties Dominate (v48 AutoML)

AutoML v48 searched 25 reward parameters across 15 trials. The winning pattern was overwhelmingly clear:

- **lin_vel_z**: -0.195 → -0.027 (7.2× lighter) — bumps REQUIRE vertical motion; heavy z-penalty suppresses it
- **torque_saturation**: -0.025 → -0.012 (2.1× lighter) — aggressive terrain needs high torques
- **swing_contact_penalty**: -0.031 → -0.003 (10× lighter) — bump terrain causes unavoidable contacts
- **termination**: -200 → -150 (25% lighter) — less catastrophic falling fear

Simultaneously, **navigation pull was strengthened**: waypoint_approach 1.68×, zone_approach 2.13×. The combination means: "Don't punish the robot for moving aggressively over terrain, and reward it more for making progress."

5 parameters hit search boundaries, indicating the Bayesian optimizer was constrained. Future searches should expand these ranges (see [Task_Reference.md](Task_Reference.md) Section 10).

### Discovery 7: Short-Horizon AutoML ≠ Long-Horizon Success (v48-T14 Failure)

T14 was the best trial at 15M cold-start steps (wp_idx_mean=0.484). But when trained to 100M steps:
- **Collapsed into backward-dragging local optimum** — foot_clearance=0, robot never lifts legs
- **LR crushed by KL-adaptive** — from 4.5e-4 → 5.9e-5 (policy stopped learning)
- **Lighter penalties enabled survival exploitation** — without strong penalties, alive_bonus dominates

**Lesson**: Configs optimized for early training (15M) can catastrophically fail at scale. The penalty lightening that accelerates early learning removes the gradient signal needed to escape local optima at 50M+.

**Fix (v49)**: Added `drag_foot_penalty` (penalizes exactly the degenerate behavior) and `stagnation_penalty` (provides gradient signal before stagnation truncation). These target the specific failure mode rather than reverting to heavier general penalties.

---

## 8. Failed Experiments (What NOT To Do)

| Experiment | Change | Result | Why It Failed |
|------------|--------|--------|---------------|
| v30 | torque_saturation -0.025→-0.010 | 4.02 | Reward distribution shift corrupted value function |
| v31 | LR 1e-4→5e-5 only | 1.68 | Too slow recovery from optimizer state reset |
| v33 | LR 2e-4 + KL-adaptive | Same as v32 | KL scheduler normalizes effective LR regardless |
| v34 | Granular ckpt, no scheduler | 3.22 | Confirms KL-adaptive is essential for warm-start |
| Manual `train.py` iteration | One-at-a-time HP search | Slow | Use AutoML for batch comparison |
| Bonus-first training | Start with large bonuses | Stuck at 0 | Sparse rewards → no learning signal |

---

## 9. Training Infrastructure

| Resource | Specification |
|----------|--------------|
| GPU | NVIDIA RTX 5080 (16 GB) |
| Backend | PyTorch (JAX not available) |
| Parallel envs | 2048 |
| Training speed | ~7,500-12,500 steps/sec |
| 50M steps | ~70 min |
| Eval (ranking) | `eval_checkpoint.py --rank <run_dir>` |
| Monitoring | `monitor_training.py --env <env> --deep` |
| AutoML | `automl.py --mode hp-search --hp-trials 15` |

---

## 10. File Inventory

| File | Purpose |
|------|---------|
| `starter_kit/navigation2/vbot/cfg.py` | Environment config + reward scales |
| `starter_kit/navigation2/vbot/rl_cfgs.py` | PPO hyperparameters |
| `starter_kit/navigation2/vbot/vbot_section011_np.py` | Environment implementation |
| `starter_kit/navigation2/vbot/xmls/scene_section011.xml` | MuJoCo scene definition |
| `starter_kit_schedule/checkpoints/vbot_navigation_section011/best_agent.pt` | **Deployed best checkpoint (v35)** |
| `starter_kit_schedule/configs/vbot_navigation_section011/v35_best_wpidx591.json` | Full config archive |
| `starter_kit_docs/navigation2/section011/REPORT_NAV2_section011.md` | Chronological experiment log (§1-§69) |
| `starter_kit_docs/navigation2/section011/Task_Reference.md` | Task-specific reference data |

---

## 11. Warm-Start Chain (Full Lineage)

### Legacy Chain (v15 → v35, warm-start)

```
Stage 15 (γ=0.999, λ=0.99 curriculum — wp_idx=1.98)
    │
    ▼  [fresh AutoML search, gradient-only rewards]
v23b-T7 (AutoML winner — wp_idx=0.58 @ 5M steps)
    │
    ▼  [add ordered targeting, multi-jump celebration]
v25 → v27 (structural improvements, no metric lift)
    │
    ▼  [re-enable discrete bonuses on trained walker]
v28 (smiley/red_packet/waypoint bonuses — wp_idx=5.05)
    │
    ▼  [boost later-phase bonus magnitudes]
v29 (boosted bonuses — wp_idx=5.86 @ agent_6000.pt)
    │
    │   agent_5000.pt (pre-peak, wp_idx=5.41)
    ▼                        │
v29 agent_6000.pt            │
    (PEAK — but stale        │
     optimizer direction)    │
                             ▼  [KL-adaptive scheduler + pre-peak start]
                          v35 (KL-adaptive + pre-peak — wp_idx=5.91 @ agent_6000.pt) ★ BEST (warm-start)
```

### Fresh-Start Chain (v46 → v48-T14 → v49, no warm-start)

```
v46 (fresh config: v35 rewards + PHASE_APPROACH=-1 + waypoint_facing 0.061→0.61)
    │
    ▼  [policy net (256,128,64) → (512,256,128) symmetric]
v47 (50M fresh — wp_idx_mean=1.40, max=7.0, 100% term rate)
    │
    ▼  [AutoML 15 trials × 15M, joint HP+reward search]
v48-T14 (AutoML winner — wp_idx_mean=0.484 @ 15M cold-start)
    │    Lighter penalties + stronger navigation pull
    │    LR=4.5e-4, entropy=0.0078, (512,256,128) both
    ▼
T14 100M deployment — ★ FAILED at 78% (backward-dragging local optimum)
    │    foot_clearance=0, LR crushed to 5.9e-5
    ▼
v49 (anti-local-optimum: +drag_foot_penalty, +stagnation_penalty)
    │    AutoML search space expanded: 29 params (was 27)
    ▼
v49 AutoML search (NEXT) → best trial → 50M+ long-horizon validation
```

---

*Generated 2026-02-17. Updated 2026-02-20 with v48-T14 failure analysis and v49 anti-local-optimum fixes. Source of truth for Section011 training methodology.*
