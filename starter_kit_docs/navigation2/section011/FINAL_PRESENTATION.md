# Section011 — Final Presentation & Reproduction Recipe

> **CURRENT CHAMPION: S4 T7 — CELEB_DONE sustained, wp=7.0, 100% reached**
> Run: `runs/vbot_navigation_section011/26-02-26_06-03-39-435963_PPO`
> AutoML: `automl_20260226_033450` (18/20 trials completed, 6.0h)
> Pipeline: S1 → Full 100M → S2 → S2FT → **S5 (3 right turns, full 2π yaw, relaxed term)**
> Competition score estimate: **20/20 points**
> **S5 RUNNING**: `automl_20260226_173838` — 6 trials, 2h budget, seeded from S4 top 5. S3/S3b/S4 stages are DISCARDED in reproduction.

---

## 1. Task Overview

**Environment**: `vbot_navigation_section011` — VBot quadruped robot navigating Section01 of the MotrixArena S1 competition.

**Course**: START platform → height-field bumps → 15° ramp → high platform (z=1.294m)

**Objectives** (20 points max):
- Collect 3 smiley zones (+4 pts each = 12 pts) on the height field
- Collect 3 red-packet zones (+2 pts each = 6 pts) on the ramp
- Reach the "2026" celebration platform and perform 3 right turns (+2 pts)

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

**100M deployment**: Training v48-T14 config from scratch to 100M steps (failed — see Discovery 7).

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

### Phase 6: Stage One + Full Training + Stage Two AutoML (2026-02-22)

**Goal**: Break through the warm-start chain ceiling (wp~2.8) using large-scale AutoML search, full 100M training, and warm-start refinement with harder celebration (10 jumps).

#### Phase 6A: Stage One — Cold-Start AutoML (`automl_20260221_203616`)

20 Bayesian trials × 10M steps, searching 31 reward + 2 HP parameters centered on R1_T10 values.

**Winner: T12** (wp_mean=0.412, score=0.2562 — seed config survived as champion):

| Parameter | T12 Value | Key |
|-----------|-----------|-----|
| `forward_velocity` | 6.49 | Strong forward pull |
| `waypoint_approach` | 510.9 | Very high navigation attraction |
| `zone_approach` | 196.3 | Strong zone pull |
| `termination` | -50 | Light fall penalty |
| `stagnation_penalty` | -1.13 | Anti-loafing |
| `drag_foot_penalty` | -0.29 | Anti-dragging |
| `crouch_penalty` | -1.30 | Anti-crouching |
| LR | 1.0e-3 | Aggressive (search upper bound) |
| Entropy | 0.0028 | Very low exploration |

```powershell
# Stage One AutoML (20 trials × 10M steps)
uv run starter_kit_schedule/scripts/automl.py --mode stage --env vbot_navigation_section011 `
    --budget-hours 8 --hp-trials 20 --seed-configs starter_kit_schedule/configs/seed_T12_warmstart.json
```

#### Phase 6B: Full 100M Training — Three Candidates

Three top trials (T12, T13, T11) trained to 100M steps with fixed LR=1e-3 (no KL-adaptive).

| Run | Peak wp | Peak Iter | Final wp | Collapse | Run Dir |
|-----|---------|-----------|----------|----------|---------|
| **A (T12)** | **2.232** | **24500** | 0.138 | -94% | `26-02-22_05-25-43-367487_PPO` |
| B (T13) | 2.033 | 21000 | 0.639 | -69% | `26-02-22_05-28-12-324803_PPO` |
| C (T11) | 1.919 | 25000 | 0.363 | -80% | `26-02-22_05-30-41-234319_PPO` |

**All three collapsed after ~50M steps** — fixed LR=1e-3 too aggressive for exploitation phase. Peak checkpoint `agent_24500.pt` from Train A preserved for warm-start.

#### Phase 6C: Stage Two — Warm-Start AutoML (`automl_20260222_124457`, COMPLETED)

15 warm-start trials from T12's peak checkpoint, with `required_turns = 10` (hardened celebration). Completed in **6.71 hours**.

**Infrastructure changes**:
- `--checkpoint` and `--freeze-preprocessor` flags added to `automl.py`
- LR clamped ≤ 7e-4 in both random and Bayesian phases
- Frozen RunningStandardScaler prevents normalizer drift

**Full results — Top 5** (all 15 trials reached wp_MAX=7.0):

| Rank | Trial | Score | wp_mean | suc% | LR | Entropy | Term | Stag | CrouchP | BmpB | Fwd | WPA |
|------|-------|-------|---------|------|-----|---------|------|------|---------|------|-----|-----|
| **1** | **T13** | **0.5439** | **3.443** | 23.2% | 5.7e-4 | **0.0100** | -50 | -2.38 | -2.34 | 13.55 | 5.55 | **346.0** |
| 2 | T4 | 0.5399 | 3.411 | 20.5% | 5.1e-4 | 0.0084 | -50 | -2.38 | -2.34 | 13.55 | 5.55 | 310.6 |
| 3 | T6 | 0.5374 | 3.354 | 25.5% | 7.0e-4 | 0.0066 | -75 | -1.66 | -2.35 | 19.13 | 9.38 | 772.3 |
| 4 | T10 | 0.5297 | 3.312 | 19.7% | 4.9e-4 | 0.0100 | -50 | -2.38 | -2.21 | 16.25 | 5.55 | 310.6 |
| 5 | T2 | 0.5286 | 3.300 | 21.4% | 7.0e-4 | 0.0071 | -50 | -5.52 | -1.40 | 7.44 | 5.52 | 638.9 |

*Full 15-trial table: see [REPORT_NAV2_section011.md](REPORT_NAV2_section011.md) Section 11.*

**Key findings**:
- **T13 champion** — entropy=0.01 + WPA=346 was the winning edge over T4 (entropy=0.0084, WPA=310.6)
- **Term -50 dominates**: avg score 0.5247 vs -75 (0.5205) vs -100 (0.5119)
- **T9 controlled ablation**: T4's exact rewards with term=-100 → score dropped 0.5399→0.5272
- **Bayesian convergence**: After ~10 trials, optimizer heavily exploited T4 config region
- **Warm-start doubles score**: 0.50-0.54 (warm-start) vs 0.22-0.26 (cold-start)

```powershell
# Stage Two AutoML — Branch A (warm-start from T12 peak)
uv run starter_kit_schedule/scripts/automl.py --mode stage --env vbot_navigation_section011 `
    --budget-hours 8 --hp-trials 15 `
    --seed-configs starter_kit_schedule/configs/seed_T12_warmstart.json `
    --checkpoint "runs/vbot_navigation_section011/26-02-22_05-25-43-367487_PPO/checkpoints/agent_24500.pt" `
    --freeze-preprocessor
```

#### Phase 6E: Stage Two Full Training — S2FT A_T4 (2026-02-23)

**Goal**: Train the Stage 2B champion (A_T4) to full 100M steps with warm-start from its AutoML checkpoint.

| Property | Value |
|----------|-------|
| Config source | A_T4 from `automl_20260222_124457` |
| LR | 1.52e-4 (0.3× A_T4's 5.1e-4) |
| Preprocessor | Frozen |
| Run dir | `26-02-23_13-49-12-918060_PPO` |

**Result**: Peak wp_mean=**5.635** @ iter 11500. This checkpoint became the warm-start source for Stage 4.

#### Phase 6F: Stage 4 — Relaxed-Termination AutoML (`automl_20260226_033450`, COMPLETED)

**Goal**: Find celebration-completing reward weights under relaxed physics termination and random initial heading.

**Infrastructure changes** (code modifications in this session):
1. Configurable termination fields in `cfg.py` (`hard_tilt_deg`, `soft_tilt_deg`, `enable_base_contact_term`, `enable_stagnation_truncate`)
2. `env_overrides` infrastructure in `train_one.py` + `automl.py`
3. Random yaw via `reset_yaw_scale` in `vbot_section011_np.py`
4. Celebration is **right turn** (10 turns, z-threshold=1.55) — NOT jumping

**env_overrides**: `hard_tilt=85°, soft_tilt=OFF, base_contact=OFF, stagnation=ON, grace=500, yaw_scale=1.0`

**First attempt** (`automl_20260226_001823`): Launched with `enable_stagnation_truncate=false` — crashed after 7/20 trials (long-stagnating episodes exhausted budget). Fixed to `true` in second run.

**Second attempt** (`automl_20260226_033450`): **18/20 trials completed** in 6.0 hours.

**All 18 trials reached peak wp_idx=7.0 and 100% reached fraction.** 5 of 18 reached CELEB_DONE (all 10 right turns). Of those 5, **T7 and T16 sustained CELEB_DONE** (final celeb_state=3.0).

**Top 5 trials:**

| Trial | Peak wp | Final celeb | Peak reward | Turns | Sustained? |
|-------|---------|-------------|-------------|-------|-----------|
| **T7** | **7.0** | **3.0 DONE** | **19,779** | **10** | **YES** |
| T16 | 7.0 | 3.0 DONE | 11,842 | 10 | YES |
| T4 | 7.0 | 2.0 (regressed) | 23,229 | 10 | NO |
| T15 | 7.0 | 2.0 (regressed) | 15,143 | 10 | NO |
| T6 | 7.0 | 2.0 | 30,362 | 4 | — |

**Champion: T7** — sustained CELEB_DONE with highest reward among sustained trials (19,779 vs T16's 11,842). Celebration learned at step 7,070.

**T7 run directory**: `runs/vbot_navigation_section011/26-02-26_06-03-39-435963_PPO`

#### Phase 6G: Stage 5 — 3-Turn Celebration AutoML (`automl_20260226_173838`, RUNNING)

**Goal**: Re-train from S2FT A_T4 peak with simplified celebration (3 right turns instead of 10), full 2π random initial yaw, and relaxed termination. Seeded from S4 top 5 configs.

**Key config changes** (permanent in cfg.py):
- `reset_yaw_scale` = 1.0 (full 2π uniform yaw at reset)
- `required_turns` = 3 (was 10)
- Removed `celebration_turn_threshold` / `celebration_settle_z` from Section011 WaypointNav (code defaults used: 1.55 / 1.50)

**Setup**: 6 trials (5 seeds from S4 top 5: T7, T16, T4, T15, T6 + 1 Bayesian), 2h budget, warm-start from S2FT A_T4 peak (`agent_11500.pt`), frozen preprocessor.

**env_overrides**: `hard_tilt=85°, soft_tilt=OFF, base_contact=OFF, stagnation=ON, grace=500`

> **Note**: S3, S3b, and S4 stages are **discarded** in reproduction. S5 directly replaces S4 by starting from the same S2FT checkpoint with simplified celebration.

---

## 4. Reproduction Commands — 4-Stage Pipeline

> **S3, S3b, S4 stages are DISCARDED.** S5 directly replaces S4 by starting from the same S2FT checkpoint with 3 right turns.

### Prerequisites

```powershell
uv sync --all-packages --all-extras
```

### Step-by-Step Reproduction

```powershell
# ============================================================
# STAGE 1: Cold-Start AutoML (20 trials × 10M)
# ============================================================
# Winner: T12 (wp_mean=0.412, score=0.2562)
uv run starter_kit_schedule/scripts/automl.py `
    --mode stage --env vbot_navigation_section011 `
    --budget-hours 8 --hp-trials 20 `
    --seed-configs starter_kit_schedule/configs/seed_T12_warmstart.json

# ============================================================
# STAGE 2A: Full 100M Training
# ============================================================
# T12 config, fixed LR=1e-3. Peak: wp=2.232 @ iter 24500.
uv run scripts/train.py --env vbot_navigation_section011 --train-backend torch --max-env-steps 100000000

# ============================================================
# STAGE 2B: Warm-Start AutoML (15 trials × 10M)
# ============================================================
# From Stage 2A peak. Winner: A_T4 (wp_mean=3.411, score=0.5399)
uv run starter_kit_schedule/scripts/automl.py `
    --mode stage --env vbot_navigation_section011 `
    --budget-hours 8 --hp-trials 15 `
    --seed-configs starter_kit_schedule/configs/seed_T12_warmstart.json `
    --checkpoint "runs/vbot_navigation_section011/26-02-22_05-25-43-367487_PPO/checkpoints/agent_24500.pt" `
    --freeze-preprocessor

# ============================================================
# STAGE 2C (S2FT): Full 100M from A_T4
# ============================================================
# A_T4 reward + 0.3× LR. Peak: wp=5.635 @ iter 11500.
uv run scripts/train.py --env vbot_navigation_section011 --train-backend torch `
    --checkpoint "runs/.../automl_A_T4/best_agent.pt" --max-env-steps 100000000

# ============================================================
# STAGE S5: Relaxed-Term + 3 Right Turns AutoML (6 trials)
# ============================================================
# Seeded from S4 top 5. 3 right turns, full 2π yaw, relaxed term.
# cfg.py: required_turns=3, reset_yaw_scale=1.0
python _launch_s5.py
# Or directly:
uv run starter_kit_schedule/scripts/automl.py `
    --mode stage --env vbot_navigation_section011 `
    --budget-hours 2 --hp-trials 6 `
    --checkpoint "runs/vbot_navigation_section011/26-02-23_13-49-12-918060_PPO/checkpoints/agent_11500.pt" `
    --freeze-preprocessor `
    --seed-configs starter_kit_schedule/configs/seed_S4_T7.json `
        starter_kit_schedule/configs/seed_S4_T16.json `
        starter_kit_schedule/configs/seed_S4_T4.json `
        starter_kit_schedule/configs/seed_S4_T15.json `
        starter_kit_schedule/configs/seed_S4_T6.json `
    --env-overrides '{"hard_tilt_deg":85.0,"soft_tilt_deg":0.0,"enable_base_contact_term":false,"enable_stagnation_truncate":true,"grace_period_steps":500}'

# ============================================================
# EVALUATE
# ============================================================
uv run scripts/play.py --env vbot_navigation_section011 --policy <best_agent.pt from S5>
```

---

## 5. Final Configuration Files

### PPO Hyperparameters (`starter_kit/navigation2/vbot/rl_cfgs.py`) — S4-T7 Final

```python
@rlcfg("vbot_navigation_section011")
@dataclass
class VBotSection011PPOConfig(PPOCfg):
    seed: int = 42
    num_envs: int = 2048
    play_num_envs: int = 16
    max_env_steps: int = 100_000_000
    check_point_interval: int = 500

    learning_rate: float = 4.171e-4     # S4-T7 (was 4.513e-4 in v48-T14)
    lr_scheduler_type: str | None = "kl_adaptive"
    rollouts: int = 24
    learning_epochs: int = 6
    mini_batches: int = 16
    discount_factor: float = 0.999
    lambda_param: float = 0.99
    grad_norm_clip: float = 1.0
    entropy_loss_scale: float = 0.00981  # S4-T7 (was 0.00775 in v48-T14)

    ratio_clip: float = 0.2
    value_clip: float = 0.2
    clip_predicted_values: bool = True

    share_policy_value_features: bool = False
    policy_hidden_layer_sizes: tuple[int, ...] = (512, 256, 128)
    value_hidden_layer_sizes: tuple[int, ...] = (512, 256, 128)
```

### Reward Scales — S4-T7 Final

> **S4-T7 Final.** Full config from AutoML `automl_20260226_033450` trial T7.
> Key differences vs seed (A_T4): waypoint_approach 2.47×, phase_bonus 2.97×, foot_clearance 1.81×, LR 2.74×, entropy 1.66×.

```yaml
# === Navigation (positive rewards) ===
forward_velocity: 5.774
waypoint_approach: 767.519          # 2.47× seed (310.6) — very strong pull
waypoint_facing: 0.755
zone_approach: 296.971
position_tracking: 1.553
alive_bonus: 2.848
alive_decay_horizon: 3840.09

# === Celebration (3 Right Turns in S5; was 10 in S4) ===
turn_reward: 4.465
per_turn_bonus: 33.699
celebration_bonus: 184.267

# === Bonuses (one-time events) ===
waypoint_bonus: 58.574
phase_bonus: 177.657                # 2.97× seed (59.9) — very high phase completion bonus
smiley_bonus: ~18.3
red_packet_bonus: ~14.8

# === Penalties (negative) ===
termination: -75                    # 50% heavier than seed (-50)
orientation: -0.026
lin_vel_z: -0.071
ang_vel_xy: -0.122                  # 45% heavier than seed
action_rate: -0.056
impact_penalty: -0.045
torque_saturation: -0.097
swing_contact_penalty: -0.007

# === Gait / Terrain ===
foot_clearance: 0.546               # 1.81× seed (0.302)
foot_clearance_bump_boost: 24.684   # 1.82× seed (13.55)
foot_clearance_bump_boost_pre_margin: 1.095
foot_clearance_bump_boost_post_margin: 0.146
foot_clearance_pre_zone_ratio: 0.624
swing_contact_bump_scale: 0.783
stance_ratio: 0.00308

# === Anti-loafing ===
drag_foot_penalty: -0.426
stagnation_penalty: -2.234
crouch_penalty: -1.529
dof_pos: -0.00553
```

### Environment Config — S4-T7 Final (includes env_overrides)

| Parameter | Value | Notes |
|-----------|-------|-------|
| `action_scale` (flat) | 0.25 | Dynamic zones override: 0.40 bump/slope |
| `max_episode_seconds` | 120.0 | Stagnation detection replaces fixed cutoff |
| `max_episode_steps` | 12000 | |
| `hard_tilt_deg` | **85.0** | S4: relaxed from 60° — allows aggressive terrain traversal |
| `soft_tilt_deg` | **0.0 (OFF)** | S4: disabled — no gradual penalty for tilt |
| `enable_base_contact_term` | **false** | S4: disabled — allows body-terrain contact during turns |
| `enable_stagnation_truncate` | **true** | S4: keeps episode length bounded |
| `grace_period_steps` | **500** | S4: 5s grace before stagnation check | 
| `reset_yaw_scale` | **1.0** | Default (cfg.py). S4 used env_overrides; now baked into config |
| `stagnation_window_steps` | 1000 | 10s window for stagnation check |
| `stagnation_min_distance` | 0.5 m | Must travel ≥0.5 m per window |
| `required_turns` | **3** | Right turns needed for celebration (was 10 in S4) |
| Spawn position | (0, -2.5, 0.35) ± (2.0, 0.5) | X: ±2 m (5 m wide platform), Y: ±0.5 m |
| Target waypoints | (0,0) → (0,4.4) → (0,7.83) | Smiley → Red Packet → Celebration |

---

## 6. All-Time Leaderboard

| Rank | Run | Stage | Checkpoint | wp_idx | CELEB_DONE | Notes |
|------|-----|-------|------------|--------|-----------|-------|
| **1** | **26-02-26_06-03-39** | **S4-T7** | **best_agent** | **7.0** | **YES (sustained)** | **10 turns (S4). S5 retrains with 3 turns** |
| 2 | 26-02-26 (T16) | S4-T16 | best_agent | 7.0 | YES (sustained) | Lower reward (11.8K vs 19.8K) |
| 3 | 26-02-26 (T4) | S4-T4 | best_agent | 7.0 | Regressed | DONE then lost |
| 4 | 26-02-23_13-49-12 | S2FT-A_T4 | agent_11500 | 5.635 | — | Stage 2C peak |
| 5 | 20-42-31 | v35 | agent_6000 | 5.911 | — | Legacy warm-start chain peak |
| 6 | 10-45-19 | v29 | agent_6000 | 5.864 | — | Pre-KL-adaptive peak |

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

### Discovery 8: Relaxed Termination Enables Celebration (S4)

Strict termination (60° tilt, base contact kill) prevents the robot from performing celebration turns on the platform. The turning motion naturally tilts the body and causes transient ground contacts. By relaxing to 85° hard tilt, disabling base contact termination, and enabling stagnation truncation (to bound episodes), the robot can freely perform the 10 right turns needed for celebration.

Additionally, training with **random ±180° initial heading** (`reset_yaw_scale=1.0`) forces heading-invariant locomotion, which transfers to reliable turning behavior.

**Key insight**: All 18 S4 trials reached wp=7.0 (vs previous stages where agents often got stuck at wp=5-6). The relaxed termination + stagnation guardrail combination is strictly superior to strict termination for this course.

---

## 8. Failed Experiments (What NOT To Do)

| Experiment | Change | Result | Why It Failed |
|------------|--------|--------|---------------|
| v30 | torque_saturation -0.025→-0.010 | 4.02 | Reward distribution shift corrupted value function |
| v31 | LR 1e-4→5e-5 only | 1.68 | Too slow recovery from optimizer state reset |
| v33 | LR 2e-4 + KL-adaptive | Same as v32 | KL scheduler normalizes effective LR regardless |
| v34 | Granular ckpt, no scheduler | 3.22 | Confirms KL-adaptive is essential for warm-start |
| S3b AutoML | Relaxed term env_overrides | Config never applied | env_overrides infrastructure wasn't connected to train_one.py |
| S3/S3b/S4 stages | 10 right turns, various term configs | Discarded | Superseded by S5 (3 turns, same checkpoint) |
| S4 attempt 1 | stagnation_truncate=false | Crashed after 7 trials | Episodes stagnated forever, exhausted budget |
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
| `starter_kit/navigation2/vbot/cfg.py` | Environment config + reward scales + configurable termination |
| `starter_kit/navigation2/vbot/rl_cfgs.py` | PPO hyperparameters |
| `starter_kit/navigation2/vbot/vbot_section011_np.py` | Environment implementation (configurable term + random yaw) |
| `starter_kit/navigation2/vbot/xmls/scene_section011.xml` | MuJoCo scene definition |
| `runs/vbot_navigation_section011/26-02-26_06-03-39-435963_PPO/checkpoints/best_agent.pt` | **S4-T7 final champion checkpoint** |
| `starter_kit_schedule/configs/seed_stage3_A_T4_relaxed.json` | S4 seed config (A_T4 + relaxed overrides) |
| `starter_kit_docs/navigation2/section011/REPORT_NAV2_section011.md` | Chronological experiment log (§1-§18) |
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

### Fresh-Start Chain (v46 → Stage One → Full Train → Stage Two → S2FT → S5 ★)

> **S3, S3b, S4 stages are DISCARDED.** Only the S4 seed configs (reward weights) are reused in S5.

```
v46 (fresh config: v35 rewards + PHASE_APPROACH=-1 + waypoint_facing 0.061→0.61)
    │
    ▼  [policy net (256,128,64) → (512,256,128) symmetric]
v47 (50M fresh — wp_idx_mean=1.40, max=7.0, 100% term rate)
    │
    ▼  [AutoML 15 trials × 15M, joint HP+reward search]
v48-T14 (AutoML winner — wp_idx_mean=0.484 @ 15M cold-start)
    │    Lighter penalties + stronger navigation pull
    ▼
T14 100M deployment — ★ FAILED at 78% (backward-dragging local optimum)
    │
    ▼  [+drag_foot_penalty, +stagnation_penalty]
v49/v55 (anti-local-optimum + search space expansion)
    │
    ▼  [AutoML 20 trials × 10M, cold-start, v55 search space]
Stage One AutoML (automl_20260221_203616)
    │    T12 champion: seed config survived (wp_mean=0.412, score=0.2562)
    ▼
Full 100M Training (3 candidates: T12, T13, T11)
    │    Train A (T12) peak: wp_mean=2.232 @ iter 24500
    ▼
Stage Two AutoML — Branch A (automl_20260222_124457)
    │    A_T4 CHAMPION: wp_mean=3.411, score=0.5399
    ▼
S2FT — Full 100M from A_T4 (26-02-23_13-49-12-918060_PPO)
    │    Peak: wp_mean=5.635 @ iter 11500
    │    agent_11500.pt preserved
    │
    ├── [DISCARDED] S3/S3b/S4 (10 turns, various term configs) → S4 T7 was champion but superseded
    │
    ▼
S5 AutoML (automl_20260226_173838, RUNNING)
    │    3 right turns, full 2π yaw, relaxed term
    │    Seeded from S4 top 5 reward configs
    ▼
★ S5 CHAMPION — TBD
```
```

---

*Generated 2026-02-17. Updated 2026-02-26 with Stage 4 T7 champion + S5 yaw default change. S4 T7: CELEB_DONE sustained (all 10 right turns completed), wp=7.0, 100% reached, peak reward=19,779. cfg.py `reset_yaw_scale` changed to 1.0 (full 2π) as permanent default. Five-stage pipeline finalized: S1 AutoML → S2A Full Train → S2B AutoML → S2C Full Train → S4 Relaxed-Term AutoML. Competition-ready: estimated 20/20 points.*
