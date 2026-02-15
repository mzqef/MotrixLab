# Section 011 Task Reference — Slopes + Phase-Based Zones + Celebration Spin

> **This file contains task-specific concrete values** for Section 011 (Stage 2A — slopes, height field, ramp, high platform).
> For abstract methodology, see `.github/copilot-instructions.md` and `.github/skills/`.
> For full-course reference, see `starter_kit_docs/navigation2/long_course/Task_Reference.md`.

---

## Environment ID

| Environment ID | Terrain | Status |
|----------------|---------|--------|
| `vbot_navigation_section011` | Section01: START→hfield→ramp→platform (spawn y=-2.5) | **ACTIVE** — v20: 69-dim obs (trunk_acc + raw PD torques), sensor-driven penalties, code cleanup |

## Competition Scoring — Section 1 (20 pts total)

Source: `MotrixArena_S1_计分规则讲解.md`

```
Section 1 (20 pts):
├── 3 × 笑脸区 (smiley zones)     = 3×4 = 12 pts
│   ├── LEFT  smiley: center=(-3, 0), OBJ=V_ScorePoint_001a.obj
│   ├── CENTER smiley: center=( 0, 0), OBJ=V_ScorePoint_001b.obj
│   └── RIGHT smiley: center=( 3, 0), OBJ=V_ScorePoint_001c.obj
│   (位于height field凹凸地形上, y≈0处, 机器人经过即可得分)
├── 3 × 红包区 (red packet zones)  = 3×2 = 6 pts
│   ├── LEFT  red pkt: center=(-3, 4.4), OBJ=V_ScorePoint_002a.obj
│   ├── CENTER red pkt: center=( 0, 4.4), OBJ=V_ScorePoint_002b.obj
│   └── RIGHT red pkt: center=( 3, 4.4), OBJ=V_ScorePoint_002c.obj
│   (悬浮在"GO"字样上, 位于15°坡道上, y≈4.4处)
└── 庆祝动作 (celebration)         = 2 pts
    (在"2026"平台=高台顶部做出庆祝动作)
```

**关键规则:**
- 笑脸区: "坑洼地形的趣味得分区有三个笑脸区域, 机器人经过笑脸时每经过一个+4分"
- 红包区: "斜坡地形的趣味得分区有三个悬浮在'GO'字样上的祝福红包, 机器人经过一个红包得分+2分"
- 庆祝: "在'2026'平台处做出庆祝动作+2分"
- 起点: "初始点位置随机分布在'START'平台区域" (Adiban_001, y∈[-3.5, -1.5])

### Scoring Zone Positions (from OBJ mesh vertex extraction)

| Zone | OBJ File | XY Center | Bounding Box | Detection Radius |
|------|----------|-----------|--------------|------------------|
| Smiley LEFT | V_ScorePoint_001a.obj | (-3, 0) | x∈[-4,-2], y∈[-1,1] | 1.2m |
| Smiley CENTER | V_ScorePoint_001b.obj | (0, 0) | x∈[-1,1], y∈[-1,1] | 1.2m |
| Smiley RIGHT | V_ScorePoint_001c.obj | (3, 0) | x∈[2,4], y∈[-1,1] | 1.2m |
| Red Pkt LEFT | V_ScorePoint_002a.obj | (-3, 4.4) | x∈[-4,-2], y∈[3.4,5.4] | 1.2m |
| Red Pkt CENTER | V_ScorePoint_002b.obj | (0, 4.4) | x∈[-1,1], y∈[3.4,5.4] | 1.2m |
| Red Pkt RIGHT | V_ScorePoint_002c.obj | (3, 4.4) | x∈[2,4], y∈[3.4,5.4] | 1.2m |
| Celebration | High platform (Adiban_004) | (0, 7.83) | x∈[-2.5,2.5], z>1.0 | 1.5m |

### Scoring Tactics

| Element | Points | Location | Strategy |
|---------|--------|----------|----------|
| Smileys | 4 pts each (×3=12) | height field at y≈0, x=-3/0/3 | Robot passes through each zone (radius~1.2m) |
| Red packets | 2 pts each (×3=6) | Ramp at y≈4.4, x=-3/0/3 | Touch/pass through on way up ramp |
| Celebration | 2 pts | High platform y≈7.83, z>1.0 | Perform celebration spin at platform top |

## Terrain Description — Section 01

| Element | Center (x, y, z) | Size | Top z | Notes |
|---------|-------------------|------|-------|-------|
| Starting platform | (0, -2.5, -0.25) | 5.0×1.0×0.25 box | 0 | Flat start |
| Central platform | (0, 0, -0.25) | 5.0×1.5×0.25 box | 0 | With hfield (±5m×±1.5m, h=0-0.277m) |
| **Ramp (15°)** | (0, 4.48, 0.41) | 5.0×2.5×0.25 box | ~0.66 | Tilted ~15° around x-axis |
| **High platform** | (0, 7.83, 1.04) | 5.0×1.0×0.25 box | **1.294** | Top of ramp |
| Boundary walls | x=±5.25 | 0.25m thick, 2.45m tall | — | Left/right/rear |

**Robot spawn (section011 CURRENT)**: y=-2.5, z=0.5 (START platform, competition-correct spawn). Target: 4-phase zone collection → celebration. Distance: ~10.3m.

Challenge: height field bumps (max 0.277m at y≈0) + 3 smiley zones + 15° upslope + 3 red packet zones + platform edge transition + celebration at top.

## Phase-Based Zone Collection System (v15 + Stage 1B relaxation)

```python
# 4-phase zone collection matching competition scoring rules
PHASE_SMILEYS = 0       # Collect smileys (any order, nearest-first)
PHASE_RED_PACKETS = 1   # Collect red packets (gated on >=1 smiley, Stage 1B)
PHASE_CLIMB = 2         # Reach high platform (gated on all red packets)
PHASE_CELEBRATION = 3   # Celebration spin on platform

# Target selection: nearest uncollected zone in current phase
# Heading observation: wrap_angle(desired_heading - robot_heading)
# wp_idx = smileys_collected + red_packets_collected + platform_reached (0-7)
final_radius = 0.5      # platform target requires more precision

# Phase completion bonus (30.0) awarded when:
#   - All 3 smileys collected (phase 0 → 1)
#   - All 3 red packets collected (phase 1 → 2)

# Stage 1B CHANGES (critical for learning):
#   - Phase 0→1 gate: np.any(smileys_reached) — only 1 smiley needed (was np.all)
#   - Smiley collection: allowed in Phase 0 + Phase 1 (was Phase 0 only)
#   - Zone approach: phase-independent (smileys attract in Phase 0+1, red packets in Phase 1+)
# Rationale: strict "all 3" gate blocked 95% of robots from Phase 1.
#   Relaxing to "any 1" unlocked the full course for learning.
```

## Celebration Spin State Machine

```
CELEB_IDLE(0) → CELEB_SPIN_RIGHT(1) → CELEB_SPIN_LEFT(2) → CELEB_HOLD(3) → CELEB_DONE(4)
```

| Parameter | Value |
|-----------|-------|
| Spin angle per phase | 180° (π radians) |
| Heading tolerance | 0.3 rad (≈17°) |
| Speed limit during spin | 0.3 m/s |
| Hold duration | 30 steps |
| **Position anchor** | Robot XY recorded at CELEB_IDLE→SPIN_RIGHT transition (v14) |
| **Drift penalty** | `clip(-0.5 × drift², -0.3, 0.0)` — capped at -0.3/step (Stage 1C fix; was `-2.0 × drift²` uncapped) |

**Observation**: dim 53 encodes celebration progress:
- 0.0 = navigating, 0.25 = spin_right, 0.5 = spin_left, 0.75 = hold, 1.0 = done

## Terrain Traversal Strategy — Slopes

- **Height progress reward**: Reward z-axis gain (climbing) separately from Y-axis forward progress
- **Slope-aware orientation**: Don't penalize pitch on 15° slope — robot SHOULD lean forward
- **Platform edge**: Fall risk at ramp→platform transition; tilt termination at 65°

## Curriculum Stages

```
Stage 0: Baseline v15 (zone_approach=0, strict gate)
├── Run: 26-02-14_01-30-51-285668_PPO
├── Warm-start: section001 best checkpoint
├── LR=2.5e-4, max_steps=80M (killed at 50M — plateau)
├── Result: wp_idx=1.10 plateau — zone_approach=0 + strict gate blocked 95%

Stage 1: Enable zone_approach + reduce swing penalty
├── Run: 26-02-14_03-49-43-914529_PPO
├── zone_approach 0→3.0, swing_contact -0.05→-0.025, LR→1.5e-4
├── Result: wp_idx_max 3→6 but mean still 1.10 — gate still blocking

Stage 1B: Phase gate relaxation (BREAKTHROUGH)
├── Run: 26-02-14_05-06-31-614918_PPO
├── Phase 0→1: np.any instead of np.all (1 smiley sufficient)
├── Smileys collectible in Phase 0+1, zone approach phase-independent
├── Result: wp_idx 1.10→2.37, reached 0%→4.3%, celeb_drift=-494 (issue)

Stage 1C: Celebration drift fix (COMPLETE)
├── Run: 26-02-14_06-48-49-478937_PPO
├── celeb_drift: clip(-0.5*d², -0.3, 0) — was -2.0*d² uncapped
├── Result: wp_idx=2.90, reached=7.45%, net celeb=+66.1/ep

Stage 2: Fresh LR restart (COMPLETE)
├── Run: 26-02-14_09-21-25-795082_PPO
├── LR: 1.0e-4 (0.67× warm-start reduction), no code/reward changes
├── Result: wp_idx=3.10, reached=16.0%, net celeb=+162.4/ep
├── Best checkpoint: agent_24000.pt

Stage 3: Smiley incentive boost (COMPLETE — CURRENT BEST)
├── Run: 26-02-14_11-41-04-200078_PPO
├── smiley_bonus: 40→150, score_clear: 0.6→0.3 + cap -100, LR: 7e-5
├── Peak (iter 12500): wp_idx=3.76, reached=17.84%, celeb=7.88, term=62.4%
├── ENTROPY COLLAPSE after iter 12500: wp_idx degraded 3.76→3.10
├── Best checkpoint: agent_12000.pt (before collapse)
├── Lesson: smiley count unchanged (0.91/ep) — geometric barrier, not reward magnitude
```

## Termination Strategy — Hard/Soft Split

Grace period only protects **soft** terminations. **Hard** terminations always fire immediately.

| Category | Condition | Grace Protected? | Notes |
|----------|-----------|-----------------|-------|
| **HARD** | Severe tilt > 70° | ❌ Never | Robot clearly fallen/flipped |
| **HARD** | Out-of-bounds (CourseBounds) | ❌ Never | Left the course entirely |
| **HARD** | Joint velocity overflow | ❌ Never | Physics explosion |
| **HARD** | Joint acceleration > 80 rad/s² | ❌ Never | Single-step Δvel safety net (v14) |
| **HARD** | NaN in observations | ❌ Never | Simulation instability |
| **SOFT** | Base contact sensor > 0.01 | ✅ Yes (100 steps) | May be bumpy landing from spawn |
| **SOFT** | Medium tilt 50°-70° | ✅ Yes (100 steps) | May recover on uneven terrain |

### Grace Period Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| `grace_period_steps` | 100 | 1 second of simulation time (dt=0.01) |
| Purpose | Allow robot to land and stabilize after spawn | Only applies to soft terminations |
| **Anti-exploit** | Hard terminations bypass grace | Prevents fallen robots from collecting free alive_bonus |

### Score Clear on Termination

When the robot is terminated (fall/OOB), a penalty of **60% of accumulated bonus** is applied (not 100%). This is softer than full clearing to avoid excessive risk-aversion where robots refuse to attempt difficult terrain.

```python
score_clear_penalty = -0.6 × accumulated_bonus  # on termination only
```

## Alive Bonus — Segmented by Upright Posture (v14)

The alive_bonus uses a **segmented threshold** instead of continuous sqrt scaling:

```python
# gz = clip(-projected_gravity[:, 2], 0, 1)
# gz > 0.9 (tilt < 26°): full bonus
# gz > 0.7 (tilt < 45°): half bonus  
# gz < 0.7 (tilt > 45°): zero
gz = np.clip(-projected_gravity[:, 2], 0.0, 1.0)
upright_factor = np.where(gz > 0.9, 1.0, np.where(gz > 0.7, 0.5, 0.0))
alive_bonus = scale × upright_factor
```

| Posture | gz | upright_factor | alive_bonus (×0.15) |
|---------|----|---------------|---------------------|
| Perfectly upright | 1.0 | 1.0 | 0.15 |
| 15° tilt (on ramp) | 0.966 | 1.0 | 0.15 |
| 26° tilt | 0.899 | 0.5 | 0.075 |
| 45° tilt | 0.707 | 0.5 | 0.075 |
| 46° tilt | 0.694 | 0.0 | 0.0 |
| Lying on side (90°) | 0.0 | 0.0 | 0.0 |

**Rationale:** Sharp cliff at 45° aligns with 50° soft termination. Full bonus maintained on 15° ramp (gz=0.966>0.9).

## Current Reward Config (v20: v13 base + v14-v20 improvements)

```python
# ===== Navigation (v10 proven values) =====
forward_velocity: 3.0
waypoint_approach: 100.0
waypoint_facing: 0.15
position_tracking: 0.05
alive_bonus: 0.15               # 0.15×4000=600 (segmented by upright posture)

# ===== One-time Bonuses =====
waypoint_bonus: 100.0           # platform arrival
smiley_bonus: 150.0             # Stage 3: 3×150=450
red_packet_bonus: 20.0          # 3×20=60
celebration_bonus: 100.0
phase_completion_bonus: 30.0    # v15: completing all smileys / all red packets

# ===== Zone Attraction =====
zone_approach: 5.0              # v16: stronger lateral pull for all-smiley collection

# ===== Terrain / Height =====
height_progress: 12.0
traversal_bonus: 30.0           # milestone for ramp traversal

# ===== Foot / Gait =====
foot_clearance: 0.02
foot_clearance_bump_boost: 2.5  # v19: stronger lift cue in bump area (y ∈ [-1.5, 1.5])
stance_ratio: 0.08              # Stage 7B: minimal unconditional stance

# ===== Celebration (v16: jump) =====
jump_reward: 8.0                # continuous z elevation during celebration

# ===== Swing Contact Penalty =====
swing_contact_penalty: -0.025   # Stage 1: halved (was -0.05)
swing_contact_bump_scale: 0.6   # v19: reduce swing penalty in bump area

# ===== v20: Sensor-Driven Penalties =====
impact_penalty: -0.02           # trunk accelerometer impact (>15 m/s²)
torque_saturation: -0.01        # joint torque saturation (>90% forcerange)

# ===== Stability Penalties =====
orientation: -0.015
lin_vel_z: -0.06
ang_vel_xy: -0.01
torques: -5e-6
dof_vel: -3e-5
dof_acc: -1.5e-7
action_rate: -0.005
termination: -100.0             # STRONG fall deterrent
```

**Removed zero-weight keys (v20 cleanup):** `spin_progress`, `spin_hold`, `feet_contact_pattern`, `lateral_velocity`, `body_balance`, `fine_position_tracking`, `heading_tracking`, `distance_progress`, `approach_scale`, `arrival_bonus`, `stop_scale`, `zero_ang_bonus`, `near_target_speed`, `departure_penalty` — all were legacy 0.0 entries, never active.

**Disabled but kept as 0.0:** `height_approach`, `height_oscillation`, `slope_orientation` — v17 experiments, may be re-enabled.

### v14/v15 Code Improvements (formula-level, not scale changes)

| Change | Version | Description | Impact |
|--------|---------|-------------|--------|
| **Quadratic swing penalty** | v14 | `np.square(foot_vel)/10.0` vs linear `foot_vel/10.0` | Heavier at high velocity (10× at vel=10 rad/s) |
| **Segmented alive bonus** | v14 | gz>0.9:100%, gz>0.7:50%, else:0% (was sqrt) | Sharp cliff at 45° tilt instead of gradual |
| **Celebration position anchor** | v14 | `celeb_anchor_xy` + drift penalty `clip(-0.5×drift², -0.3, 0)` | Stage 1C: capped (was `-2.0×drift²` uncapped → -494/ep) |
| **Joint accel termination** | v14 | Hard terminate if Δvel > 80 rad/s per step | Physics explosion safety net |
| **Phase-based zone collection** | v15 | 4-phase gated collection (smileys→red packets→platform→celeb) | Matches competition rules, nearest-uncollected targeting |
| **Relaxed phase gate** | Stage 1B | Phase 0→1: `np.any(smileys_reached)` (was `np.all`) | Only 1 smiley needed; smileys collectible in Phase 0+1 |
| **Zone approach phase-indep** | Stage 1B | Smileys attract Phase 0+1, red packets Phase 1+ | No longer gated by current phase |
| **Drift penalty cap** | Stage 1C | `clip(-0.5×drift², -0.3, 0.0)` per step | Was `-2.0×drift²` uncapped → -494/ep exploding penalty |
| **Heading observation fix** | v15 | `wrap_angle(desired_heading - robot_heading)` | Was always pointing East, now points toward target |
| **Phase completion bonuses** | v15 | 30.0 reward per phase completed | Incentivizes completing each zone type fully |

### Reward Budget Audit (v15: v13 config + v14/v15 code)

```
STANDING STILL for 4000 steps:
  alive_bonus: 0.15 × 4000 × 1.0 = 600
  passive signals: ~320
  Total: ~920

COMPLETING COURSE in ~2500 steps:
  alive_bonus: 0.15 × 2500 = 375
  forward_velocity (3.0): ~600
  waypoint_approach (100): ~400
  waypoint_bonus: 100 (platform only, was 3×100)
  smiley_bonus: 3 × 40 = 120
  red_packet_bonus: 3 × 20 = 60
  phase_completion: 2 × 30 = 60 (smileys done + red packets done)
  celebration_bonus: 100
  height_progress: ~100
  traversal_bonus: ~60
  spin rewards: ~200
  Total: ~2,175

RATIO: 2175/920 = 2.4:1 ✅
```

## PD Control System

### Architecture

VBot uses software PD control over `<motor>` type XML actuators (raw torque). No cascaded PD issue.

```python
# _compute_torques()
action_scaled = actions * action_scale  # [-0.5, 0.5] rad deviation
target_pos = default_angles + action_scaled
torques = kp * (target_pos - current_pos) - kv * current_vel
torques = clip(torques, -torque_limits, torque_limits)
```

### PD Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| kp | 100.0 | Position gain (N·m/rad) |
| kv | 8.0 | Velocity damping (N·m·s/rad) |
| action_scale | 0.5 | Max deviation from default angles (rad) |
| torque_limits | [17, 17, 34]×4 | Aligned with XML `forcerange` |
| Action filter α | configurable | Exponential smoothing on raw actions |

### Torque Limits (XML forcerange — binding constraint)

| Joint | XML forcerange | Joint actuatorfrcrange | Notes |
|-------|---------------|----------------------|-------|
| Hip (FR/FL/RR/RL) | ±17 Nm | ±17 Nm | Abduction/adduction |
| Thigh (FR/FL/RR/RL) | ±17 Nm | ±17 Nm | Flexion/extension |
| Calf (FR/FL/RR/RL) | ±34 Nm | ±34 Nm | Knee flexion/extension |

### Saturation Regime

Max PD torque = kp × action_scale = 100 × 0.5 = 50 Nm, but clipped to 17 Nm (hip/thigh) or 34 Nm (calf). For large actions, the controller operates as a bang-bang at torque limits. Fine control only for small action magnitudes.

## PPO Hyperparameters (v13: Back to Basics)

| Parameter | v12 Value | v13 Value | Rationale |
|-----------|-----------|-----------|----------|
| learning_rate | 5e-4 | **1.5e-4** | Stage 1: 0.6× of v13 (warm-start LR reduction) |
| lr_scheduler | linear | linear | — |
| rollouts | 32 | **24** | v10 proven |
| learning_epochs | 4 | **8** | v10 proven (more learning per rollout) |
| mini_batches | 16 | **32** | v10 proven (less noisy gradients) |
| entropy_loss_scale | 0.01 | **0.005** | v10 proven (less random exploration) |
| ratio_clip | 0.2 | 0.2 | — |
| value_clip | 0.2 | 0.2 | — |
| max_env_steps | 80M | **50M** | Stage 1: reduced for faster iteration |
| discount_factor | 0.99 | 0.99 | — |
| lambda_param | 0.95 | 0.95 | — |
| grad_norm_clip | 1.0 | 1.0 | — |
| num_envs | 2048 | 2048 | — |
| checkpoint_interval | 500 | 500 | — |
| policy_net | (256,256,256) | **(256,128,64)** | v10 proven |
| value_net | (512,256,128) | (512,256,128) | — |

## Episode Configuration

| Parameter | Value |
|-----------|-------|
| max_episode_seconds | 40.0 |
| max_episode_steps | 4000 |
| grace_period_steps | 100 (hard/soft split) |
| action_scale | 0.5 |
| sim_dt | 0.01 |
| sim_substeps | 4 |

## Key Files

| File | Purpose |
|------|---------|
| `starter_kit/navigation2/vbot/cfg.py` | Environment config + reward scales for section011 |
| `starter_kit/navigation2/vbot/vbot_section011_np.py` | Section 01 environment implementation |
| `starter_kit/navigation2/vbot/rl_cfgs.py` | PPO hyperparameters for all navigation2 sections |
| `starter_kit/navigation2/vbot/xmls/scene_section011.xml` | Section 01 MJCF scene |
| `starter_kit/navigation2/vbot/xmls/0126_C_section01.xml` | Section 01 collision model |
