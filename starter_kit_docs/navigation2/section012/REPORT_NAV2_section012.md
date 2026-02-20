# Section 012 Experiment Report — Stairs + Bridge + Spheres + Cones

**Date**: February 2026
**Environment**: `vbot_navigation_section012`
**Terrain**: Entry platform → stairs (left steep / right gentle) → arch bridge / spheres → stairs down → exit platform
**Competition**: MotrixArena S1 Stage 2, Section 2 — 60 points max
**Framework**: SKRL PPO, PyTorch backend, 2048 parallel envs, torch.compile (reduce-overhead)

---

## 1. Starting Point & Inherited State

### Task Overview

Section 012 is the hardest and highest-value section of Navigation2's obstacle course — a ~14.5m path through stairs, an arch bridge, scattered sphere and cone obstacles, and stair descent. Worth **60 pts** (57% of total Stage 2 score), this section demands precise foot placement, narrow-path traversal, and obstacle avoidance.

### Key Differences from Section 011

| Aspect | Section 011 | Section 012 |
|--------|------------|------------|
| **Terrain** | Bumps → 15° slope → high platform | Stairs (10-step) → bridge → spheres/cones → stairs down |
| **Elevation** | z=0 → 1.294 (monotonic up) | z=1.294 → 2.794 → 1.294 (up-then-down) |
| **Navigation** | Multi-waypoint + celebration spin | Forward traversal through complex terrain |
| **Distance** | ~10.3m | ~14.5m |
| **Episode** | 3000 steps (30s) | 6000 steps (60s) |
| **Points** | 20 pts | **60 pts** |
| **Key challenge** | Slope climbing | Stair climbing/descending + narrow bridge |

### Codebase State at Start

- Environment `VBotSection012Env` with 54-dim obs, 12-dim actions
- Default reward config: alive=0.3, arrival=80 — **broken budget** (see Section 3)
- No prior training runs for section012
- Warm-start candidate: section011 best checkpoint (slope climbing skills)

---

## 2. Terrain Analysis — Section 02

### Two-Route Layout

```
Y: 8.8   12.4  14.2  15~20  21.4  23.2  24.3
    |--entry--|--stairs up--|--bridge/spheres--|--stairs down--|--exit--|
    z=1.294   z→2.79         z≈2.86              z→1.37        z=1.294
```

#### Left Route (harder stairs, arch bridge)
| Element | Details |
|---------|---------|
| Left stairs up (10 steps) | x=-3.0, ΔZ≈0.15/step, z: 1.37→2.79 |
| Arch bridge | 23 segments, peak z≈2.86, width ~2.64m, with railings |
| Bridge support | 4 cylindrical pillars (R=0.4), 4 platform bases |
| Left stairs down (10 steps) | x=-3.0, z: 2.79→1.37 |

#### Right Route (easier stairs, obstacles)
| Element | Details |
|---------|---------|
| Right stairs up (10 steps) | x=2.0, ΔZ≈0.10/step, z: 1.32→2.29 |
| 5 spheres | R=0.75, scattered at y=15.8-19.7, z=0.8-1.2 |
| 8 cones (STL mesh) | Scattered obstacles |
| Right stairs down (10 steps) | x=2.0, z: 2.29→1.32 |

**End platform**: (0, 24.33, z≈1.294).

**Predicted difficulty**: Very Hard. Stairs require precise foot placement and knee lift. Bridge is narrow (~2.64m) with railings. Sphere obstacles (R=0.75m) block right path.

---

## 3. Reward Budget Analysis

### Current Config (BROKEN)

```
STANDING STILL for 6000 steps (alive=0.3):
  alive = 0.3 × 6000 = 1,800
  position_tracking ≈ 400
  Total standing ≈ 2,200+

COMPLETING TASK:
  arrival_bonus = 80

⚠️ STANDING WINS! Ratio: 27:1 — lazy robot strongly favored.
```

### TODO: Fix Required

Apply anti-laziness trifecta before training:
- Reduce alive_bonus to ≤0.05
- Increase arrival_bonus to ≥200
- Add terrain-specific progress rewards (stair completion, Y-axis checkpoints)
- Add termination penalty ≤-100

---

## 4. Training Experiments

*No experiments conducted yet. Section 012 training begins after section011 reaches stable performance.*

---

## 5. Current Config State

See `Task_Reference.md` in this folder for full reward config, PPO hyperparameters, and terrain details.

---

## 6. Architecture Redesign — Bridge-Priority State Machine (v1.0)

**Date**: Session 2, February 2026

### Motivation

The original section012 env was a single-target 54-dim environment with broken reward budget (alive=0.3 >> arrival=80). It lacked:
- Multi-phase navigation (no state machine)
- Bridge-specific routing (no left/right route commitment)
- Competition scoring alignment (no hongbao zones, no celebration)
- Warm-start compatibility (54-dim vs section011's 69-dim obs)

### Changes Made

| File | Change | Status |
|------|--------|--------|
| `cfg.py` | Complete rewrite of `VBotSection012EnvCfg` — added ScoringZones, BridgeNav, CourseBounds, new RewardConfig | ✅ Done |
| `vbot_section012_np.py` | Full rewrite (~900 lines) — 7-phase state machine, 69-dim obs, bridge sub-WPs, celebration jump, gait rewards, trunk_acc/torque sensing | ✅ Done |
| `rl_cfgs.py` | Aligned PPO config with section011 for warm-start (γ=0.999, λ=0.99, lr=5e-5) | ✅ Done |
| `automl.py` | Added REWARD_SEARCH_SPACE_SECTION012 (~35 params) + section012 scoring branch (max_wp=9.0) | ✅ Done |

### Bridge-Priority Strategy

Fixed left route: entry → left stairs → arch bridge → left stairs down → under-bridge collect → exit → celebrate.

7 navigation phases with 9 total waypoints (3 bridge sub-WPs + 2 under-bridge targets):
```
Phase 0: WAVE_TO_STAIR    → [-3, 12.3]
Phase 1: CLIMB_STAIR      → [-3, 14.5] z>2.3
Phase 2: CROSS_BRIDGE     → 3 sub-WPs (entry/mid/exit), z>2.3
Phase 3: DESCEND_STAIR    → [-3, 23.2]
Phase 4: COLLECT_UNDER    → nearest-uncollected under-bridge hongbao
Phase 5: REACH_EXIT       → [0, 24.33] r=0.8
Phase 6: CELEBRATION      → jump sequence
```

### Reward Budget (Fixed)

```
Standing: 0.05 × 3000 (conditional) ≈ 150
Completing: milestones(370) + approach(200) + alive(150) ≈ 720+
Ratio: 4.8:1 in favor of completing ✅
```

### 69-dim Observation Layout

Fully aligned with section011 v20 for warm-start checkpoint loading:
- 3 linvel + 3 gyro + 3 gravity + 12 joint_pos + 12 joint_vel + 12 last_actions
- 2 pos_error + 1 heading_error + 1 base_height + 1 celeb_progress + 4 foot_contact
- 3 trunk_acc + 12 torques_normalized = **69 total**

### Key Design Decisions

1. **Bridge-priority over right route**: Bridge has +10 pts (crossing) + +10 pts (hongbao) = 20 pts guaranteed. Right route only offers gentler stairs but obstacle dodging risk.
2. **Sequential phase enforcement**: State machine prevents phase-skipping exploits.
3. **Score-clear on termination**: 30% of accumulated milestones deducted on fall — prevents fall-reset farming.
4. **Celebration jump ported from section011**: Same IDLE→JUMP→DONE sub-state machine.
5. **Conditional alive bonus**: Only awarded when robot is upright (not fallen) — prevents lazy standing.

---

## 7. Next Steps

1. ⬜ **Smoke test** — Verify env creates, steps, obs shape (69), reward computes without errors
2. ⬜ **From-scratch baseline** — Run 5M step smoke test to verify learning signal exists
3. ⬜ **Warm-start from section011** — Load best section011 checkpoint, verify no shape mismatch
4. ⬜ **VLM visual analysis** — `capture_vlm.py --env vbot_navigation_section012` on initial policy
5. ⬜ **AutoML reward search** — `automl.py --hp-trials 15` to tune milestone/penalty weights
6. ⬜ **Curriculum bridge focus** — If stairs prove too hard from scratch, add intermediate env with only stairs (no bridge)

---

*This report is append-only. Never overwrite existing content — the history is a permanent record.*
