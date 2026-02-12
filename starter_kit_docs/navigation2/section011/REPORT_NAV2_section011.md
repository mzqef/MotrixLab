# Section 011 Experiment Report — Slopes + Multi-Waypoint + Celebration Spin

**Date**: February 2026
**Environment**: `vbot_navigation_section011`
**Terrain**: START platform → height field → 15° ramp → high platform (z=1.294)
**Competition**: MotrixArena S1 Stage 2, Section 1 — 20 points max
**Framework**: SKRL PPO, PyTorch backend, 2048 parallel envs, torch.compile (reduce-overhead)

---

## 1. Starting Point & Inherited State

### Task Overview

Section 011 is the first section of Navigation2's obstacle course — a 10.3m path from the START platform through height field bumps, up a 15° ramp, to a high platform at z=1.294m. Scoring includes 3 smiley zones (12 pts), 3 red packet zones (6 pts), and a celebration action (2 pts) = 20 pts total.

### Key Differences from Navigation1

| Aspect | Navigation1 | Section 011 |
|--------|------------|------------|
| **Terrain** | Flat circular platform | Linear: bumps → 15° slope → high platform |
| **Navigation** | Radial to center | Multi-waypoint (3 WPs) → celebration spin |
| **Elevation** | z=0 everywhere | z=0 → 1.294 |
| **Scoring** | 20 pts (reach center) | 20 pts (smileys + red packets + celebration) |
| **Episode length** | 1000 steps | 3000 steps |

### Codebase State at Start

- Environment `VBotSection011Env` with 54-dim obs, 12-dim actions
- Default reward config: alive=1.0, arrival=50 — broken budget (80:1 ratio)
- No prior training runs for Navigation2

---

## 2. Terrain Analysis — Section 01

```
Y: -3.5    0    4.5   7.8
    |---flat---|--ramp--|--high platform--|
    z=0        z=0.41   z=1.294
    
    Challenge: Height field bumps (max 0.277m) + 15° upslope + platform edge
```

**Predicted difficulty**: Medium. 15° slope is manageable, but height field bumps can trip a flat-ground policy. Platform edge step-up requires foot clearance and balance.

---

## 3. Reward Budget Analysis

### Initial Config (Broken)

```
STANDING STILL for 4000 steps (alive=1.0):
  alive = 1.0 × 4000 = 4,000
  position_tracking ≈ 640
  Total standing ≈ 4,640

COMPLETING TASK:
  arrival_bonus = 50
  Total completing ≈ 250

⚠️ STANDING WINS BY ~4,390! Ratio: 80:1
```

### Fixed Config (v3 — Current)

```
STANDING STILL for 3000 steps (alive=0.05):
  alive = 0.05 × 3000 = 150
  position_tracking ≈ 570
  Total standing ≈ 720

COMPLETING FULL COURSE:
  waypoint_bonus = 3 × 25 = 75
  smiley_bonus = 3 × 20 = 60
  red_packet_bonus = 3 × 10 = 30
  celebration_bonus = 30
  arrival_bonus = 160
  traversal = 2 × 15 = 30
  spin rewards ≈ 450
  wp_approach + forward ≈ 300-500
  Total completing ≈ 1,135-1,335

✅ COMPLETING > STANDING — incentive aligned
```

---

## 4. Training Experiments

### Experiment 011-1: Scratch Training on Wrong Terrain (DISCARDED)

**Date**: Feb 11, 2026
**Discovery**: `scene_section011.xml` referenced `0131_C_section01.xml` (flat cylinder, NO terrain!) instead of `0126_C_section01.xml` (real terrain: height field + 15° ramp + high platform z=1.294). ALL prior section011 ideas were based on wrong terrain.

**Fix**: Updated XML to `0126_C_section01.xml`, fixed sensor refs from `C_B_BODY` to `C_ground_root`.

### Experiment 011-2: First Real-Terrain Run (26-02-11_22-32-18)

**Config**: Scratch training, spawn at y=-2.4, Nav1 proven reward (approach=50, arrival=160, alive=0.08, term=-75)
**Result**: Catastrophic collapse
- Episode length: 115 → 3 steps over 3000 iterations
- 0% reached, mean reward -71.7
- 25% termination rate per step
- Diagnosis: Height field bumps (y=-1.5 to +1.5, max 0.277m) tripping robots immediately

### Experiment 011-3: Warm-Start from Nav1 (26-02-11_22-55-40)

**Config**: Warm-start from Nav1 best (`stage3_continue_agent1600_reached100_4608.pt`), spawn y=-2.4, LR=1.5e-4 (0.5×), entropy=0.012, ratio_clip=0.15, optimizer reset
**Result**: Still collapsed
- Episode length: 9.5 → 3.6 steps
- distance_to_target_min: 0.80m (some nav progress) but 0% reached
- Diagnosis: Same terrain issue — bumps before ramp base kill flat-ground policy

### Spawn Fix: Move Past Bumps (Temporary)

**Analysis**: Zero-action test at y=-2.4 = 200 steps, 0 terminations. The POLICY causes crashes on bumps, not the terrain itself. Solution: skip the height field for initial curriculum.

**Fix**: Moved spawn from y=-2.4 → y=2.0 (past height field, at ramp base). Distance shortened from 10.2m → 5.8m.

### Experiment 011-4: Warm-Start + Spawn Fix + Terrain Rewards (26-02-11_23-26-12)

**Run ID**: `26-02-11_23-26-12-584658_PPO`
**Checkpoint**: Warm-start from Nav1 best, optimizer reset
**Spawn**: y=2.0 (past height field bumps, ramp base), ±0.3m randomization
**Target**: y=7.8 (platform top center, z=1.294), distance=5.8m
**PPO**: LR=1.5e-4 (linear), entropy=0.012, ratio_clip=0.15, epochs=6

**Config Changes**:
1. **Height progress reward** (`height_progress=8.0`): Reward positive z-delta per step
2. **Traversal milestones** (`traversal_bonus=15.0` each): Mid-ramp (y>4, z>0.3) + ramp-top (y>6.5, z>0.8)
3. **Stricter fall detection**: Tilt threshold 75° → 65°

**Training Results** (50M steps target):

| Iter | Env Steps | Ep Len | Reward Mean | Dist Mean | Dist Min | Reached% | Notes |
|------|-----------|--------|-------------|-----------|----------|----------|-------|
| 500 | 1.0M | 40.4 | -72.4 | 5.13 | 2.00 | 0.0% | Early exploration |
| 1000 | 2.0M | 23.4 | -15.2 | 3.63 | 0.002 | 0.7% | **First reaches!** |
| 1500 | 3.1M | 31.0 | 30.3 | 1.27 | 0.0004 | **56.3%** | Breakthrough |
| 3000 | 6.1M | 15.8 | 33.0 | 1.64 | 0.0004 | **60.3%** | Solid |
| 5000 | 10.2M | **141.5** | **973.3** | 2.32 | 0.0002 | 37.5% | Reward climbing |

**Key Observations:**
1. First-ever successful reaches on real terrain — 0.7% at iter 1000, surging to 56% by 1500
2. Reward curve: -72 → +973 (iter 0→5000), no sign of plateauing
3. Survival mastery: Episode length 40 → 141 steps
4. Reached% dilution: Apparent drop from 60%→37% is artifact — longer episodes dilute per-step fraction
5. Terrain rewards working: height_progress steady, traversal_bonus growing, forward_velocity increasing
6. Stop behavior learned
7. No catastrophic collapse — warm-start + spawn fix + terrain rewards work

**Conclusion**: Run 011-4 is the **first viable training run** for section011.

---

## 5. Competition Scoring Correction (Feb 12, 2026)

### Critical Discoveries

#### 1. START Point Was WRONG

| | Old (wrong) | New (correct) |
|--|-------------|---------------|
| Spawn Y | 2.0 (past height field) | **-2.5** (START platform) |
| Platform | Adiban_003 (ramp base) | **Adiban_001** (START box) |
| Y range | N/A | **[-3.5, -1.5]** |
| Distance | 5.8m | **~10.3m** |

#### 2. Missing Scoring Zones (60% of Section Score!)

3 smiley zones (12 pts) and 3 red packet zones (6 pts) were completely missing from the reward function. Fixed by adding `ScoringZones` dataclass and zone detection in `update_state()`.

#### 3. Missing Celebration Pose (+2 pts)

Implemented as celebration spin state machine (right 180° → left 180° → hold 30 steps).

### Code Changes Made

**cfg.py**: Spawn corrected, episode=3000 steps, alive=0.05, added ScoringZones, new reward scales for zones + celebration.

**vbot_section011_np.py**: Added foot contact detection, swing contact penalty, scoring zone detection, multi-waypoint navigation, celebration spin state machine.

---

## 6. Multi-Waypoint Navigation Redesign (Feb 12, 2026)

| Component | Old (v2) | New (v3) |
|-----------|---------|---------|
| Navigation | Single target (0, 7.8) | 3 waypoints: (0,0)→(0,4.4)→(0,7.83) |
| Waypoint switching | N/A | Auto-advance when within radius |
| Celebration | Front thigh deviation | Spin state machine |
| Observation dim 53 | `stop_ready` (binary) | `celeb_progress` (0→1) |
| Episode end | Timeout only | Timeout OR CELEB_DONE |

### Experiment 011-5: Multi-Waypoint + Celebration Spin Training

**Run ID**: `26-02-12_02-29-54-982761_PPO`
**Checkpoint**: Warm-start from Nav1 best, optimizer reset
**Spawn**: y=-2.5 (competition START platform), ±0.5m randomization
**Config**: v3 multi-waypoint + celebration spin
**PPO**: LR=1.5e-4 (linear), entropy=0.012, ratio_clip=0.15, epochs=6
**Steps**: 50M target

**Smoke test**: obs shape (4,54) ✅, waypoints loaded ✅, celebration state machine ✅

**Status**: Training in progress. Awaiting results for waypoint progression, celebration success rate, and reward curve.

**Key metrics to watch**:
- `wp_idx_mean`: Waypoints reached
- `celeb_state_mean`: Celebration progress
- `smiley_bonus` / `red_packet_bonus`: Zone collection
- `spin_progress` / `spin_hold`: Celebration rewards

---

## 7. Current Config State

See `Task_Reference.md` in this folder for full reward config, PPO hyperparameters, and terrain details.

---

## 8. Next Steps

1. ✅ ~~Fix reward budgets~~ — Done (alive=0.05, arrival=160)
2. ✅ ~~Launch Stage 2A training~~ — Run 011-4 completed
3. ✅ ~~Fix start point~~ — Corrected to y=-2.5 (competition START)
4. ✅ ~~Add scoring zone rewards~~ — Smileys, red packets, celebration
5. ✅ ~~Add swing-phase contact penalty~~
6. ✅ ~~Launch 011-5 with multi-waypoint + celebration spin~~
7. ⬜ **VLM visual analysis on 011-5 at 25M+** — Diagnose waypoint + celebration behavior
8. ⬜ **AutoML reward weight search** — Tune waypoint_bonus, spin_progress, spin_hold scales
9. ⬜ **Evaluate warm-start strategy for section012** — From 011-5 best or fresh from Nav1
10. ⬜ **Height field traversal curriculum** — Can the robot handle bumps from y=-2.5 spawn?

---

*This report is append-only. Never overwrite existing content — the history is a permanent record.*
