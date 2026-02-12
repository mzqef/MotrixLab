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

## 6. Next Steps

1. ⬜ **Fix reward budget** — Apply anti-laziness trifecta (alive=0.05, arrival≥200, add stair/bridge milestones)
2. ⬜ **Design stair-specific rewards** — Knee lift bonus, foot slip penalty, height progress
3. ⬜ **Design bridge-specific rewards** — Lateral deviation penalty, narrow path stability
4. ⬜ **Evaluate warm-start strategy** — Option A: from section011 best, Option B: fresh from Nav1
5. ⬜ **VLM visual analysis** — Capture frames of section011 policy on section012 terrain to assess transfer
6. ⬜ **AutoML reward weight search** — Tune stair/bridge/obstacle reward scales
7. ⬜ **Scoring zone analysis** — Identify section02 scoring zones from competition docs

---

*This report is append-only. Never overwrite existing content — the history is a permanent record.*
