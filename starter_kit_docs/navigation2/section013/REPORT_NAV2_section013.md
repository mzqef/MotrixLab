# Section 013 Experiment Report — Gold Balls + Steep Ramp + High Step

**Date**: February 2026
**Environment**: `vbot_navigation_section013`
**Terrain**: Entry platform → 0.75m high step → 21.8° steep ramp → height field → 3 gold balls → final platform (z=1.494)
**Competition**: MotrixArena S1 Stage 2, Section 3 — 25 points max
**Framework**: SKRL PPO, PyTorch backend, 2048 parallel envs, torch.compile (reduce-overhead)

---

## 1. Starting Point & Inherited State

### Task Overview

Section 013 is the final section of Navigation2's obstacle course — a ~6.3m path featuring a major 0.75m high step, a steep 21.8° ramp, and 3 gold balls (R=0.75) blocking the approach to the final platform at z=1.494. Worth **25 pts**, this section tests the robot's ability to handle extreme terrain challenges: the 0.75m step is taller than the robot itself (~0.35m), and the 21.8° ramp is significantly steeper than Section 011's 15°.

### Key Differences from Previous Sections

| Aspect | Section 011 | Section 012 | Section 013 |
|--------|------------|------------|------------|
| **Primary challenge** | 15° slope | Stairs + bridge | 0.75m step + 21.8° ramp |
| **Elevation profile** | 0 → 1.294 | 1.294 → 2.794 → 1.294 | 1.294 → 1.494 |
| **Obstacles** | Height field bumps | Spheres + cones | 3 gold balls (R=0.75) |
| **Distance** | ~10.3m | ~14.5m | ~6.3m |
| **Episode** | 3000 steps | 6000 steps | 5000 steps |
| **Points** | 20 pts | 60 pts | 25 pts |
| **Difficulty** | Medium | Very Hard | Hard |

### Codebase State at Start

- Environment `VBotSection013Env` with 54-dim obs, 12-dim actions
- Default reward config: alive=0.3, arrival=60 — **broken budget** (see Section 3)
- No prior training runs for section013
- Warm-start candidate: section012 best checkpoint (stair climbing + obstacle skills)

---

## 2. Terrain Analysis — Section 03

```
Y: 24.3  26.3  27.6  29.3  31.2  32.3  34.3
    |--entry--|--step+ramp--|--hfield--|--gold balls--|--final--|--wall--|
    z=1.294   z↗?           z=1.294   z=0.844(balls) z=1.494
```

| Element | Center (x, y, z) | Size | Notes |
|---------|-------------------|------|-------|
| Entry platform | (0, 26.33, 1.044) | 5.0×1.0×0.25 box | z=1.294, from S02 |
| **0.75m high step** | (0, 27.58, 0.544) | 5.0×0.25×**0.75** box | Major obstacle — taller than robot! |
| **21.8° steep ramp** | (0, 27.62, 1.301) | Tilted 21.8° | Steeper than Section 01's 15° |
| Middle platform | (0, 29.33, 0.794) | 5.0×1.5×0.5 box | z=1.294, with height field |
| **3 gold balls** | x={-3, 0, 3}, y=31.23 | R=0.75 each | Blocking, ~2.5m gaps |
| **Final platform** | (0, 32.33, 0.994) | 5.0×1.5×0.5 box | **z=1.494** (highest point in course) |
| End wall | (0, 34.33, 2.564) | Blocking wall | Course end |

**Predicted difficulty**: Hard. The 0.75m step is the single hardest obstacle on the entire course — the robot cannot step over it naturally. The 21.8° ramp requires aggressive leaning. Gold balls with 2.5m gaps demand precise navigation.

---

## 3. Reward Budget Analysis

### Current Config (BROKEN)

```
STANDING STILL for 5000 steps (alive=0.3):
  alive = 0.3 × 5000 = 1,500
  position_tracking ≈ 300
  Total standing ≈ 1,800+

COMPLETING TASK:
  arrival_bonus = 60

⚠️ STANDING WINS! Ratio: 25:1 — lazy robot strongly favored.
```

### TODO: Fix Required

Apply anti-laziness trifecta before training:
- Reduce alive_bonus to ≤0.05
- Increase arrival_bonus to ≥150
- Add step/ramp milestone bonuses
- Add gold ball gap navigation rewards

---

## 4. Training Experiments

*No experiments conducted yet. Section 013 training begins after section012 reaches stable performance.*

---

## 5. Current Config State

See `Task_Reference.md` in this folder for full reward config, PPO hyperparameters, and terrain details.

---

## 6. Next Steps

1. ⬜ **Fix reward budget** — Apply anti-laziness trifecta (alive=0.05, arrival≥150, add step/ramp milestones)
2. ⬜ **Analyze 0.75m step traversal** — Can VBot physically climb a 0.75m step? VLM analysis needed
3. ⬜ **Design step-specific rewards** — Step-up detection, ramp climbing (steeper than Section 011)
4. ⬜ **Design gold ball avoidance** — Contact penalty, gap detection, observation extension
5. ⬜ **Evaluate warm-start strategy** — From section012 best (stair skills may transfer to step)
6. ⬜ **VLM visual analysis** — Capture frames of section012 policy on section013 terrain
7. ⬜ **AutoML reward weight search** — Tune step/ramp/ball reward scales
8. ⬜ **Scoring zone analysis** — Identify section03 scoring zones from competition docs

---

*This report is append-only. Never overwrite existing content — the history is a permanent record.*
