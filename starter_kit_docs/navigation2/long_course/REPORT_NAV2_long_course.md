# Long Course Experiment Report — Full 34m Three-Section Navigation

**Date**: February 2026
**Environment**: `vbot_navigation_long_course`
**Terrain**: START → Section01 (hfield+ramp) → Section02 (stairs+bridge) → Section03 (balls+ramp) → FINISH
**Competition**: MotrixArena S1 Full Course — 105 points max (20 + 60 + 25)
**Framework**: SKRL PPO, PyTorch backend, 2048 parallel envs, torch.compile (reduce-overhead)

---

## 1. Starting Point & Inherited State

### Task Overview

Long Course is the competition submission environment — a 34m path from START through all 3 sections to the FINISH platform. Uses a waypoint navigation system (7 waypoints) to guide the robot through terrain transitions. Scoring = Section 1 (20 pts) + Section 2 (60 pts) + Section 3 (25 pts) = 105 pts total.

### Prerequisite Training

Long Course should ONLY be trained after completing per-section training:
1. `vbot_navigation_section011` — slopes, hfield, ramp (20 pts)
2. `vbot_navigation_section012` — stairs, bridge, obstacles (60 pts)
3. `vbot_navigation_section013` — balls, steep ramp, high step (25 pts)

Transfer chain: section011 → section012 → section013 → long_course (warm-start with LR reduction).

### Current Reward Budget Status

```
STANDING STILL for 9000 steps (alive=0.5):
  alive = 0.5 × 9000 = 4,500
  passive (heading/position tracking) ≈ 2,000–5,000
  Total standing ≈ 6,500–9,500

COMPLETING TASK:
  7 waypoints × 30.0 = 210
  arrival_bonus = 100
  Total completion bonuses = 310

⚠️ STANDING (6,500+) >> COMPLETION (310) — Ratio ~21:1
LAZY ROBOT GUARANTEED — Must fix before training.
```

### Codebase State at Start

- Environment `VBotLongCourseEnv` with 54-dim obs, 12-dim actions
- 7-waypoint navigation system with auto-advance
- Default reward budget is broken (alive dominates arrival by 21:1)
- No prior training runs for long_course

---

## 2. Experiments

*(No experiments run yet — awaiting per-section training completion)*

### Experiment Template

```
### Experiment LC-N: <Title>

**Date**: YYYY-MM-DD
**Run**: `runs/vbot_navigation_long_course/<run_id>`
**Config**: <what changed>
**Steps**: <total steps>
**Warm-start**: <checkpoint path if any>

#### Results

| Metric | Value |
|--------|-------|
| Best reward | |
| Waypoints reached (avg) | |
| Final reached % | |
| Max Y-progress | |
| Episode length (avg) | |

#### Observations

-

#### Lessons Learned

-
```

---

## 3. Key Findings & Lessons

*(To be populated as experiments run)*

---

## 4. Next Steps

- [ ] Complete per-section training: section011, section012, section013
- [ ] Fix reward budget: alive_bonus ≤ 0.05, increase waypoint/arrival bonuses
- [ ] Run first warm-start long_course experiment from section013 checkpoint
- [ ] Validate waypoint progression rate across all 7 waypoints
- [ ] VLM visual analysis of section transitions (critical failure points)
