# Section 012 Experiment Report — Stairs + Bridge + Hongbaos

**Initialized**: February 27, 2026
**Environment**: `vbot_navigation_section012`
**Terrain**: Entry platform → right stairs → stone hongbaos → under-bridge → bridge out-and-back → exit → celebration
**Competition**: MotrixArena S1 Stage 2, Section 2 — 60 points max
**Framework**: SKRL PPO, PyTorch backend, 2048 parallel envs, torch.compile (reduce-overhead)
**Celebration**: Walk to X-axis endpoint + sit (v58, same as section011)

---

## 1. Starting Point

### Task Overview

Section 012 is the highest-value section of Navigation2 — a ~14.5m path through stairs, an arch bridge, scattered sphere/cone obstacles, and stair descent. Worth **60 pts** (57% of total Stage 2 score). Uses ordered multi-waypoint full-collection with 15 waypoints.

### Key Differences from Section 011

| Aspect | Section 011 | Section 012 |
|--------|------------|------------|
| **Terrain** | Bumps → 15° slope → high platform | Stairs (10-step) → bridge → spheres/cones → stairs down |
| **Elevation** | z=0 → 1.294 (monotonic up) | z=1.294 → 2.794 → 1.294 (up-then-down) |
| **Navigation** | Multi-waypoint + celebration (walk+sit) | 15 ordered waypoints + celebration (walk+sit) |
| **Distance** | ~10.3m | ~14.5m actual, ~25m+ route |
| **Episode** | 3000 steps (30s) | 6000 steps (60s) |
| **Points** | 20 pts | **60 pts** |
| **Celebration** | Walk to X endpoint + sit | Walk to X endpoint + sit (identical FSM) |
| **Key challenge** | Slope climbing | Stair climbing/descending + narrow bridge |

### Current Codebase State

- Environment `VBotSection012Env` — 69-dim obs, 12-dim actions, ordered waypoint navigation
- Reward config: BASE_REWARD_SCALES shared across sections (v48-T14 AutoML winner base)
- Celebration: v58 walk+sit FSM (IDLE→WALKING→SITTING→DONE), celeb_x_target=(4.0, 24.33)
- 15 ordered waypoints: WP0 right_approach → WP14 exit_platform (goal)
- Hard termination: hard_tilt_deg=70°, enable_base_contact_term=True
- Warm-start compatible: 69-dim obs aligned with section011

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
| Left stairs down (10 steps) | x=-3.0, z: 2.79→1.37 |

#### Right Route (easier stairs, obstacles)
| Element | Details |
|---------|---------|
| Right stairs up (10 steps) | x=2.0, ΔZ≈0.10/step, z: 1.32→2.29 |
| 5 spheres (R=0.75) | Scattered at y=15.8-19.7 |
| 8 cones (STL mesh) | Scattered obstacles |
| Right stairs down (10 steps) | x=2.0, z: 2.29→1.32 |

**Exit platform**: (0, 24.33, z≈1.294). **Spawn**: (2.0, 12.0, 1.8) ±(0.5, 0.3)m.

---

## 3. Architecture — Ordered Multi-Waypoint Full-Collection

### Route Strategy (Right-Side First)

```
WP  0: right_approach       → Guide robot to right side
WP  1: stair_top            → Climb right stairs (ΔZ≈0.10/step)
WP 2-6: stone_hongbao_1~5   → Zigzag through 5 stone hongbaos (+15 pts)
WP 7-8: under_bridge         → Collect 2 under-bridge hongbaos (+10 pts)
WP  9: bridge_climb_base    → Walk to far stair base
WP 10: bridge_far_entry     → Climb left stairs, enter bridge (z>2.3)
WP 11: bridge_hongbao       → Collect bridge hongbao (+10 pts)
WP 12: bridge_turnaround    → Turn around on bridge (z>2.3)
WP 13: bridge_descent       → Descend stairs to ground
WP 14: exit_platform        → Goal → triggers celebration
```

### Celebration FSM (v58: Walk + Sit)

Identical to section011:
```
CELEB_IDLE → CELEB_WALKING → CELEB_SITTING → CELEB_DONE
```
- **WALKING**: Robot walks to celeb_x_target = (4.0, 24.33). Delta-based approach reward.
- **SITTING**: Robot lowers z below celeb_sit_z=1.40. Counter increments each step where z < threshold. After celeb_sit_steps=30 steps → DONE.
- **Rewards**: celeb_walk_approach (200.0), celeb_walk_bonus (30.0), celeb_sit_reward (5.0/step), celebration_bonus (50.0).

### Reward Budget

```
STANDING STILL for 6000 steps:
  alive_bonus (decayed): 1.013 × ~2383 effective ≈ 150-300
  No milestone bonuses
  Total standing ≈ 300 max

COMPLETING ALL 15 WPs + CELEBRATION:
  alive_bonus: ~300
  waypoint_approach (dominant): ~500+ cumulative
  Milestones (15 WPs): 10+20+10×5+15×2+10+20+30+5+10+30 ≈ 230
  Celebration: 30+50+5×30 = 230
  forward_velocity: ~200
  Total completing ≈ 1,400+

✅ Ratio 4.5:1+ — completing dominates
```

### Terrain Zones (action_scale modulation)

| Zone | Y range | action_scale | Clearance boost | Notes |
|------|---------|-------------|----------------|-------|
| s012_wave | 8.83-11.83 | 0.40 | foot_clearance_wave_boost=3.0 | Entry hfield |
| s012_stairs_up | 12.33-14.33 | **0.80** | foot_clearance_stair_boost=**20.0** | Max leg amplitude |
| s012_bridge_valley | 14.33-21.33 | 0.20 | — | Bridge + river |
| s012_stairs_down | 21.33-23.33 | 0.20 | foot_clearance_stair_boost=20.0 | Descent |

---

## 4. Prior Experiment History (Pre-Fresh-Start)

> All prior experiments used old architecture versions. Summarized for reference.

### AutoML Runs (8 runs, ~35 trials total, all success_rate=0.0)

| Run ID | Trials | Best Reward | WP Progress | Notes |
|--------|--------|-------------|-------------|-------|
| automl_20260226_015954 | 1 | 1.38 | wp_idx≈1.0 | First section012 trial |
| automl_20260226_032658 | 4 | 1.55 | wp_idx≈1.0 | Reward search |
| automl_20260226_063605 | **15** | **2.30** | wp_idx≈1.0 | Largest batch — best reward but still stuck at stair base |
| automl_20260226_165658 | 3 | 1.14 | stuck | Relaxed termination attempt |
| automl_20260226_210051 | 1 | 1.06 | stuck | Single trial |
| automl_20260226_214637 | 3 | 1.57 | stuck | More relaxed term |
| automl_20260226_232851 | 1 | **4.30** | wp_idx≈1.0 | Highest reward (relaxed term joint-drag exploit) |
| automl_20260227_001357 | 2 | 2.08 | stuck | Hard term restart — stopped by user |

**Key Finding**: Robot consistently learns to survive at stair base (wp_idx≈1.0) but never climbs stairs. The relaxed-termination run (automl_20260226_232851) achieved highest reward by exploiting joint-dragging along the ground.

### Discovered Issues
1. **Joint-dragging exploit**: With relaxed termination, robot drags along ground collecting per-step rewards without climbing
2. **Stair climbing barrier**: Even with foot_clearance_stair_boost=20.0, slope_orientation=0.04, the robot cannot learn stair climbing from scratch in 50M steps
3. **Warm-start needed**: Section011 checkpoint provides baseline locomotion skills but slope-walking ≠ stair-climbing

### Reward Engineering Changes Applied (v58)
- `foot_clearance_stair_boost`: 3.0 → **20.0** (extreme knee lift incentive)
- `slope_orientation`: 0.0 → **0.04** (compensate forward-lean on stairs)
- `lin_vel_z`: -0.027 → **-0.005** (allow vertical leg push for stair stepping)
- `action_scale` on stairs: → **0.80** (max leg amplitude)
- Celebration: jump x3 → **walk+sit** (v58, aligned with section011)

---

## 5. Fresh Start — Next Steps

This report is now at a clean starting point. All code is updated (celebration walk+sit, 15 waypoints, terrain zones). No successful stair climbing has been achieved yet.

### Priority Tasks

1. ⬜ **VLM visual diagnosis** — `capture_vlm.py --env vbot_navigation_section012` to see what current policy does at stair base
2. ⬜ **Warm-start from section011** — Load best section011 checkpoint → test if slope skills transfer to stairs
3. ⬜ **Curriculum: stair-only sub-env** — If warm-start alone fails, create a simplified env with just stair climbing (no bridge/hongbaos) to isolate the skill
4. ⬜ **AutoML stair-focused search** — Tune stair-specific rewards (foot_clearance_stair_boost, action_scale, height_progress) with `automl.py --hp-trials 15`
5. ⬜ **Bridge crossing** — Only after stair climbing works, extend to full route
6. ⬜ **Full-route + celebration** — Complete ordered collection + walk+sit celebration

### Open Questions

- Is step_height ΔZ≈0.10 (right stairs) within VBot's physical capability with action_scale=0.80?
- Does the section011 slope-climbing gait transfer at all to discrete stair steps?
- Should we reduce spawn randomization further to focus learning on the stair approach angle?

---

*This report is append-only. All future experiments append below this line.*

---

## 6. v59 — Valley Zone Full Optimization (2026-03-01)

### Diagnosis

The `s012_bridge_valley` zone (y=14.33→21.33) was severely under-configured:
- `action_scale=0.20` (lowest of all zones) — yet contains 0.75m stone spheres + 1.0m valley depth
- No `clearance_boost` — foot clearance reward only 0.219 (1/20th of stairs)
- No `swing_scale` — full penalty for stone contact collisions
- No slope orientation compensation — valley slopes (11.6°) get orientation penalized
- `s012_stairs_down` had `action_scale=0.20` — but robot must CLIMB UP left stairs (ΔZ=0.15m) to reach bridge

### Changes Applied (v59)

#### cfg.py — TerrainZone update
| Parameter | Before | After | Rationale |
|-----------|--------|-------|----------|
| `s012_bridge_valley.action_scale` | 0.20 | **0.50** | Between bump(0.40) and stairs(0.80); stones need significant leg swing |
| `s012_bridge_valley.clearance_boost_key` | (empty) | `"foot_clearance_valley_boost"` | Enable terrain-specific clearance boost |
| `s012_bridge_valley.swing_scale_key` | (empty) | `"swing_contact_valley_scale"` | Reduce penalty for stone contacts |
| `s012_bridge_valley.pre_zone_margin` | 0.0 | 0.5 | Transition zone before valley |
| `s012_bridge_valley.post_zone_margin` | 0.0 | 0.3 | Cover back legs transitioning out |
| `s012_stairs_down.action_scale` | 0.20 | **0.80** | Must climb UP left stairs to bridge (ΔZ=0.15m/step) |

#### cfg.py — New reward keys in BASE_REWARD_SCALES
| Key | Value | Rationale |
|-----|-------|----------|
| `foot_clearance_valley_boost` | 10.0 | Between bump(7.2) and stair(20.0); stones are large but rounded |
| `swing_contact_valley_scale` | 0.3 | Between bump(0.21) and stair(0.5); frequent stone contacts expected |

#### vbot_section012_np.py — Slope orientation extension (v59b)
- Extended slope_orientation from entry stairs only → entry stairs + valley slopes + far-end stairs
- Entry stairs: y∈12.0-14.5, expected |sin|=0.447 (26.5°)
- Valley south slope: y∈14.33-17.0, expected |sin|=0.201 (11.6°)
- Valley north slope: y∈19.0-21.33, expected |sin|=0.201 (11.6°)
- Far-end stairs: y∈21.33-23.33, expected |sin|=0.600 (37°, left stairs steeper)
- Uses **direction-independent** matching: `min(|gy-mag|, |gy+mag|)` since robot traverses far-end stairs and valley both ways

#### automl.py — Search space additions
- `foot_clearance_valley_boost`: uniform [3.0, 30.0]
- `swing_contact_valley_scale`: uniform [0.05, 0.8]

### Smoke Test

5M steps completed successfully, no crashes. All terrain zone transitions smooth.

### Next Steps

1. ⬜ Run AutoML with valley-optimized config: `uv run starter_kit_schedule/scripts/automl.py --mode stage --hp-trials 15 --budget-hours 8`
2. ⬜ VLM visual analysis of stone traversal behavior
3. ⬜ Compare valley foot_clearance reward magnitude vs stairs in TensorBoard
| `vbot_section012_np.py` | Replaced 7-phase FSM with generic ordered waypoint progression; removed `_init_scoring_zones`/`_init_bridge_nav`; new `_init_ordered_route`, vectorized `_update_waypoint_state`, simplified `_get_current_target`, `_compute_reward`, `reset` | ✅ Done |
| `Task_Reference.md` | Full rewrite: new strategy, 14-WP route table, updated reward config, celebration config | ✅ Done |
| `Tutorial.md` | Rewritten for ordered route strategy, removed broken-budget warnings, updated config section | ✅ Done |
| `Tutorial_RL_Reward_Engineering.md` | Rewritten for generic waypoint rewards, removed per-phase code examples, added reusable pattern docs | ✅ Done |

### New Right-Side-First Route (14 Waypoints)

```
WP 0: right_approach         (2.0, 12.0)   virtual
WP 1-5: stone_hongbao_1~5    zigzag         reward  ← +15 pts total
WP 6-7: under_bridge_far/near (-3.0, ...)   reward  ← +10 pts total
WP 8: bridge_climb_base      (-3.0, 22.5)   virtual (far end)
WP 9: bridge_far_entry       (-3.0, 20.0)   virtual z>2.3
WP10: bridge_hongbao         (-3.0, 17.83)  reward  ← +10 pts
WP11: bridge_turnaround      (-3.0, 20.0)   virtual z>2.3
WP12: bridge_descent         (-3.0, 22.5)   virtual
WP13: exit_platform          (0.0, 24.33)   goal    ← +5 pts (celebration)
CELEBRATION: 3 right turns (configurable)
```

### Key Design Changes from v1.0

1. **Right-side first**: Collects all 5 stone hongbaos (15 pts) before going to bridge area
2. **Under-bridge before bridge**: Collect under-bridge hongbaos (10 pts) at ground level before climbing
3. **Out-and-back on bridge**: Climb from far end (y≈22.5), walk to center (y≈17.83) for hongbao, turn around, descend same stairs
4. **Multi-turn celebration**: 3 right turns (configurable) instead of single-jump, with IDLE→TURNING→SETTLING→TURNING... FSM
5. **Generic implementation**: `Waypoint`/`OrderedRoute` are reusable dataclasses; `_update_waypoint_state` handles any ordered route without per-WP code
6. **No zone_approach**: Removed the per-zone approach reward (was sector012-specific); generic `waypoint_approach` suffices
7. **Vectorized numpy**: All waypoint checks use batch operations with `np.clip(wp_current, 0, N-1)` indexing

### Reward Budget (v2.0)

```
Standing: 0.05 × 3000 (conditional) ≈ 150
Completing: milestones(~217) + celebration(125) + approach(200) + alive(150) ≈ 700+
Ratio: 5.3:1 in favor of completing ✅
```
