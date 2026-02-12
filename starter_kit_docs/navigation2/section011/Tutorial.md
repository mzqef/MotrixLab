# Section 011 Tutorial — Slopes + Multi-Waypoint + Celebration Spin

Welcome! This tutorial covers the **Section 011** environment in Navigation2 — training a quadruped robot (VBot) to traverse slopes, navigate through scoring zones via multi-waypoint guidance, and perform a celebration spin on the high platform.

> **Prerequisite**: Read `starter_kit_docs/navigation1/Tutorial.md` for framework fundamentals, and `starter_kit_docs/navigation2/long_course/Tutorial.md` for the overall Navigation2 course overview.

---

## 1. Section 011 Overview

| Aspect | Value |
|--------|-------|
| **Environment** | `vbot_navigation_section011` |
| **Terrain** | START platform → height field (bumps) → 15° ramp → high platform (z=1.294) |
| **Distance** | ~10.3m (y=-2.5 → y=7.83) |
| **Episode** | 3000 steps (30s) |
| **Points** | 20 pts (12 smileys + 6 red packets + 2 celebration) |
| **Status** | v3: Multi-waypoint + celebration spin |

### What Skills Does This Section Train?

- Basic locomotion over uneven terrain (height field bumps)
- 15° slope climbing (uphill walking)
- Platform edge transitions (step-up onto high platform)
- Multi-waypoint navigation (3 sequential targets)
- Scoring zone pass-through collection (smileys, red packets)
- Celebration spin (controlled rotation at platform top)

---

## 2. Terrain Map — Section 01

```
Y →  -3.5    -1.5    0    1.5    4.5   7.8
      |--START--|---hfield---|--ramp--|--platform--|
      z=0       z=0~0.277    z=0.41   z=1.294
      
      Smileys at y≈0 (x=-3, 0, 3) — on height field
      Red packets at y≈4.4 (x=-3, 0, 3) — on ramp
      Celebration at y≈7.83 — high platform top
```

| Element | Location | Z-height | Challenge |
|---------|----------|----------|-----------|
| Starting flat (START) | y=-3.5 ~ -1.5 | z=0 | Safe start area |
| Height field | y=-1.5 ~ +1.5 | z=0–0.277 | Mild undulation, 3 smiley zones |
| 15° ramp | y≈4.48 | z=0.41–0.66 | Uphill walking, 3 red packet zones |
| High platform | y≈7.83 | z=1.294 | Celebration spin target |

**Robot spawn**: (0, -2.5, 0.5) on START platform, ±0.5m randomization.

---

## 3. Multi-Waypoint Navigation System

Section 011 uses a **3-waypoint guidance system** instead of a single target:

| Waypoint | Position (x, y) | Purpose | Scoring Zones Nearby |
|----------|-----------------|---------|----------------------|
| WP0 | (0, 0) | Height field → smiley zones | 3 smileys at y≈0 |
| WP1 | (0, 4.4) | Ramp → red packet zones | 3 red packets at y≈4.4 |
| WP2 | (0, 7.83) | High platform (final) | Celebration zone |

- **Auto-advance**: When robot enters `waypoint_radius` (1.0m) of current WP, switches to next
- **Final precision**: WP2 uses `final_radius` (0.5m) — must reach platform accurately
- **Observation**: Robot always sees direction/distance to current waypoint (obs dims 48-52)

### Scoring Zone Collection

Smileys and red packets are collected **passively** as the robot passes through zones — they are NOT gated by waypoint index. Side zones (x=±3) can be collected even while the central waypoint guides the path.

---

## 4. Celebration Spin

After reaching WP2 (high platform), the **celebration state machine** activates:

```
CELEB_IDLE(0) → CELEB_SPIN_RIGHT(1) → CELEB_SPIN_LEFT(2) → CELEB_HOLD(3) → CELEB_DONE(4)
```

| Phase | Action | Completion Criteria |
|-------|--------|---------------------|
| SPIN_RIGHT | Rotate 180° clockwise | Heading within 0.3 rad of target |
| SPIN_LEFT | Rotate 180° counter-clockwise | Heading within 0.3 rad of initial |
| HOLD | Stay still | 30 steps with speed < 0.15 m/s |
| DONE | Episode truncates | Automatic — success! |

**Key design**: All four feet stay planted during spin — safe and stable. Speed limited to 0.3 m/s during spin phases.

**Observation dim 53** encodes celebration progress: 0.0=navigating, 0.25=spin_right, 0.5=spin_left, 0.75=hold, 1.0=done.

---

## 5. Configuration

### Where to Edit

| What | File |
|------|------|
| Spawn, target, episode length, reward scales | `starter_kit/navigation2/vbot/cfg.py` (class `VBotSection011EnvCfg`) |
| PPO hyperparameters | `starter_kit/navigation2/vbot/rl_cfgs.py` |
| Environment logic (reward computation, waypoints, celebration) | `starter_kit/navigation2/vbot/vbot_section011_np.py` |
| Terrain geometry | `starter_kit/navigation2/vbot/xmls/scene_section011.xml` |

### Current Config Highlights

```python
# Spawn
pos = [0.0, -2.5, 0.5]  # START platform (competition-correct)
max_episode_steps = 3000  # 30 seconds

# Anti-laziness
alive_bonus = 0.05        # 0.05 × 3000 = 150 < arrival(160) ✅
arrival_bonus = 160.0
termination = -75.0

# Terrain-specific
height_progress = 8.0     # z-delta climbing reward
traversal_bonus = 15.0    # milestone bonuses (×2: mid-ramp + ramp-top)

# Scoring zones
smiley_bonus = 20.0       # 3×20=60 potential
red_packet_bonus = 10.0   # 3×10=30 potential
celebration_bonus = 30.0

# Waypoint navigation
waypoint_bonus = 25.0     # 3×25=75 one-time per waypoint
waypoint_approach = 40.0  # step-delta toward current waypoint
waypoint_facing = 0.6

# Celebration spin
spin_progress = 3.0       # continuous heading progress reward
spin_hold = 5.0           # reward stillness in HOLD phase

# Tilt termination: 65° (allows 15° ramp + 50° margin)
```

---

## 6. Reward Engineering — Slopes

### Key Challenges

1. **Height field bumps** (y≈0, max 0.277m): Can trip the robot. Swing-phase contact penalty (-0.15) helps.
2. **15° ramp**: Forward velocity alone doesn't reward climbing effort. Height progress (z-delta) provides continuous gradient.
3. **Platform edge**: Step-up from ramp to platform is a fall risk. Tighter tilt termination (65°) helps catch falls early.
4. **Slope orientation**: Body naturally tilts ~15° on slope. Orientation penalty reduced to -0.03 (from standard -0.05).

### Slope-Specific Rewards

| Reward | Scale | Purpose |
|--------|-------|---------|
| `height_progress` | 8.0 | Reward z-axis gain per step (climbing) |
| `traversal_bonus` | 15.0 | One-time milestones: mid-ramp (y>4, z>0.3), ramp-top (y>6.5, z>0.8) |
| `forward_velocity` | 3.5 | Forward drive (Y-axis velocity) |
| `orientation` | -0.03 | Reduced penalty (slope tilt is correct behavior) |
| `lin_vel_z` | -0.15 | Reduced z-velocity penalty (climbing has z-velocity) |

### Budget Audit

```
STANDING STILL at y=-2.5 for 3000 steps:
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

## 7. Training Workflow

### Smoke Test

```powershell
uv run scripts/train.py --env vbot_navigation_section011 --max-env-steps 200000
```

### Visual Debug

```powershell
uv run scripts/train.py --env vbot_navigation_section011 --render
```

### Full Training (Warm-Start from Nav1)

```powershell
uv run scripts/train.py --env vbot_navigation_section011 \
    --checkpoint runs/vbot_navigation_section001/<best_run>/checkpoints/best_agent.pt
```

### AutoML Batch Search

```powershell
uv run starter_kit_schedule/scripts/automl.py --mode stage --budget-hours 8 --hp-trials 15
```

### Evaluation

```powershell
uv run scripts/play.py --env vbot_navigation_section011
uv run scripts/capture_vlm.py --env vbot_navigation_section011 --max-frames 25
```

### TensorBoard

```powershell
uv run tensorboard --logdir runs/vbot_navigation_section011
```

**Key metrics to watch**:
- `wp_idx_mean`: Waypoints being reached (should increase)
- `celeb_state_mean`: Celebration progress (should rise above 0)
- `smiley_bonus` / `red_packet_bonus`: Zone collection (should become non-zero)
- `spin_progress` / `spin_hold`: Celebration rewards
- `height_progress`: Climbing behavior
- `traversal_bonus`: Milestone achievements

---

## 8. Debugging Tips

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| Robot stands still | alive_bonus dominates (lazy robot) | Check budget: alive_budget < arrival |
| Robot falls on height field bumps | Nav1 flat-ground gait trips on bumps | Increase swing_contact_penalty, reduce forward_velocity initially |
| Robot climbs ramp but falls at edge | Platform step-up too abrupt | Tune termination angle, add height_progress near z=1.0 |
| Robot reaches platform but no celebration | WP2 radius too tight | Check final_radius (currently 0.5m) |
| Celebration spin doesn't complete | Heading tolerance too tight or speed too fast | Check spin_tolerance (0.3 rad), speed_limit (0.3 m/s) |
| Robot collects 0 smileys | Path doesn't cross smiley zones at y≈0 | Smiley detection radius is 1.2m — central waypoint at (0,0) should guide through center zone |

---

## 9. File Reference

| File | Purpose |
|------|---------|
| `starter_kit/navigation2/vbot/cfg.py` | Section011 config + reward scales |
| `starter_kit/navigation2/vbot/vbot_section011_np.py` | Section 01 environment (waypoints + celebration) |
| `starter_kit/navigation2/vbot/xmls/scene_section011.xml` | Section 01 MJCF scene |
| `starter_kit_docs/navigation2/section011/Task_Reference.md` | Terrain geometry, scoring, reward config |
| `starter_kit_docs/navigation2/section011/REPORT_NAV2_section011.md` | Experiment history |
| `starter_kit_docs/navigation2/section011/Tutorial_RL_Reward_Engineering.md` | Reward engineering guide |
