# Tutorial: RL Reward Engineering for Section 012

**Case Study: VBot collecting ALL rewards on Section 02 via ordered multi-waypoint navigation**

> **Prerequisite**: Read `starter_kit_docs/navigation1/Tutorial_RL_Reward_Engineering.md` for foundational lessons.

---

## 1. The Task

| Aspect | Value |
|--------|-------|
| Environment | `vbot_navigation_section012` |
| Strategy | Ordered 15-waypoint route, right-side-first |
| Terrain | Right stairs + stones → under-bridge → bridge out-and-back → exit |
| Distance | ~14.5m straight, ~25m+ actual route |
| Episode | 6000 steps (60s) |
| Points | **60 pts** (57% of Stage 2) |
| Celebration | Walk to X endpoint + sit (v58) |

**Architecture**: Generic ordered waypoint progression. `_update_waypoint_state` handles all waypoints uniformly — no per-waypoint special cases in the reward function.

---

## 2. Reward Budget Audit (Verified)

```
STANDING STILL for 6000 steps (alive=1.013, decayed horizon=2383):
  alive ≈ 300 max
  Total standing ≈ 300

COMPLETING ALL 15 WPs + WALK+SIT CELEBRATION:
  alive: ~300
  waypoint_approach: ~500+ cumulative
  Milestones (15 WPs): ~230
  Celebration: 30 + 50 + 150 = 230
  forward_velocity: ~200
  Total completing ≈ 1,400+

✅ Completing (1,400+) >> Standing (300) — budget is sound
```

---

## 3. Reward Architecture

### 3.1 Generic Waypoint Progression

The core signals:
- **waypoint_approach** (280.534): Step-delta toward current WP — dominant gradient signal
- **forward_velocity** (3.163): Speed toward current WP
- **waypoint_facing** (0.637): Face toward current WP
- **zone_approach** (74.727): Zone-gated approach within terrain zones

Plus one-time milestone bonuses per waypoint (from `bonus_key → scales` mapping).

### 3.2 Z-Constraint Enforcement

Prevents cheating:
- Under-bridge WP7-8: `z_max=2.2` — must be below bridge
- Bridge WP10-12: `z_min=2.3` — must be on the bridge

Arrival check: `dist < radius AND z_min ≤ z ≤ z_max`.

### 3.3 Celebration Rewards (Walk + Sit)

After reaching WP14 (exit_platform):
- `celeb_walk_approach` (200.0): Delta approach to X endpoint (4.0, 24.33)
- `celeb_walk_bonus` (30.0): One-time on reaching X endpoint
- `celeb_sit_reward` (5.0): Per-step reward × z_below_threshold while sitting
- `celebration_bonus` (50.0): Final bonus when sit counter reaches 30 steps

---

## 4. Stair-Specific Reward Engineering

The primary unsolved challenge. Key reward components:

### 4.1 Foot Clearance Amplification

```python
foot_clearance_stair_boost = 20.0  # ×20 multiplier on stair zone
# Robot gets 20× the base foot_clearance reward (0.219) when in stair zone
# Effective: 0.219 × 20 = 4.38 per eligible swing leg per step
```

The stair zone (y=12.33-14.33) has `pre_zone_margin=1.0m` — the boost starts ramping up 1m before the stairs.

### 4.2 Action Scale

```python
# s012_stairs_up zone: action_scale = 0.80 (vs 0.20 default)
# Maximum leg amplitude to enable high knee lift for step edges
```

### 4.3 Vertical Motion Allowance

```python
lin_vel_z = -0.005      # Near-zero penalty (was -0.027)
# Allow vertical body motion needed for stair climbing
slope_orientation = 0.04 # Positive reward for forward lean on stairs
# Compensates orientation penalty when robot tilts forward on incline
```

### 4.4 Swing Contact on Stairs

```python
swing_contact_stair_scale = 0.5  # Halved swing penalty on stairs
# Foot-edge contact is expected on stair steps
```

---

## 5. Terrain Zone System

| Zone | Y range | action_scale | Effect |
|------|---------|-------------|--------|
| s012_wave | 8.83-11.83 | 0.40 | Wave hfield entry zone |
| s012_stairs_up | 12.33-14.33 | **0.80** | Max amplitude, ×20 clearance boost |
| s012_bridge_valley | 14.33-21.33 | 0.20 | Default for bridge/valley |
| s012_stairs_down | 21.33-23.33 | 0.20 | Descent with clearance boost |

Zones modulate `action_scale` (joint target magnitude) and `foot_clearance` boost per terrain region. The robot's Y position determines the active zone.

---

## 6. Anti-Exploit Measures

| Exploit | Prevention |
|---------|------------|
| Standing still | alive_bonus decayed (horizon=2383), milestones + approach dominate |
| Joint dragging | Hard termination (base_contact=True, hard_tilt=70°) |
| Waypoint skipping | Strict sequential order enforcement |
| Z-constraint bypass | z_min/z_max checks on bridge/under-bridge WPs |
| Fall-reset farming | termination=-150 penalty |
| Crouch exploit | crouch_penalty=-1.5 (binary, z-based) |
| Foot dragging | drag_foot_penalty=-0.15 per dragging leg |

---

## 7. Experiment Workflow

### ALWAYS use AutoML for parameter exploration:

```powershell
# HP + reward weight search
uv run starter_kit_schedule/scripts/automl.py --mode stage --budget-hours 8 --hp-trials 15
```

### Verify visually after any change:

```powershell
# VLM analysis
uv run scripts/capture_vlm.py --env vbot_navigation_section012

# Live render
uv run scripts/train.py --env vbot_navigation_section012 --render
```

### Follow the reward engineering cycle:

```
DIAGNOSE (VLM) → HYPOTHESIZE → IMPLEMENT → TEST (AutoML) → EVALUATE → ARCHIVE
```

---

## 8. Known Issues & Lessons

### Stair Climbing Not Yet Solved

All prior AutoML runs (8 runs, ~35 trials) achieved `success_rate=0.0`. Robot consistently:
- Learns to survive at stair base (wp_idx≈1.0)
- Never climbs the first step
- With relaxed termination, exploits joint-dragging along the ground

### Joint-Dragging Exploit

Discovered with relaxed termination: robot learns to drag itself along the ground, collecting per-step approach rewards without actually walking. **Prevention**: hard termination with `enable_base_contact_term=True`.

### Reward Budget Was Originally Broken

Original config had `alive=0.3 × 6000 = 1800 >> arrival=80`. Fixed by reducing alive, increasing milestones, adding decay horizon.

---

## 9. Next Research Directions

1. **Warm-start from section011**: Does slope-walking skill transfer to stair-stepping?
2. **Curriculum: stair-only env**: Isolate stair climbing as a sub-skill
3. **Step height analysis**: Is ΔZ≈0.10m (right stairs) within VBot's physical capability?
4. **Alternative approaches**: Reference position targets for stair gait, motion imitation
