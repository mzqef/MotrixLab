# Section 012 Tutorial — Ordered Multi-Waypoint Full-Collection Navigation

Welcome! This tutorial covers the **Section 012** environment in Navigation2 — training a quadruped robot (VBot) to collect ALL rewards on the obstacle course using a strict ordered route: right-side stones first, then under-bridge hongbaos, bridge out-and-back, and finally exit + celebration jumps.

> **Prerequisite**: Read `starter_kit_docs/navigation1/Tutorial.md` for framework fundamentals, and `starter_kit_docs/navigation2/long_course/Tutorial.md` for the overall Navigation2 course overview.

---

## 1. Section 012 Overview

| Aspect | Value |
|--------|-------|
| **Environment** | `vbot_navigation_section012` |
| **Strategy** | Ordered multi-waypoint: 14 waypoints, strict fixed route |
| **Terrain** | Entry → right stairs → stone hongbaos → under-bridge → bridge out-and-back → exit |
| **Distance** | ~14.5m (y=9.5 → y=24.0), but route covers more due to zigzag + bridge traversal |
| **Episode** | 6000 steps (60s) |
| **Points** | **60 pts** (highest value section — 57% of total Stage 2 score) |
| **Celebration** | 3 configurable right turns at exit platform |

### Key Insight: Multi-Navigation Problem

Every competition scoring zone is treated as a **waypoint** in an ordered list. The robot navigates to each in turn:
- **Reward waypoints**: stone hongbaos, under-bridge hongbaos, bridge hongbao — map to competition points
- **Virtual waypoints**: guide the route between reward zones (approach angles, stair bases, bridge entry/exit)
- **Goal waypoint**: exit platform — triggers celebration

The policy sees its current target change as it reaches each waypoint, creating a natural curriculum: early training learns to walk toward a nearby target, then extends to the full route.

---

## 2. Terrain Map — Section 02

### Two Routes

```
              LEFT ROUTE (harder)              RIGHT ROUTE (easier)
              x ≈ -3.0                         x ≈ +2.0
              
Y=12.4  ── 10-step stairs UP (ΔZ=0.15) ──    10-step stairs UP (ΔZ=0.10) ──
            z: 1.37 → 2.79                    z: 1.32 → 2.29
            
Y=15~20 ── Arch bridge ──────────────────    5 spheres (R=0.75) + 8 cones ──
            23 segments, peak z≈2.86          scattered obstacles
            width ~2.64m, railings
            
Y=21~23 ── 10-step stairs DOWN ──────────    10-step stairs DOWN ───────────
            z: 2.79 → 1.37                    z: 2.29 → 1.32
            
Y=24.3  ──────────────── EXIT PLATFORM (z≈1.294) ────────────────────────────
```

### Terrain Details

| Element | Left Route | Right Route |
|---------|-----------|-------------|
| Stairs up | 10 steps, ΔZ≈0.15/step (steep) | 10 steps, ΔZ≈0.10/step (gentle) |
| Middle | Arch bridge (narrow, elevated) | Spheres + cones (obstacles) |
| Stairs down | 10 steps, ΔZ≈0.15/step | 10 steps, ΔZ≈0.10/step |
| Difficulty | Hard (steep stairs + narrow bridge) | Medium (gentle stairs + obstacle avoidance) |

**Robot spawn**: (0, 9.5, 1.8) on entry platform, ±0.3m randomization.

---

## 3. Ordered Route — Full Collection Strategy

The route is **fixed and strict** (not learned by the policy). The waypoint list in `cfg.py` defines the exact order:

```
  WP 0: right_approach         → Enter right side of course
  WP 1-5: stone_hongbao_1~5    → Zigzag through 5 stone hongbaos (+15 pts)
  WP 6-7: under_bridge_far/near → Collect 2 under-bridge hongbaos (+10 pts)
  WP 8: bridge_climb_base      → Walk to far stair base
  WP 9: bridge_far_entry       → Climb stairs, enter bridge (z>2.3)
  WP10: bridge_hongbao         → Collect bridge hongbao (+10 pts)
  WP11: bridge_turnaround      → Turn around on bridge (z>2.3)
  WP12: bridge_descent         → Descend stairs back to ground
  WP13: exit_platform          → Reach exit → 10-jump celebration (+5 pts)
```

**Why right-side first?** The right stairs are gentler (ΔZ≈0.10 vs 0.15) and the stone hongbaos are scattered among spheres that serve as stepping stones. After collecting all ground-level rewards, the robot crosses to the left side for the bridge out-and-back.

**Why out-and-back on the bridge?** The bridge hongbao is at the center (y≈17.83). The robot climbs from the far (y≈22.5) end, walks to the center, then turns around and descends from the same end. This avoids the need to descend steep stairs from the near end.

---

## 4. Terrain-Specific Challenges

### 4.1 Stair Climbing

Stairs require fundamentally different locomotion than flat ground or slopes:

| Parameter | Flat Ground | 15° Slope (Section 011) | Stairs (Section 012) |
|-----------|------------|------------------------|---------------------|
| Foot clearance needed | Minimal | Low | **High** (step edge) |
| Knee flexion | Standard | Slightly increased | **Significantly increased** |
| Speed | Fast | Moderate | **Slow** |
| Foot placement precision | Low | Low | **High** |
| Fall risk | Low | Medium | **High** |

**Key reward components for stairs**:
- **Knee lift bonus**: Reward higher calf joint flexion when on stairs
- **Height progress**: Reward z-axis gain per step (like slopes but more granular)
- **Foot slip penalty**: Penalize lateral/backward foot sliding on stair edges
- **Reduced forward_velocity**: Stability over speed

### 4.2 Arch Bridge

The bridge is only ~2.64m wide with railings on both sides:

```
|  railing  |--- 2.64m usable width ---|  railing  |
```

VBot's body width is ~0.3m, but with leg spread it occupies ~0.5m. The margin is sufficient but requires tight lateral control.

**Key reward components for bridge**:
- **Lateral deviation penalty**: Penalize deviation from bridge centerline
- **Reduced angular velocity**: Limit turning on the bridge
- **Height stability**: Bridge is at z≈2.86 — falls are catastrophic

### 4.3 Obstacle Avoidance

Right route has 5 spheres (R=0.75m) and 8 cones scattered in the path:

**Key reward components for obstacles**:
- **Contact penalty**: Penalize collision with obstacle bodies
- **Edge preference**: Encourage paths along terrain edges where obstacles are fewer
- **Reactive avoidance**: Robot must learn to detour around obstacles

---

## 5. Configuration

### Where to Edit

| What | File |
|------|------|
| Spawn, target, episode length, reward scales | `starter_kit/navigation2/vbot/cfg.py` (class `VBotSection012EnvCfg`) |
| PPO hyperparameters | `starter_kit/navigation2/vbot/rl_cfgs.py` |
| Environment logic (reward computation) | `starter_kit/navigation2/vbot/vbot_section012_np.py` |
| Terrain geometry | `starter_kit/navigation2/vbot/xmls/scene_section012.xml` |

### Current Config Highlights

```python
# Spawn
pos = [0.0, 9.5, 1.8]  # Entry platform (z≈1.294 + 0.5m)
max_episode_steps = 6000  # 60 seconds

# Ordered route: 14 waypoints (see cfg.py Section012Route)
# Celebration: 3 right turns at exit platform
# Reward budget: completing (800+) >> standing (150) ✅
```

---

## 6. Reward Engineering — Ordered Waypoints

### 6.1 Reward Budget (Verified)

```
STANDING STILL for 6000 steps:
  alive = 0.05 × 3000 (conditional) = 150
  Total standing ≈ 150

COMPLETING ALL 14 WAYPOINTS + CELEBRATION:
  alive = 0.05 × 3000 = 150
  Milestones (14 WPs): ~217
  Celebration: 15×3 + 80 = 125
  Navigation rewards: ~200+
  Total completing ≈ 700+

✅ Ratio 4.5:1 — completing dominates
```

### 6.2 Waypoint Progression Rewards

The primary navigation signal comes from `waypoint_approach` (step-delta toward current WP) and milestone bonuses (one-time on first arrival). The reward function is generic — no per-waypoint special cases.

### 6.3 Stair-Specific Rewards

Stair climbing/descending challenges are handled by generic rewards:
- **height_progress**: Rewards z-axis gain when climbing
- **foot_clearance + stair_boost**: Terrain-zone driven, amplified on stairs
- **swing_contact_penalty + stair_scale**: Reduced on stairs (foot-edge contact expected)

---

## 7. Training Workflow

### This Section Uses Curriculum — Train After Section 011

```
Stage 2A: Section 011 (slopes) — MUST COMPLETE FIRST
  ↓ warm-start best checkpoint
Stage 2B: Section 012 (stairs + bridge) — THIS SECTION
```

### Smoke Test

```powershell
uv run scripts/train.py --env vbot_navigation_section012 --max-env-steps 200000
```

### Visual Debug

```powershell
uv run scripts/train.py --env vbot_navigation_section012 --render
```

### Full Training (Warm-Start from Section 011)

```powershell
uv run scripts/train.py --env vbot_navigation_section012 \
    --checkpoint runs/vbot_navigation_section011/<best_run>/checkpoints/best_agent.pt
```

### AutoML Batch Search

```powershell
uv run starter_kit_schedule/scripts/automl.py --mode stage --budget-hours 8 --hp-trials 15
```

### Evaluation

```powershell
uv run scripts/play.py --env vbot_navigation_section012
uv run scripts/capture_vlm.py --env vbot_navigation_section012 --max-frames 25
```

### TensorBoard

```powershell
uv run tensorboard --logdir runs/vbot_navigation_section012
```

**Key metrics to watch**:
- `max_y`: Forward progress (should approach 24.0)
- `max_z`: Height reached (should approach 2.79 for left route, 2.29 for right)
- `height_progress`: Stair climbing behavior
- `ep_len_mean`: Episode length (too short = falling on stairs, max = lazy)

---

## 8. Debugging Tips

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| Robot stands still at entry | alive_bonus dominates (lazy robot) | Fix reward budget |
| Robot falls on first step | Insufficient knee lift / wrong foot placement | Add knee lift bonus, reduce forward_velocity on stairs |
| Robot reaches stair top but falls on bridge | Lateral instability | Add bridge lateral penalty, reduce speed |
| Robot avoids stairs entirely | Stairs too punishing / termination too harsh | Reduce orientation penalty on stairs |
| Robot can't descend stairs | Front-heavy gait | Train descent separately, reduce forward_velocity |
| Robot collides with spheres (right route) | No obstacle avoidance signal | Add contact penalty for sphere bodies |

---

## 9. File Reference

| File | Purpose |
|------|---------|
| `starter_kit/navigation2/vbot/cfg.py` | Section012 config + reward scales |
| `starter_kit/navigation2/vbot/vbot_section012_np.py` | Section 02 environment implementation |
| `starter_kit/navigation2/vbot/rl_cfgs.py` | Section012 PPO hyperparameters |
| `starter_kit/navigation2/vbot/xmls/scene_section012.xml` | Section 02 MJCF scene |
| `starter_kit_docs/navigation2/section012/Task_Reference.md` | Terrain, scoring, reward config |
| `starter_kit_docs/navigation2/section012/REPORT_NAV2_section012.md` | Experiment history |
| `starter_kit_docs/navigation2/section012/Tutorial_RL_Reward_Engineering.md` | Reward engineering guide |
