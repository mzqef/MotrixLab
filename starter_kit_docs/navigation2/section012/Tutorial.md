# Section 012 Tutorial — Ordered Multi-Waypoint Full-Collection Navigation

Train a quadruped robot (VBot) to collect ALL rewards on Section 02's obstacle course using a strict ordered route: right-side stones first, then under-bridge hongbaos, bridge out-and-back, and exit + walk+sit celebration.

> **Prerequisite**: Read `starter_kit_docs/navigation1/Tutorial.md` for framework fundamentals.

---

## 1. Section 012 Overview

| Aspect | Value |
|--------|-------|
| **Environment** | `vbot_navigation_section012` |
| **Strategy** | Ordered multi-waypoint: 15 waypoints, strict fixed route |
| **Terrain** | Right stairs → stone hongbaos → under-bridge → bridge out-and-back → exit |
| **Distance** | ~14.5m straight, ~25m+ actual route |
| **Episode** | 6000 steps (60s) |
| **Points** | **60 pts** (highest value section — 57% of Stage 2 score) |
| **Celebration** | Walk to X-axis endpoint + sit (v58, same as section011) |
| **Spawn** | (2.0, 12.0, 1.8) at right stair base |

### Multi-Navigation Problem

Every competition scoring zone is a **waypoint** in an ordered list. The robot navigates to each in turn:
- **Reward WPs** (7): stone hongbaos ×5, under-bridge ×2, bridge hongbao ×1 — award competition points
- **Virtual WPs** (7): guide the route between reward zones (approach angles, stair entry/exit)
- **Goal WP** (1): exit platform — triggers walk+sit celebration

The policy sees its current target change as it reaches each waypoint, creating a natural curriculum.

---

## 2. Terrain Map — Section 02

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

### Why Right-Side First?

The right stairs are gentler (ΔZ≈0.10 vs 0.15). Stone hongbaos (+3 pts each) are scattered on the right riverbed. After collecting all ground-level rewards, the robot crosses to the left side for the bridge out-and-back.

---

## 3. Route: 15 Ordered Waypoints

```
WP  0: right_approach       → Guide to right side
WP  1: stair_top            → Climb right stairs
WP 2-6: stone_hongbao_1~5   → Collect 5 stones (+15 pts)
WP 7-8: under_bridge         → Collect 2 under-bridge (+10 pts)
WP  9: bridge_climb_base    → Walk to far stair base
WP 10: bridge_far_entry     → Climb left stairs, enter bridge (z>2.3)
WP 11: bridge_hongbao       → Collect bridge hongbao (+10 pts)
WP 12: bridge_turnaround    → Turn around on bridge
WP 13: bridge_descent       → Descend stairs
WP 14: exit_platform        → Goal → walk+sit celebration (+5 pts)
```

### Z-Constraint Waypoints

Some waypoints have altitude constraints to prevent cheating:
- **Under-bridge (WP7-8)**: `z_max=2.2` — must be below bridge deck
- **Bridge (WP10-12)**: `z_min=2.3` — must be on the bridge, not underneath

---

## 4. Celebration: Walk + Sit (v58)

After reaching exit_platform (WP14), the celebration FSM activates:

```
CELEB_IDLE → CELEB_WALKING → CELEB_SITTING → CELEB_DONE
```

1. **WALKING**: Robot walks toward `celeb_x_target = (4.0, 24.33)`. Delta approach reward guides it.
2. **SITTING**: Once within `celeb_walk_radius=1.0` of X target, robot must lower z below `celeb_sit_z=1.40`. A counter increments each step when z is low enough. After `celeb_sit_steps=30` (0.3s) → DONE.

This is identical to section011's celebration.

---

## 5. Configuration

| What | File |
|------|------|
| Spawn, waypoints, reward scales | `starter_kit/navigation2/vbot/cfg.py` → `VBotSection012EnvCfg` |
| PPO hyperparameters | `starter_kit/navigation2/vbot/rl_cfgs.py` |
| Environment logic | `starter_kit/navigation2/vbot/vbot_section012_np.py` |
| Terrain geometry | `starter_kit/navigation2/vbot/xmls/scene_section012.xml` |

See `Task_Reference.md` for full reward config, waypoint table, and terrain zone details.

---

## 6. Training Workflow

### Step 1: Smoke Test

```powershell
uv run python -c "
from starter_kit.navigation2.vbot import cfg as _
from motrix_envs.registry import make
env = make('vbot_navigation_section012', num_envs=4)
obs, info = env.reset()
print(f'obs shape: {obs.shape}')
for i in range(10):
    actions = env.action_space.sample()
    obs, reward, term, trunc, info = env.step(actions)
print('Smoke test passed!')
"
```

### Step 2: AutoML Reward Search

```powershell
uv run starter_kit_schedule/scripts/automl.py --mode stage --budget-hours 8 --hp-trials 15
```

### Step 3: Warm-Start from Section011

```powershell
uv run scripts/train.py --env vbot_navigation_section012 --policy <section011_best.pt>
```

### Step 4: Visual Debugging

```powershell
# Live visualization
uv run scripts/train.py --env vbot_navigation_section012 --render

# VLM analysis
uv run scripts/capture_vlm.py --env vbot_navigation_section012
```

### Step 5: Evaluate

```powershell
uv run scripts/play.py --env vbot_navigation_section012
```

---

## 7. Key Challenges

### 7.1 Stair Climbing

The primary unsolved challenge. Right stairs have ΔZ≈0.10m/step — requires high knee lift and precise foot placement.

Key reward components:
- `foot_clearance_stair_boost=20.0`: Extreme knee lift incentive on stair zone
- `action_scale=0.80` on stairs: Maximum leg amplitude
- `slope_orientation=0.04`: Compensate forward-lean penalty on stairs
- `lin_vel_z=-0.005`: Near-zero vertical velocity penalty (allow upward push)

### 7.2 Arch Bridge (narrow, ~2.64m wide)

Requires tight lateral control. Z-constraint waypoints enforce being on the bridge.

### 7.3 Stair Descent

Descending stairs is often harder than ascending — the robot must control forward momentum.

---

## 8. Monitoring

```powershell
# Monitor latest training run
uv run starter_kit_schedule/scripts/monitor_training.py --env vbot_navigation_section012

# Deep analysis
uv run starter_kit_schedule/scripts/monitor_training.py --env vbot_navigation_section012 --deep

# TensorBoard
uv run tensorboard --logdir runs/vbot_navigation_section012
```

Key metrics to watch:
- `wp_idx_mean`: Average waypoints reached (target: >2 means climbing stairs)
- `ep_len`: Episode length trend (increasing = learning, maxed = stagnating)
- `celeb_walk_reward`: Non-zero only when robot reaches exit and celebrates
