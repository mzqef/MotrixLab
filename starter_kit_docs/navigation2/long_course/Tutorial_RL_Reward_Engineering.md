# Tutorial: RL Reward Engineering for Long Course — Full 34m Navigation

**Case Study: VBot navigating the complete MotrixArena S1 obstacle course**

> This tutorial covers reward engineering specific to the Long Course — the competition submission environment combining all 3 sections with a 7-waypoint navigation system.

> **Prerequisite**: Read per-section reward engineering tutorials first:
> - `starter_kit_docs/navigation2/section011/Tutorial_RL_Reward_Engineering.md` — slopes
> - `starter_kit_docs/navigation2/section012/Tutorial_RL_Reward_Engineering.md` — stairs/bridge
> - `starter_kit_docs/navigation2/section013/Tutorial_RL_Reward_Engineering.md` — balls/ramp
> - `starter_kit_docs/navigation1/Tutorial_RL_Reward_Engineering.md` — foundational

---

## 1. The Task

| Aspect | Value |
|--------|-------|
| Environment | `vbot_navigation_long_course` |
| Total distance | ~34m |
| Episode | 9000 steps (90 seconds) |
| Waypoints | 7 (auto-advancing) |
| Competition points | 105 max |
| Terrain diversity | hfield, 15° slope, stairs, bridge, 21.8° ramp, 0.75m step, gold balls |

### Why Long Course Reward Engineering Is Different

| Challenge | Per-Section | Long Course |
|-----------|-------------|-------------|
| Horizon | 3000-6000 steps | 9000 steps |
| Terrain types | 1-2 | 6+ |
| Waypoints | 0-3 | 7 |
| Discount factor | 0.99 | 0.995 |
| Passive reward trap | Bad | **Critical** |
| Skill diversity | Moderate | **Extreme** |

---

## 2. Reward Budget Audit (CRITICAL — Not Yet Fixed)

### Current Config (BROKEN)

```
STANDING STILL for 9000 steps:
  alive = 0.5 × 9000 = 4,500
  position_tracking ≈ 0.2/step → 1,800
  heading_tracking ≈ 0.5/step → 4,500
  Total standing ≈ 10,800+

COMPLETING TASK:
  7 waypoints × 30.0 = 210
  arrival_bonus = 100
  forward + distance_progress ≈ 200-500
  Total completion ≈ 510-810

⚠️ STANDING (10,800) >> COMPLETION (810) — Ratio: 13:1
LAZY ROBOT IS 100% GUARANTEED
```

### Required Fix

```python
# Anti-laziness config for long course:
alive_bonus = 0.02           # 0.02 × 9000 = 180
waypoint_bonus = 50.0        # 7 × 50 = 350 (dominates alive)
arrival_bonus = 300.0        # Dominates everything
termination = -150.0         # Heavy penalty for falling

# Progressive waypoint bonuses (optional):
# WP0-WP4: 30.0 each (easy sections)
# WP5: 50.0 (section02 complete — hardest section)
# WP6: 300.0 (final arrival)
```

### Fixed Budget Projection

```
STANDING STILL:
  alive = 0.02 × 9000 = 180
  tracking ≈ 200-400
  Total standing ≈ 380-580

COMPLETING TASK:
  waypoints = 7 × 50 = 350
  arrival = 300
  navigation rewards ≈ 300-600
  Total completing ≈ 950-1,250

✅ COMPLETING (950+) >> STANDING (580) — incentive aligned
```

---

## 3. The Long-Horizon Problem

### 3.1 Discount Factor Analysis

With `discount_factor=0.995` and 9000 steps:

| Delay (steps) | Discount | Perceived Value of r=100 |
|---------------|----------|--------------------------|
| 0 | 1.000 | 100.0 |
| 1000 | 0.007 | 0.7 |
| 2000 | 0.00005 | 0.005 |
| 5000 | ~0 | ~0 |
| 9000 | ~0 | ~0 |

**Implication**: The arrival_bonus at the end is invisible at the start, no matter how large. Waypoints are essential to create intermediate reward signals within the effective discount horizon (~500-1000 steps).

### 3.2 Waypoint Spacing Analysis

| Transition | Distance | Est. Steps | Discount at Arrival |
|------------|----------|------------|---------------------|
| Start → WP0 | ~8.4m | ~1500 | 0.0006 |
| WP0 → WP1 | ~7.2m | ~1200 | 0.002 |
| WP1 → WP2 | ~3.0m | ~500 | 0.08 |
| WP2 → WP3 | ~5.5m | ~900 | 0.01 |
| WP3 → WP4 | ~2.5m | ~400 | 0.13 |
| WP4 → WP5 | ~3.4m | ~600 | 0.05 |
| WP5 → WP6 | ~7.8m | ~1500 | 0.0006 |

**Key insight**: WP0 and WP6 are the hardest transitions (longest, lowest discount). These need the strongest progress signals.

### 3.3 Anti-Laziness Trifecta (Enhanced for Long Horizon)

1. **Minimal alive bonus**: `alive_bonus ≤ 0.02` (budget = 180 for 9000 steps)
2. **Conditional per-step bonus**: Only award dense rewards when moving toward waypoint
3. **Time decay**: `time_factor = max(0, 1 - step/max_steps)` — early progress worth more

```python
# Time-decayed forward velocity reward:
time_factor = max(0.0, 1.0 - current_step / max_episode_steps)
forward_reward = forward_velocity_scale * forward_vel_component * time_factor
```

---

## 4. Waypoint-Specific Reward Design

### 4.1 Progressive Waypoint Bonuses

Not all waypoints are equally hard. Consider variable bonuses:

```python
WAYPOINT_BONUSES = {
    0: 30.0,    # WP0: Section01 exit (ramp climb — well-trained)
    1: 40.0,    # WP1: Left stair entrance (terrain transition)
    2: 30.0,    # WP2: Bridge start (stair climb)
    3: 40.0,    # WP3: Bridge end (bridge crossing)
    4: 50.0,    # WP4: Stair descent (hard — different skill from ascent)
    5: 60.0,    # WP5: Section02 exit (hardest section complete!)
    6: 300.0,   # WP6: FINISH (arrival_bonus)
}
```

### 4.2 Section-Specific Reward Zones

Different terrain types need different reward weightings:

```python
# Contextual reward scaling based on robot Y-position:
if robot_y < 10.3:  # Section 01 (slopes)
    # Relaxed orientation for ramp climbing
    orientation_scale = -0.02
    forward_target = 3.0  # Can be faster on slopes
elif robot_y < 24.3:  # Section 02 (stairs/bridge)
    # Strict balance for bridge crossing
    orientation_scale = -0.08
    forward_target = 1.5  # Slow and steady on stairs
elif robot_y < 34.0:  # Section 03 (balls/ramp)
    # Very relaxed for 21.8° ramp
    orientation_scale = -0.02
    forward_target = 1.0  # Very cautious near balls
```

### 4.3 Heading Change Reward

Several waypoint transitions require sharp heading changes:
- WP0 (0,6) → WP1 (-3,12): ~26° left turn
- WP4 (-3,23) → WP5 (0,24.5): ~63° right turn

```python
# Reward for tracking heading toward next waypoint
heading_error = abs(desired_heading - current_heading)
heading_reward = heading_scale * (1.0 - heading_error / np.pi)
```

---

## 5. Cross-Section Skill Retention

### The Catastrophic Forgetting Problem

Training on the full course means the robot practices later sections less (only reaches them if earlier sections succeed). This creates a **distribution shift**:

| Section | Fraction of Training Time | Risk |
|---------|--------------------------|------|
| Section 01 | 50-70% (reached every episode) | Over-optimized |
| Section 02 | 20-40% (reached if Section 01 OK) | Under-trained |
| Section 03 | 5-15% (reached if both previous OK) | Severely under-trained |

### Mitigation Strategies

1. **Per-section pre-training**: Train each section independently first, then warm-start long_course
2. **Random section start** (optional): Occasionally spawn the robot at section transitions — but this changes the task distribution
3. **High entropy_loss_scale**: Keep exploration alive so the robot doesn't collapse to a narrow Section 01 strategy
4. **Large rollouts (48)**: More diverse experience per update, captures cross-section behavior

---

## 6. Predicted Exploits (Long Course-Specific)

| Exploit | Description | Prevention |
|---------|-------------|------------|
| **Section 01 camper** | Robot masters Section 01 but stops at WP0 | Progressive waypoint bonuses, conditional alive |
| **Bridge avoider** | Robot falls off bridge repeatedly, earning alive + Section 01 rewards | Bridge-specific balance rewards, heavy termination |
| **Speed demon** | Robot runs too fast on stairs, falls off | Speed caps near elevation changes |
| **Passive surfer** | Robot walks forward slowly, collecting alive + tracking | alive_bonus ≤ 0.02, time decay |
| **Section hopper** | Robot reaches Section 02 then immediately dies (short episodes, fast resets) | Termination penalty > Section 01 waypoint bonus |
| **Waypoint oscillator** | Robot circles near waypoint 1.5m boundary | One-time waypoint bonus (no re-entry reward) |

---

## 7. Competition Score Alignment

### Training Reward vs Competition Score

| Competition Points | Training Reward Signal | Gap? |
|-------------------|----------------------|------|
| Smiley zones (12 pts) | Not modeled | ⚠️ Yes — robot may skip optimal path |
| Red packet zones (6 pts) | Not modeled | ⚠️ Yes |
| Celebration (2+5+5 pts) | Not modeled | ⚠️ Yes — 12 pts uncollected |
| Bridge red packets (10 pts) | Not modeled | ⚠️ Yes |
| Riverbed red packets (15 pts) | Not modeled | ⚠️ Yes |
| Ball avoidance (10-15 pts) | Not modeled | ⚠️ Yes |
| Section traversal | Modeled by waypoints | ✅ Aligned |
| Falls/boundary (−all section pts) | Termination penalty | ⚠️ Scale may be wrong |

**Missing competition score signals**: Up to 60 pts from bonus items are not represented in training rewards. Consider adding:
- Score zone detection rewards (smiley, red packet proximity)
- Celebration action trigger at section endpoints
- Ball avoidance/contact detection

### Priority

1. **First**: Get the robot to complete the full course reliably
2. **Then**: Add bonus item collection rewards
3. **Finally**: Optimize celebration actions for +12 pts

---

## 8. Optimal Training Sequence

```
Phase 1: Per-section training (independent, parallelizable)
├── section011: Train to >90% reached rate
├── section012: Train to >70% reached rate
└── section013: Train to >60% reached rate

Phase 2: Long course warm-start
├── Load section013 checkpoint (has full locomotion repertoire)
├── LR = 0.3× section013 LR
├── Reset optimizer state
├── Train 50M steps, check waypoint progression

Phase 3: Long course AutoML
├── Fix reward budget (alive=0.02, waypoint=50, arrival=300)
├── Run automl.py --hp-trials 8+
├── Select best config based on max waypoints reached

Phase 4: Full training
├── Best config from Phase 3
├── Train 300M steps
├── VLM checkpoint analysis every 50M steps
├── Submission evaluation
```

---

## 9. Config Verification Script

```powershell
uv run python -c "
from starter_kit.navigation2.vbot import cfg as _
from motrix_envs.registry import make
env = make('vbot_navigation_long_course', num_envs=1)
cfg = env._cfg
s = cfg.reward_config.scales
max_steps = cfg.max_episode_steps
alive = s.get('alive_bonus', 0) * max_steps
wp_total = s.get('waypoint_bonus', 0) * 7
arrival = s.get('arrival_bonus', 0)
completion = wp_total + arrival
ratio = alive / max(completion, 0.01)
print(f'=== Long Course Budget Audit ===')
print(f'max_steps={max_steps}')
print(f'alive_budget={alive:.0f}  (alive_bonus={s.get(\"alive_bonus\",\"?\")} × {max_steps})')
print(f'waypoint_total={wp_total:.0f}  (waypoint_bonus={s.get(\"waypoint_bonus\",\"?\")} × 7)')
print(f'arrival_bonus={arrival:.0f}')
print(f'completion_total={completion:.0f}')
print(f'ratio={ratio:.1f}:1  (should be <2)')
print()
if ratio > 5:
    print('⚠️  BROKEN: Standing still is {:.0f}× more rewarding than completing course!'.format(ratio))
    print('   Fix alive_bonus to ≤0.02, increase waypoint_bonus to ≥50, arrival_bonus to ≥300')
else:
    print('✅  Budget looks reasonable')
"
```
