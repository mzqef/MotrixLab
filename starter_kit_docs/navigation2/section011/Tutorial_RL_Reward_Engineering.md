# Section 011 — RL Reward Engineering Guide (v17)

---

## 1. Reward Architecture Overview

Section 011 uses a **multi-signal reward** combining:
- **Navigation rewards**: waypoint approach, forward velocity, zone bonuses
- **Height rewards**: height progress, height approach, height oscillation
- **Stability rewards**: orientation (+ slope compensation), stance ratio
- **Celebration rewards**: jump reward, celebration bonus
- **Penalties**: termination, lateral velocity, lin_vel_z

---

## 2. Current Reward Scales (v17)

| Signal | Scale | Category | Notes |
|--------|-------|----------|-------|
| `forward_velocity` | 3.0 | Navigation | Per-step forward progress |
| `waypoint_approach` | 100.0 | Navigation | Delta distance to current target |
| `zone_approach` | 5.0 | Navigation | Proximity to nearest zone |
| `smiley_bonus` | 150.0 | Bonus | One-time per smiley collected |
| `red_packet_bonus` | 150.0 | Bonus | One-time per red packet collected |
| `celebration_bonus` | 100.0 | Bonus | One-time on successful celebration |
| `jump_reward` | 8.0 | Celebration | Continuous z bonus during jump |
| `height_progress` | 8.0 | Height | Raw z-delta upward |
| `height_approach` | 5.0 | Height | Target-z approach (new v17) |
| `height_oscillation` | -2.0 | Height | Z-bounce penalty (new v17) |
| `orientation` | -0.05 | Stability | Tilt penalty (was -0.015) |
| `slope_orientation` | 0.04 | Stability | Ramp tilt compensation (new v17) |
| `stance_ratio` | 0.08 | Stability | Foot contact regularity |
| `lateral_velocity` | 0.0 | Disabled | (Available if needed) |
| `lin_vel_z` | -0.06 | Stability | Vertical velocity penalty |
| `termination` | -100.0 | Penalty | Fall or body contact |

---

## 3. v17 Changes from Previous Version

### 3.1 Orientation Strengthened (-0.015 → -0.05)

**Problem**: Weak orientation penalty allowed the robot to become dangerously tilted without sufficient cost, leading to falls especially on height field bumps.

**Solution**: 3.3× increase in orientation penalty. Combined with slope compensation so ramp climbing isn't punished.

### 3.2 Slope Orientation Compensation (+0.04)

**Problem**: Strengthened orientation penalty would punish correct body tilt on the 15° ramp.

**Solution**: On ramp (y ∈ [2.0, 7.0]), compute expected gravity projection for 15° tilt (sin(15°) ≈ 0.259), reward alignment: `exp(-gy_error²/0.05)`. Scaled at +0.04 to partially offset the -0.05 orientation penalty when correctly tilted.

Result: Net orientation cost on flat ground ≈ -0.05 (strong). On ramp with correct tilt ≈ -0.01 (mild).

### 3.3 Height Approach (scale=5.0)

**Problem**: `height_progress` (raw z-delta) only rewards positive z change. Robot gets no gradient signal about *target height*.

**Solution**: Track |z_target - z_robot| and reward any reduction. Target z estimated from target y-position via linear interpolation on ramp geometry. Creates a smooth gradient toward the correct elevation.

### 3.4 Height Oscillation (-2.0)

**Problem**: Robots can bounce/hop to exploit height_progress (up→down→up earns double credit on z-delta).

**Solution**: Penalize |z_delta| > 0.015m/step threshold. Only rapid z changes are punished; normal walking gait passes through.

### 3.5 Height Progress Reduced (12.0 → 8.0)

Reduced because height_approach now shares the height-incentive role. Total height motivation is maintained or increased; the signal is better distributed.

---

## 4. Sweep Ordering (Zone Targeting)

### The 90° Turn Problem

With nearest-first zone selection, a robot at center x=0 could target a side zone at x=3, then the next nearest at x=-3, requiring a 180° reversal. Even partial cases created 90° turns — dangerous on bumpy/sloped terrain where sideways forces cause falls.

### Solution: Coordinate-Sorted Sweep

Sort zones by x-coordinate based on spawn position:
- Spawn x < 0: sweep L(-3) → C(0) → R(3)
- Spawn x ≥ 0: sweep R(3) → C(0) → L(-3)

Red packets sweep in **reverse direction**, creating a continuous zigzag:
```
Example (spawn x≥0):
  S-R(3,0) → S-C(0,0) → S-L(-3,0) → RP-L(-3,4.4) → RP-C(0,4.4) → RP-R(3,4.4)
```

Maximum heading change between consecutive targets: ~35° (vs 90°+ with nearest-first).

---

## 5. Reward Budget Audit

### Standing Still Strategy
```
Per step: stance_ratio(0.08) + possible slope_orientation(0.04) ≈ 0.12
Over 4000 steps: 0.12 × 4000 = 480
```

### Successful Navigation
```
3 smileys: 3 × 150 = 450
3 red packets: 3 × 150 = 450
celebration_bonus: 100
waypoint_approach: ~100 × 10 = ~1000 (varies)
forward_velocity: ~3.0 × 1000 = ~3000 (active steps)
height_progress: ~50-80
height_approach: ~30-50
Total: ~5000+
```

**Budget check**: Successful navigation (~5000+) >> standing still (~480). ✅ No degenerate incentive.

---

## 6. Experiment History

See [REPORT_NAV2_section011.md](REPORT_NAV2_section011.md) for the full chronological record of all experiments, from v1 through v17.

### Key Milestones

| Version/Stage | wp_idx | Key Change |
|-------------|--------|------------|
| Stage 5C | 1.559 | Baseline |
| Stage 7B | 1.631 | HP tuning |
| Stage 11-12 | 1.712-1.723 | γ/λ axis search |
| Stage 13 | 1.866 | γ=0.999, λ=0.97 |
| Stage 14 | 1.956 | γ=0.999, λ=0.98 |
| Stage 15 | 1.977 | γ=0.999, λ=0.99 ★★★★★ |
| v17 | ? | Sweep ordering + new rewards |

---

## 7. Debugging Tips

1. **Check new reward channels**: `monitor_training.py --deep` and grep for `height_approach`, `height_oscillation`, `slope_orientation`
2. **Sweep direction**: Available in reset info as `sweep_direction` (-1 or +1)
3. **Height oscillation too punitive?**: If robots stop moving on bumpy terrain, reduce threshold from 0.015 or reduce scale from -2.0
4. **Slope orientation not activating?**: Only triggers for y ∈ [2.0, 7.0] — verify robot reaches the ramp
5. **Zone collection stuck?**: Check `smiley_bonus` and `red_packet_bonus` in TensorBoard — should increment over training
