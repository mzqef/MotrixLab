# Section 011 Tutorial — Slopes + Sweep-Order Zones + Jump Celebration

---

## 1. Section 011 Overview

| Aspect | Value |
|--------|-------|
| **Environment** | `vbot_navigation_section011` |
| **Terrain** | START → height field (bumps) → 15° ramp → high platform (z=1.294) |
| **Distance** | ~10.3m (y=-2.5 → y=7.83) |
| **Episode** | 4000 steps (40s) |
| **Points** | 20 pts (12 smileys + 6 red packets + 2 celebration) |
| **Architecture** | v17: Sweep-order zone collection + jump celebration |

### Skills Trained

- Locomotion over uneven terrain (height field bumps, max 0.277m)
- 15° slope climbing
- Platform edge transitions (step-up onto high platform z=1.294)
- Phase-based zone collection with **sweep ordering** (diagonal zigzag, ~35° turns)
- Jump celebration at platform top

---

## 2. Terrain Map

```
Y →  -3.5    -1.5    0    1.5    4.5   7.8
      |--START--|---hfield---|--ramp--|--platform--|
      z=0       z=0~0.277    z≈0.4    z=1.294
      
      Smileys at y≈0 (x=-3, 0, 3) — on height field
      Red packets at y≈4.4 (x=-3, 0, 3) — on ramp
      Celebration at y≈7.83 — high platform top
```

| Element | Y-range | Z-height | Challenge |
|---------|---------|----------|-----------|
| START flat | y=-3.5 ~ -1.5 | z=0 | Safe start area |
| Height field | y=-1.5 ~ +1.5 | z=0–0.277 | Bumps, 3 smiley zones |
| 15° ramp | y≈2.0 ~ 7.0 | z=0→1.294 | Uphill, 3 red packets |
| High platform | y≈7.0 ~ 8.8 | z=1.294 | Jump celebration |

**Robot spawn**: (0, -2.5, 0.5), X: ±2.0m, Y: ±0.5m randomization.

---

## 3. Phase-Based Zone Collection (v17 Sweep Ordering)

4-phase system matching competition scoring:

| Phase | Collect | Gate | Target Selection |
|-------|---------|------|-----------------|
| **SMILEYS (0)** | 3 smiley zones | Start here | **Sweep order** (L→C→R or R→C→L based on spawn) |
| **RED_PACKETS (1)** | 3 red packets | All 3 smileys | **Reverse sweep** (continuous zigzag) |
| **CLIMB (2)** | Reach platform | All red packets | Platform center (0, 7.83) |
| **CELEBRATION (3)** | Jump | On platform | Jump in place |

### v17 Sweep Ordering (replaces nearest-first)

**Problem**: Nearest-first targeting causes 90° turns (dangerous on bumpy/sloped terrain).

**Solution**: Sort zones by x-coordinate, creating a diagonal sweep with ~35° turns.

```
Spawn at x≥0 → Smileys: R(3,0) → C(0,0) → L(-3,0) → Red packets: L(-3,4.4) → C(0,4.4) → R(3,4.4)
Spawn at x<0 → Smileys: L(-3,0) → C(0,0) → R(3,0) → Red packets: R(3,4.4) → C(0,4.4) → L(-3,4.4)
```

Red packets sweep in **reverse direction** → continuous zigzag path, maximum heading change ~35°.

### Key Constants

- `wp_idx = smileys_collected + red_packets_collected + platform_reached` (range 0-7)
- Smiley/red packet radius: 1.2m
- Platform final radius: 0.5m + z > 1.0m check

---

## 4. Celebration — Jump (v16+)

After reaching the high platform, the robot does a simple **jump celebration**:

```
CELEB_IDLE(0) → CELEB_JUMP(1) → CELEB_DONE(2)
```

| State | What Happens | Completion |
|-------|-------------|------------|
| **IDLE→JUMP** | Enters celebration phase | Automatic on platform arrival |
| **JUMP** | Reward z elevation above 1.5m | z > `celebration_jump_threshold` (1.55) |
| **DONE** | Episode truncates (success!) | Automatic after jump detected |

**Rewards**: `jump_reward=8.0` (continuous z bonus), `celebration_bonus=100.0` (one-time).

---

## 5. Reward Engineering — v17 Changes

### Orientation + Slope Compensation

| Signal | Scale | Purpose |
|--------|-------|---------|
| `orientation` | **-0.05** | Strong tilt penalty (prevents falls) |
| `slope_orientation` | **+0.04** | Compensates pitch on ramp |

On flat ground: -0.05 penalty, no compensation → strong anti-fall.
On ramp with correct 15° tilt: penalty largely cancelled by compensation.

### Height Rewards

| Signal | Scale | Purpose |
|--------|-------|---------|
| `height_progress` | 8.0 | Raw z-delta upward |
| `height_approach` | 5.0 | Reduce |z_target - z_robot| |
| `height_oscillation` | -2.0 | Penalize rapid z bouncing |

### PPO Hyperparameters (Stage 15)

| Param | Value | Rationale |
|-------|-------|-----------|
| γ | 0.999 | Long planning horizon |
| λ | 0.99 | GAE ~460 steps |
| LR | 5e-5 | Stable fine-tuning |

---

## 6. Configuration Files

| What | File |
|------|------|
| Spawn, targets, reward scales | `starter_kit/navigation2/vbot/cfg.py` |
| PPO hyperparameters | `starter_kit/navigation2/vbot/rl_cfgs.py` |
| Environment logic | `starter_kit/navigation2/vbot/vbot_section011_np.py` |
| Terrain geometry | `starter_kit/navigation2/vbot/xmls/scene_section011.xml` |

---

## 7. Commands

```powershell
# Smoke test
uv run scripts/train.py --env vbot_navigation_section011 --max-env-steps 2000000

# Full training (warm-start)
uv run scripts/train.py --env vbot_navigation_section011 --checkpoint <path/to/best.pt>

# Evaluate / Monitor
uv run scripts/play.py --env vbot_navigation_section011
uv run starter_kit_schedule/scripts/monitor_training.py --env vbot_navigation_section011 --deep
uv run starter_kit_schedule/scripts/eval_checkpoint.py --rank runs/vbot_navigation_section011/<run_dir>
```

**Key metrics**: `wp_idx_mean`, `smiley_bonus`, `red_packet_bonus`, `height_approach`, `slope_orientation`.
