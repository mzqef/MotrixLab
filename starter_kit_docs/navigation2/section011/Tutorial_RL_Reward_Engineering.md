# Section 011 — RL Reward Engineering Guide

---

## 1. Reward Architecture Overview

Section 011 uses a **multi-signal reward** combining:
- **Navigation rewards**: waypoint approach, forward velocity, zone bonuses
- **Height rewards**: height progress (height_approach/oscillation disabled)
- **Stability rewards**: orientation, stance ratio (0.08)
- **Celebration rewards**: jump reward, celebration bonus
- **Sensor-driven penalties**: impact penalty (trunk_acc), torque saturation (raw PD torques)
- **Standard penalties**: termination, lin_vel_z, action_rate, swing contact

---

## 2. Current Reward Scales (v48-T14 — Active)

See [Task_Reference.md](Task_Reference.md) Section 9 for the full table with all 28 parameters and comparison vs v47.

**Key v48-T14 changes from v47:**
- `lin_vel_z`: -0.195 → **-0.027** (7.2× lighter — bumps need vertical motion)
- `torque_saturation`: -0.025 → **-0.012** (2.1× lighter)
- `termination`: -200 → **-150** (sweet spot)
- `waypoint_approach`: 166.5 → **280.5** (1.68× stronger pull)
- `zone_approach`: 35.06 → **74.7** (2.13× stronger)
- `alive_decay_horizon`: 1500 → **2383** (longer motivation)
- `swing_contact_penalty`: -0.031 → **-0.003** (10× lighter)
- HP: LR 1e-4 → **4.5e-4**, entropy 0.0043 → **0.0078**

Source: AutoML `automl_20260220_071134` trial T14 (best of 15 cold-start trials at 15M steps).

---

## 6. Experiment History

See [REPORT_NAV2_section011.md](REPORT_NAV2_section011.md) for the full chronological record of all experiments, from v1 through v20.

### Key Milestones

| Version/Stage | wp_idx | Key Change |
|-------------|--------|------------|
| Stage 5C | 1.559 | Baseline |
| Stage 7B | 1.631 | HP tuning |
| Stage 11-12 | 1.712-1.723 | γ/λ axis search |
| Stage 13 | 1.866 | γ=0.999, λ=0.97 |
| Stage 14 | 1.956 | γ=0.999, λ=0.98 |
| Stage 15 | 1.977 | γ=0.999, λ=0.99 ★★★★★ |
| v18 | — | Boundary targeting + bump boost |
| v19 | — | Bump proprioception (bump_approach_boost) |
| v20 | — | Obs 54→69 (trunk_acc + raw PD torques) |
| v23b-T7 | 0.58 @5M | AutoML Phase 1 (gradient-only) |
| v29 | 5.86 | Bonus re-enablement (warm-start chain) |
| v35 | **5.91** | KL-adaptive + pre-peak warm-start ★ ALL-TIME BEST (old chain) |
| v47 | 1.40 mean, 7.0 max @50M | Fresh v46 config, (512,256,128) policy |
| **v48-T14** | **0.484 @15M** (cold) | **AutoML winner: lighter penalties + stronger navigation** |

> **Note**: v48-T14's 0.484@15M cold-start is not directly comparable to v35's 5.91 (warm-start chain). v48-T14 config is expected to significantly outperform v47 at equal step counts due to better penalty/reward balance.

---

## 7. Debugging Tips

1. **Check sensor penalties**: `monitor_training.py --deep` and grep for `impact_penalty`, `torque_saturation`
2. **Impact penalty too harsh?**: If robot freezes on bumpy terrain, reduce scale from -0.02 or raise threshold from 15 m/s²
3. **Torque saturation always zero?**: Threshold 0.9 means penalty only fires when torque > 90% of joint limit. Check `self._raw_torques` values vs `self.torque_limits`
4. **Obs dimension mismatch on checkpoint load?**: v20 is 69-dim — older 54-dim checkpoints cannot be warm-started
5. **Zone collection stuck?**: Check `smiley_bonus` and `red_packet_bonus` in TensorBoard — should increment over training
6. **Height oscillation / slope_orientation disabled?**: Both scales are 0.0 in cfg.py v20 — re-enable with small weight if needed
