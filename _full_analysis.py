"""Comprehensive cross-automl analysis of all 16 trials with full metrics."""
import os, yaml, sys

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except ImportError:
    print("ERROR: tensorboard not importable"); sys.exit(1)

RUNS = [
    ('R1', r'd:\MotrixLab\starter_kit_log\automl_20260221_012214'),
    ('R2', r'd:\MotrixLab\starter_kit_log\automl_20260221_075758'),
]

all_trials = []

for run_label, automl_dir in RUNS:
    state = yaml.safe_load(open(os.path.join(automl_dir, 'state.yaml')))
    exp_dir = os.path.join(automl_dir, 'experiments')
    trial_run_dirs = {}
    if os.path.isdir(exp_dir):
        for d in os.listdir(exp_dir):
            summary_path = os.path.join(exp_dir, d, 'summary.yaml')
            if os.path.isfile(summary_path):
                s = yaml.safe_load(open(summary_path))
                idx = s.get('trial_index')
                run_dir = s.get('results', {}).get('run_dir', '')
                if idx is not None and run_dir:
                    trial_run_dirs[idx] = os.path.join(r'd:\MotrixLab', run_dir.replace('\\', '/'))
    
    for h in state['hp_search_history']:
        all_trials.append({
            'run_label': run_label,
            'trial_idx': h['trial'],
            'id': f"{run_label}_T{h['trial']}",
            'score': h['score'],
            'ms': h.get('metrics', {}),
            'rc': h.get('reward_config', {}),
            'hp': h.get('hp_config', {}),
            'run_dir': trial_run_dirs.get(h['trial'], ''),
        })

print(f"Total trials: {len(all_trials)}\n")

# TB metric tags to extract
TAGS = {
    'metrics / wp_idx_mean (max)': ('wp_max', 'Navigation: max waypoint reached'),
    'metrics / wp_idx_mean (mean)': ('wp_mean', 'Navigation: mean waypoint index'),
    'metrics / distance_to_target (mean)': ('dist_m', 'Navigation: mean dist to target'),
    'metrics / distance_to_target (min)': ('dist_min', 'Navigation: min dist to target'),
    'Episode / Total timesteps (mean)': ('ep_len', 'Episode: mean length'),
    'Episode / Total timesteps (max)': ('ep_len_mx', 'Episode: max length'),
    'Reward / Total reward (mean)': ('rew_mean', 'Reward: mean total episode reward'),
    'Reward / Total reward (max)': ('rew_max', 'Reward: max total episode reward'),
    'Reward / Instantaneous reward (mean)': ('inst_rew', 'Reward: mean instantaneous reward'),
    'Reward Instant / wp_approach (mean)': ('ri_wp_app', 'RI: waypoint approach'),
    'Reward Instant / wp_approach (max)': ('ri_wp_app_x', 'RI: wp approach (max)'),
    'Reward Instant / zone_approach (mean)': ('ri_zone', 'RI: zone approach'),
    'Reward Instant / zone_approach (max)': ('ri_zone_x', 'RI: zone approach (max)'),
    'Reward Instant / forward_velocity (mean)': ('ri_fwd', 'RI: forward velocity'),
    'Reward Instant / forward_velocity (max)': ('ri_fwd_x', 'RI: forward velocity (max)'),
    'Reward Instant / alive_bonus (mean)': ('ri_alive', 'RI: alive bonus'),
    'Reward Instant / heading_tracking (mean)': ('ri_head', 'RI: heading tracking'),
    'Reward Instant / position_tracking (mean)': ('ri_pos', 'RI: position tracking'),
    'Reward Instant / foot_clearance (mean)': ('ri_fc_m', 'RI: foot clearance (mean)'),
    'Reward Instant / foot_clearance (max)': ('ri_fc_x', 'RI: foot clearance (max)'),
    'Reward Instant / penalties (mean)': ('ri_pen', 'RI: total penalties'),
    'Reward Instant / penalties (max)': ('ri_pen_x', 'RI: penalties (max/least)'),
    'Reward Instant / stagnation_penalty (mean)': ('ri_stag', 'RI: stagnation penalty'),
    'Reward Instant / drag_foot_penalty (mean)': ('ri_drag', 'RI: drag foot penalty'),
    'Reward Instant / crouch_penalty (mean)': ('ri_crouch', 'RI: crouch penalty'),
    'Reward Instant / torque_saturation (mean)': ('ri_torque', 'RI: torque saturation'),
    'Reward Instant / impact_penalty (mean)': ('ri_impact', 'RI: impact penalty'),
    'Reward Instant / termination (mean)': ('ri_term', 'RI: termination penalty'),
    'Reward Instant / swing_contact_penalty (mean)': ('ri_swing', 'RI: swing contact'),
    'Reward Instant / orientation (mean)': ('ri_orient', 'RI: orientation'),
    'Reward Instant / jump_reward (mean)': ('ri_jump', 'RI: jump reward'),
    'Reward Instant / phase_completion_bonus (mean)': ('ri_phase', 'RI: phase bonus'),
    'Reward Instant / wp_bonus (mean)': ('ri_wpbon', 'RI: waypoint bonus'),
    'Reward Instant / smiley_bonus (mean)': ('ri_smiley', 'RI: smiley bonus'),
    'Reward Instant / celeb_bonus (mean)': ('ri_celeb', 'RI: celebration bonus'),
    'Reward Instant / gait_stance (mean)': ('ri_gait', 'RI: gait stance'),
}

# Load TB data for all trials
for t in all_trials:
    sys.stderr.write(f"  Loading {t['id']}...\n")
    t['tb'] = {}
    if not t['run_dir'] or not os.path.isdir(t['run_dir']):
        continue
    ea = EventAccumulator(t['run_dir'])
    ea.Reload()
    tags = ea.Tags().get('scalars', [])
    for tag, (short, _) in TAGS.items():
        if tag in tags:
            events = ea.Scalars(tag)
            if events:
                vals = [e.value for e in events]
                t['tb'][short] = {'last': vals[-1], 'max': max(vals), 'min': min(vals), 'n': len(vals)}

# Sort by score
all_trials.sort(key=lambda x: x['score'], reverse=True)

def v(t, name, key='last', default=0.0):
    r = t.get('tb', {}).get(name)
    return r[key] if r else default

# =========================================================================
# TABLE 1: Navigation & Episode Metrics
# =========================================================================
print("=" * 130)
print("TABLE 1: NAVIGATION & EPISODE METRICS (sorted by score)")
print("=" * 130)
print(f"{'ID':>8} {'Score':>7} | {'wp_max':>6} {'wp_mean':>7} {'dist':>7} {'dmin':>7} | {'ep_len':>7} {'ep_max':>7} | {'rew_m':>8} {'rew_mx':>8} {'inst_r':>7}")
print("-" * 130)
for t in all_trials:
    print(f"{t['id']:>8} {t['score']:>7.4f} | "
          f"{v(t,'wp_max'):>6.1f} {v(t,'wp_mean'):>7.3f} {v(t,'dist_m'):>7.3f} {v(t,'dist_min'):>7.3f} | "
          f"{v(t,'ep_len'):>7.0f} {v(t,'ep_len_mx'):>7.0f} | "
          f"{v(t,'rew_mean'):>8.1f} {v(t,'rew_max'):>8.1f} {v(t,'inst_rew'):>7.3f}")

# =========================================================================
# TABLE 2: Positive Reward Components
# =========================================================================
print("\n" + "=" * 150)
print("TABLE 2: POSITIVE REWARD COMPONENTS (Instantaneous, sorted by score)")
print("=" * 150)
print(f"{'ID':>8} {'Score':>7} | {'wp_app':>7} {'zone':>7} {'fwd_v':>6} {'alive':>6} {'head':>6} {'pos_t':>6} | "
      f"{'fc_mean':>7} {'fc_max':>7} | {'phase':>7} {'wpbon':>7} {'smiley':>7} {'celeb':>7} {'jump':>7} {'gait':>7}")
print("-" * 150)
for t in all_trials:
    print(f"{t['id']:>8} {t['score']:>7.4f} | "
          f"{v(t,'ri_wp_app'):>7.3f} {v(t,'ri_zone'):>7.3f} {v(t,'ri_fwd'):>6.3f} {v(t,'ri_alive'):>6.3f} {v(t,'ri_head'):>6.3f} {v(t,'ri_pos'):>6.3f} | "
          f"{v(t,'ri_fc_m'):>7.4f} {v(t,'ri_fc_x'):>7.2f} | "
          f"{v(t,'ri_phase'):>7.4f} {v(t,'ri_wpbon'):>7.4f} {v(t,'ri_smiley'):>7.4f} {v(t,'ri_celeb'):>7.4f} {v(t,'ri_jump'):>7.4f} {v(t,'ri_gait'):>7.4f}")

# =========================================================================
# TABLE 3: Penalty Components
# =========================================================================
print("\n" + "=" * 140)
print("TABLE 3: PENALTY COMPONENTS (Instantaneous, sorted by score)")
print("=" * 140)
print(f"{'ID':>8} {'Score':>7} | {'total_p':>8} {'pen_mx':>8} | {'stag':>8} {'drag':>8} {'crouch':>8} {'torque':>8} {'impact':>8} {'term':>8} {'swing':>8} {'orient':>8}")
print("-" * 140)
for t in all_trials:
    print(f"{t['id']:>8} {t['score']:>7.4f} | "
          f"{v(t,'ri_pen'):>8.3f} {v(t,'ri_pen_x'):>8.4f} | "
          f"{v(t,'ri_stag'):>8.4f} {v(t,'ri_drag'):>8.4f} {v(t,'ri_crouch'):>8.4f} {v(t,'ri_torque'):>8.3f} {v(t,'ri_impact'):>8.4f} {v(t,'ri_term'):>8.4f} {v(t,'ri_swing'):>8.4f} {v(t,'ri_orient'):>8.4f}")

# =========================================================================
# TABLE 4: MAX-EVER Values (peak during training)
# =========================================================================
print("\n" + "=" * 140)
print("TABLE 4: MAX-EVER VALUES (best achieved at any point during training)")
print("=" * 140)
print(f"{'ID':>8} {'Score':>7} | {'wp_max':>6} {'wp_mx':>7} {'d_min':>7} {'el_max':>7} {'r_max':>8} | {'wpa_mx':>7} {'zon_mx':>7} {'fv_mx':>6} {'hd_mx':>6} | {'fc_mx':>7} {'fcx_mx':>7}")
print("-" * 140)
for t in all_trials:
    print(f"{t['id']:>8} {t['score']:>7.4f} | "
          f"{v(t,'wp_max','max'):>6.1f} {v(t,'wp_mean','max'):>7.3f} {v(t,'dist_m','min'):>7.3f} {v(t,'ep_len','max'):>7.0f} {v(t,'rew_mean','max'):>8.1f} | "
          f"{v(t,'ri_wp_app','max'):>7.3f} {v(t,'ri_zone','max'):>7.3f} {v(t,'ri_fwd','max'):>6.3f} {v(t,'ri_head','max'):>6.3f} | "
          f"{v(t,'ri_fc_m','max'):>7.4f} {v(t,'ri_fc_x','max'):>7.3f}")

# =========================================================================
# TABLE 5: Summary Metrics (from state.yaml)
# =========================================================================
print("\n" + "=" * 100)
print("TABLE 5: SUMMARY METRICS FROM STATE.YAML")
print("=" * 100)
print(f"{'ID':>8} {'Score':>7} | {'wp_mean':>8} {'succ%':>7} {'ep_len':>8} {'ep_rew':>9} {'rew_std':>8} {'term%':>7}")
print("-" * 100)
for t in all_trials:
    ms = t['ms']
    print(f"{t['id']:>8} {t['score']:>7.4f} | "
          f"{ms.get('wp_idx_mean',0):>8.4f} {ms.get('success_rate',0)*100:>6.2f}% {ms.get('episode_length_mean',0):>8.1f} "
          f"{ms.get('episode_reward_mean',0):>9.4f} {ms.get('episode_reward_std',0):>8.4f} {ms.get('termination_rate',0)*100:>6.3f}%")

# =========================================================================
# TABLE 6: HP Config Comparison
# =========================================================================
print("\n" + "=" * 120)
print("TABLE 6: HYPERPARAMETER CONFIG")
print("=" * 120)
print(f"{'ID':>8} {'Score':>7} | {'lr':>10} {'entropy':>8} {'rollouts':>8} {'minibat':>8} {'epochs':>7} {'discount':>9} {'lambda':>7} {'clip':>5}")
print("-" * 120)
for t in all_trials:
    hp = t['hp']
    print(f"{t['id']:>8} {t['score']:>7.4f} | "
          f"{hp.get('learning_rate',0):>10.2e} {hp.get('entropy_loss_scale',0):>8.4f} "
          f"{hp.get('rollouts','?'):>8} {hp.get('mini_batches','?'):>8} "
          f"{hp.get('learning_epochs','?'):>7} {hp.get('discount_factor',0):>9.4f} "
          f"{hp.get('lambda_param',0):>7.3f} {hp.get('ratio_clip',0):>5.2f}")

# =========================================================================
# TABLE 7: KEY REWARD WEIGHTS
# =========================================================================
print("\n" + "=" * 200)
print("TABLE 7: KEY REWARD WEIGHTS (top 8 trials)")
print("=" * 200)
keys = ['forward_velocity', 'waypoint_approach', 'zone_approach', 'alive_bonus', 'alive_decay_horizon',
        'foot_clearance', 'foot_clearance_bump_boost', 'waypoint_bonus', 'phase_bonus',
        'celebration_bonus', 'per_jump_bonus', 'jump_reward', 'termination',
        'stagnation_penalty', 'drag_foot_penalty', 'crouch_penalty', 'position_tracking',
        'torque_saturation', 'impact_penalty', 'swing_contact_penalty']
header = f"{'ID':>8} {'Score':>7}"
for k in keys:
    w = min(len(k), 8)
    header += f" | {k[:8]:>{w}}"
print(header)
print("-" * 200)
for t in all_trials[:8]:
    rc = t['rc']
    row = f"{t['id']:>8} {t['score']:>7.4f}"
    for k in keys:
        val = rc.get(k)
        if val is None:
            row += f" | {'N/A':>8}"
        elif abs(val) >= 100:
            row += f" | {val:>8.1f}"
        elif abs(val) >= 1:
            row += f" | {val:>8.3f}"
        else:
            row += f" | {val:>8.4f}"
    print(row)

# =========================================================================
# CORRELATION ANALYSIS
# =========================================================================
print("\n" + "=" * 100)
print("CORRELATION: Score vs Each TB Metric (Pearson r)")
print("=" * 100)

import statistics
scores = [t['score'] for t in all_trials]

def pearson(x, y):
    n = len(x)
    if n < 3: return 0
    mx, my = statistics.mean(x), statistics.mean(y)
    sx, sy = statistics.stdev(x), statistics.stdev(y)
    if sx == 0 or sy == 0: return 0
    return sum((xi-mx)*(yi-my) for xi,yi in zip(x,y)) / ((n-1)*sx*sy)

correlations = []
for tag, (short, desc) in TAGS.items():
    vals = [v(t, short) for t in all_trials]
    if all(x == 0 for x in vals):
        continue
    r = pearson(scores, vals)
    correlations.append((r, short, desc))

correlations.sort(key=lambda x: abs(x[0]), reverse=True)
for r, short, desc in correlations:
    bar_len = int(abs(r) * 25)
    bar = ("+" * bar_len) if r > 0 else ("-" * bar_len)
    print(f"  r={r:>+.3f} [{bar:>25s}]  {short:>12s}  {desc}")

# =========================================================================
# FINAL RANKING SUMMARY
# =========================================================================
print("\n" + "=" * 100)
print("FINAL RANKING")
print("=" * 100)
for i, t in enumerate(all_trials):
    ms = t['ms']
    star = " ***" if i == 0 else ""
    print(f"  #{i+1:>2} {t['id']:>8}  score={t['score']:.4f}  wp={ms.get('wp_idx_mean',0):.3f}  "
          f"succ={ms.get('success_rate',0)*100:.1f}%  eplen={ms.get('episode_length_mean',0):.0f}{star}")

# Convergence check
top5 = [t['score'] for t in all_trials[:5]]
print(f"\nTop 5 score std: {statistics.stdev(top5):.4f}")
if statistics.stdev(top5) < 0.003:
    print("VERDICT: Scores have converged. Ready for FULL TRAIN with best config.")
else:
    print("VERDICT: Still significant variation. Consider more search.")
