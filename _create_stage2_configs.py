"""Create Stage 2 full-train configs from Stage 1 AutoML T0 and T1."""
import json, os

# Load T0 and T1 configs from Stage 1 (automl_20260228_160819)
with open('starter_kit_log/automl_20260228_160819/configs/automl_20260228_160819_vbot_navigation_section012_i0_1772266099.json') as f:
    t0 = json.load(f)
with open('starter_kit_log/automl_20260228_160819/configs/automl_20260228_160819_vbot_navigation_section012_i1_1772267545.json') as f:
    t1 = json.load(f)

os.makedirs('starter_kit_schedule/configs_full_train', exist_ok=True)

FULL_STEPS = 100_000_000
CKPT_INTERVAL = 500

# ---- T1 full config ----
t1_full = json.loads(json.dumps(t1))
t1_full['run_tag'] = 's012_stage2_T1'
t1_full['rl_overrides']['max_env_steps'] = FULL_STEPS
t1_full['rl_overrides']['check_point_interval'] = CKPT_INTERVAL
with open('starter_kit_schedule/configs_full_train/s012_stage2_T1.json', 'w') as f:
    json.dump(t1_full, f, indent=2)
lr1 = t1_full['rl_overrides']['learning_rate']
ent1 = t1_full['rl_overrides']['entropy_loss_scale']
print(f'Created: s012_stage2_T1.json  LR={lr1}, entropy={ent1:.6f}')

# ---- T0 full config ----
t0_full = json.loads(json.dumps(t0))
t0_full['run_tag'] = 's012_stage2_T0'
t0_full['rl_overrides']['max_env_steps'] = FULL_STEPS
t0_full['rl_overrides']['check_point_interval'] = CKPT_INTERVAL
with open('starter_kit_schedule/configs_full_train/s012_stage2_T0.json', 'w') as f:
    json.dump(t0_full, f, indent=2)
lr0 = t0_full['rl_overrides']['learning_rate']
ent0 = t0_full['rl_overrides']['entropy_loss_scale']
print(f'Created: s012_stage2_T0.json  LR={lr0}, entropy={ent0:.6f}')

# ---- Weighted average config (60% T1, 40% T0 since T1 performed better) ----
w1, w0 = 0.6, 0.4
avg = json.loads(json.dumps(t1))
avg['run_tag'] = 's012_stage2_AVG'
avg['rl_overrides']['max_env_steps'] = FULL_STEPS
avg['rl_overrides']['check_point_interval'] = CKPT_INTERVAL

# Average reward scales
for key in avg['reward_scales']:
    v1 = t1['reward_scales'][key]
    v0 = t0['reward_scales'][key]
    if key == 'termination':
        avg['reward_scales'][key] = t1['reward_scales'][key]  # discrete, keep T1's
    else:
        avg['reward_scales'][key] = round(v1 * w1 + v0 * w0, 6)

# Average RL HPs
avg['rl_overrides']['learning_rate'] = round(
    t1['rl_overrides']['learning_rate'] * w1 + t0['rl_overrides']['learning_rate'] * w0, 8)
avg['rl_overrides']['entropy_loss_scale'] = round(
    t1['rl_overrides']['entropy_loss_scale'] * w1 + t0['rl_overrides']['entropy_loss_scale'] * w0, 8)

with open('starter_kit_schedule/configs_full_train/s012_stage2_AVG.json', 'w') as f:
    json.dump(avg, f, indent=2)
lr_a = avg['rl_overrides']['learning_rate']
ent_a = avg['rl_overrides']['entropy_loss_scale']
print(f'Created: s012_stage2_AVG.json  LR={lr_a}, entropy={ent_a:.6f}')

# Summary
print('\n=== Stage 2 Config Comparison ===')
fmt = '%25s  %12s  %12s  %12s'
print(fmt % ('param', 'T1', 'T0', 'AVG(60/40)'))
print('-' * 65)
for key in ['forward_velocity', 'waypoint_approach', 'zone_approach', 'waypoint_bonus',
            'alive_bonus', 'alive_decay_horizon', 'stagnation_penalty', 'crouch_penalty',
            'foot_clearance_stair_boost', 'termination']:
    v_t1 = t1['reward_scales'][key]
    v_t0 = t0['reward_scales'][key]
    v_avg = avg['reward_scales'][key]
    if isinstance(v_t1, float):
        print(fmt % (key, f'{v_t1:.3f}', f'{v_t0:.3f}', f'{v_avg:.3f}'))
    else:
        print(fmt % (key, str(v_t1), str(v_t0), str(v_avg)))

print(fmt % ('learning_rate', f'{lr1:.6f}', f'{lr0:.6f}', f'{lr_a:.6f}'))
print(fmt % ('entropy', f'{ent1:.6f}', f'{ent0:.6f}', f'{ent_a:.6f}'))
