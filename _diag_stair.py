"""Diagnostic: where do robots get stuck in section012 with stair-base spawn?"""
import numpy as np, sys, torch, torch.nn.functional as F
sys.path.insert(0, 'starter_kit/navigation2')
import vbot
from motrix_envs.registry import make
import glob, os

device = torch.device('cuda')

# Find T0's checkpoint
run_dirs = sorted(glob.glob('runs/vbot_navigation_section012/26-02-26_16-57-*'))
if not run_dirs:
    run_dirs = sorted(glob.glob('runs/vbot_navigation_section012/26-02-26_17-*'))
ckpt_path = os.path.join(run_dirs[0], 'checkpoints', 'best_agent.pt')
print('Using:', ckpt_path)

ckpt = torch.load(ckpt_path, map_location=device)
pol = ckpt['policy']
sp = ckpt['state_preprocessor']

rm = sp['running_mean'].float().to(device)
rv = sp['running_variance'].float().to(device)
w0 = pol['_orig_mod.policy_net.0.weight'].float().to(device)
b0 = pol['_orig_mod.policy_net.0.bias'].float().to(device)
w1 = pol['_orig_mod.policy_net.2.weight'].float().to(device)
b1 = pol['_orig_mod.policy_net.2.bias'].float().to(device)
w2 = pol['_orig_mod.policy_net.4.weight'].float().to(device)
b2 = pol['_orig_mod.policy_net.4.bias'].float().to(device)
wm = pol['_orig_mod.mean_layer.weight'].float().to(device)
bm = pol['_orig_mod.mean_layer.bias'].float().to(device)

def policy_forward(obs_np):
    with torch.no_grad():
        x = torch.from_numpy(obs_np).float().to(device)
        x = (x - rm) / (torch.sqrt(rv) + 1e-8)
        x = F.elu(F.linear(x, w0, b0))
        x = F.elu(F.linear(x, w1, b1))
        x = F.elu(F.linear(x, w2, b2))
        x = torch.tanh(F.linear(x, wm, bm))
    return x.cpu().numpy()

N = 512
env = make('vbot_navigation_section012', num_envs=N)
env._cfg.hard_tilt_deg = 85.0
env._cfg.soft_tilt_deg = 0.0
env._cfg.enable_base_contact_term = False
env._cfg.enable_stagnation_truncate = False
env._cfg.grace_period_steps = 500
env.init_state()
state = env._state

max_y = np.full(N, -999.0)
max_z = np.full(N, -999.0)
min_z = np.full(N, 999.0)
max_x = np.full(N, -999.0)
min_x = np.full(N, 999.0)

for step in range(2000):
    actions = policy_forward(state.obs)
    state = env.step(actions)
    pose = env._body.get_pose(state.data)
    y, x, z = pose[:, 1], pose[:, 0], pose[:, 2]
    alive = ~state.terminated & ~state.truncated
    max_y = np.where(alive, np.maximum(max_y, y), max_y)
    max_z = np.where(alive, np.maximum(max_z, z), max_z)
    min_z = np.where(alive, np.minimum(min_z, z), min_z)
    max_x = np.where(alive, np.maximum(max_x, x), max_x)
    min_x = np.where(alive, np.minimum(min_x, x), min_x)
    if step in (499, 999, 1999):
        v = max_y > -900
        na = int(np.sum(alive))
        print(f'step {step+1}: alive={na} max_y={np.max(max_y[v]):.2f} mean_y={np.mean(max_y[v]):.2f} '
              f'max_z={np.max(max_z[v]):.2f} min_z={np.min(min_z[v]):.2f}')

v = max_y > -900
print(f'\n=== FINAL (2000 steps, {N} envs) ===')
print(f'Max Y: {np.max(max_y[v]):.3f} | Mean max Y: {np.mean(max_y[v]):.3f}')
print(f'Max Z: {np.max(max_z[v]):.3f} | Mean max Z: {np.mean(max_z[v]):.3f} | Min Z global: {np.min(min_z[v]):.3f}')
print(f'X span: [{np.min(min_x[v]):.2f}, {np.max(max_x[v]):.2f}]')

bins = [10, 11, 11.5, 12, 12.2, 12.4, 12.6, 12.8, 13, 13.5, 14, 15, 20]
counts, _ = np.histogram(max_y[v], bins=bins)
print('\nY distribution:')
for j in range(len(bins)-1):
    bar = '#' * min(counts[j], 60)
    print(f'  y {bins[j]:5.1f}-{bins[j+1]:5.1f}: {counts[j]:>4} {bar}')

# Frontier analysis: robots that got past y=12.4
for ythresh in [12.0, 12.4, 12.6, 13.0, 14.0]:
    f = v & (max_y > ythresh)
    cnt = int(np.sum(f))
    if cnt > 0:
        print(f'\nFrontier y>{ythresh}: {cnt} envs | z=[{np.min(min_z[f]):.3f}, {np.max(max_z[f]):.3f}] '
              f'| x=[{np.min(min_x[f]):.2f}, {np.max(max_x[f]):.2f}]')
    else:
        print(f'\nFrontier y>{ythresh}: 0 envs')
