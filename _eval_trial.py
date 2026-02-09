"""Quick evaluation of trial metrics from TensorBoard."""
import sys
sys.path.insert(0, 'starter_kit_schedule/scripts')
from evaluate import read_tb_scalars

run_dir = 'runs/vbot_navigation_section001/26-02-08_00-19-09-319793_PPO'

tags = [
    'Reward / Instantaneous reward (mean)',
    'metrics / distance_to_target (mean)',
    'metrics / reached_fraction (mean)',
    'Reward Instant / heading_tracking (mean)',
    'Reward Instant / forward_velocity (mean)',
    'Reward Instant / approach_reward (mean)',
    'Reward Instant / arrival_bonus (mean)',
    'Reward Instant / alive_bonus (mean)',
    'Reward Instant / penalties (mean)',
    'Reward Total/ penalties (mean)',
    'Episode / Total timesteps (mean)',
]

for tag in tags:
    data = read_tb_scalars(run_dir, tag)
    if data:
        steps, vals = zip(*data)
        short = tag.split("/")[1].strip()
        print(f"{short:40s} first={vals[0]:8.3f}  last={vals[-1]:8.3f}  max={max(vals):8.3f}  min={min(vals):8.3f}")
