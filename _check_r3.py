"""Quick check of Round 3 training metrics."""
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ea = EventAccumulator(r'D:\MotrixLab\runs\vbot_navigation_section001\26-02-08_03-58-18-624357_PPO')
ea.Reload()

output = []
for tag in [
    'Reward / Instantaneous reward (mean)',
    'metrics / distance_to_target (mean)',
    'metrics / reached_fraction (mean)',
    'Reward Instant / fine_position_tracking (mean)',
    'Reward Instant / approach_reward (mean)',
    'Reward Instant / arrival_bonus (mean)',
    'Reward Instant / stop_bonus (mean)',
    'Reward Total/ arrival_bonus (mean)',
    'Episode / Total timesteps (mean)',
]:
    events = ea.Scalars(tag) if tag in ea.Tags()['scalars'] else []
    if events:
        label = tag.split('/')[-1].strip()
        vals = ', '.join([f'{e.step}={e.value:.4f}' for e in events])
        output.append(f'{label}: {vals}')

with open(r'D:\MotrixLab\_r3_metrics.txt', 'w') as f:
    f.write('\n'.join(output))
    f.write('\n')
print('Wrote', len(output), 'metrics')
