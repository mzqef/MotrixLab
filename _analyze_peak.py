"""Reward breakdown analysis at peak performance."""
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ea = EventAccumulator('runs/vbot_navigation_section011/26-02-21_13-56-35-936742_PPO')
ea.Reload()

tags_of_interest = [
    'Reward Instant / wp_approach (mean)',
    'Reward Instant / zone_approach (mean)',
    'Reward Instant / forward_velocity (mean)',
    'Reward Instant / wp_bonus (mean)',
    'Reward Instant / smiley_bonus (mean)',
    'Reward Instant / red_packet_bonus (mean)',
    'Reward Instant / height_progress (mean)',
    'Reward Instant / height_approach (mean)',
    'Reward Instant / phase_completion_bonus (mean)',
    'Reward Instant / celebration_bonus (mean)',
    'Reward Instant / jump_reward (mean)',
    'Reward Instant / traversal_bonus (mean)',
    'Reward Instant / alive_bonus (mean)',
    'Reward Instant / penalties (mean)',
    'Reward Instant / termination (mean)',
    'Reward Instant / stagnation_penalty (mean)',
    'Reward Instant / heading_tracking (mean)',
    'Reward Instant / position_tracking (mean)',
    'Reward Instant / slope_orientation (mean)',
]

print('REWARD BREAKDOWN at peak (step 7000):')
print('=' * 60)
for tag in tags_of_interest:
    try:
        evts = ea.Scalars(tag)
        closest = min(evts, key=lambda e: abs(e.step - 7000))
        label = tag.replace('Reward Instant / ', '').replace(' (mean)', '')
        print(f'  {label:35s} = {closest.value:10.4f}')
    except KeyError:
        pass

# wp metrics
for tag in ['metrics / wp_idx_mean (mean)', 'metrics / wp_idx_mean (max)']:
    evts = ea.Scalars(tag)
    c = min(evts, key=lambda e: abs(e.step - 7000))
    short = tag.split('/')[-1].strip().replace(' (mean)', '').replace(' (max)', '_max')
    print(f'  {short:35s} = {c.value:10.4f}')

# Total reward
evts = ea.Scalars('Reward / Total reward (mean)')
c = min(evts, key=lambda e: abs(e.step - 7000))
print(f'  {"total_reward":35s} = {c.value:10.4f}')
