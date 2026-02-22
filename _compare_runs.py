import os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

base = 'runs/vbot_navigation_section011'
runs = {
    'FullA_T12':  '26-02-22_05-25-43-367487_PPO',
    'FullB_T13':  '26-02-22_05-28-12-324803_PPO',
    'FullC_T11':  '26-02-22_05-30-41-234319_PPO',
    'BrA_T4':     '26-02-22_14-22-21-888968_PPO',
    'BrA_T6':     '26-02-22_15-10-42-347014_PPO',
    'BrA_T13':    '26-02-22_18-36-38-433889_PPO',
    'BrB_T2':     '26-02-22_21-13-33-900557_PPO',
    'BrB_T4':     '26-02-22_22-43-05-381961_PPO',
    'BrB_T8':     '26-02-23_00-22-34-269791_PPO',
}

results = {}
all_tags = set()
for label, rd in runs.items():
    path = os.path.join(base, rd)
    if not os.path.isdir(path):
        print(f'MISSING: {label} -> {rd}')
        continue
    ea = EventAccumulator(path)
    ea.Reload()
    tags = ea.Tags().get('scalars', [])
    all_tags.update(tags)

    data = {}
    for tag in tags:
        vals = ea.Scalars(tag)
        if vals:
            last_val = vals[-1].value
            best_val = max(v.value for v in vals)
            worst_val = min(v.value for v in vals)
            best_step = max(vals, key=lambda v: v.value).step
            data[tag] = {
                'last': last_val,
                'best': best_val,
                'worst': worst_val,
                'best_step': best_step,
                'n': len(vals),
            }
    results[label] = data

# Find interesting tags
interest = []
for tag in sorted(all_tags):
    t = tag.lower()
    if any(k in t for k in ['wp_idx', 'reward_mean', 'success', 'termin', 'episode_length', 'reached']):
        interest.append(tag)

print("Available interesting tags:", interest)
print()

labels = list(runs.keys())

# Print BEST values table
print("=" * 120)
print("BEST values across training")
print("=" * 120)
header = f"{'Tag':<40}"
for l in labels:
    header += f" | {l:>10}"
print(header)
print("-" * len(header))

for tag in interest:
    row = f"{tag:<40}"
    for l in labels:
        d = results.get(l, {}).get(tag, None)
        if d:
            row += f" | {d['best']:>10.4f}"
        else:
            row += f" | {'N/A':>10}"
    print(row)

print()

# Print LAST (final) values table
print("=" * 120)
print("FINAL (last logged) values")
print("=" * 120)
header = f"{'Tag':<40}"
for l in labels:
    header += f" | {l:>10}"
print(header)
print("-" * len(header))

for tag in interest:
    row = f"{tag:<40}"
    for l in labels:
        d = results.get(l, {}).get(tag, None)
        if d:
            row += f" | {d['last']:>10.4f}"
        else:
            row += f" | {'N/A':>10}"
    print(row)

print()

# Print best step for wp_idx
print("=" * 120)
print("Best step (iteration) for key metrics")
print("=" * 120)
for tag in interest:
    if 'wp_idx' in tag.lower() or 'reward_mean' in tag.lower():
        row = f"{tag:<40}"
        for l in labels:
            d = results.get(l, {}).get(tag, None)
            if d:
                row += f" | {d['best_step']:>10}"
            else:
                row += f" | {'N/A':>10}"
        print(row)
