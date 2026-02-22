import os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

runs = [
    ("T0", "runs/vbot_navigation_section011/26-02-21_20-36-19-294823_PPO"),
    ("T1", "runs/vbot_navigation_section011/26-02-21_20-58-18-080749_PPO"),
    ("T2", "runs/vbot_navigation_section011/26-02-21_21-20-00-694875_PPO"),
    ("T3", "runs/vbot_navigation_section011/26-02-21_21-42-54-804787_PPO"),
    ("T4", "runs/vbot_navigation_section011/26-02-21_22-00-38-630708_PPO"),
]

instant_tags = [
    "Reward Instant / swing_contact_penalty (mean)",
    "Reward Instant / impact_penalty (mean)",
    "Reward Instant / drag_foot_penalty (mean)",
    "Reward Instant / foot_clearance (mean)",
]
total_tags = [
    "Reward Total/ swing_contact_penalty (mean)",
    "Reward Total/ impact_penalty (mean)",
    "Reward Total/ drag_foot_penalty (mean)",
    "Reward Total/ foot_clearance (mean)",
]

extra_kws = ["foot_clearance_bump", "swing_contact_bump"]

all_tags = list(instant_tags + total_tags)

# Gather data
data = {}
for name, path in runs:
    ea = EventAccumulator(path)
    ea.Reload()
    avail = ea.Tags().get("scalars", [])
    for t in avail:
        if any(k in t.lower() for k in extra_kws) and t not in all_tags:
            all_tags.append(t)
    row = {}
    for tag in all_tags:
        if tag in avail:
            events = ea.Scalars(tag)
            row[tag] = events[-1].value
        else:
            row[tag] = None
    data[name] = row

names = ["T0", "T1", "T2", "T3", "T4"]
hdr = f"{'Tag':<50} " + " ".join(f"{n:>8}" for n in names)
sep = "=" * len(hdr)

print(sep)
print(hdr)
print(sep)

print("--- Per-Step Instant (mean) ---")
for tag in [t for t in all_tags if "Instant" in t]:
    vals = []
    for n in names:
        v = data[n].get(tag)
        vals.append(f"{v:>8.4f}" if v is not None else f"{'N/A':>8}")
    short = tag.replace("Reward Instant / ", "").replace(" (mean)", "").replace(" (max)", "").replace(" (min)", "")
    print(f"{short:<50} " + " ".join(vals))

print()
print("--- Cumulative Total per Episode (mean) ---")
for tag in [t for t in all_tags if "Total" in t]:
    vals = []
    for n in names:
        v = data[n].get(tag)
        vals.append(f"{v:>8.1f}" if v is not None else f"{'N/A':>8}")
    short = tag.replace("Reward Total/ ", "").replace(" (mean)", "").replace(" (max)", "").replace(" (min)", "")
    print(f"{short:<50} " + " ".join(vals))

print()
print("--- Extra (bump boost / bump scale) ---")
for tag in [t for t in all_tags if any(k in t.lower() for k in extra_kws)]:
    vals = []
    for n in names:
        v = data[n].get(tag)
        if v is not None:
            if abs(v) > 10:
                vals.append(f"{v:>8.1f}")
            else:
                vals.append(f"{v:>8.4f}")
        else:
            vals.append(f"{'N/A':>8}")
    short = tag.replace("Reward Instant / ", "").replace("Reward Total/ ", "").replace(" (mean)", "").replace(" (max)", "").replace(" (min)", "")
    print(f"{short:<50} " + " ".join(vals))
