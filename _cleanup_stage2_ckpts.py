"""Safely clean Stage 2 full-train and Branch C AutoML checkpoints.

Preserves:
  - Stage 2 full-train: peak checkpoint + 1 neighbor each side + best_agent.pt
  - Branch C AutoML: best_agent.pt only (TensorBoard logs preserved)
  - All other directories: untouched (already clean from prior cleanup)

DRY RUN by default. Pass --execute to actually delete.
"""
import os
import sys
from pathlib import Path

DRY_RUN = "--execute" not in sys.argv

base = Path("runs/vbot_navigation_section011")

# Stage 2 Full Train runs — keep peak ± 500 iters + best_agent.pt
fulltrain_runs = {
    "A_T13": {"dir": "26-02-23_11-08-08-977856_PPO", "keep_iters": [9000, 9500, 10000]},
    "A_T4":  {"dir": "26-02-23_13-49-12-918060_PPO", "keep_iters": [11000, 11500, 12000]},
    "B_T10": {"dir": "26-02-23_16-27-16-527358_PPO", "keep_iters": [11500, 12000, 12500]},
    "C_T6":  {"dir": "26-02-23_19-08-02-906445_PPO", "keep_iters": [11500, 12000, 12500]},
}

# Branch C AutoML runs (22 dirs with 73 ckpts each) — keep best_agent.pt only
branch_c_automl_dirs = [
    "26-02-23_01-28-19-303571_PPO", "26-02-23_01-29-13-192706_PPO",
    "26-02-23_02-03-31-852619_PPO", "26-02-23_02-04-44-372084_PPO",
    "26-02-23_02-39-21-018417_PPO", "26-02-23_02-40-34-368055_PPO",
    "26-02-23_03-15-07-492235_PPO", "26-02-23_03-16-11-734328_PPO",
    "26-02-23_03-50-48-633512_PPO", "26-02-23_03-51-43-461660_PPO",
    "26-02-23_04-25-48-452014_PPO", "26-02-23_04-26-43-678926_PPO",
    "26-02-23_05-01-42-566129_PPO", "26-02-23_05-32-27-270962_PPO",
    "26-02-23_06-06-53-006948_PPO", "26-02-23_06-34-03-819789_PPO",
    "26-02-23_06-59-05-666069_PPO", "26-02-23_07-23-58-271931_PPO",
    "26-02-23_07-49-21-313249_PPO", "26-02-23_08-14-08-183631_PPO",
    "26-02-23_08-39-01-415239_PPO", "26-02-23_09-04-23-895623_PPO",
]

total_removed = 0
total_bytes = 0

prefix = "[DRY RUN] " if DRY_RUN else ""

# 1. Clean Stage 2 Full Train runs
print("=== Stage 2 Full Train Cleanup ===")
for label, info in fulltrain_runs.items():
    ckpt_dir = base / info["dir"] / "checkpoints"
    keep_names = {f"agent_{i}.pt" for i in info["keep_iters"]}
    keep_names.add("best_agent.pt")

    removed = 0
    freed = 0
    for f in ckpt_dir.glob("agent_*.pt"):
        if f.name not in keep_names:
            freed += f.stat().st_size
            if not DRY_RUN:
                f.unlink()
            removed += 1

    total_removed += removed
    total_bytes += freed
    print(f"  {label}: {prefix}removed {removed} ckpts, freed {freed / 1024 / 1024:.0f} MB, kept {keep_names}")

# 2. Clean Branch C AutoML runs
print("\n=== Branch C AutoML Cleanup ===")
for d in branch_c_automl_dirs:
    ckpt_dir = base / d / "checkpoints"
    if not ckpt_dir.exists():
        continue

    removed = 0
    freed = 0
    for f in ckpt_dir.glob("agent_*.pt"):
        freed += f.stat().st_size
        if not DRY_RUN:
            f.unlink()
        removed += 1

    total_removed += removed
    total_bytes += freed
    if removed > 0:
        print(f"  {d}: {prefix}removed {removed} agent ckpts, freed {freed / 1024 / 1024:.0f} MB, kept best_agent.pt")

print(f"\n{'=' * 60}")
print(f"TOTAL: {prefix}{total_removed} files, {total_bytes / 1024 / 1024 / 1024:.2f} GB")
if DRY_RUN:
    print("\nThis was a DRY RUN. Pass --execute to actually delete.")
