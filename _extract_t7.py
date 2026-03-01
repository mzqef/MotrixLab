"""Extract HP config for T7 (best trial) from state.yaml."""
import yaml, json

with open("starter_kit_log/automl_20260226_033450/state.yaml") as f:
    state = yaml.safe_load(f)

hist = state.get("hp_search_history", [])
# T7 is trial index 7
t7 = [h for h in hist if h["trial"] == 7][0]

print("=== T7 HP Config ===")
print(json.dumps(t7["hp_config"], indent=2))
print("\n=== T7 Reward Config ===")
print(json.dumps(t7["reward_config"], indent=2, sort_keys=True))
print("\n=== T7 Metrics (from state.yaml, old avg method) ===")
print(json.dumps(t7["metrics"], indent=2))
print(f"\n=== T7 Score: {t7['score']:.4f} ===")

# Also show T4 for comparison
t4 = [h for h in hist if h["trial"] == 4][0]
print("\n\n=== T4 HP Config ===")
print(json.dumps(t4["hp_config"], indent=2))
print("\n=== T4 Reward Config ===")
print(json.dumps(t4["reward_config"], indent=2, sort_keys=True))
print(f"\n=== T4 Score: {t4['score']:.4f} ===")

# Print best trial index
print(f"\nBest trial in state.yaml: {state.get('best_trial_index', '?')}")

# Also print run directory for T7
import glob, os
run_base = "runs/vbot_navigation_section011/"
all_runs = sorted(glob.glob(run_base + "26-02-26_*_PPO"))
our_runs = [r for r in all_runs if os.path.basename(r) >= "26-02-26_03-34"]
if len(our_runs) > 7:
    print(f"\nT7 run dir: {our_runs[7]}")
    # List checkpoints
    ckpt_dir = os.path.join(our_runs[7], "checkpoints")
    if os.path.exists(ckpt_dir):
        ckpts = sorted(os.listdir(ckpt_dir))
        print(f"Checkpoints: {ckpts}")
