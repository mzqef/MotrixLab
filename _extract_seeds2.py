"""Extract top 5 S4 trial configs as individual seed files for next AutoML."""
import yaml
import json
from pathlib import Path

with open("starter_kit_log/automl_20260226_033450/state.yaml") as f:
    state = yaml.safe_load(f)

top_trials = [7, 16, 4, 15, 6]
hist = state["hp_search_history"]
outdir = Path("starter_kit_schedule/configs")

for t in hist:
    idx = t["trial"]
    if idx not in top_trials:
        continue
    hp = t["hp_config"]
    rw = t["reward_config"]
    
    seed = {
        "reward_scales": rw,
        "rl_overrides": {
            "learning_rate": hp["learning_rate"],
            "entropy_loss_scale": hp["entropy_loss_scale"],
            "policy_hidden_layer_sizes": hp.get("policy_hidden_layer_sizes", [512, 256, 128]),
            "value_hidden_layer_sizes": hp.get("value_hidden_layer_sizes", [512, 256, 128]),
            "rollouts": hp.get("rollouts", 24),
            "learning_epochs": hp.get("learning_epochs", 6),
            "mini_batches": hp.get("mini_batches", 16),
            "discount_factor": hp.get("discount_factor", 0.999),
        },
    }
    
    path = outdir / f"seed_S4_T{idx}.json"
    with open(path, "w") as f:
        json.dump(seed, f, indent=2)
    
    lr = hp["learning_rate"]
    ent = hp["entropy_loss_scale"]
    score = t.get("score", "?")
    print(f"T{idx}: LR={lr:.4e}, entropy={ent:.4e}, score={score:.4f} -> {path.name}")

print(f"\nSaved {len(top_trials)} seed configs to {outdir}/seed_S4_T*.json")
