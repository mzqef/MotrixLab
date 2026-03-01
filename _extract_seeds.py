"""Extract top 5 S4 trial configs as seed file for next AutoML."""
import yaml
import json

with open("starter_kit_log/automl_20260226_033450/state.yaml") as f:
    state = yaml.safe_load(f)

top_trials = [7, 16, 4, 15, 6]
hist = state["hp_search_history"]
seeds = []
for t in hist:
    idx = t["trial"]
    if idx in top_trials:
        # Merge hp_config + reward_config into one flat dict
        cfg = {}
        cfg.update(t.get("hp_config", {}))
        cfg.update(t.get("reward_config", {}))
        cfg["_source"] = f"S4_T{idx}"
        seeds.append(cfg)
        lr = cfg.get("learning_rate", "?")
        ent = cfg.get("entropy_loss_scale", "?")
        score = t.get("score", "?")
        print(f"T{idx}: LR={lr}, entropy={ent}, score={score}")

with open("starter_kit_schedule/configs/seed_S4_top5.json", "w") as f:
    json.dump(seeds, f, indent=2)
print(f"\nSaved {len(seeds)} seed configs to seed_S4_top5.json")
