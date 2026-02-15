#!/usr/bin/env python3
"""Quick script to check automl state."""
import yaml

d = yaml.safe_load(open("starter_kit_schedule/progress/automl_state.yaml"))
hist = d.get("hp_search_history", [])
for h in hist:
    t = h["trial"]
    s = h["score"]
    r = h["metrics"]["episode_reward_mean"]
    w = h["metrics"].get("wp_idx_mean", 0)
    lr = h["hp_config"]["learning_rate"]
    pol = h["hp_config"]["policy_hidden_layer_sizes"]
    val = h["hp_config"]["value_hidden_layer_sizes"]
    term = h["reward_config"].get("termination", "?")
    fv = h["reward_config"].get("forward_velocity", 0)
    print(f"Trial {t:2d}: score={s:.4f}  reward={r:6.2f}  wp_idx={w:.2f}  "
          f"lr={lr:.2e}  pol={pol}  val={val}  term={term}  fv={fv:.1f}")

print(f"\nTotal trials: {len(hist)}")
print(f"Current iteration: {d['current_iteration']}")
print(f"Elapsed: {d['elapsed_hours']:.2f}h / {d['budget_hours']}h")
print(f"Status: {d['status']}")

if d.get("best_results"):
    for env, res in d["best_results"].items():
        m = res.get("metrics", {})
        print(f"\nBest for {env}:")
        print(f"  reward={m.get('episode_reward_mean', 0):.2f}")
        print(f"  wp_idx={m.get('wp_idx_mean', 0):.2f}")
