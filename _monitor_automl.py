"""Quick AutoML monitor script."""
import yaml

with open("starter_kit_log/automl_20260226_033450/state.yaml") as f:
    state = yaml.safe_load(f)

history = state.get("hp_search_history", [])
print(f"AutoML: {state['automl_id']} | Status: {state['status']} | Iter: {state['current_iteration']}/20 | Elapsed: {state['elapsed_hours']:.1f}h")
print()
print(f"{'T':>3} {'Score':>7} {'wp_idx':>6} {'celeb':>6} {'sr%':>6} {'term%':>7} {'ep_len':>7} {'Reward':>8} {'lr':>10} {'ent':>10}")
print("-" * 85)
best_score = 0
best_t = -1
for entry in history:
    t = entry["trial"]
    s = entry["score"]
    m = entry["metrics"]
    hp = entry["hp_config"]
    wp = m.get("wp_idx_mean", 0)
    celeb = m.get("celeb_state", 0)
    sr = m.get("success_rate", 0) * 100
    tr = m.get("termination_rate", 0) * 100
    el = m.get("episode_length_mean", 0)
    rw = m.get("episode_reward_mean", 0)
    lr = hp.get("learning_rate", 0)
    ent = hp.get("entropy_loss_scale", 0)
    if s > best_score:
        best_score = s
        best_t = t
    marker = " <-- BEST" if t == best_t and s >= best_score else ""
    print(f"T{t:>2} {s:>7.3f} {wp:>6.1f} {celeb:>6.0f} {sr:>5.1f}% {tr:>6.3f}% {el:>7.0f} {rw:>8.2f} {lr:>10.6f} {ent:>10.6f}{marker}")
print(f"\nBest: T{best_t} (score={best_score:.3f})")
print(f"Training T{state['current_iteration']} now...")
