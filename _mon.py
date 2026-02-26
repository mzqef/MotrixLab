import yaml

with open("starter_kit_log/automl_20260226_033450/state.yaml") as f:
    state = yaml.safe_load(f)

print(f"Phase: {state['current_phase']}")
print(f"Iter:  {state['current_iteration']}/20")
print(f"Elapsed: {state['elapsed_hours']:.1f}h / {state['budget_hours']}h")

hist = state.get("hp_search_history", [])
print(f"Completed: {len(hist)} trials\n")

print(f"{'T':>3} {'Score':>7} {'wp':>5} {'reached':>8} {'reward':>8} {'term':>9} {'ep_len':>7} {'celeb':>6}")
print("-" * 62)

best_s, best_t = 0, -1
for h in hist:
    t = h["trial"]
    m = h["metrics"]
    s = h["score"]
    wp = m.get("wp_idx_mean", 0)
    reached = m.get("success_rate", 0) * 100
    rew = m.get("episode_reward_mean", 0)
    term = m.get("termination_rate", 0)
    ep = m.get("episode_length_mean", 0)
    celeb = m.get("celeb_state", "-")
    if s > best_s:
        best_s = s
        best_t = t
    c_str = f"{celeb:.0f}" if isinstance(celeb, (int, float)) else str(celeb)
    star = " *" if t == state.get("best_trial_index", -1) else ""
    print(f"T{t:<2} {s:>7.4f} {wp:>5.1f} {reached:>7.1f}% {rew:>8.2f} {term:>9.5f} {ep:>7.0f} {c_str:>6}{star}")

print(f"\nBest: T{best_t} score={best_s:.4f}")
