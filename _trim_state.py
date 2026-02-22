"""Trim automl state to keep only T0-T2, removing failed T3/T4."""
import yaml

STATE_PATH = "starter_kit_schedule/progress/automl_state.yaml"

with open(STATE_PATH, "r") as f:
    state = yaml.safe_load(f)

print(f"Before: {len(state['hp_search_history'])} trials")
print(f"current_iteration: {state['current_iteration']}")

# Trim to T0-T2 only
state["hp_search_history"] = state["hp_search_history"][:3]
state["current_iteration"] = 2  # last completed trial index

print(f"After: {len(state['hp_search_history'])} trials")
print(f"Trials kept: {[h['trial'] for h in state['hp_search_history']]}")
print(f"Scores: {[round(h['score'], 4) for h in state['hp_search_history']]}")

with open(STATE_PATH, "w") as f:
    yaml.dump(state, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

print("State file updated successfully.")
