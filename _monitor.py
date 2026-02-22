"""Monitor automl trials with full config and metrics."""
import yaml

with open("starter_kit_schedule/progress/automl_state.yaml") as f:
    state = yaml.safe_load(f)

n = len(state["hp_search_history"])
cur = state.get("current_iteration", 0)
print(f"=== Trials completed: {n}, current_iteration: {cur} ===\n")

for entry in state["hp_search_history"]:
    t = entry["trial"]
    m = entry["metrics"]
    rc = entry["reward_config"]
    hp = entry["hp_config"]
    print(f"--- T{t} (score={entry['score']:.4f}) ---")
    print(f"  wp_idx={m['wp_idx_mean']:.3f}  success={m['success_rate']*100:.1f}%  term_rate={m['termination_rate']*100:.2f}%  ep_len={m['episode_length_mean']:.0f}")
    print(f"  lr={hp['learning_rate']:.6f}  entropy={hp['entropy_loss_scale']:.4f}")
    print(f"  term={rc.get('termination','?')}  forward_vel={rc.get('forward_velocity',0):.2f}  wp_approach={rc.get('waypoint_approach',0):.1f}  zone_approach={rc.get('zone_approach',0):.1f}")
    print(f"  alive_bonus={rc.get('alive_bonus',0):.2f}  alive_decay={rc.get('alive_decay_horizon',0):.0f}  stagnation={rc.get('stagnation_penalty',0):.3f}")
    print(f"  foot_clearance={rc.get('foot_clearance',0):.3f}  bump_boost={rc.get('foot_clearance_bump_boost',0):.2f}")
    pre_m = rc.get("foot_clearance_bump_boost_pre_margin", "default")
    post_m = rc.get("foot_clearance_bump_boost_post_margin", "default")
    pzr = rc.get("foot_clearance_pre_zone_ratio", "default")
    print(f"  pre_margin={pre_m}  post_margin={post_m}  pre_zone_ratio={pzr}")
    print(f"  phase_bonus={rc.get('phase_bonus',0):.1f}  celebration={rc.get('celebration_bonus',0):.1f}  jump_reward={rc.get('jump_reward',0):.2f}  per_jump={rc.get('per_jump_bonus',0):.1f}")
    print(f"  wp_bonus={rc.get('waypoint_bonus',0):.1f}  wp_facing={rc.get('waypoint_facing',0):.3f}  pos_tracking={rc.get('position_tracking',0):.3f}")
    print(f"  crouch={rc.get('crouch_penalty',0):.3f}  drag_foot={rc.get('drag_foot_penalty',0):.3f}  impact={rc.get('impact_penalty',0):.4f}")
    print(f"  swing_contact={rc.get('swing_contact_penalty',0):.4f}  swing_bump_scale={rc.get('swing_contact_bump_scale',0):.3f}")
    print(f"  action_rate={rc.get('action_rate',0):.4f}  torque_sat={rc.get('torque_saturation',0):.4f}  dof_pos={rc.get('dof_pos',0):.4f}")
    print(f"  orientation={rc.get('orientation',0):.4f}  lin_vel_z={rc.get('lin_vel_z',0):.4f}  ang_vel_xy={rc.get('ang_vel_xy',0):.4f}")
    print(f"  stance_ratio={rc.get('stance_ratio',0):.6f}")
    print()

print(f"Next trial: T{n}" if n < 20 else "All 20 trials complete!")
