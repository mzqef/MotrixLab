import yaml
with open('starter_kit_schedule/progress/automl_state.yaml') as f:
    state = yaml.safe_load(f)
n = len(state['hp_search_history'])
print(f"Total entries: {n}")
for i in range(n):
    t = state['hp_search_history'][i]
    m = t['metrics']
    rc = t['reward_config']
    hp = t['hp_config']
    print(f"\nT{t['trial']}: wp={m['wp_idx_mean']:.3f} suc={m['success_rate']:.3f} sc={t['score']:.4f}")
    print(f"  lr={hp['learning_rate']:.2e} ent={hp['entropy_loss_scale']:.4f} term={rc['termination']}")
    print(f"  bump={rc['foot_clearance_bump_boost']:.2f} pre_m={rc.get('foot_clearance_bump_boost_pre_margin', '-')} post_m={rc.get('foot_clearance_bump_boost_post_margin', '-')}")
    print(f"  ratio={rc.get('foot_clearance_pre_zone_ratio', '-')} fwd={rc['forward_velocity']:.2f} stag={rc['stagnation_penalty']:.2f}")
    print(f"  alive={rc['alive_bonus']:.2f} wp_appr={rc['waypoint_approach']:.1f} zone={rc['zone_approach']:.1f}")
    print(f"  ep_len={m['episode_length_mean']:.0f} rew={m['episode_reward_mean']:.2f} term_r={m['termination_rate']:.4f}")
    print(f"  wp_bonus={rc.get('waypoint_bonus', 0):.1f} celeb={rc.get('celebration_bonus', 0):.1f}")
    print(f"  crouch={rc.get('crouch_penalty', 0):.2f} drag={rc.get('drag_foot_penalty', 0):.2f}")
