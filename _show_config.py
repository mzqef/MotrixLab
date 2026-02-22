import json, sys
d = json.load(open(sys.argv[1]))
rs = d["reward_scales"]
rl = d["rl_overrides"]
print("Key params:")
print(f"  lr={rl['learning_rate']:.5f}")
print(f"  termination={rs['termination']}")
print(f"  wp_approach={rs['waypoint_approach']:.1f}")
print(f"  zone_approach={rs['zone_approach']:.1f}")
print(f"  fwd_vel={rs['forward_velocity']:.2f}")
print(f"  bump_boost={rs['foot_clearance_bump_boost']:.2f}")
print(f"  pre_margin={rs.get('foot_clearance_bump_boost_pre_margin', 'NOT SET')}")
print(f"  post_margin={rs.get('foot_clearance_bump_boost_post_margin', 'NOT SET')}")
print(f"  pre_zone_ratio={rs.get('foot_clearance_pre_zone_ratio', 'NOT SET')}")
print(f"  alive_bonus={rs['alive_bonus']:.2f}")
print(f"  foot_clearance={rs['foot_clearance']:.4f}")
print(f"  swing_contact_penalty={rs['swing_contact_penalty']:.5f}")
print(f"  swing_contact_bump_scale={rs['swing_contact_bump_scale']:.3f}")
