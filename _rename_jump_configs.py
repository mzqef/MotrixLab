"""Batch rename jump -> turn in Python source files and config JSONs."""
import os
import re

base_dir = r'd:\MotrixLab'

# Files to process
py_files = [
    os.path.join(base_dir, 'starter_kit', 'navigation2', 'vbot', 'vbot_section012_np.py'),
    os.path.join(base_dir, 'starter_kit', 'navigation2', 'vbot', 'vbot_section013_np.py'),
]

# Pattern replacements for Python source files (order matters - longer patterns first)
py_replacements = [
    ('CELEB_JUMP', 'CELEB_TURNING'),
    ('CELEB_LANDING', 'CELEB_SETTLING'),
    ('celeb_jump_threshold', 'celeb_turn_threshold'),
    ('celebration_jump_threshold', 'celebration_turn_threshold'),
    ('celeb_landing_z', 'celeb_settle_z'),
    ('celebration_landing_z', 'celebration_settle_z'),
    ('required_jumps', 'required_turns'),
    ('all_jumps_done', 'all_turns_done'),
    ('still_jumping', 'still_turning'),
    ('jump_count_mean', 'turn_count_mean'),
    ('jump_count', 'turn_count'),
    ('jump_reward', 'turn_reward'),
    ('per_jump_bonus', 'per_turn_bonus'),
    # Variables
    ('jumping', 'turning'),
    ('jumped', 'turned'),
]

for f in py_files:
    with open(f, 'r', encoding='utf-8') as fh:
        content = fh.read()
    new_content = content
    for old, new in py_replacements:
        new_content = new_content.replace(old, new)
    if new_content != content:
        with open(f, 'w', encoding='utf-8') as fh:
            fh.write(new_content)
        print(f'Updated: {os.path.basename(f)}')
    else:
        print(f'No changes: {os.path.basename(f)}')

