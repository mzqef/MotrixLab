import yaml
with open('starter_kit_log/automl_20260227_220608/state.yaml') as f:
    state = yaml.safe_load(f)
history = state.get('hp_search_history', [])
print(f'History entries: {len(history)}')
if history:
    print('Keys in first entry:', list(history[0].keys()))
    for i, e in enumerate(history[:3]):
        for k, v in e.items():
            print(f'  i{i}.{k} = {v}')
        print()
