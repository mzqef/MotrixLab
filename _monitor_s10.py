import time
import sys
import os
import subprocess
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

RUNS = {
    'T2': r'runs\vbot_navigation_section011\26-03-01_12-02-19-953959_PPO',
    'T8': r'runs\vbot_navigation_section011\26-03-01_12-02-22-089718_PPO'
}

INTERVAL = 300  # 5 minutes
TOTAL_ITERS = 24414  # 50M steps / (2048 * 1) 

def check():
    r = subprocess.run(['powershell', '-c', 'Get-Process python* -ErrorAction SilentlyContinue | Measure-Object | Select-Object -ExpandProperty Count'], capture_output=True, text=True)
    n_procs = int(r.stdout.strip() or 0)
    
    ts = time.strftime('%H:%M:%S')
    print(f"\n[{ts}] Python processes active: {n_procs}")
    print("-" * 75)
    print(f"{'Run':<4} | {'Iter':<6} | {'%':<5} | {'Max WP':<6} | {'Mean WP':<8} | {'Smiley':<7} | {'Reward':<8}")
    print("-" * 75)
    
    for name, run_dir in RUNS.items():
        if not os.path.exists(run_dir):
            print(f"{name:<4} | Directory not found")
            continue
            
        ea = EventAccumulator(run_dir, size_guidance={'scalars': 0})
        ea.Reload()
        tags = ea.Tags().get('scalars', [])
        
        if 'metrics / wp_idx_mean (mean)' not in tags:
            print(f"{name:<4} | No data yet")
            continue
            
        wp = ea.Scalars('metrics / wp_idx_mean (mean)')
        mx = ea.Scalars('metrics / wp_idx_mean (max)')
        sm = ea.Scalars('metrics / smiley_reached_frac (mean)') if 'metrics / smiley_reached_frac (mean)' in tags else []
        rwd = ea.Scalars('Reward / Total reward (mean)') if 'Reward / Total reward (mean)' in tags else []
        
        step = wp[-1].step
        pct = (step / TOTAL_ITERS) * 100
        wp_now = wp[-1].value
        wp_max = max(e.value for e in mx)
        sm_now = sm[-1].value * 100 if sm else 0
        rwd_now = rwd[-1].value if rwd else 0
        
        print(f"{name:<4} | {step:<6} | {pct:>4.1f}% | {wp_max:<6.0f} | {wp_now:<8.3f} | {sm_now:>5.1f}% | {rwd_now:<8.1f}")
        
        if len(wp) >= 3:
            last_3 = ' -> '.join(f'{e.value:.3f}' for e in wp[-3:])
            print(f"       Trend: {last_3}")

    print("-" * 75)
    
    if n_procs == 0:
        print(f"[{ts}] NO PYTHON PROCESSES - training finished or crashed!")
        return False
    return True

if __name__ == '__main__':
    print('=== Stage 10 (Full Training) Monitor ===')
    print('Updating every 5 minutes. Ctrl+C to stop.')
    while True:
        try:
            alive = check()
            if not alive:
                break
            sys.stdout.flush()
            time.sleep(INTERVAL)
        except KeyboardInterrupt:
            print('\nMonitor stopped.')
            break
