"""
Run ALL Proposed Model experiments: 5 sites × 6 horizons × 2 targets = 60 runs.
Uses the unified Proposed_Models/main.py (adapted from GitHub).
"""
import subprocess
import sys
import time

SITES = [1463500, 14211720, 3321500, 3216070, 1646500]
HORIZONS = [6, 12, 24, 48, 96, 168]
TARGETS = ['EC', 'pH']

total = len(SITES) * len(HORIZONS) * len(TARGETS)
done = 0
start_all = time.time()

print(f"{'='*60}")
print(f"PROPOSED MODEL (SpikeDLinear) — FULL RUN: {total} experiments")
print(f"Sites: {SITES}")
print(f"Horizons: {HORIZONS}")
print(f"Targets: {TARGETS}")
print(f"{'='*60}\n")

for target in TARGETS:
    for site in SITES:
        for h in HORIZONS:
            done += 1
            print(f"\n[{done}/{total}] {target} | Site {site} | H={h}")
            print("-" * 40)
            
            start = time.time()
            cmd = [
                sys.executable,
                "Proposed_Models/main.py",
                "--target", target,
                "--horizon", str(h),
                "--site", str(site),
            ]
            
            result = subprocess.run(cmd, capture_output=False)
            elapsed = time.time() - start
            
            status = "✅" if result.returncode == 0 else "❌"
            print(f"{status} Done in {elapsed:.1f}s (exit={result.returncode})")

total_time = time.time() - start_all
print(f"\n{'='*60}")
print(f"ALL DONE! {done}/{total} experiments in {total_time/60:.1f} minutes")
print(f"{'='*60}")
