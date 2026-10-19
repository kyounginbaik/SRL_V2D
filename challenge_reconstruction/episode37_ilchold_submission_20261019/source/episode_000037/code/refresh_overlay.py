from pathlib import Path
import sys,fcntl,subprocess
b=Path(__file__).resolve().parents[1]
phase,ep,variant,seed,version=sys.argv[1:6]
out=b/f'runs/{phase}/{ep}/{variant}/seed_{seed}/reference_overlays/version_{int(version):04d}'
out.mkdir(parents=True,exist_ok=True)
with (out/'transparent_closeups.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 if not (out/'transparent_closeups/complete.json').exists():
  for script in ['render_reference_overlay.py','compose_overlay.py']:
   subprocess.run([sys.executable,str(b/'code'/script),*sys.argv[1:6]],check=True)
