"""Isolated seed-specific smoke followed by pilot; reject concurrent duplicates."""
from pathlib import Path
import argparse,fcntl,subprocess,sys,json,atexit,os
B=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--seed',type=int,default=42);p.add_argument('--phase',choices=['smoke','pilot'],default='pilot');a=p.parse_args()
if a.seed < 0:p.error('Seed must be nonnegative')
locks=B/'locks';locks.mkdir(exist_ok=True)
with (locks/f'seed_{a.seed}.lock').open('a+') as lock:
 try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 except BlockingIOError:raise SystemExit(f'Seed {a.seed} is already running; use a different seed')
 def run(phase):
  subprocess.run([sys.executable,str(B/'code/run_mixed.py'),'--phase',phase,'--seed',str(a.seed)],check=True)
 watcher=None
 if os.environ.get('WANDB_MODE', 'disabled') != 'disabled':
  watcher_log=(B/'slurm'/f'wandb_watch_seed_{a.seed}.log').open('a')
  watcher=subprocess.Popen([sys.executable,str(B/'code/watch_wandb.py'),str(a.seed),a.phase],stdout=watcher_log,stderr=subprocess.STDOUT)
  atexit.register(watcher.terminate)
 try:
  run(a.phase)
 finally:
  # Allow the live logger to upload the final checkpoint and finish its run.
  if watcher is not None:
   try: watcher.wait(timeout=120)
   except subprocess.TimeoutExpired:
    (B/'slurm'/f'wandb_flush_timeout_{a.phase}_seed_{a.seed}.json').write_text(json.dumps({'phase':a.phase,'seed':a.seed,'local_results_preserved':True,'needs_wandb_backfill':True}))
    watcher.terminate()
