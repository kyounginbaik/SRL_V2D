"""Keep W&B runs active for the duration of each training phase."""
from pathlib import Path
import fcntl,json,subprocess,sys,time
base=Path(__file__).resolve().parents[1]
seed=int(sys.argv[1])
lock=(base/'locks'/f'wandb_watch_{seed}.lock').open('a')
try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
except BlockingIOError: raise SystemExit('A W&B watcher is already active for this seed')
for phase in (sys.argv[2] if len(sys.argv)>2 else 'pilot',):
 root=base/'runs'/phase/'episode_000037/chunk'/f'seed_{seed}'
 while not (root/'run_config.json').exists() or not (root/'stage.json').exists():
  time.sleep(5)
 while True:
  with (root/'live_sync.log').open('a') as log:
   result=subprocess.run([sys.executable,str(base/'code/wandb_sync.py'),phase,'episode_000037','chunk',str(seed)],stdout=log,stderr=subprocess.STDOUT)
  try: stage=json.loads((root/'stage.json').read_text()).get('stage')
  except (FileNotFoundError,json.JSONDecodeError): stage=None
  if result.returncode==0 and stage in ('complete','failed','cancelled_by_user'):
   if stage!='complete':raise SystemExit(0)
   break
  time.sleep(10)
