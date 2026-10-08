"""Persist live progress and pack the submission only after all rollouts validate."""
from pathlib import Path
import datetime as dt,json,os,re,subprocess,sys,time
BASE=Path(os.environ.get('V2D_TRANSFER_RUN',Path(__file__).resolve().parent.parent))
refs=json.loads((BASE/'reference_manifest.json').read_text())
while True:
 states=[]
 for ref in refs:
  p=BASE/ref['episode']/'stage.json'
  try:s=json.loads(p.read_text()) if p.exists() else {'episode':ref['episode'],'stage':'queued'}
  except (OSError,json.JSONDecodeError):s={'episode':ref['episode'],'stage':'status_updating'}
  s['objects']=ref['objects'];states.append(s)
 d={'updated_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'expected_episodes':20,
  'completed_rollouts':sum(s['stage']=='complete' for s in states),'episodes':states,'uploaded_to_kaggle':False}
 try:
  q=subprocess.run(['squeue','-u','exaFLOPs','-h','-o','%i %j %T %M %R'],capture_output=True,text=True,timeout=20)
  d['scheduler']=q.stdout
 except Exception as e:d['scheduler_error']=str(e)
 if d['completed_rollouts']==20:
  if (BASE/'CODE_COMMIT.txt').exists():
   try:
    with (BASE/'packing.log').open('a') as log:
     subprocess.run([sys.executable,str(BASE/'code/finalize.py')],stdout=log,stderr=subprocess.STDOUT,check=True)
    d['submission_ready_zip']=str(BASE/'track3_submission_ready.zip')
   except Exception as e:d['packing_error']=str(e)
  else:d['packing_error']='Producing code commit not yet recorded'
 tmp=BASE/'run_status.tmp';tmp.write_text(json.dumps(d,indent=2));tmp.replace(BASE/'run_status.json')
 print(d['updated_utc'],d['completed_rollouts'],'/20 validated rollouts',flush=True)
 if 'submission_ready_zip' in d:break
 time.sleep(60)
