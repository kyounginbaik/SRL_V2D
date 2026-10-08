"""Persist live progress and pack the submission only after all rollouts validate."""
from pathlib import Path
import datetime as dt,json,os,re,subprocess,sys,time
BASE=Path(os.environ.get('V2D_TRANSFER_RUN',Path(__file__).resolve().parent.parent))
refs=json.loads((BASE/'reference_manifest.json').read_text())
while True:
 states=[]
 for index,ref in enumerate(refs):
  p=BASE/ref['episode']/'stage.json'
  try:s=json.loads(p.read_text()) if p.exists() else {'episode':ref['episode'],'stage':'queued'}
  except (OSError,json.JSONDecodeError):s={'episode':ref['episode'],'stage':'status_updating'}
  jobs=json.loads((BASE/'training_jobs.json').read_text())
  job=jobs.get('episode_jobs',{}).get(str(index),f'{jobs["episode_array"]}_{index}')
  s['job_id']=job
  log=Path('/data1/SRL_V2D/slurm_logs')/f't3_transfer_{job}.out'
  if log.exists():
   with log.open('rb') as f:
    f.seek(max(0,log.stat().st_size-65536));tail=f.read().decode(errors='replace')
   matches=re.findall(r'training progress: ([\d,]+)/([\d,]+) environment steps \(([\d.]+)%\); ([\d,]+) env steps/s; elapsed ([\d:]+); ETA ([\d:]+)',tail)
   if matches:
    steps,total,percent,rate,elapsed,eta=matches[-1]
    s['progress']={'steps':int(steps.replace(',','')),'total':int(total.replace(',','')),'percent':float(percent),'steps_per_second':int(rate.replace(',','')),'elapsed':elapsed,'training_eta':eta}
  s['objects']=ref['objects']
  video=BASE/'videos'/(ref['episode']+'.mp4')
  if video.exists() and video.stat().st_size>1000 and (BASE/'videos'/(ref['episode']+'.complete')).exists():s['video']='videos/'+video.name
  states.append(s)
 d={'updated_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'expected_episodes':20,
  'completed_rollouts':sum(s['stage']=='complete' for s in states),'episodes':states,'uploaded_to_kaggle':False}
 try:
  q=subprocess.run(['squeue','-r','-u','exaFLOPs','-h','-o','%i %j %T %M %R'],capture_output=True,text=True,timeout=20)
  d['scheduler']=q.stdout
  scheduled={line.split()[0]:line.split()[2] for line in q.stdout.splitlines() if len(line.split())>=3}
  for state in states:
   if scheduled.get(state['job_id'])=='PENDING' and state['stage'] in ('failed','awaiting_stability_retry'):
    state['stage']='queued_retry';state['previous_attempt_error']=state.pop('error',None)
 except Exception as e:d['scheduler_error']=str(e)
 # Render each completed policy once; failed previews remain visible for diagnosis.
 try:
  vp=BASE/'video_jobs.json';video_jobs=json.loads(vp.read_text()) if vp.exists() else {}
  for index,state in enumerate(states):
   if state['stage']=='complete' and state['episode'] not in video_jobs:
    result=subprocess.run(['sbatch','--parsable',f'--array={index}',str(BASE/'code/render.sbatch')],capture_output=True,text=True,check=True,timeout=20)
    video_jobs[state['episode']]=result.stdout.strip()
    vp.write_text(json.dumps(video_jobs,indent=2))
 except Exception as e:d['preview_scheduling_error']=str(e)
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
