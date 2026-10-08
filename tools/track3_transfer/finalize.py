"""Pack only complete, official, unassisted rollouts. Never submit to Kaggle."""
from pathlib import Path
from checkpoint_health import check_checkpoint
import csv,datetime as dt,hashlib,json,os,shutil,struct,subprocess,sys,zipfile
BASE=Path(os.environ.get('V2D_TRANSFER_RUN',Path(__file__).resolve().parent.parent))
KIT=Path('/data1/video_to_data_challenge/v2d_submission_kit')
MESH=BASE.parent/'eval_mesh_refined_v2_20261008/track3_meshes.parquet'
OUT=BASE/'submission_ready'
refs=json.loads((BASE/'reference_manifest.json').read_text())
assert len(refs)==20 and sum(len(r['objects']) for r in refs)==36
commit=(BASE/'CODE_COMMIT.txt').read_text().strip()
if not commit.startswith('https://github.com/') or '/commit/' not in commit:raise ValueError('Producing commit URL required')
quality=[]
reproduction=[]
for r in refs:
 ep=r['episode'];state=json.loads((BASE/ep/'stage.json').read_text())
 if state['stage']!='complete':raise RuntimeError(f'{ep}: incomplete {state["stage"]}')
 check_checkpoint(state['checkpoint'])
 check_checkpoint(BASE/ep/'training'/f"state_{state['training_steps']}.safetensors")
 report=json.loads((BASE/'results'/(ep+'.json')).read_text())
 assert report['reset_mode']=='explicit' and report['start_frame']==0
 assert report['world_count']==1 and report['step_count']==r['frames']
 assert not report['non_finite_worlds'] and report['sim_control_fps']==20
 assert report['environment_steps']==state['training_steps']
 with Path(state['checkpoint']).open('rb') as f:
  n=struct.unpack('<Q',f.read(8))[0];metadata=json.loads(f.read(n))['__metadata__']
 assert int(metadata['user.environment_steps'])==state['training_steps']
 assert float(metadata['user.voc_scale'])==0, f'{ep}: final checkpoint still assisted'
 command=json.loads((BASE/ep/'training/command.json').read_text())
 reproduction.append({'episode':ep,'checkpoint_sha256':hashlib.sha256(Path(state['checkpoint']).read_bytes()).hexdigest(),
  'reference_sha256':hashlib.sha256((BASE/ep/'reference.parquet').read_bytes()).hexdigest(),
  'training_command':command,'evaluation_report':report})
 metrics=report['metrics'];updates=[json.loads(line) for line in (BASE/ep/'training/upper_updates.jsonl').read_text().splitlines()]
 for i,obj in enumerate(r['objects']):
  quality.append({'episode':ep,'object':obj,'position_error_cm':100*metrics['object_position_error_per_body_m'][i],
   'orientation_error_rad':metrics['object_orientation_error_per_body_rad'][i],
   'upper_updates_tested':len(updates),'upper_updates_accepted':sum(u['accepted'] for u in updates),
   'reference':'video reconstruction; not hidden ground truth'})

 subprocess.run([sys.executable,'/data1/SRL_V2D/tools/track3_validate_trajectory.py',str(BASE/'results'/(ep+'.parquet')),'--kit',str(KIT)],check=True)
OUT.mkdir(exist_ok=True)
(OUT/'reproduction.json').write_text(json.dumps(reproduction,indent=2))
with (OUT/'local_reference_tracking.csv').open('w',newline='') as f:
 writer=csv.DictWriter(f,fieldnames=list(quality[0]));writer.writeheader();writer.writerows(quality)
subprocess.run([sys.executable,str(KIT/'tools/pack_submission.py'),'--track','track_3','--results',str(BASE/'results'),'--code-commit',commit,'-o',str(OUT/'track3_poses.csv')],cwd=KIT,check=True)
shutil.copy2(MESH,OUT/'track3_meshes.parquet')
shutil.copy2(BASE/'REPOSITORY_ACCESS.txt',OUT/'REPOSITORY_ACCESS.txt')
report={'validated_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'episodes':20,'object_tracks':36,'code_commit':commit,'uploaded_to_kaggle':False,'source_results':str(BASE/'results'),'files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.glob('track3_*')}}
(OUT/'validation.json').write_text(json.dumps(report,indent=2))
(OUT/'README.txt').write_text('Track 3 submission files\n\ntrack3_poses.csv: official unassisted policy rollout; 20 episodes, 36 objects.\ntrack3_meshes.parquet: reconstructed meshes for CD-O.\n\nNo files have been submitted to Kaggle.\nAccept the competition rules, then run from the official submission kit:\nuv run --with kaggle --with pandas --with pyarrow --with scipy submit/v2d_submit.py /path/to/track3_poses.csv --track track_3 -m "HaWoR FoundationPose floating Sharpa chunk1s"\nuv run --with kaggle --with pandas --with pyarrow --with scipy submit/v2d_submit.py /path/to/track3_meshes.parquet --track track_3_mesh -m "Track3 reconstructed meshes"\n\nLocal rollout metrics compare to reconstructed references, not hidden leaderboard ground truth.\nProducing transfer code: '+commit+'\n')
with zipfile.ZipFile(BASE/'track3_submission_ready.zip','w',zipfile.ZIP_DEFLATED) as z:
 for p in sorted(OUT.iterdir()):z.write(p,p.name)
print(BASE/'track3_submission_ready.zip')
