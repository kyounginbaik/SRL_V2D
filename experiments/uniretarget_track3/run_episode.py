"""Train a floating Sharpa policy, then export and validate its official rollout."""
from pathlib import Path
import datetime as dt, hashlib, json, os, subprocess, sys
from checkpoint_health import check_checkpoint

BASE=Path(os.environ.get('V2D_TRANSFER_RUN',Path(__file__).resolve().parent.parent))
KIT=Path('/data1/video_to_data_challenge/v2d_submission_kit')
ROOT=Path('/workspace')
EP=sys.argv[1]
TOTAL=int(os.environ.get('V2D_TRAIN_STEPS','251658240'))
WORLDS=2048
RUN=BASE/EP/'training'
RUN.mkdir(parents=True,exist_ok=True)
RESULTS=BASE/'results';RESULTS.mkdir(exist_ok=True)

def status(stage,**extra):
 d={'episode':EP,'stage':stage,'updated_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'training_steps':TOTAL,'worlds':WORLDS,**extra}
 p=BASE/EP/'stage.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(d,indent=2));tmp.replace(p)
 print(json.dumps(d),flush=True)

def run(args):
 print('RUN',args,flush=True)
 subprocess.run([sys.executable,*map(str,args)],cwd=ROOT,check=True)

try:
 checkpoint=RUN/f'policy_{TOTAL}.safetensors'
 if not checkpoint.exists():
  if list(RUN.glob('policy_*.safetensors')):
   raise RuntimeError('Partial bi-level run exists; refusing to silently reset editor state. Select a fresh attempt directory.')
  status('training')
  args=['scripts/train_flash_sac.py',f'task.parquet={BASE/EP/"reference.parquet"}',
   'task.motion_speed=1.0','reset.seed=42','training.seed=42',f'scene.world_count={WORLDS}','scene.support=false',
   'sim.njmax=2048','sim.nconmax=512',f'training.total_environment_steps={TOTAL}',
   'training.updates_per_collection=8','training.replay.capacity=1048576',
   'training.replay.minimum_size=102400','training.replay.batch_size=1024',
   'training.checkpoint.save_interval_environment_steps=5033984',
   'curriculum.thresholds=[3960832,6930432,9895936,12865536,15835136,18804736,21770240,24739840,27709440,40267776,50331648]',
   '+adaptive_voc.interval=1048576','+adaptive_voc.required_passes=3','+bilevel.engine=ttsa_flashsac','+bilevel.mode=reactor_wrench',
   '+bilevel.study_variant=chunk','+bilevel.chunk_seconds=1.0',
   '+bilevel.warmup=5033984','+bilevel.interval=5033984',
   '+bilevel.eval_worlds=1','+bilevel.upper_trials=4',
   'logging.mode=disabled',f'output_dir={RUN}',f'hydra.run.dir={RUN/"hydra"}']
  if os.environ.get('V2D_FLOAT32') == '1':
   os.environ['V2D_CHECK_FINITE']='1'
   args += ['training.mixed_precision=false','training.compute_dtype=float32']
  start_substeps=int(os.environ.get('V2D_SUBSTEPS','5'))
  substep_trials=[n for n in (5,10,20,40) if n>=start_substeps] if os.environ.get('V2D_FLOAT32') == '1' else [start_substeps]
  for substeps in substep_trials:
   status('training',substeps=substeps,compute_dtype='float32' if os.environ.get('V2D_FLOAT32')=='1' else 'float16')
   attempt_args=args+[f'sim.substeps={substeps}']
   (RUN/'command.json').write_text(json.dumps(attempt_args,indent=2))
   try:
    run(attempt_args)
    break
   except subprocess.CalledProcessError:
    diagnostic=RUN/'nonfinite_transition.json'
    if not diagnostic.exists() or substeps==substep_trials[-1]:raise
    archive=RUN.with_name(f'failed_nonfinite_substeps{substeps}')
    if archive.exists():raise RuntimeError(f'Refusing to overwrite {archive}')
    RUN.rename(archive);RUN.mkdir()
    status('retrying_simulation',failed_substeps=substeps,diagnostic=str(archive/'nonfinite_transition.json'))

 if not checkpoint.is_file():raise RuntimeError('Final checkpoint missing')
 check_checkpoint(checkpoint)
 check_checkpoint(RUN/f"state_{TOTAL}.safetensors")
 status('official_unassisted_evaluation',checkpoint=str(checkpoint))
 run([KIT/'flash_chord/scripts/evaluate_policy.py',f'evaluation.checkpoint={checkpoint}',
  'evaluation.world_count=1','evaluation.reset_mode=explicit',
  f'evaluation.metrics_output={RESULTS/(EP+".json")}',
  f'evaluation.object_trajectories_output={RESULTS/(EP+".parquet")}',
  f'hydra.run.dir={RESULTS/"hydra"/EP}','logging.mode=disabled'])
 status('validating_rollout')
 run(['/data1/SRL_V2D/tools/track3_validate_trajectory.py',RESULTS/(EP+'.parquet'),'--kit',KIT])
 status('complete',checkpoint=str(checkpoint),trajectory=str(RESULTS/(EP+'.parquet')),
  checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest())
except Exception as e:
 status('failed',error=str(e));raise
