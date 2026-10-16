"""Runs while lower training is paused at a completed checkpoint boundary."""
from pathlib import Path
import os,sys,subprocess,json
from flash_chord.training.checkpoint import read_checkpoint_metadata
B=Path(__file__).resolve().parents[1];phase=os.environ['UNIRETARGET_PHASE'];step=int(sys.argv[1]);seed=os.environ.get('UNIRETARGET_SEED','42');episode=os.environ.get('UNIRETARGET_EPISODE','episode_000001');args=[phase,episode,'chunk',seed];run=B/'runs'/phase/episode/'chunk'/f'seed_{seed}'
meta=read_checkpoint_metadata(run/f'training/policy_{step}.safetensors');version=int(meta['robot_reference_version'])
def call(script,*extra):subprocess.run([sys.executable,str(B/'code'/script),*args,*map(str,extra)],cwd='/workspace',check=True)
call('render_checkpoint.py',step,'both')
for saved_version in range(version+1):
 if not (run/'reference_overlays'/f'version_{saved_version:04d}'/'transparent_closeups'/'complete.json').is_file():
  call('refresh_overlay.py',saved_version)
# Local media is required; network logging failure is recorded and does not discard it.
with (run/'media_sync.log').open('a') as log:
 result=subprocess.run([sys.executable,str(B/'code/wandb_sync.py'),*args,'--once'],cwd='/workspace',stdout=log,stderr=subprocess.STDOUT)
(run/'last_media_status.json').write_text(json.dumps({'checkpoint_step':step,'reference_version':version,'voc_scale':float(meta['voc_scale']),'local_media_complete':True,'wandb_sync_returncode':result.returncode},indent=2)+'\n')
