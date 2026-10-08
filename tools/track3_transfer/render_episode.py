"""Render the final checkpoint; this preview is separate from official exports."""
from pathlib import Path
import json,os,subprocess,sys
import numpy as np,pyarrow.parquet as pq
BASE=Path(os.environ.get('V2D_TRANSFER_RUN',Path(__file__).resolve().parent.parent));ep=sys.argv[1]
state=json.loads((BASE/ep/'stage.json').read_text());assert state['stage']=='complete'
reference=pq.read_table(BASE/ep/'reference.parquet').to_pylist()[0]
positions=np.asarray(reference['object_body_position']).reshape(-1,3)
center=np.median(positions,axis=0);radius=max(.5,float(np.percentile(np.linalg.norm(positions-center,axis=1),95))+.35)
camera=center+radius*np.array([1.8,-2.3,1.7])
out=BASE/'videos';out.mkdir(exist_ok=True);path=out/(ep+'.mp4')
args=['scripts/view_policy.py',f'evaluation.checkpoint={state["checkpoint"]}',
 'evaluation.reset_mode=explicit','scene.world_count=1','training.world_count=1',
 'viewer.backend=mp4',f'viewer.output_path={path}',f'viewer.num_frames={len(reference["object_body_position"])}',
 'viewer.video_fps=20.0','viewer.width=1280','viewer.height=720',
 'viewer.headless=true','viewer.realtime=false','markers.enabled=false',
 'viewer.camera_position='+json.dumps(camera.tolist(),separators=(',',':')),
 'viewer.camera_target='+json.dumps(center.tolist(),separators=(',',':')),
 f'hydra.run.dir={out/"hydra"/ep}']
subprocess.run([sys.executable,*args],cwd='/workspace',check=True)
assert path.is_file() and path.stat().st_size>1000
(out/(ep+'.complete')).write_text('Rendered final checkpoint successfully\n')
print(path)
