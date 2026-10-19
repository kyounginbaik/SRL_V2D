"""Checkpoint reference playback: fixed initial ghost vs accepted current reference."""
from pathlib import Path
import sys,json,subprocess,shutil,csv
import numpy as np
import pyarrow.parquet as pq
import newton,warp as wp
# GL's CPU fallback asks for CUDA-pinned host buffers; use ordinary host buffers
# on render-only CPU jobs with no CUDA driver. This does not affect FK values.
if not wp.is_cuda_available():
 _original_empty=wp.empty
 def _cpu_empty(*args,**kwargs):
  kwargs['pinned']=False
  return _original_empty(*args,**kwargs)
 wp.empty=_cpu_empty
import flash_chord.embodiments.sharpa_hands
from flash_chord.embodiments.base import get_embodiment
from flash_chord.scene.setup import setup_scene
from flash_chord.scene.collision import CollisionPolicy
from flash_chord.lifecycle.reset import build_reference_joint_q
from flash_chord.visualization.viewer import ViewerConfig,create_viewer,configure_camera
B=Path(__file__).resolve().parent.parent
phase,ep,variant,seed,version=sys.argv[1:6];version=int(version)
mode=sys.argv[6] if len(sys.argv)>6 else 'combine'
root=B/'runs'/phase/ep/variant/f'seed_{seed}'
run_config=json.loads((root/'run_config.json').read_text())
refit_path=(B/'refits'/ep/'reference.npz' if run_config.get('contact_refit') else
            B/'surface_refits'/ep/'reference.npz' if run_config.get('surface_refit') else
            B/'local_refined'/ep/'reference.npz' if run_config.get('local_refined') else None)
_snap=[x.split('=',1)[1] for x in run_config.get('training_command',[]) if x.startswith('task.robot_reference_path=')]
_snap_sha=[x.split('=',1)[1] for x in run_config.get('training_command',[]) if x.startswith('task.robot_reference_sha256=')]
if refit_path is None and _snap:
 # Warm-started runs: version 0 is the supplied snapshot, not the parquet IK reference.
 refit_path=Path(_snap[0]);run_config['contact_refit_sha256']=_snap_sha[0] if _snap_sha else None
out=root/'reference_overlays'/f'version_{version:04d}';out.mkdir(parents=True,exist_ok=True)
out=out/'transparent_closeups';out.mkdir(exist_ok=True)
paths=[root/'training/references/reference_0000.npz',root/f'training/references/reference_{version:04d}.npz']
data=[np.load(p,allow_pickle=False) for p in paths]
if mode=='combine':
 for label in ['initial','current','initial_hands']:
  subprocess.run([sys.executable,__file__,phase,ep,variant,seed,str(version),label],check=True)
if mode=='combine':sys.exit(0)
reference_path=B/'frozen'/ep/'reference.parquet'
fps=float(pq.read_table(reference_path,columns=['fps']).to_pylist()[0]['fps'])
setup=setup_scene(parquet=reference_path,control_fps=fps,motion_speed=1.,embodiment=get_embodiment('sharpa_hands')(),collision=CollisionPolicy(),world_count=1,include_support=bool(run_config.get('support_table')),decompose_objects=False,
                  robot_reference_path=str(refit_path) if refit_path else None,
                  robot_reference_sha256=run_config.get('contact_refit_sha256') if refit_path else None)
s=setup.scene;ref=setup.reference
assert np.allclose(data[0]['joint_q'],s.robot_reference.joint_q)
flags=s.model.shape_flags.numpy()
if mode=='initial_hands':
 # The original-hand ghost must not include objects, support, or ground.
 flags[:s.collision_layout.robot.start]=0
 flags[s.collision_layout.robot.stop:]=0
elif s.collision_policy.ground:
 flags[s.world_count*s.collision_layout.shape_count:]=0
s.model.shape_flags.assign(flags)
pos=ref.object_body_pos_w().reshape(-1,3);center=np.median(pos,axis=0);radius=max(.5,float(np.percentile(np.linalg.norm(pos-center,axis=1),95))+.35);cam=center+radius*np.array([1.8,-2.3,1.7])
base=build_reference_joint_q(s,ref);qs=[]
for d in data:
 q=base.copy();q[:,:s.robot_reference.num_joint_q]=d['joint_q'];assert np.isfinite(q).all();qs.append(q)
state=s.model.state()
def pose(q,i):
 state.joint_q.assign(q[i].astype(np.float32));newton.eval_fk(s.model,state.joint_q,state.joint_qd,state)
if mode!='combine':
 cfg=ViewerConfig(backend='mp4',output_path=str(out/f'{mode}.mp4'),num_frames=ref.num_frames,video_fps=fps,width=1920,height=1080,headless=True,realtime=False,camera_position=tuple(cam),camera_target=tuple(center))
 v=create_viewer(cfg);v.set_model(s.model);configure_camera(v,cfg)
 (out/'projection.json').write_text(json.dumps({'view':v.camera.get_view_matrix().reshape(4,4).transpose().tolist(),'projection':v.camera.get_projection_matrix().reshape(4,4).transpose().tolist(),'width':1920,'height':1080,'support_table':run_config.get('support_table')}))
 try:
  for i in range(ref.num_frames):
   pose(qs[0 if mode in ('initial','initial_hands') else 1],i);v.begin_frame(i/fps);v.log_state(state);v.end_frame()
 finally:v.close()
 sys.exit(0)
