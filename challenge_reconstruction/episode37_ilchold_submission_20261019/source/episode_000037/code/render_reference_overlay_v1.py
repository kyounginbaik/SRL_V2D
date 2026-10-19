"""Checkpoint reference playback: fixed initial ghost vs accepted current reference."""
from pathlib import Path
import sys,json,subprocess,shutil,csv
import numpy as np
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
out=root/'reference_overlays'/f'version_{version:04d}';out.mkdir(parents=True,exist_ok=True)
paths=[root/'training/references/reference_0000.npz',root/f'training/references/reference_{version:04d}.npz']
data=[np.load(p,allow_pickle=False) for p in paths]
if mode=='combine':
 for label in ['initial','current']:
  subprocess.run([sys.executable,__file__,phase,ep,variant,seed,str(version),label],check=True)
setup=setup_scene(parquet=B/'frozen'/ep/'reference.parquet',control_fps=20.,motion_speed=1.,embodiment=get_embodiment('sharpa_hands')(),collision=CollisionPolicy(),world_count=1,include_support=False,decompose_objects=False,
                  robot_reference_path=str(refit_path) if refit_path else None,
                  robot_reference_sha256=run_config.get('contact_refit_sha256') if refit_path else None)
s=setup.scene;ref=setup.reference
assert np.allclose(data[0]['joint_q'],s.robot_reference.joint_q)
flags=s.model.shape_flags.numpy();flags[s.world_count*s.collision_layout.shape_count]=0;s.model.shape_flags.assign(flags)
pos=ref.object_body_pos_w().reshape(-1,3);center=np.median(pos,axis=0);radius=max(.5,float(np.percentile(np.linalg.norm(pos-center,axis=1),95))+.35);cam=center+radius*np.array([1.8,-2.3,1.7])
base=build_reference_joint_q(s,ref);qs=[]
for d in data:
 q=base.copy();q[:,:s.robot_reference.num_joint_q]=d['joint_q'];assert np.isfinite(q).all();qs.append(q)
state=s.model.state()
def pose(q,i):
 state.joint_q.assign(q[i].astype(np.float32));newton.eval_fk(s.model,state.joint_q,state.joint_qd,state)
if mode!='combine':
 cfg=ViewerConfig(backend='mp4',output_path=str(out/f'{mode}.mp4'),num_frames=ref.num_frames,video_fps=20,width=960,height=540,headless=True,realtime=False,camera_position=tuple(cam),camera_target=tuple(center))
 v=create_viewer(cfg);v.set_model(s.model);configure_camera(v,cfg)
 try:
  for i in range(ref.num_frames):
   pose(qs[0 if mode=='initial' else 1],i);v.begin_frame(i/20);v.log_state(state);v.end_frame()
 finally:v.close()
 sys.exit(0)
# True translucent geometry in the interactive recording.
from newton.viewer import ViewerViser
class GhostViewer(ViewerViser):
 ghost=False
 def log_instances(self,name,mesh,xforms,scales,colors,materials,hidden=False):
  key=('/initial_ghost'+name) if self.ghost else name
  super().log_instances(key,mesh,xforms,scales,colors,materials,hidden)
  h=self._scene_handles.get(key)
  if self.ghost and h is not None:
   h.opacity=.65
   h.batched_colors=np.tile(np.array([75,185,255],np.uint8),(len(h.batched_positions),1))
player=B/'player';shutil.copytree(ViewerViser.get_viser_client_dir(),player,dirs_exist_ok=True)
v=GhostViewer(record_to_viser=str(out/'overlay.viser'));v.set_model(s.model)
for i in range(ref.num_frames):
 v.begin_frame(i/20)
 for ghost,q in [(False,qs[1]),(True,qs[0])]:
  v.ghost=ghost;pose(q,i);v.log_state(state)
 v.end_frame()
v.save_recording();v.close()
# MP4 overlap: same camera, original tinted cyan at 65% opacity over current.
print("REFERENCE_RENDERED",out,flush=True)
