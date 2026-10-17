"""Curl only unused middle/ring/pinky digits of the physical spoon grasp.

Thumb, index, wrist commands, left hand, objects, clock, scene, servo gains and
force caps remain unchanged. Every candidate undergoes full physical replay.
"""
from pathlib import Path
import argparse,json,sys
import numpy as np,mujoco
B=Path(__file__).resolve().parents[1];TEACHER=B/'refinement_20261018/episode_000003/retention_after45'
sys.path.insert(0,str(B.parents[1]/'third_party/DexAgent'))
from tool_library.common import hand as H,tracking as TR
from floating_runtime import execute_reference
from evaluate_mp_sr import evaluate_files
from grasp_stability import audit_grasp_stability

def blend(u):
 u=np.clip(u,0,1);return u**3*(10-15*u+6*u*u)

def run(mode,amount,start_s=.8):
 out=B/f'refinement_20261018/episode_000003/unused_fingers_v2/{mode}_{amount:.2f}_start{start_s:.1f}'
 out.mkdir(parents=True,exist_ok=True)
 if (out/'result.json').exists():raise FileExistsError(out)
 scene=TEACHER/'execution/scene_physics.xml';m=mujoco.MjModel.from_xml_path(str(scene));ref=TR.load_reference(TEACHER/'reference.npz',m);Q=ref['qpos'].copy();base=Q.copy();original=TR.load_reference(B/'episodes/episode_000003/tracking_reference.npz',m)['qpos'];rig=H.HandRig.build(m,'robot_right_','right')
 columns={n:c for n,c in rig.finger_qpos.items() if H.finger_of_joint(n) in ['middle','ring','pinky']}
 assert len(columns)==13, columns
 t=np.arange(len(Q))/ref['fps'];weight=amount*blend((t-start_s)/.6)*(1-blend((t-14.45)/.35))
 for n,c in columns.items():
  if mode=='reference':target=original[:,c]
  elif mode=='neutral':
   target=np.full(len(Q),np.deg2rad(45. if n.endswith('MCP_FE') else 55. if n.endswith('PIP') else 25. if n.endswith('DIP') else 8. if n.endswith('CMC') else 0.))
  else:raise ValueError(mode)
  a=rig.finger_act[n];j=m.actuator_trnid[a,0];Q[:,c]=np.clip((1-weight)*base[:,c]+weight*target,*m.jnt_range[j])
 others=np.setdiff1d(np.arange(m.nq),list(columns.values()));assert np.array_equal(Q[:,others],base[:,others]);assert np.array_equal(Q[0],base[0])
 path=TR.save_reference(out/'reference.npz',m,Q,ref['labels'],ref['fps'],ref['segment'],ref['segments'],source='Only unused right middle/ring/pinky posture refinement; physical grasp and immutable object targets preserved',arms_coupled=False,floating_runtime=True)
 ex=execute_reference(scene,path,out/'execution');inp=B/'episodes/episode_000003';manifest=json.loads((inp/'input_manifest.json').read_text());metrics=evaluate_files(manifest['source'],ex['trajectory'],inp/'input_manifest.json',ex['scene'],inp/'metric_geometry.json')
 audit=audit_grasp_stability(ex['scene'],ex['trajectory'],frames_only=True);(out/'audit').mkdir(exist_ok=True);(out/'audit/grasp_stability_frames.json').write_text(json.dumps(audit,indent=2)+'\n')
 r=dict(episode='episode_000003',mode=mode,amount=amount,start_s=start_s,changed_robot_joints=list(columns),protected_commands_bitwise_equal=True,reference=str(path),execution=ex,metrics=metrics)
 (out/'result.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(dict(output=str(out),mp_sr=metrics['reference_proxy_mp_sr'],objects=metrics['official_style_reference_proxy']['objects'])),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--mode',choices=['reference','neutral'],required=True);p.add_argument('--amount',type=float,required=True);p.add_argument('--start-s',type=float,default=.8);run(**vars(p.parse_args()))
