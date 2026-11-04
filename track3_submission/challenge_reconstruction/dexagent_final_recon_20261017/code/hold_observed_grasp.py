"""Test retention of a physically observed grasp using robot commands only.

The first part of the original physical trial is reproduced, then either finger
targets are held or the observed commanded wrist/object transform is carried
along the immutable target object path. There are no object constraints/forces.
"""
from pathlib import Path
import argparse,json,sys
import numpy as np,mujoco
from scipy.spatial.transform import Rotation
B=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(B.parents[1]/'third_party/DexAgent'))
from tool_library.common import tracking as TR
from floating_runtime import execute_reference
from wrist_reference import continuous_euler_xyz
from evaluate_mp_sr import evaluate_files

def run(episode,anchor,mode):
    name=f'episode_{episode:06d}';inp=B/'episodes'/name
    root=B/'refinement_20261017'
    if episode==8:
        side,obj='right','beige_cup'
        recdir=B/'runs/pilot_v2'/name/'seed_44/refined'
        refpath=B/'runs/pilot_v2'/name/'seed_44/mpc/reference.npz'
    elif episode==5:
        side,obj='left','white_pot'
        recdir=root/'seed_physics_v2'/name/'kp5000_r200_d5/execution'
        refpath=root/'geometry_v2'/name/'tracking_reference_geometry.npz'
    else:raise ValueError('Only independently observed partial grasps in episodes5/8')
    model=mujoco.MjModel.from_xml_path(str(recdir/'scene_physics.xml'))
    ref=TR.load_reference(refpath,model);original=TR.load_reference(inp/'tracking_reference.npz',model)
    measured=dict(np.load(recdir/'trajectory.npz'));base=ref['qpos'];Q=base.copy();fps=ref['fps']
    k=round(anchor*fps);prefix=f'robot_{side}_{side}_'
    wrist=[model.joint(prefix+s).id for s in ('pos_x','pos_y','pos_z','rot_x','rot_y','rot_z')]
    finger=[j for j in range(model.njnt) if model.joint(j).name.startswith(prefix) and j not in wrist]
    c=model.jnt_qposadr[wrist+finger].astype(int);adr=int(model.jnt_qposadr[model.joint(obj+'_free').id])
    # Anchor was selected from a prior genuine physical recording. Use desired
    # finger positions, preserving the finite motor preload at that instant.
    actual=measured['qpos'][k];orot=Rotation.from_quat(actual[adr+3:adr+7][[1,2,3,0]])
    wrot=Rotation.from_euler('XYZ',base[k,c[3:6]])
    localp=orot.inv().apply(base[k,c[:3]]-actual[adr:adr+3]);localr=orot.inv()*wrot
    target=base[:,c].copy();target[:,6:]=base[k,c[6:]]
    if mode=='rigid':
        r=Rotation.from_quat(original['qpos'][:,adr+3:adr+7][:,[1,2,3,0]])
        target[:,:3]=original['qpos'][:,adr:adr+3]+r.apply(np.tile(localp,(len(Q),1)))
        target[:,3:6]=continuous_euler_xyz(r*localr,initial_angles=base[0,c[3:6]])
    t=np.arange(len(Q))/fps
    # Preserve the actual successful pickup; blend only after the anchor.
    w=np.clip((t-anchor)/.35,0.,1.)
    # Release near the original final return, not while the object is airborne.
    w*=np.clip((t[-1]-t)/.35,0.,1.);w=w*w*(3-2*w)
    Q[:,c]=(1-w[:,None])*base[:,c]+w[:,None]*target
    for j in wrist+finger:
        if model.jnt_limited[j]:Q[:,model.jnt_qposadr[j]]=np.clip(Q[:,model.jnt_qposadr[j]],*model.jnt_range[j])
    assert np.array_equal(Q[:k+1],base[:k+1])
    nonrobot=[a for j in range(model.njnt) if not model.joint(j).name.startswith('robot_') for a in range(int(model.jnt_qposadr[j]),int(model.jnt_qposadr[j])+7)]
    assert np.array_equal(Q[:,nonrobot],original['qpos'][:,nonrobot])
    out=root/'observed_hold'/name/f'{mode}_{anchor:.1f}s';out.mkdir(parents=True,exist_ok=True)
    path=TR.save_reference(out/'reference.npz',model,Q,ref['labels'],fps,ref['segment'],ref['segments'],source=f'Observed-grasp retention test: {refpath}',floating_runtime=True)
    execution=execute_reference(recdir/'scene_physics.xml',path,out/'execution')
    manifest=json.loads((inp/'input_manifest.json').read_text())
    metrics=evaluate_files(manifest['source'],execution['trajectory'],inp/'input_manifest.json',execution['scene'],inp/'metric_geometry.json')
    # A retained-prefix test is valid only if pickup actually reproduces.
    done=dict(np.load(execution['trajectory']));prefix_error=float(np.max(np.abs(done['qpos'][:k+1]-measured['qpos'][:k+1])))
    report=dict(episode=name,mode=mode,anchor_seconds=anchor,side=side,object=obj,reference=path,execution=execution,metrics=metrics,
                prefix_replay_max_qpos_error=prefix_error,unchanged_objects=True,learned_policy=False)
    (out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--episode',type=int,required=True);p.add_argument('--anchor',type=float,required=True)
    p.add_argument('--mode',choices=['fingers','rigid'],required=True);run(**vars(p.parse_args()))
