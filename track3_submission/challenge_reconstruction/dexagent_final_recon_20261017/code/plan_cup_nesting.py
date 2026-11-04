"""Test physically fitted cup placement via robot commands, with fixed object GT."""
from pathlib import Path
import argparse,json,sys
import numpy as np,mujoco
from scipy.spatial.transform import Rotation
B=Path(__file__).resolve().parents[1];R=B/'refinement_20261017'
sys.path.insert(0,str(B.parents[1]/'third_party/DexAgent'))
from tool_library.common import tracking as TR
from floating_runtime import execute_reference
from evaluate_mp_sr import evaluate_files
from wrist_reference import continuous_euler_xyz

def smooth(x):
    x=np.clip(x,0,1);return x**3*(10-15*x+6*x*x)

def run(index):
    anchor,lag,release=[(a,l,r) for a in (2.,2.2,2.4) for l in (0.,.5,1.) for r in (3.5,4.)][index]
    ep='episode_000010';src=R/'hand_combinations'/ep/'original_left_cem_right';inp=B/'episodes'/ep
    model=mujoco.MjModel.from_xml_path(str(src/'execution/scene_physics.xml'))
    ref=TR.load_reference(src/'reference.npz',model);base=ref['qpos'].copy();Q=base.copy();fps=ref['fps'];k=round(anchor*fps)
    recorded=np.load(src/'execution/trajectory.npz');actual=recorded['qpos'][k]
    prefix='robot_right_right_';wr=[model.joint(prefix+s).id for s in ('pos_x','pos_y','pos_z','rot_x','rot_y','rot_z')]
    fingers=[j for j in range(model.njnt) if model.joint(j).name.startswith(prefix) and j not in wr]
    cols=model.jnt_qposadr[wr+fingers].astype(int);a=int(model.joint('beige_cup_free').qposadr[0])
    ar=Rotation.from_quat(actual[a+3:a+7],scalar_first=True)
    wrist=actual[cols[:3]]+lag*(base[k,cols[:3]]-actual[cols[:3]])
    localp=ar.inv().apply(wrist-actual[a:a+3])
    actualwr=Rotation.from_euler('XYZ',actual[cols[3:6]])
    cmdwr=Rotation.from_euler('XYZ',base[k,cols[3:6]])
    localr=ar.inv()*actualwr*Rotation.from_rotvec(lag*(actualwr.inv()*cmdwr).as_rotvec())
    plan=json.loads((B/'refinement_20261018/nesting_plan'/ep/'report.json').read_text())['best_settled']['final_pose']
    goal=np.array(plan['beige_cup']['position_setup_m'])
    planr=Rotation.from_quat(plan['beige_cup']['quaternion_wxyz'],scalar_first=True)
    # Keep the acquired axial spin, rotating only the tilt to upright.
    oldaxis=ar.apply([0.,0.,1.]);newaxis=planr.apply([0.,0.,1.])
    cross=np.cross(oldaxis,newaxis);theta=np.arctan2(np.linalg.norm(cross),np.dot(oldaxis,newaxis))
    align=Rotation.from_rotvec(cross/max(np.linalg.norm(cross),1e-12)*theta)
    goalr=align*ar
    t=np.arange(len(Q))/fps
    objectp=np.tile(actual[a:a+3],(len(Q),1))
    via=goal.copy();via[2]=max(.28,float(actual[a+2]))
    move=smooth((t-anchor)/max(.25,2.8-anchor))
    objectp=(1-move[:,None])*objectp+move[:,None]*via
    down=smooth((t-2.8)/.6)
    objectp=(1-down[:,None])*objectp+down[:,None]*goal
    orient=ar*Rotation.from_rotvec(smooth((t-anchor)/.7)[:,None]*(ar.inv()*goalr).as_rotvec())
    wristp=objectp+orient.apply(np.tile(localp,(len(Q),1)))
    wristr=orient*localr
    blend=smooth((t-anchor)/.25)
    br=Rotation.from_euler('XYZ',base[:,cols[3:6]])
    Q[:,cols[:3]]=(1-blend[:,None])*base[:,cols[:3]]+blend[:,None]*wristp
    Q[:,cols[3:6]]=continuous_euler_xyz(br*Rotation.from_rotvec(blend[:,None]*(br.inv()*wristr).as_rotvec()),initial_angles=base[0,cols[3:6]])
    Q[:,cols[6:]]=(1-blend[:,None])*base[:,cols[6:]]+blend[:,None]*base[k,cols[6:]]
    opening=smooth((t-release)/.3)
    Q[:,cols[6:]]=(1-opening[:,None])*Q[:,cols[6:]]+opening[:,None]*base[round(5*fps),cols[6:]]
    Q[:,cols[2]]+=.16*smooth((t-release-.15)/.5)
    # Withdraw toward the original final hand pose only after releasing above
    # the fitted cup seat. Evaluation object poses are never edited.
    retreat=smooth((t-release-.7)/.7)
    Q[:,cols[:3]]=(1-retreat[:,None])*Q[:,cols[:3]]+retreat[:,None]*base[:,cols[:3]]
    currentr=Rotation.from_euler('XYZ',Q[:,cols[3:6]])
    Q[:,cols[3:6]]=continuous_euler_xyz(currentr*Rotation.from_rotvec(retreat[:,None]*(currentr.inv()*br).as_rotvec()),initial_angles=base[0,cols[3:6]])
    Q[:,cols[6:]]=(1-retreat[:,None])*Q[:,cols[6:]]+retreat[:,None]*base[:,cols[6:]]
    Q[:k+1]=base[:k+1]
    unchanged=np.setdiff1d(np.arange(model.nq),cols)
    assert np.array_equal(Q[:,unchanged],base[:,unchanged])
    out=B/'refinement_20261018/cup_nesting'/f'candidate_{index:02d}';out.mkdir(parents=True,exist_ok=True)
    path=TR.save_reference(out/'reference.npz',model,Q,ref['labels'],fps,ref['segment'],ref['segments'],source='Robot placement toward gravity-tested nest; original object GT untouched',floating_runtime=True)
    ex=execute_reference(src/'execution/scene_physics.xml',path,out/'execution')
    done=np.load(ex['trajectory']);prefix_error=float(np.max(np.abs(done['qpos'][:k+1]-recorded['qpos'][:k+1])))
    if prefix_error!=0.:raise RuntimeError('Pickup prefix changed')
    manifest=json.loads((inp/'input_manifest.json').read_text())
    metrics=evaluate_files(manifest['source'],ex['trajectory'],inp/'input_manifest.json',ex['scene'],inp/'metric_geometry.json')
    report=dict(episode=ep,anchor_s=anchor,wrist_lag_fraction=lag,release_s=release,reference=path,execution=ex,metrics=metrics,
                prefix_max_qpos_error=prefix_error,learned_policy=False,original_object_targets_preserved=True)
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(output=str(out),mp_sr=metrics['reference_proxy_mp_sr'],objects=metrics['official_style_reference_proxy']['objects'])),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--index',type=int,required=True);run(**vars(p.parse_args()))
