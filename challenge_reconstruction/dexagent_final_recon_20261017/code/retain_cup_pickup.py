"""Preserve episode10's observed pickup, then test retained robot grasp commands."""
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

def s5(x):
    x=np.clip(x,0,1);return x**3*(10-15*x+6*x*x)

def run(anchor,clearance):
    ep='episode_000010';src=R/'hand_combinations'/ep/'original_left_cem_right';inp=B/'episodes'/ep
    model=mujoco.MjModel.from_xml_path(str(src/'execution/scene_physics.xml'))
    ref=TR.load_reference(src/'reference.npz',model);Q=ref['qpos'].copy();base=Q.copy();fps=ref['fps'];k=round(anchor*fps)
    recorded=np.load(src/'execution/trajectory.npz');actual=recorded['qpos'][k]
    prefix='robot_right_right_';wr=[model.joint(prefix+s).id for s in ('pos_x','pos_y','pos_z','rot_x','rot_y','rot_z')]
    finger=[j for j in range(model.njnt) if model.joint(j).name.startswith(prefix) and j not in wr]
    cols=model.jnt_qposadr[wr+finger].astype(int);a=int(model.joint('beige_cup_free').qposadr[0])
    ar=Rotation.from_quat(actual[a+3:a+7],scalar_first=True)
    localp=ar.inv().apply(base[k,cols[:3]]-actual[a:a+3]);localr=ar.inv()*Rotation.from_euler('XYZ',base[k,cols[3:6]])
    t=np.arange(len(Q))/fps;object_r=Rotation.from_quat(Q[:,a+3:a+7],scalar_first=True)
    goalp=Q[:,a:a+3]+object_r.apply(np.tile(localp,(len(Q),1)))
    goalp[:,2]+=clearance*s5((t-2.3)/.4)*(1-s5((t-4.3)/.3))
    goalr=object_r*localr;baser=Rotation.from_euler('XYZ',base[:,cols[3:6]])
    w=s5((t-anchor)/.35)*(1-s5((t-4.3)/.35))
    Q[:,cols[:3]]=(1-w[:,None])*base[:,cols[:3]]+w[:,None]*goalp
    Q[:,cols[3:6]]=continuous_euler_xyz(baser*Rotation.from_rotvec(w[:,None]*(baser.inv()*goalr).as_rotvec()),initial_angles=base[0,cols[3:6]])
    Q[:,cols[6:]]=(1-w[:,None])*base[:,cols[6:]]+w[:,None]*base[k,cols[6:]]
    Q[w==0]=base[w==0]
    assert np.array_equal(Q[:k+1],base[:k+1])
    untouched=np.setdiff1d(np.arange(model.nq),cols);assert np.array_equal(Q[:,untouched],base[:,untouched])
    out=R/'cup_observed_retention'/ep/f'a{anchor:.1f}_clear{clearance:.3f}';out.mkdir(parents=True,exist_ok=True)
    path=TR.save_reference(out/'reference.npz',model,Q,ref['labels'],fps,ref['segment'],ref['segments'],
        source='Retain observed cup pickup; unchanged object targets and left commands',floating_runtime=True)
    ex=execute_reference(src/'execution/scene_physics.xml',path,out/'execution')
    done=np.load(ex['trajectory']);prefix_error=float(np.max(np.abs(done['qpos'][:k+1]-recorded['qpos'][:k+1])))
    assert prefix_error==0.,'Observed pickup did not reproduce exactly'
    manifest=json.loads((inp/'input_manifest.json').read_text())
    metrics=evaluate_files(manifest['source'],ex['trajectory'],inp/'input_manifest.json',ex['scene'],inp/'metric_geometry.json')
    report=dict(episode=ep,anchor_s=anchor,clearance_m=clearance,reference=path,execution=ex,metrics=metrics,
        prefix_max_qpos_error=prefix_error,learned_policy=False,original_object_targets_preserved=True)
    (out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps({'output':str(out),'mp_sr':metrics['reference_proxy_mp_sr']}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--anchor',type=float,required=True);p.add_argument('--clearance',type=float,required=True)
    run(**vars(p.parse_args()))
