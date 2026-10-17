"""Bounded thumb/index position-target overdrive, checked only in real physics.

Position targets may pass the contact surface to develop finite pinch force;
actual joints/contacts are simulated and replay verified. No force cap changes.
"""
from pathlib import Path
import argparse,json,sys
import numpy as np,mujoco
B=Path(__file__).resolve().parents[1];sys.path.insert(0,str(B.parents[1]/'third_party/DexAgent'))
from floating_runtime import execute_reference
from evaluate_mp_sr import evaluate_files
from predict_settled_grasp import smooth


def run(candidate,scale):
    source=B/f'refinement_20261017/lid_seed/episode_000001/candidate_{candidate:02d}.npz'
    scene=B/f'refinement_20261017/lid_seed/episode_000001/candidate_{candidate:02d}/execution/scene_physics.xml'
    out=B/f'refinement_20261017/lid_pressure/episode_000001/candidate_{candidate:02d}_press_{scale:g}'
    out.mkdir(parents=True,exist_ok=True)
    m=mujoco.MjModel.from_xml_path(str(scene));ref=dict(np.load(source));q=ref['qpos'].copy();old=q.copy()
    t=np.arange(len(q))/float(ref['fps'])
    weight=smooth((t-.95)/.65)*(1-smooth((t-4.45)/.25))
    offsets={'thumb_CMC_FE':.025,'thumb_MCP_FE':.04,'thumb_IP':.04,
             'index_MCP_FE':.05,'index_PIP':.12,'index_DIP':.08}
    changed=[];clipped=0
    for joint,value in offsets.items():
        jid=m.joint('robot_right_right_'+joint).id;adr=int(m.jnt_qposadr[jid])
        raw=q[:,adr]+weight*scale*value;limited=np.clip(raw,*m.jnt_range[jid]);clipped+=int(np.sum(raw!=limited));q[:,adr]=limited;changed.append(adr)
    others=np.setdiff1d(np.arange(m.nq),changed)
    assert np.array_equal(q[:,others],old[:,others]);assert np.array_equal(q[0],old[0])
    ref['qpos']=q;reference=out/'reference.npz';np.savez_compressed(reference,**ref)
    execution=execute_reference(scene,reference,out/'execution')
    metrics=evaluate_files(B.parent/'final_recon/episode_000001',execution['trajectory'],B/'episodes/episode_000001/input_manifest.json',execution['scene'])
    report=dict(candidate=candidate,pressure_scale=scale,joint_offsets_rad=offsets,clipped_samples=clipped,
                force_caps_unchanged=True,object_reference_unchanged=True,initial_state_unchanged=True,
                execution=execution,metrics=metrics)
    (out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2));return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--task',type=int,required=True);a=p.parse_args()
    run(a.task//4,[.5,1.,1.5,2.][a.task%4])
