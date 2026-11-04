"""Transfer a measured stable pot grasp as a proposal, never as object state."""
from pathlib import Path
import json,sys
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from wrist_reference import continuous_euler_xyz
from physics_pickup_search import dump

B=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(B.parents[1]/'third_party/DexAgent'))
from tool_library.common import hand as H,tracking as TR


def prepare():
    source=B/'refinement_20261018/episode_000003/retention_after45'
    record=np.load(source/'execution/trajectory.npz')
    sm=mujoco.MjModel.from_xml_path(str(source/'execution/scene_physics.xml'))
    base=B/'refinement_20261018/episode_000001/pot_inputs'
    out=B/'refinement_20261018/episode_000001/pot_donor_inputs';out.mkdir(parents=True,exist_ok=True)
    spec=json.loads((base/'pot_00.json').read_text())
    tm=mujoco.MjModel.from_xml_path(spec['scene'])
    target=TR.load_reference(spec['original_reference'],tm)['qpos'][34].copy()
    sr=H.HandRig.build(sm,'robot_left_','left');tr=H.HandRig.build(tm,'robot_left_','left')
    sc=np.array([sr.wrist_qpos[k] for k in ('x','y','z','rx','ry','rz')])
    tc=np.array([tr.wrist_qpos[k] for k in ('x','y','z','rx','ry','rz')])
    so=int(sm.joint('white_pot_free').qposadr[0]);to=int(tm.joint('white_pot_free').qposadr[0])
    i=int(np.argmin(abs(record['timestamps_seconds']-5.)))
    actual=record['qpos'][i].copy();command=actual.copy()
    command[sm.jnt_qposadr[sm.actuator_trnid[:,0]]]=record['ctrl'][i]
    object_R=Rotation.from_quat(actual[so+3:so+7],scalar_first=True)
    target_R=Rotation.from_quat(target[to+3:to+7],scalar_first=True)
    sref=TR.load_reference(B/'refinement_20261017/cem_v2/episode_000003/reference.npz',sm)
    opened=sref['qpos'][round(1.*sref['fps']),list(sr.finger_qpos.values())].copy()
    for variant,q in enumerate((actual,command)):
        pose=target.copy()
        local_p=object_R.inv().apply(q[sc[:3]]-actual[so:so+3])
        local_R=object_R.inv()*Rotation.from_euler('XYZ',q[sc[3:]])
        pose[tc[:3]]=target[to:to+3]+target_R.apply(local_p)
        wrist_R=target_R*local_R
        pose[tc[3:]]=continuous_euler_xyz(Rotation.from_matrix([wrist_R.as_matrix()]),target[tc[3:]])[0]
        pose[list(tr.finger_qpos.values())]=q[list(sr.finger_qpos.values())]
        path=out/f'pot_anchor_{variant:02d}.npz';np.savez_compressed(path,qpos=pose)
        this=dict(spec,anchor_pose=str(path),open_fingers=opened.tolist(),
                  donor_source=str(source),donor_time_seconds=float(record['timestamps_seconds'][i]),
                  donor_posture=('actual measured qpos' if variant==0 else 'recorded motor target'),
                  donor_wrist_actual_minus_commanded_m=(actual[sc[:3]]-command[sc[:3]]).tolist(),
                  donor_local_wrist_position=local_p.tolist(),donor_local_wrist_rotation=local_R.as_matrix().tolist(),
                  physical_initial_state='unchanged original episode1 scene keyframe; donor defines robot proposal only')
        dump(out/f'pot_{variant:02d}.json',this)
    print(str(out),flush=True)


if __name__=='__main__':prepare()
