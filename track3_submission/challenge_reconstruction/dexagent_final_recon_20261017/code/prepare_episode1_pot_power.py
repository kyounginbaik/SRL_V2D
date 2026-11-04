"""Native DexAgent grasp-pocket proposals on the pot handle, for physics CEM.

Proposal FK may use a passive forecast to locate the handle. Execution always
starts from the original scene keyframe; object targets are never changed.
"""
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
    base=B/'refinement_20261018/episode_000001/pot_inputs'
    out=B/'refinement_20261018/episode_000001/pot_power_inputs';out.mkdir(parents=True,exist_ok=True)
    spec=json.loads((base/'pot_00.json').read_text());m=mujoco.MjModel.from_xml_path(spec['scene'])
    rig=H.HandRig.build(m,'robot_left_','left');d=mujoco.MjData(m)
    original=TR.load_reference(spec['original_reference'],m)['qpos'];q=original[30].copy()
    forecast_path=B/'refinement_20261017/settled_seed/episode_000001/passive_forecast/trajectory.npz'
    forecast=np.load(forecast_path)
    for j in range(m.njnt):
        if m.jnt_type[j]==mujoco.mjtJoint.mjJNT_FREE:
            a=int(m.jnt_qposadr[j]);q[a:a+7]=forecast['qpos'][30,a:a+7]
    d.qpos[:]=q;mujoco.mj_forward(m,d);obj=m.body('white_pot').id
    op=d.xpos[obj].copy();R=d.xmat[obj].reshape(3,3).copy()
    pocket=H.grasp_pocket(m,d,rig,.85)
    point_local=np.array([.015,.003,.17]);point=op+R@point_local
    wc=np.array([rig.wrist_qpos[k] for k in ('x','y','z','rx','ry','rz')])
    for variant,sign in enumerate((1.,-1.)):
        position,euler=H.pocket_pose(pocket,point,-R[:,0],sign*R[:,1])
        pose=q.copy();pose[wc[:3]]=position
        pose[wc[3:]]=continuous_euler_xyz(Rotation.from_euler('XYZ',[euler]),original[30,wc[3:]])[0]
        closed=rig.finger_targets(.90);opened=rig.finger_targets(.20)
        pose[list(rig.finger_qpos.values())]=[closed[k] for k in rig.finger_qpos]
        path=out/f'pot_anchor_{variant:02d}.npz';np.savez_compressed(path,qpos=pose)
        candidate=dict(spec,anchor_pose=str(path),open_fingers=[opened[k] for k in rig.finger_qpos],
            grasp_generator='unmodified DexAgent H.grasp_pocket/H.pocket_pose/H.finger_targets',
            handle_point_local=point_local.tolist(),palm_direction_world=(-R[:,0]).tolist(),
            finger_direction_world=(sign*R[:,1]).tolist(),passive_forecast_for_proposal_only=str(forecast_path),
            physical_initial_state='unchanged original scene keyframe',
            scale=np.r_[[.012]*3,[.15]*3,[.02]*3,[.12]*3,[.15]*22,[.20]*22,[.015]*3,[.10]*3,[.10,.08,.12]].tolist())
        dump(out/f'pot_{variant:02d}.json',candidate)
    print(str(out),flush=True)


if __name__=='__main__':prepare()
