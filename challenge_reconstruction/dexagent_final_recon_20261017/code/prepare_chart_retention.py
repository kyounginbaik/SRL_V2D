"""Version the raw wrist coordinates for a previously validated chart test.

No object target, source clock, finger command, mesh or physics parameter is
edited. This adapter is for explicitly changed virtual wrist axes only.
"""
from pathlib import Path
import json
import shutil
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from rechart_floating_wrist import B, TR, transformed, dump


def run():
    source=B/'refinement_20261018/wrist_chart/episode_000035/chart_0'
    result=json.loads((source/'result.json').read_text())
    inp=Path(result['input_dir'])
    out=source/'adapter'
    out.mkdir(exist_ok=False)
    for name in ['input_manifest.json','metric_geometry.json','camera_trajectory.npz']:
        if (inp/name).exists(): shutil.copy2(inp/name,out/name)
    old=mujoco.MjModel.from_xml_path(json.loads((Path(result['source'])/'result.json').read_text())['execution']['scene'])
    new=mujoco.MjModel.from_xml_path(result['execution']['scene'])
    ref=TR.load_reference(inp/'tracking_reference.npz',old)
    frame=Rotation.from_quat(result['kinematic_validation']['chart_quat_wxyz'],scalar_first=True)
    cols=np.array([old.joint('robot_right_right_rot_'+a).qposadr[0] for a in 'xyz'])
    Q=ref['qpos'].copy();Q[:,cols]=transformed(Q,cols,frame)
    physical=[i for i in range(old.nbody) if 'robot_right_right_' in old.body(i).name and '_base_' not in old.body(i).name]
    od,nd=mujoco.MjData(old),mujoco.MjData(new)
    maxp=maxr=0.
    for before,after in zip(ref['qpos'],Q):
        od.qpos[:]=before;nd.qpos[:]=after
        mujoco.mj_kinematics(old,od);mujoco.mj_kinematics(new,nd)
        maxp=max(maxp,float(np.max(np.abs(od.xpos[physical]-nd.xpos[physical]))))
        maxr=max(maxr,float(np.max(np.abs(od.xmat[physical]-nd.xmat[physical]))))
    assert maxp<1e-6 and maxr<1e-5, (maxp,maxr)
    TR.save_reference(out/'tracking_reference.npz',new,Q,ref['labels'],ref['fps'],ref['segment'],ref['segments'],
        source='Original targets in a nonsingular virtual wrist chart; source object poses unchanged',floating_runtime=True)
    manifest=json.loads((out/'input_manifest.json').read_text())
    manifest['virtual_wrist_chart']=dict(scene=result['execution']['scene'],quaternion_wxyz=frame.as_quat(scalar_first=True).tolist(),
        source_adapter=str(inp),virtual_robot_axes_changed=True,physical_reference_hand_poses_preserved_to_xml_precision=True,
        max_position_difference_m=maxp,max_rotation_matrix_difference=maxr,objects_unchanged=True)
    dump(out/'input_manifest.json',manifest)
    dump(out/'chart_validation.json',manifest['virtual_wrist_chart'])
    print(out)


if __name__=='__main__':run()
