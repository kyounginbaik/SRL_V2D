"""Verify the handedness convention against the actual compiled Sharpa model."""
from pathlib import Path
import argparse,hashlib,json,sys
import numpy as np,mujoco
B=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(B.parents[1]/'third_party/DexAgent'))
from tool_library.common import hand as H

def audit(scene,output):
    scene=Path(scene);m=mujoco.MjModel.from_xml_path(str(scene));d=mujoco.MjData(m)
    l=H.HandRig.build(m,'robot_left_','left');r=H.HandRig.build(m,'robot_right_','right')
    keys=list(r.finger_qpos);rows=[];rng=np.random.default_rng(360)
    for sample in range(16):
        mujoco.mj_resetDataKeyframe(m,d,0)
        for rig in [l,r]:d.qpos[list(rig.wrist_qpos.values())]=0.
        angles=[]
        for key in keys:
            jid=m.actuator_trnid[r.finger_act[key],0];lo,hi=m.jnt_range[jid];angle=float(rng.uniform(lo,hi));angles.append(angle)
            d.qpos[r.finger_qpos[key]]=angle;d.qpos[l.finger_qpos[key]]=angle
        mujoco.mj_kinematics(m,d)
        lp=np.array([d.site_xpos[l.fingertip_sites[f]] for f in H.FINGERS]);rp=np.array([d.site_xpos[r.fingertip_sites[f]] for f in H.FINGERS]);mirror=np.diag([1.,-1.,1.])
        error=np.linalg.norm(lp-rp@mirror,axis=1)
        rows.append(dict(sample=sample,finger_angles_rad=angles,errors_m=error.tolist(),maximum_error_m=float(error.max())))
    result=dict(scene=str(scene.resolve()),scene_sha256=hashlib.sha256(scene.read_bytes()).hexdigest(),joint_keys=keys,local_reflection_matrix=mirror.tolist(),finger_mapping='same joint angles; do not flip abduction/adduction signs',samples=rows,max_fingertip_mirror_error_m=max(row['maximum_error_m'] for row in rows),scope='Kinematic fingertip convention check; does not certify a transferred grasp or reflective object symmetry')
    if result['max_fingertip_mirror_error_m']>5e-6:raise AssertionError('Sharpa mirror convention did not match this compiled model')
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='samples'},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--output',required=True);audit(**vars(p.parse_args()))
