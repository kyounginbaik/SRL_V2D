#!/usr/bin/env python3
"""Delay thumb engagement until the other fingers surround the spoon.

The isolated seed edits only thumb servo commands. Source objects and CEM's
successful left-hand commands stay exact; every candidate needs physics replay.
"""
import argparse,json
import numpy as np,mujoco
from grasp_geometry_seeds_v6 import B,ROOT,H,TR,surface


def smooth(x):
    x=np.clip(x,0.,1.);return x*x*(3.-2.*x)


def main(close_start):
    ep='episode_000003';source=B/'refinement_20261017/geometry_combo_v8_clamp'/ep/'tracking_reference_geometry.npz'
    rep=json.loads((B/'refinement_20261017/geometry_v8'/ep/'geometry_report.json').read_text())['grasp_intervals'][0]
    m=mujoco.MjModel.from_xml_path(str(B/'refinement_20261017/cem_v2'/ep/'execution/scene_physics.xml'));d=mujoco.MjData(m);rig=H.HandRig.build(m,'robot_right_','right')
    ref=TR.load_reference(source,m);Q=ref['qpos'].copy();fps=ref['fps'];anchor=round(2.*fps)
    d.qpos[:]=Q[anchor];oc=int(m.joint('wooden_spoon_free').qposadr[0]);d.qpos[oc:oc+7]=rep['predicted_settled_object_pose'];mujoco.mj_kinematics(m,d)
    bid=m.body('wooden_spoon').id;op=d.xpos[bid].copy();orr=d.xmat[bid].reshape(3,3).copy();sid=rig.fingertip_sites['thumb'];pos=d.site_xpos[sid].copy()
    points,normals,tree=surface(ROOT/f'challenge_reconstruction/final_recon/{ep}/assets/wooden_spoon.obj');idx=tree.query((pos-op)@orr)[1];normal=orr@normals[idx]
    cols=np.array([col for key,col in rig.finger_qpos.items() if H.finger_of_joint(key)=='thumb']);closed=d.qpos[cols].copy()
    residual=H.finger_ik(m,d,rig,{'thumb':pos+.015*normal},iters=100);opened=d.qpos[cols].copy();delta=np.clip(opened-closed,-.6,.6)
    d.qpos[cols]=closed+delta;mujoco.mj_kinematics(m,d);outward=float((d.site_xpos[sid]-pos)@normal)
    for t in range(len(Q)):
        sec=t/fps;weight=smooth((sec-1.30)/.25)*(1.-smooth((sec-close_start)/.20))
        Q[t,cols]+=weight*delta
    for j in range(m.njnt):
        if m.joint(j).name.startswith('robot_'):
            col=int(m.jnt_qposadr[j]);Q[:,col]=np.clip(Q[:,col],*m.jnt_range[j])
    assert np.array_equal(Q[0],ref['qpos'][0]);unchanged=np.setdiff1d(np.arange(m.nq),cols);assert np.array_equal(Q[:,unchanged],ref['qpos'][:,unchanged])
    name=f'geometry_stagger_thumb_{round(close_start*100):03d}';out=B/'refinement_20261017'/name/ep;out.mkdir(parents=True,exist_ok=True)
    path=out/'tracking_reference_geometry.npz';TR.save_reference(path,m,Q,ref['labels'],fps,ref['segment'],ref['segments'],source='thumb engagement delayed until opposite fingers approach; full physics execution required',arms_coupled=False)
    (out/'geometry_report.json').write_text(json.dumps({'source':str(source),'thumb_only':True,'closure_start_seconds':close_start,'closure_end_seconds':close_start+.20,'requested_outward_tip_m':.015,'achieved_outward_tip_m':outward,'ik_residual':residual,'thumb_joint_delta':delta.tolist(),'objects_unchanged':True,'left_CEM_commands_unchanged':True,'initial_state_unchanged':True},indent=2)+'\n')
    print(json.dumps({'reference':str(path),'outward_m':outward}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--close-start',type=float,nargs='+',default=[1.85,1.95,2.05]);a=p.parse_args()
    for x in a.close_start:main(x)
