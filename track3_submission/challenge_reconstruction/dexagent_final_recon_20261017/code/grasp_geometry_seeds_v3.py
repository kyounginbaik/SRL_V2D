#!/usr/bin/env python3
"""Physical-seed hypotheses: earlier closure, bounded squeeze, observed settling.

Only the hand reference changes. Input objects, duration and initial state remain
exact. These priors require full physical execution; no grasp is certified here.
"""
import argparse
import json
from pathlib import Path
import sys

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from grasp_geometry_seeds_v2 import B, H, TR, nearest_euler


def smooth(x):
    x=np.clip(x,0.,1.)
    return x*x*(3.-2.*x)


def main(episode):
    ep=f'episode_{episode:06d}'
    source=B/'refinement_20261017/geometry_v2'/ep
    report=json.loads((source/'geometry_report.json').read_text())
    model=mujoco.MjModel.from_xml_path(report['scene'])
    data=mujoco.MjData(model)
    ref=TR.load_reference(B/'episodes'/ep/'tracking_reference.npz',model)
    original=ref['qpos'].copy();fps=ref['fps'];count=len(original)
    for side in ['left','right']:
        rig=H.HandRig.build(model,f'robot_{side}_',side)
        cols=[rig.wrist_qpos[k] for k in ['rx','ry','rz']]
        for t in range(1,count):original[t,cols]=nearest_euler(original[t,cols],original[t-1,cols])
    result=original.copy()
    physical=np.load(B/f'refinement_20261017/seed_physics_v2/{ep}/kp5000_r200_d5/execution/trajectory.npz')
    edits=[]
    for interval in report['grasp_intervals']:
        if not interval['applied']:continue
        side=interval['side'];name=interval['object']
        rig=H.HandRig.build(model,f'robot_{side}_',side)
        cols=np.array([rig.wrist_qpos[k] for k in ['x','y','z','rx','ry','rz']]+list(rig.finger_qpos.values()))
        start=interval['start_frame'];end=interval['end_frame']
        begin=max(1,start-round(.35*fps))
        localp=np.asarray(interval['object_from_wrist_position'])
        localR=np.asarray(interval['object_from_wrist_rotation'])
        closed=interval['finger_targets'];grip=interval['selected']['opposing_fingers']
        press=H.press_targets(rig,closed,grip,depth_rad=.12,max_travel_m=.003)
        # Reject extra pressure that increases self-overlap by more than 1 mm.
        anchor=interval['anchor_frame'];data.qpos[:]=original[anchor]
        for key,value in closed.items():data.qpos[rig.finger_qpos[key]]=value
        mujoco.mj_fwdPosition(model,data)
        def self_depth():
            return max([0.]+[-float(c.dist) for c in data.contact
                            if int(model.geom_bodyid[c.geom1]) in rig.body_ids
                            and int(model.geom_bodyid[c.geom2]) in rig.body_ids])
        initial_pen=self_depth();scale=1.
        for scale in [1.,.75,.5,.25,0.]:
            for key,value in closed.items():data.qpos[rig.finger_qpos[key]]=value+scale*(press[key]-value)
            mujoco.mj_fwdPosition(model,data)
            if self_depth()<=max(.002,initial_pen+.001):break
        fingers=np.array([closed[k]+scale*(press[k]-closed[k]) for k in rig.finger_qpos])
        # A completed physical rollout supplies pre-contact settling evidence.
        # Never use a displacement after the intended pickup to chase a failure.
        sample=max(0,start-round(.35*fps));oc=int(model.joint(name+'_free').qposadr[0])
        dp=physical['qpos'][sample,oc:oc+3]-original[sample,oc:oc+3]
        rr=Rotation.from_quat(original[sample,oc+3:oc+7],scalar_first=True)
        ar=Rotation.from_quat(physical['qpos'][sample,oc+3:oc+7],scalar_first=True)
        rv=(ar*rr.inv()).as_rotvec()
        drift_applied=bool(np.linalg.norm(dp)<.04 and np.linalg.norm(rv)<np.deg2rad(20))
        if not drift_applied:dp=np.zeros(3);rv=np.zeros(3)
        last=original[begin,cols[3:6]]
        bid=model.body(name).id
        for t in range(begin,end+1):
            data.qpos[:]=original[t];mujoco.mj_kinematics(model,data)
            op=data.xpos[bid].copy();orr=data.xmat[bid].reshape(3,3).copy()
            fade=1.-smooth((t-start)/(1.0*fps))
            dr=Rotation.from_rotvec(rv*fade).as_matrix()
            pos=op+dp*fade+dr@orr@localp
            angles=nearest_euler(H.mat_to_euler(dr@orr@localR),last);last=angles
            target=np.r_[pos,angles,fingers]
            approach=smooth((t-begin)/max(1,start-begin))
            release=smooth((end-t)/max(1,round(.25*fps)))
            alpha=min(approach,release)
            result[t,cols]=(1-alpha)*original[t,cols]+alpha*target
        edits.append({'side':side,'object':name,'begin_frame':begin,'closed_frame':start,'end_frame':end,
                      'pressure_scale':scale,'max_pad_press_travel_m':.003,
                      'settling_sample_frame':sample,'settling_applied':drift_applied,
                      'settling_translation_m':dp.tolist(),'settling_rotation_vector_rad':rv.tolist(),
                      'settling_correction_fade_seconds':1.0,'interval_source':str(source/'geometry_report.json')})
    for j in range(model.njnt):
        if model.joint(j).name.startswith('robot_'):
            c=int(model.jnt_qposadr[j]);result[:,c]=np.clip(result[:,c],*model.jnt_range[j])
    assert np.array_equal(result[0],original[0])
    for n in physical['object_names']:
        c=int(model.joint(str(n)+'_free').qposadr[0]);assert np.array_equal(result[:,c:c+7],original[:,c:c+7])
    out=B/'refinement_20261017/geometry_v3'/ep;out.mkdir(parents=True,exist_ok=True)
    target=out/'tracking_reference_geometry.npz'
    TR.save_reference(target,model,result,ref['labels'],fps,ref['segment'],ref['segments'],
                      source='geometry v3: earlier clamp and observed precontact drift; physics validation required',arms_coupled=False)
    (out/'geometry_report.json').write_text(json.dumps({'episode':ep,'edits':edits,'source':str(source),
        'objects_unchanged':True,'initial_state_unchanged':True,'time_unchanged':True,'physical_validation':False},indent=2)+'\n')
    print(json.dumps({'episode':ep,'output':str(target),'edits':edits}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--episodes',type=int,nargs='+',default=[3,5,8]);a=p.parse_args()
    for episode in a.episodes:main(episode)
