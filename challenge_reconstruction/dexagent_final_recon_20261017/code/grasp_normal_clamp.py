#!/usr/bin/env python3
"""Add bounded inward pad targets to an antipodal grasp reference.

These are position-servo commands, never injected physical states. The source
objects, initial state, timestamp grid, collision model and forces stay fixed.
"""
import argparse,json
import numpy as np,mujoco
from scipy.spatial.transform import Rotation
from grasp_geometry_seeds_v6 import B,ROOT,H,TR,surface,FINGERS


def main(episode,depth=.003,source_version='v6'):
    ep=f'episode_{episode:06d}';inp=B/f'refinement_20261017/geometry_{source_version}'/ep
    report=json.loads((inp/'geometry_report.json').read_text())
    m=mujoco.MjModel.from_xml_path(report['scene']);d=mujoco.MjData(m)
    ref=TR.load_reference(inp/'tracking_reference_geometry.npz',m);Q=ref['qpos'].copy();fps=ref['fps'];edits=[]
    for interval in report['grasp_intervals']:
        if not interval['applied']:continue
        side=interval['side'];name=interval['object'];rig=H.HandRig.build(m,f'robot_{side}_',side)
        anchor=interval['anchor_frame'];close=interval['fully_closed_frame'];begin=interval['approach_start_frame'];end=interval['end_frame']
        oc=int(m.joint(name+'_free').qposadr[0]);settled=np.asarray(interval['predicted_settled_object_pose']);d.qpos[:]=Q[anchor]
        z=np.clip((anchor/fps-(close/fps+.15)),0.,1.);fade=z*z*(3-2*z)
        d.qpos[oc:oc+3]=(1-fade)*settled[:3]+fade*Q[anchor,oc:oc+3]
        sr=Rotation.from_quat(settled[3:],scalar_first=True);rr=Rotation.from_quat(Q[anchor,oc+3:oc+7],scalar_first=True)
        d.qpos[oc+3:oc+7]=(sr*Rotation.from_rotvec((sr.inv()*rr).as_rotvec()*fade)).as_quat(scalar_first=True)
        mujoco.mj_kinematics(m,d);bid=m.body(name).id;op=d.xpos[bid].copy();orr=d.xmat[bid].reshape(3,3).copy()
        points,normals,tree=surface(ROOT/f'challenge_reconstruction/final_recon/{ep}/assets/{name}.obj')
        targets={};before={};directions={}
        for finger in interval['selected']['opposing_fingers']:
            p=d.site_xpos[rig.fingertip_sites[finger]].copy();idx=tree.query((p-op)@orr)[1];normal=orr@normals[idx]
            before[finger]=p;directions[finger]=normal;targets[finger]=p-depth*normal
        cols=np.array(list(rig.finger_qpos.values()));closed=d.qpos[cols].copy();qstart=d.qpos.copy()
        H.finger_ik(m,d,rig,targets,iters=80);requested=d.qpos[cols].copy();delta=np.clip(requested-closed,-.20,.20)
        accepted=False
        for scale in [1.,.75,.5,.25,0.]:
            d.qpos[:]=qstart;d.qpos[cols]=closed+scale*delta;mujoco.mj_fwdPosition(m,d)
            pen={'self':0.,'table':0.,'target':0.,'other':0.}
            for c in d.contact:
                b1,b2=int(m.geom_bodyid[c.geom1]),int(m.geom_bodyid[c.geom2]);h1=b1 in rig.body_ids;h2=b2 in rig.body_ids
                if not(h1 or h2):continue
                if h1 and h2:kind='self'
                else:
                    other=b2 if h1 else b1;kind='target' if other==bid else 'table' if m.body(other).name=='table' else 'other'
                pen[kind]=max(pen[kind],-float(c.dist))
            if pen['self']<.003 and pen['table']<.0015 and pen['target']<.006 and pen['other']<.003:
                accepted=True;break
        into={f:float((before[f]-d.site_xpos[rig.fingertip_sites[f]])@directions[f]) for f in before}
        if accepted:
            for t in range(begin,end+1):
                alpha=min(1.,(t-begin)/max(1,close-begin),(end-t)/max(2,int(.25*fps)));alpha=max(0.,alpha);alpha=alpha*alpha*(3-2*alpha)
                Q[t,cols]+=alpha*scale*delta
        edits.append({'side':side,'object':name,'requested_pad_press_m':depth,'accepted':accepted,'scale':scale,
                      'actual_inward_pad_travel_m':into,'command_penetration_m':pen,'max_joint_increment_rad':float(np.max(abs(scale*delta)))})
    assert np.array_equal(Q[0],ref['qpos'][0])
    for name in np.load(B/'episodes'/ep/'reference_trajectory.npz')['object_names']:
        oc=int(m.joint(str(name)+'_free').qposadr[0]);assert np.array_equal(Q[:,oc:oc+7],ref['qpos'][:,oc:oc+7])
    out_version='v7' if source_version=='v6' else source_version+'_clamp'
    out=B/f'refinement_20261017/geometry_{out_version}_press{round(depth*1000)}'/ep;out.mkdir(parents=True,exist_ok=True)
    path=out/'tracking_reference_geometry.npz'
    TR.save_reference(path,m,Q,ref['labels'],fps,ref['segment'],ref['segments'],source='antipodal grasp with bounded normal clamp servo targets; physical test required',arms_coupled=False)
    (out/'geometry_report.json').write_text(json.dumps({'episode':ep,'source':str(inp),'edits':edits,'objects_unchanged':True,'initial_state_unchanged':True,'timing_unchanged':True},indent=2)+'\n')
    print(json.dumps({'episode':ep,'reference':str(path),'edits':edits}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--episodes',type=int,nargs='+',default=[3,5]);p.add_argument('--depth',type=float,default=.003);p.add_argument('--source-version',default='v6');a=p.parse_args()
    for ep in a.episodes:main(ep,a.depth,a.source_version)
