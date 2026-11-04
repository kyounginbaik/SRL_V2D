"""Observed cup grasps as robot-only seeds for physical approach optimization.

Anchor object poses are planning coordinates only. Trials reset to the exact
frozen episode10 home; immutable object target arrays and source clock stay fixed.
"""
from pathlib import Path
import json,sys
import numpy as np,mujoco
from scipy.spatial.transform import Rotation
B=Path(__file__).resolve().parents[1];R=B/'refinement_20261017';OUT=B/'refinement_20261018/episode_000010/pickup_inputs'
sys.path.insert(0,str(B.parents[1]/'third_party/DexAgent'))
from tool_library.common import hand as H,tracking as TR
from wrist_reference import continuous_euler_xyz


def main():
    source=R/'hand_combinations/episode_000010/original_left_cem_right'
    scene=source/'execution/scene_physics.xml';m=mujoco.MjModel.from_xml_path(str(scene));rig=H.HandRig.build(m,'robot_right_','right')
    wc=np.array([rig.wrist_qpos[k] for k in ['x','y','z','rx','ry','rz']]);fc=np.array(list(rig.finger_qpos.values()))
    oc=int(m.joint('beige_cup_free').qposadr[0]);ref=TR.load_reference(source/'reference.npz',m);q=ref['qpos']
    observed=np.load(source/'execution/trajectory.npz');settled=observed['qpos'][round(.65*ref['fps'])]
    target_p=settled[oc:oc+3];target_R=Rotation.from_quat(settled[oc+3:oc+7],scalar_first=True)
    OUT.mkdir(parents=True,exist_ok=True)
    donors=[('native',source,2.0),('ep8',R/'feedback_polish/episode_000008/both_0.5',3.5)]
    for tag,donor,t in donors:
        dm=mujoco.MjModel.from_xml_path(str(donor/'execution/scene_physics.xml'));dz=np.load(donor/'execution/trajectory.npz');dq=dz['qpos'][round(t*float(dz['fps']))]
        drig=H.HandRig.build(dm,'robot_right_','right');dw=np.array([drig.wrist_qpos[k] for k in ['x','y','z','rx','ry','rz']]);dc=int(dm.joint('beige_cup_free').qposadr[0])
        dR=Rotation.from_quat(dq[dc+3:dc+7],scalar_first=True);lp=dR.inv().apply(dq[dw[:3]]-dq[dc:dc+3]);lR=dR.inv()*Rotation.from_euler('XYZ',dq[dw[3:]])
        anchor=settled.copy();anchor[wc[:3]]=target_p+target_R.apply(lp);anchor[wc[3:]]=continuous_euler_xyz(Rotation.from_matrix([(target_R*lR).as_matrix()]),q[0,wc[3:]])[0]
        # Preserve the actual donor actuator's pressure target rather than
        # replacing it with relaxed measured angles at a loaded contact.
        aids=np.array(list(drig.finger_act.values()));ctrl=dz['ctrl'][round(t*float(dz['fps']))]
        anchor[fc]=ctrl[aids]
        data=mujoco.MjData(m);data.qpos[:]=anchor;mujoco.mj_forward(m,data)
        # An object-radial opening leaves all fingers outside the cup and lets
        # CEM choose the symmetric closure timing and full22 target posture.
        tips={f:data.site_xpos[rig.fingertip_sites[f]].copy() for f in H.FINGERS}
        opening={}
        for f,p in tips.items():
            local=target_R.inv().apply(p-target_p);radial=local.copy();axis=np.array([0.,1.,0.])
            # Infer cup axial direction from its compiled surface PCA below.
            opening[f]=p
        verts=[];ob=m.body('beige_cup').id
        for g in range(m.ngeom):
            if m.geom_bodyid[g]!=ob or not m.geom_contype[g]:continue
            mi=m.geom_dataid[g]
            if mi<0:continue
            v=m.mesh_vert[m.mesh_vertadr[mi]:m.mesh_vertadr[mi]+m.mesh_vertnum[mi]]
            verts.extend(target_R.inv().apply(v@data.geom_xmat[g].reshape(3,3).T+data.geom_xpos[g]-target_p))
        pts=np.asarray(verts);_,ev=np.linalg.eigh(np.cov(pts.T));axis=ev[:,-1]
        for f,p in tips.items():
            v=target_R.inv().apply(p-target_p);v-=axis*(v@axis);v/=max(np.linalg.norm(v),1e-9)
            opening[f]=p+.005*target_R.apply(v)
        H.finger_ik(m,data,rig,opening,iters=100)
        path=OUT/f'anchor_{tag}.npz';np.savez_compressed(path,qpos=anchor)
        spec=dict(scene=str(scene),reference=str(source/'reference.npz'),original_reference=str(B/'episodes/episode_000010/tracking_reference.npz'),
            source=str(B.parents[1]/'challenge_reconstruction/final_recon/episode_000010'),geometry=str(B/'episodes/episode_000010/metric_geometry.json'),
            manifest=str(B/'episodes/episode_000010/input_manifest.json'),object_name='beige_cup',side='right',anchor_pose=str(path),
            open_fingers=data.qpos[fc].tolist(),finger_names=list(rig.finger_qpos),
            times=dict(start=.35,hover=.75,contact=1.1,closed=1.5,transport=2.15,release=4.6),horizon_s=2.8,hover_offset=[0.,0.,.08],
            donor=dict(path=str(donor),time_s=t,physical_state=True,finger_commands=True),
            planning_object_pose_time_s=.65,anchor_is_not_simulator_initial_state=True)
        spec['scale']=np.r_[[.010]*3,[.10]*3,[.02]*3,[.12]*3,[.12]*22,[.12]*22,[.012]*3,[.08]*3,[.10,.07,.10]].tolist()
        (OUT/f'pose_{tag}.json').write_text(json.dumps(spec,indent=2)+'\n');print(OUT/f'pose_{tag}.json')

if __name__=='__main__':main()
