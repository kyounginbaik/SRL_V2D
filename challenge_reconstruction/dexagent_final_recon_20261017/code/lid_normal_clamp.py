"""Normal-directed lid pad preload, using bounded physical finger servos."""
from pathlib import Path
import argparse,json,sys
import numpy as np,mujoco,trimesh
from scipy.spatial import cKDTree
B=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(B.parents[1]/'third_party/DexAgent'))
from tool_library.common import hand as H
from lid_grasp_seed import contact_penetration
from predict_settled_grasp import smooth
from floating_runtime import execute_reference
from evaluate_mp_sr import evaluate_files
from calibrate_controller import tracking_audit
from finger_gain_sweep import grasp_audit


def run(task):
    candidate=task//2;depth=(.003,.006)[task%2]
    inputs=B/'refinement_20261017/lid_seed/episode_000001'
    spec=json.loads((inputs/'report.json').read_text())['exported'][candidate]
    scene=inputs/f'candidate_{candidate:02d}/execution/scene_physics.xml'
    ref=dict(np.load(inputs/f'candidate_{candidate:02d}.npz'));q=ref['qpos'].copy();fps=float(ref['fps'])
    forecast=dict(np.load(B/'refinement_20261017/settled_seed/episode_000001/passive_forecast/trajectory.npz'))
    model=mujoco.MjModel.from_xml_path(str(scene));data=mujoco.MjData(model)
    rig=H.HandRig.build(model,'robot_right_','right');lid=model.body('white_pot_lid').id
    frame=spec['geometry']['frame'];data.qpos[:]=q[frame]
    object_cols=[]
    for name in forecast['object_names']:
        adr=int(model.jnt_qposadr[model.joint(str(name)+'_free').id]);object_cols.extend(range(adr,adr+7))
    data.qpos[object_cols]=forecast['qpos'][frame,object_cols]
    mujoco.mj_forward(model,data);op=data.xpos[lid].copy();rot=data.xmat[lid].reshape(3,3).copy()
    mesh=trimesh.load(B.parent/'final_recon/episode_000001/assets/white_pot_lid_visual.obj',force='mesh',process=False)
    mesh.fix_normals(multibody=True)
    points,faces=trimesh.sample.sample_surface(mesh,60000,seed=43);normals=mesh.face_normals[faces]
    mask=points[:,0]>.018;points=points[mask];normals=normals[mask];tree=cKDTree(points)
    targets={};before={};directions={}
    for finger in ('thumb','index'):
        p=data.site_xpos[rig.fingertip_sites[finger]].copy()
        index=tree.query((p-op)@rot)[1];normal=rot@normals[index]
        before[finger]=p;directions[finger]=normal;targets[finger]=p-depth*normal
    cols=np.array(list(rig.finger_qpos.values()));start=data.qpos.copy()
    H.finger_ik(model,data,rig,targets,iters=80)
    delta=np.clip(data.qpos[cols]-start[cols],-.2,.2)
    # Cap intended target penetration: actual physics will resolve motor targets.
    accepted=False
    for scale in (1.,.75,.5,.25,0.):
        data.qpos[:]=start;data.qpos[cols]=start[cols]+scale*delta;mujoco.mj_forward(model,data)
        pen=contact_penetration(model,data,rig,lid)
        if pen['self']<.003 and pen['table']<.0015 and pen['target']<.006 and pen['other']<.003:
            accepted=True;break
    into={f:float((before[f]-data.site_xpos[rig.fingertip_sites[f]])@directions[f]) for f in before}
    times=np.arange(len(q))/fps
    blend=smooth((times-.95)/(.75))*(1-smooth((times-4.45)/.25))
    if accepted:q[:,cols]+=blend[:,None]*scale*delta
    assert np.array_equal(q[:,object_cols],ref['qpos'][:,object_cols])
    assert np.array_equal(q[0],ref['qpos'][0])
    out=B/f'refinement_20261017/lid_normal/episode_000001/candidate_{candidate:02d}_depth{round(depth*1000)}'
    out.mkdir(parents=True,exist_ok=True);reference=out/'reference.npz'
    np.savez_compressed(reference,**dict(ref,qpos=q))
    execution=execute_reference(scene,reference,out/'execution')
    metrics=evaluate_files(B.parent/'final_recon/episode_000001',execution['trajectory'],
        B/'episodes/episode_000001/input_manifest.json',execution['scene'])
    report=dict(candidate=candidate,requested_preload_m=depth,accepted=accepted,scale=scale,
        actual_inward_pad_travel_m=into,command_penetration_m=pen,
        max_finger_increment_rad=float(np.max(np.abs(scale*delta))),
        object_initial_physics_and_timing_unchanged=True,execution=execution,metrics=metrics,
        tracking=tracking_audit(execution['scene'],execution['trajectory'],reference),
        grasp=grasp_audit(execution['scene'],execution['trajectory']))
    (out/'report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k not in ('execution','metrics','grasp','tracking')},indent=2))
    print('MP-SR',metrics['reference_proxy_mp_sr'])


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--task',type=int,required=True)
    run(parser.parse_args().task)
