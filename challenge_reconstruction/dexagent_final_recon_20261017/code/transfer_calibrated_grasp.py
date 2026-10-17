"""Test a measured object-local robot grasp against calibrated target physics.

These are command seeds, not learned policy checkpoints. All object targets,
initial states and physical properties stay fixed within the calibrated scene.
"""
from pathlib import Path
import argparse, json, hashlib
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from refine_grasp_cem import B, TR, dump, contact_envelopes
from wrist_reference import continuous_euler_xyz
from transfer_observed_grasp import hand_joints
from floating_runtime import execute_reference
from evaluate_mp_sr import evaluate_files


def run(index):
    W=B/'refinement_20261018/calibrated_scene_candidates'
    inp=W/'episodes/episode_000005'
    src=B/'refinement_20261018/episode_000003/retention_after45'
    base=W/'runs/episode_000005/temporal50'
    manifest=json.loads((inp/'input_manifest.json').read_text())
    donor=json.loads((B/'episodes/episode_000003/input_manifest.json').read_text())
    assets=lambda r: next(o for o in r['objects'] if o['id']=='wooden_spoon')
    sig=lambda a:(a['source_urdf_sha256'],[(m['kind'],m['sha256']) for m in a['meshes']])
    if sig(assets(manifest))!=sig(assets(donor)):
        raise ValueError('Grasp transfer requires identical spoon asset and frame')
    variants=[(a,b,p) for a in [2.8,3.5,5.,8.] for b in [.3,.6] for p in [0.,.04]]
    anchor,blend,preload=variants[index]
    out=W/'grasp_transfer5_v2'/f'candidate_{index:02d}'
    out.mkdir(parents=True,exist_ok=True)
    if (out/'result.json').exists():raise FileExistsError(out)
    model=mujoco.MjModel.from_xml_path(str(base/'execution/scene_physics.xml'))
    sm=mujoco.MjModel.from_xml_path(str(src/'execution/scene_physics.xml'))
    ref=TR.load_reference(base/'reference.npz',model)
    sr=TR.load_reference(src/'reference.npz',sm)
    actual=np.load(src/'execution/trajectory.npz')['qpos']
    sc=sm.jnt_qposadr[hand_joints(sm,'right')].astype(int)
    tc=model.jnt_qposadr[hand_joints(model,'right')].astype(int)
    sa=int(sm.joint('wooden_spoon_free').qposadr[0]);ta=int(model.joint('wooden_spoon_free').qposadr[0])
    k=round(anchor*sr['fps']);obs=actual[k]
    ro=Rotation.from_quat(obs[sa+3:sa+7],scalar_first=True)
    # Measured wrist plus bounded servo preload, never an object attachment.
    wrist=obs[sc].copy()
    wrist[:3]+=np.clip(sr['qpos'][k,sc[:3]]-wrist[:3],-.003,.003)
    wrist[6:]+=np.clip(sr['qpos'][k,sc[6:]]-wrist[6:],-preload,preload)
    local_p=ro.inv().apply(wrist[:3]-obs[sa:sa+3])
    local_r=ro.inv()*Rotation.from_euler('XYZ',wrist[3:6])
    Q=ref['qpos'].copy();t=np.arange(len(Q))/ref['fps']
    target_r=Rotation.from_quat(Q[:,ta+3:ta+7],scalar_first=True)
    target_p=Q[:,ta:ta+3]+target_r.apply(np.tile(local_p,(len(Q),1)))
    w=np.minimum(np.clip((t-1.0)/blend,0,1),np.clip((13.7-t)/.35,0,1))
    w=w*w*(3-2*w)
    Q[:,tc[:3]]+=(target_p-Q[:,tc[:3]])*w[:,None]
    original_r=Rotation.from_euler('XYZ',Q[:,tc[3:6]])
    corrected=original_r*Rotation.from_rotvec((original_r.inv()*target_r*local_r).as_rotvec()*w[:,None])
    Q[:,tc[3:6]]=continuous_euler_xyz(corrected,initial_angles=Q[0,tc[3:6]])
    Q[:,tc[6:]]+=(wrist[None,6:]-Q[:,tc[6:]])*w[:,None]
    for j in hand_joints(model,'right'):
        if model.jnt_limited[j]:
            a=int(model.jnt_qposadr[j]);Q[:,a]=np.clip(Q[:,a],*model.jnt_range[j])
    Q[0]=ref['qpos'][0]
    assert np.array_equal(Q[:,np.setdiff1d(np.arange(model.nq),tc)],ref['qpos'][:,np.setdiff1d(np.arange(model.nq),tc)])
    path=TR.save_reference(out/'reference.npz',model,Q,ref['labels'],ref['fps'],ref['segment'],ref['segments'],
        source='Observed spoon grasp command seed in independently calibrated scene; all object targets unchanged',floating_runtime=True)
    execution=execute_reference(base/'execution/scene_physics.xml',path,out/'execution')
    metrics=evaluate_files(manifest['source'],execution['trajectory'],inp/'input_manifest.json',execution['scene'],inp/'metric_geometry.json')
    dump(out/'result.json',dict(episode='episode_000005',input_dir=str(inp),donor=str(src),donor_anchor_s=anchor,
        blend_s=blend,preload=preload,reference=str(path),execution=execution,metrics=metrics,learned_policy=False))
    print(json.dumps(dict(index=index,metrics=metrics['declared_frame_reference_proxy'])),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--index',type=int,choices=range(16),required=True)
    run(**vars(p.parse_args()))
