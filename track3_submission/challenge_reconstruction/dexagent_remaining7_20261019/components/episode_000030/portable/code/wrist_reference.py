"""Continuous equivalent intrinsic-XYZ coordinates for floating wrist servos.

This fixes representation jumps; it does not smooth or alter physical poses.
All three angles must be permitted to wrap by 2pi, including pitch.
"""
from __future__ import annotations
from pathlib import Path
import argparse,json
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation


def continuous_euler_xyz(rotations: Rotation, initial_angles=None) -> np.ndarray:
    """Nearest equivalent branch to the previous triple, preserving SO(3).

    `initial_angles` anchors the first branch. It need not equal the first
    rotation; output is always equivalent to `rotations`, never to the anchor.
    Intrinsic XYZ has two equivalent branches, each with independent 2pi wraps.
    This removes coordinate jumps but cannot remove the XYZ gimbal singularity.
    """
    angles=np.asarray(rotations.as_euler('XYZ'),float)
    if angles.ndim!=2 or angles.shape[1]!=3 or not len(angles):
        raise ValueError('rotations must contain a nonempty sequence of poses')
    prev=angles[0].copy() if initial_angles is None else np.asarray(initial_angles,float)
    if prev.shape!=(3,) or not np.isfinite(prev).all():
        raise ValueError('initial_angles must be a finite XYZ triple')
    out=np.empty_like(angles)
    for t,e in enumerate(angles):
        alternative=np.array([e[0]+np.pi,np.pi-e[1],e[2]+np.pi])
        candidates=[v+2*np.pi*np.round((prev-v)/(2*np.pi)) for v in (e,alternative)]
        out[t]=min(candidates,key=lambda v:(np.max(np.abs(v-prev)),np.linalg.norm(v-prev)))
        prev=out[t]
    error=(rotations.inv()*Rotation.from_euler('XYZ',out)).magnitude()
    if np.max(error)>1e-10:
        raise AssertionError('Euler continuity conversion changed physical orientations')
    return out


def repair_reference(model, source, destination):
    ref=dict(np.load(source));q=ref['qpos'].copy();original=q.copy();changes={};wrist_cols=[]
    for side in ('left','right'):
        ids=[model.joint(f'robot_{side}_{side}_rot_{k}').id for k in 'xyz']
        a=model.jnt_qposadr[ids];old=q[:,a].copy()
        new=continuous_euler_xyz(Rotation.from_euler('XYZ',old),initial_angles=old[0])
        # A no-op branch conversion must be a bitwise no-op. Tiny quaternion
        # roundtrip noise can otherwise perturb later contact bifurcations.
        unchanged=np.max(np.abs(new-old),axis=1)<1e-10
        new[unchanged]=old[unchanged]
        delta=(Rotation.from_euler('XYZ',old).inv()*Rotation.from_euler('XYZ',new)).magnitude()
        if not (np.all(new>=model.jnt_range[ids,0]-1e-8) and np.all(new<=model.jnt_range[ids,1]+1e-8)):
            raise ValueError(f'{side}: continuous path exceeds the declared wrist joint limits')
        q[:,a]=new;wrist_cols.extend(a)
        changes[side]=dict(max_equivalent_pose_error_rad=float(delta.max()),
            max_before_joint_step_deg=float(np.rad2deg(np.abs(np.diff(old,axis=0))).max()),
            max_after_joint_step_deg=float(np.rad2deg(np.abs(np.diff(new,axis=0))).max()),
            changed_frames=int(np.any(np.abs(new-old)>1e-8,axis=1).sum()))
    other=np.setdiff1d(np.arange(model.nq),wrist_cols)
    assert np.array_equal(q[:,other],original[:,other])
    assert np.allclose(q[0],original[0],rtol=0,atol=1e-10)
    ref['qpos']=q;np.savez_compressed(destination,**ref)
    changes['non_wrist_coordinates_unchanged']=True
    changes['original_timing_unchanged']=True
    return changes


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--export-all',action='store_true');a=p.parse_args()
    if not a.export_all:p.error('use --export-all')
    base=Path(__file__).resolve().parents[1];out=base/'refinement_20261017/continuous_references';out.mkdir(parents=True,exist_ok=True)
    reports={}
    for i in (1,3,5,8,10):
        episode=f'episode_{i:06d}'
        model=mujoco.MjModel.from_xml_path(str(base/'runs/pilot_v2'/episode/'seed_44/baseline/scene_physics.xml'))
        report=repair_reference(model,base/'episodes'/episode/'tracking_reference.npz',out/f'{episode}.npz')
        reports[episode]=report
    (out/'report.json').write_text(json.dumps(reports,indent=2));print(json.dumps(reports,indent=2))
