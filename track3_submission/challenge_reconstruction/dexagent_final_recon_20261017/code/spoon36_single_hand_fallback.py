"""Explicit alternate physical strategy: right carry, left safely withdrawn.

This is not a handover. The exact successful pickup prefix and all original
object goals/timestamps are retained. Only subsequent robot commands change.
"""
from pathlib import Path
import argparse,json
import numpy as np,mujoco
from physics_pickup_search import B,H,TR,blend,dump,execute_reference,evaluate_files

def main(variant):
    r=B/'refinement_20261018';base=r/'spoon36_long_grasp/search'
    source=base if variant<2 else r/'episode_000036/retention_after9'
    out=r/'episode_000036/right_only_fallback'/f'variant_{variant:02d}'
    out.mkdir(parents=True,exist_ok=True)
    if (out/'result.json').exists():raise FileExistsError(out)
    scene=base/'execution/scene_physics.xml';m=mujoco.MjModel.from_xml_path(str(scene))
    ref=TR.load_reference(source/'reference.npz',m);Q=ref['qpos'].copy();old=Q.copy()
    times=np.arange(len(Q))/ref['fps'];start=9.;k=round(start*ref['fps'])
    with np.load(base/'execution/trajectory.npz') as z:
        i=np.argmin(abs(z['timestamps_seconds']-start));actual=z['qpos'][i].copy()
    left=H.HandRig.build(m,'robot_left_','left');right=H.HandRig.build(m,'robot_right_','right')
    lc=np.array([left.wrist_qpos[key] for key in ['x','y','z','rx','ry','rz']]+list(left.finger_qpos.values()))
    # The original idle wrist is below the actual table-blocked hand. Withdraw
    # from its attained pose, with outward/upward clearance, not toward that
    # infeasible command. Initial command and all preceding physics stay exact.
    target=actual[lc].copy();target[:3]+=np.array([-.05,-.12,.08])
    alpha=blend((times-start)/.65)
    Q[:,lc]=(1-alpha[:,None])*old[k,lc]+alpha[:,None]*target
    if variant%2==1:
        curl=blend((times-start)/.35)
        for finger in ['middle','ring','pinky']:
            for joint,deg in [('MCP_FE',45.),('PIP',35.),('DIP',25.)]:
                key=f'{finger}_{joint}'
                if key not in right.finger_qpos:raise KeyError(key)
                col=right.finger_qpos[key];Q[:,col]=(1-curl)*Q[:,col]+curl*np.deg2rad(deg)
    for a in range(m.nu):
        col=int(m.jnt_qposadr[m.actuator_trnid[a,0]])
        Q[:,col]=np.clip(Q[:,col],*m.actuator_ctrlrange[a])
    Q[:k+1]=old[:k+1]
    assert np.array_equal(Q[:k+1],old[:k+1])
    for j in range(m.njnt):
        if m.jnt_type[j]==mujoco.mjtJoint.mjJNT_FREE:
            a=int(m.jnt_qposadr[j]);assert np.array_equal(Q[:,a:a+7],old[:,a:a+7])
    reference=TR.save_reference(out/'reference.npz',m,Q,ref['labels'],ref['fps'],ref['segment'],ref['segments'],source='Alternative right-only spoon carry; no human-role handover claim; exact source object targets and pickup prefix',floating_runtime=True)
    inp=B/'episodes/episode_000036';manifest=json.loads((inp/'input_manifest.json').read_text())
    execution=execute_reference(scene,reference,out/'execution')
    metrics=evaluate_files(manifest['source'],execution['trajectory'],inp/'input_manifest.json',execution['scene'],inp/'metric_geometry.json')
    with np.load(execution['trajectory']) as z,np.load(base/'execution/trajectory.npz') as b:
        mask=z['timestamps_seconds']<=start+1e-10
        prefix_exact=bool(np.array_equal(z['qpos'][mask],b['qpos'][mask]))
    assert prefix_exact
    dump(out/'result.json',dict(episode='episode_000036',input_dir=str(inp),strategy='right_only_alternative_not_handover',source_reference=str(source/'reference.npz'),unused_right_fingers_curled=bool(variant%2),prefix_exact_through_seconds=start,reference=str(reference),execution=execution,metrics=metrics,learned_policy=False))
    print(json.dumps(dict(variant=variant,metrics=metrics)),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--variant',type=int,choices=range(4),required=True);main(p.parse_args().variant)
