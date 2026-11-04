#!/usr/bin/env python3
"""Search a HaWoR pitcher grip while preserving the successful cup controller."""
from pathlib import Path
import argparse,json,os,sys
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from prepare_seven import BASE,LEGACY,ROOT,write
sys.path.insert(0,str(LEGACY/'code'));sys.path.insert(0,str(ROOT/'third_party/DexAgent'))
from tool_library.common import hand as H,tracking as TR
from transfer_observed_grasp import hand_joints
from wrist_reference import continuous_euler_xyz
from physics_pickup_search_v2 import optimize

def run(args):
    inp=BASE/'physics_collision_v2/episodes/episode_000020'
    prior=BASE/'diagnostics/episode20_hold/safe_release90'
    report=json.loads((prior/'result.json').read_text());scene=report['execution']['scene']
    manifest=json.loads((inp/'input_manifest.json').read_text());m=mujoco.MjModel.from_xml_path(scene)
    ref=TR.load_reference(report['reference'],m);q=ref['qpos'];fps=float(ref['fps'])
    original=TR.load_reference(inp/'tracking_reference.npz',m)['qpos']
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    cols=m.jnt_qposadr[hand_joints(m,'right')].astype(int);oa=int(m.joint('water_pitcher_free').qposadr[0]);k=round(args.anchor*fps)
    r=Rotation.from_quat(original[k,oa+3:oa+7],scalar_first=True)
    lp=r.inv().apply(original[k,cols[:3]]-original[k,oa:oa+3]);lr=r.inv()*Rotation.from_euler('XYZ',original[k,cols[3:6]])
    baseline=json.loads((BASE/'physics_collision_v2/runs/baseline_v1/episode_000020/baseline/result.json').read_text())
    rest=np.load(baseline['execution']['trajectory'],allow_pickle=False)['qpos'][6]
    rr=Rotation.from_quat(rest[oa+3:oa+7],scalar_first=True)
    anchor=q[0].copy();anchor[oa:oa+7]=rest[oa:oa+7]
    anchor[cols[:3]]=rest[oa:oa+3]+rr.apply(lp)
    anchor[cols[3:6]]=continuous_euler_xyz(Rotation.from_matrix((rr*lr).as_matrix()[None]),q[0,cols[3:6]])[0]
    anchor[cols[6:]]=original[k,cols[6:]]
    np.savez_compressed(out/'anchor_pose.npz',qpos=anchor)
    spec=dict(episode=inp.name,scene=scene,reference=report['reference'],original_reference=str(inp/'tracking_reference.npz'),
        source=manifest['source'],geometry=str(inp/'metric_geometry.json'),manifest=str(inp/'input_manifest.json'),
        object_name='water_pitcher',side='right',anchor_pose=str(out/'anchor_pose.npz'),open_fingers=original[0,cols[6:]].tolist(),
        times=dict(start=.05,hover=.35,contact=.60,closed=.90,transport=1.25,release=8.6),hover_offset=[.035,0,.035],
        horizon_s=3.5,score_hold_start_s=.9,score_track_start_s=1.4,
        hand_source='Observed HaWoR right-hand grasp transformed into measured settled pitcher frame',
        left_controller_preserved=True,object_targets_unchanged=True)
    write(out/'spec.json',spec)
    optimize(spec=spec,output=out/'search',samples=args.samples,iterations=args.iterations,workers=args.workers,seed=args.seed,resume=None)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--anchor',type=float,default=7.5);p.add_argument('--samples',type=int,default=64)
    p.add_argument('--iterations',type=int,default=12);p.add_argument('--workers',type=int,default=8);p.add_argument('--seed',type=int,default=44)
    run(p.parse_args())
