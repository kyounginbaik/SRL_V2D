#!/usr/bin/env python3
"""Twist only the left wrist around its measured opposing pinch contact line.

Both contact points are invariant under this rigid rotation; physical rollout
still determines whether the frictional grasp survives the changed motion.
Object trajectories, scene, finger commands and the other hand stay fixed.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
os.environ['E25_WORKSPACE']='/data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/physics_slots_v3'
import argparse,json,multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np,mujoco
from scipy.spatial.transform import Rotation
from refine_episode25_geometry import W,E,H,G,TR,Experiment,init,evaluate,execute_reference,evaluate_files,dump

def run(args):
    source=W/'refinement/episode_000025/pinch_pressure_v4'
    output=W/'refinement/episode_000025/twist_v1'
    output.mkdir(parents=True,exist_ok=False)
    report=json.loads((source/'result.json').read_text())
    spec=json.loads((source/'spec.json').read_text())
    exp=Experiment(spec);model=exp.model;cols=exp.cols['left'][:6]
    ref=TR.load_reference(report['reference'],model);original=ref['qpos']
    recorded=np.load(report['execution']['trajectory'])
    anchor=round(5.0*exp.fps);d=mujoco.MjData(model)
    d.qpos[:]=recorded['qpos'][anchor];d.qvel[:]=recorded['qvel'][anchor];d.ctrl[:]=recorded['ctrl'][anchor]
    mujoco.mj_forward(model,d)
    rig=H.HandRig.build(model,'robot_left_','left');target=model.body('wooden_piece_1').id
    points={key:[] for key in ('thumb','index')}
    for c in d.contact:
        b1,b2=(int(model.geom_bodyid[g]) for g in (c.geom1,c.geom2))
        other=b2 if b1==target else b1 if b2==target else -1
        if other<0:continue
        for key in points:
            if '_'+key+'_' in model.body(other).name:points[key].append(c.pos.copy())
    if not all(points.values()):raise ValueError('Anchor does not have actual thumb/index object contacts')
    wrist=rig.wrist_pos(d);wr=H.euler_to_mat(rig.wrist_euler(d))
    contacts=(np.stack([np.mean(points[key],axis=0) for key in ('thumb','index')])-wrist)@wr
    center=contacts.mean(0);axis=contacts[0]-contacts[1];axis/=np.linalg.norm(axis)
    dump(output/'contact_anchor.json',dict(source=str(source),anchor_seconds=5.,local_contacts=contacts,
        local_center=center,local_axis=axis,object_targets_unchanged=True,initial_state_unchanged=True,
        invariant='Rigid rotation about the line through the two measured contacts preserves both contact positions; actual friction/contact must pass physics'))
    proposals=[];params=[];times=np.arange(len(original))/exp.fps
    for start,end in ((4.6,5.8),(5.2,6.4)):
        for degrees in (-150,-120,-90,-60,-30,30,60,90,120,150,180):
            Q=original.copy();fraction=np.clip((times-start)/(end-start),0,1);fraction=fraction**3*(10-15*fraction+6*fraction**2)
            residual=0.;previous=Q[0,cols[3:6]].copy()
            for k in range(len(Q)):
                if fraction[k]<=0:previous=Q[k,cols[3:6]].copy();continue
                old=H.euler_to_mat(original[k,cols[3:6]])
                new=old@Rotation.from_rotvec(axis*np.deg2rad(degrees)*fraction[k]).as_matrix()
                pivot=original[k,cols[:3]]+old@center
                Q[k,cols[:3]]=pivot-new@center
                Q[k,cols[3:6]]=G.nearest_euler(H.mat_to_euler(new),previous);previous=Q[k,cols[3:6]].copy()
                residual=max(residual,float(np.max(np.linalg.norm((contacts@new.T+Q[k,cols[:3]])-(contacts@old.T+original[k,cols[:3]]),axis=1))))
            assert residual<1e-9
            other=np.setdiff1d(np.arange(model.nq),cols)
            assert np.array_equal(Q[:,other],original[:,other]) and np.array_equal(Q[times<=start],original[times<=start])
            # Reject out-of-range wrist proposals rather than clipping away the
            # contact-line invariant and then claiming the same refinement.
            if np.any(Q[:,cols]<exp.limits['left'][:6,0]-1e-8) or np.any(Q[:,cols]>exp.limits['left'][:6,1]+1e-8):continue
            idx=len(proposals);path=output/'proposals'/f'{idx:04d}.npz';path.parent.mkdir(exist_ok=True)
            TR.save_reference(path,model,Q,ref['labels'],ref['fps'],ref['segment'],ref['segments'],
                source='Measured left pinch contact-axis wrist twist; physical verification required',floating_runtime=True)
            proposals.append((idx,str(path)));params.append(dict(index=idx,start_seconds=start,end_seconds=end,twist_degrees=degrees,contact_position_residual_m=residual))
    dump(output/'parameters.json',params);dump(output/'spec.json',spec)
    rows=[]
    def score(row):return max(row['mean_position_cm'][0]/3.,row['mean_orientation_deg'][0]/30.)+max(0.,row['max_hand_pen_m']-.01)*100
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=mp.get_context('spawn'),initializer=init,initargs=(spec,)) as pool:
        for index,row in pool.map(evaluate,proposals):
            if row.get('unstable'):row.update(mean_position_cm=[1e6,1e6],mean_orientation_deg=[180,180],max_hand_pen_m=1.)
            row=dict(index=index,**row);rows.append(row)
            dump(output/'progress.json',dict(evaluated=len(rows),total=len(proposals),best=min(rows,key=score)))
            print(json.dumps(row),flush=True)
    dump(output/'scores.json',rows);best=min(rows,key=score);path=proposals[best['index']][1]
    execution=execute_reference(spec['scene'],path,output/'execution',expected_duration_s=(len(original)-1)/exp.fps)
    manifest=json.loads((E/'input_manifest.json').read_text())
    metrics=evaluate_files(manifest['source'],execution['trajectory'],E/'input_manifest.json',execution['scene'],E/'metric_geometry.json')
    dump(output/'result.json',dict(episode='episode_000025',reference=path,best=best,parameters=params[best['index']],execution=execution,metrics=metrics,learned_policy=False,
        object_targets_unchanged=True,other_hand_unchanged=True,finger_commands_unchanged=True,source=str(source)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=8);run(p.parse_args())
