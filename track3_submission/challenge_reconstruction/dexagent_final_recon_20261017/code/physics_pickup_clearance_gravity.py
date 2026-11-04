"""Physical pickup CEM with a vertical clearance leg before table approach.

The shared physical evaluator is unchanged. This module changes robot commands
only and uses a separately named worker so concurrent searches remain stable.
"""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import time
import json
import numpy as np
import mujoco
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from physics_pickup_search_gravity import PickupExperiment,blend,rotations_between,dump
from physics_pickup_search_gravity import _freeze,execute_reference,evaluate_files,TR
from wrist_reference import continuous_euler_xyz


class ClearanceExperiment(PickupExperiment):
    def __init__(self,spec):
        super().__init__(spec)
        # A vertex-only nearest-neighbor query spuriously reports centimetre
        # gaps over broad, physically contacting mesh faces. Add deterministic
        # barycentric face samples; forces and retention still decide capture.
        fk=mujoco.MjData(self.m);fk.qpos[:]=self.anchor;mujoco.mj_kinematics(self.m,fk)
        samples=[];level=8
        weights=np.array([(i/level,j/level,1-(i+j)/level)
                          for i in range(level+1) for j in range(level+1-i)])
        for geom in self.geom_ids:
            mid=int(self.m.geom_dataid[geom])
            if mid<0:continue
            va=int(self.m.mesh_vertadr[mid]);vn=int(self.m.mesh_vertnum[mid])
            fa=int(self.m.mesh_faceadr[mid]);fn=int(self.m.mesh_facenum[mid])
            vertices=self.m.mesh_vert[va:va+vn].astype(float)
            faces=self.m.mesh_face[fa:fa+fn]
            local=np.einsum('pj,fjk->fpk',weights,vertices[faces]).reshape(-1,3)
            world=local@fk.geom_xmat[geom].reshape(3,3).T+fk.geom_xpos[geom]
            samples.append(self.anchor_R.inv().apply(world-self.anchor_p))
        if samples:
            self.surface=np.vstack(samples);self.tree=cKDTree(self.surface)

    def commands(self,x):
        Q,timing=super().commands(x)
        clearance=self.spec.get('clearance_time_s')
        if clearance is None:return Q,timing
        start=self.start_frame/self.fps;hover=self.ts['hover']
        if not start<clearance<hover:raise ValueError('clearance waypoint must be between start and hover')
        initial=self.Q[self.start_frame]
        target=Q[round(hover*self.fps)]
        pa=initial[self.wcols[:3]].copy();pc=pa.copy()
        pc[2]=max(pc[2],self.spec.get('clearance_min_wrist_z_m',.30))
        ra=Rotation.from_euler('XYZ',initial[self.wcols[3:]])
        rb=Rotation.from_euler('XYZ',target[self.wcols[3:]])
        for a,b,p0,p1,r0,r1,f0,f1 in (
            (start,clearance,pa,pc,ra,ra,initial[self.fcols],initial[self.fcols]),
            (clearance,hover,pc,target[self.wcols[:3]],ra,rb,initial[self.fcols],target[self.fcols])):
            mask=(self.times>a)&(self.times<=b);w=blend((self.times[mask]-a)/(b-a))
            Q[np.ix_(mask,self.wcols[:3])]=(1-w[:,None])*p0+w[:,None]*p1
            r=rotations_between(r0,r1,w)
            Q[np.ix_(mask,self.wcols[3:])]=continuous_euler_xyz(r,Q[self.start_frame,self.wcols[3:]])
            Q[np.ix_(mask,self.fcols)]=(1-w[:,None])*f0+w[:,None]*f1
        Q[:self.start_frame+1]=self.Q[:self.start_frame+1]
        return Q,timing


WORKER=None
def initialize(spec):
    global WORKER
    WORKER=ClearanceExperiment(spec)


def evaluate(x):return WORKER.evaluate(x)


def optimize(spec,output,samples=128,iterations=24,workers=24,seed=141,resume=None):
    out=Path(output);out.mkdir(parents=True,exist_ok=True);spec=dict(spec)
    if (out/'result.json').exists():raise FileExistsError(out/'result.json')
    frozen_dir=out/'search_physics';frozen_dir.mkdir(parents=True,exist_ok=True)
    _,frozen=_freeze(spec['scene'],frozen_dir);spec['scene']=str(frozen.resolve())
    exp=ClearanceExperiment(spec);rng=np.random.default_rng(seed)
    mean=np.zeros(exp.dim);std=exp.scale.copy();best=mean.copy();best_report=exp.evaluate(best);history=[]
    if resume:
        old=json.loads(Path(resume).read_text());best=np.asarray(old['best_parameters']);mean=best.copy();best_report=exp.evaluate(best)
    dump(out/'settings.json',dict(spec=spec,samples=samples,iterations=iterations,workers=workers,
         seed=seed,dimension=exp.dim,scale=exp.scale,bound=exp.bound,
         command_adapter=str(Path(__file__).resolve()),score_adapter='physics_pickup_search_gravity.py'))
    dump(out/'initial.json',best_report);started=time.monotonic()
    with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn'),initializer=initialize,initargs=(spec,)) as pool:
        for iteration in range(iterations):
            # Half the local proposals stay centered on the best physical
            # rollout, retaining rare grasps when elite averaging moves away.
            n=samples-2;centers=np.repeat(mean[None],n,axis=0);centers[:n//2]=best
            proposals=np.vstack([best,mean,centers+rng.normal(size=(n,exp.dim))*std])
            proposals=np.clip(proposals,-exp.bound,exp.bound)
            reports=list(pool.map(evaluate,proposals,chunksize=1));order=np.argsort([r['cost'] for r in reports])
            elite=proposals[order[:max(6,samples//8)]]
            if reports[order[0]]['cost']<best_report['cost']:best=proposals[order[0]].copy();best_report=reports[order[0]]
            mean=.2*mean+.8*elite.mean(0);std=np.maximum(.3*std+.7*elite.std(0),.15*exp.scale)
            row=dict(iteration=iteration,best=best_report,elapsed_s=time.monotonic()-started);history.append(row)
            dump(out/'progress.json',dict(history=history,best_parameters=best,mean=mean,std=std))
            dump(out/f'iteration_{iteration:03d}.json',dict(proposals=proposals,reports=reports))
            print(json.dumps(row),flush=True)
    Q,timing=exp.commands(best)
    refpath=TR.save_reference(out/'reference.npz',exp.m,Q,exp.ref['labels'],exp.fps,exp.ref['segment'],exp.ref['segments'],
        source='Physical pickup CEM with vertical clearance; original object targets and timeline',arms_coupled=False,floating_runtime=True)
    short=exp.evaluate(best,return_trace=True);trace=short.pop('trace',np.empty((0,14)))
    np.savez_compressed(out/'search_trace.npz',trace=trace)
    execution=execute_reference(spec['scene'],refpath,out/'execution')
    metrics=evaluate_files(spec['source'],execution['trajectory'],spec['manifest'],execution['scene'],spec['geometry'])
    result=dict(best=best_report,parameters=best,reference=str(refpath),execution=execution,metrics=metrics,history=history,spec=spec)
    dump(out/'result.json',result);return result
