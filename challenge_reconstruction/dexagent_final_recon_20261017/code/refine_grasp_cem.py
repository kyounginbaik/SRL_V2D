"""Contact-window full-finger search evaluated by actual bounded-servo physics.

This refines robot commands only. Object targets, timestamps, initial state,
geometry, mass and effort limits are immutable within a run. No attachment or
object actuation is used. Final candidates require full independent replay and
the separate reference-proxy MP-SR evaluator. This is not a learned checkpoint.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time
for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '1'
import mujoco
import numpy as np
import pyarrow.parquet as pq
from scipy.ndimage import gaussian_filter1d, binary_closing
from scipy.spatial.transform import Rotation

B = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(B.parents[1] / 'third_party/DexAgent'))
from tool_library.common import episode as E
from tool_library.common import tracking as TR
from floating_runtime import servo_rollout, execute_reference, _freeze
from evaluate_mp_sr import angular_errors, validate_geometry, evaluate_files
from wrist_reference import continuous_euler_xyz

_WORKER = None


def dump(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, default=lambda x: x.tolist() if isinstance(x,np.ndarray) else str(x))+'\n')
    tmp.replace(path)


def contact_envelopes(source, frames, fps):
    row = pq.read_table(Path(source)/'reference.parquet').to_pylist()[0]
    out = {}
    for side in ('left','right'):
        normals = np.asarray(row.get(f'mano_{side}_object_contact_normals', []), float)
        if normals.ndim == 3 and len(normals) == frames:
            active = np.any(np.linalg.norm(normals, axis=-1) > .1, axis=1)
        else:
            active = np.zeros(frames, bool)
        # Source contact estimates are a search prior, never a physical weld.
        if active.sum() >= 3:
            active = binary_closing(active, structure=np.ones(max(2,round(.3*fps))))
            envelope = gaussian_filter1d(active.astype(float), max(1.,.12*fps))
        else:
            envelope = np.ones(frames)
        ramp = np.minimum(np.arange(frames)/max(1.,.3*fps),1.)
        out[side] = np.clip(envelope*ramp,0.,1.)
    return out


class Experiment:
    def __init__(self, spec):
        self.spec = spec
        self.model = mujoco.MjModel.from_xml_path(spec['scene'])
        self.ref = TR.load_reference(spec['reference'],self.model)
        self.base = self.ref['qpos'].copy()
        original = TR.load_reference(spec['original_reference'],self.model)
        self.target = original['qpos']
        self.fps = original['fps']
        if len(self.base) != len(self.target) or self.ref['fps'] != self.fps:
            raise ValueError('Seed must preserve full original timestamps')
        self.cols = {}; self.rot = {}; self.limits = {}; self.dim=0
        self.scale=[];self.bound=[]
        self.envelope=contact_envelopes(spec['source'],len(self.base),self.fps)
        for side in ('left','right'):
            prefix=f'robot_{side}_{side}_'
            wrist=[self.model.joint(prefix+k).id for k in ('pos_x','pos_y','pos_z','rot_x','rot_y','rot_z')]
            finger=[j for j in range(self.model.njnt) if self.model.joint(j).name.startswith(prefix) and j not in wrist]
            if len(finger)!=22: raise ValueError('Expected all22 independently controlled finger joints')
            js=wrist+finger; self.cols[side]=self.model.jnt_qposadr[js].astype(int)
            self.limits[side]=self.model.jnt_range[js]
            self.rot[side]=Rotation.from_euler('XYZ',self.base[:,self.cols[side][3:6]])
            self.scale.extend([.015]*3+[.08]*3+[.22]*22)
            self.bound.extend([.06]*3+[.45]*3+[.85]*22)
            self.dim += 28
        self.scale=np.asarray(self.scale);self.bound=np.asarray(self.bound)
        self.objects=E.object_free_joints(self.model);self.names=list(self.objects)
        self.anchors,self.axes,self.kinds=validate_geometry(self.names,json.loads(Path(spec['geometry']).read_text()))
        self.tp,self.tq=self.object_poses(self.target)
        self.T=len(self.base)
        data=mujoco.MjData(self.model)
        if self.model.nkey:mujoco.mj_resetDataKeyframe(self.model,data,0)
        self.initial=data.qpos.copy()
        if not np.allclose(self.initial,self.target[0],atol=1e-5,rtol=0):
            raise ValueError('Scene initial state differs from the immutable reference beyond XML precision')
        for adr in self.objects.values():
            if not np.array_equal(self.base[:,adr:adr+7],self.target[:,adr:adr+7]):
                raise ValueError('Seed modified object targets')

    def object_poses(self, Q):
        # Imported rigid free-body frames are the object's pose frames.
        p=np.stack([Q[:,a:a+3] for a in self.objects.values()],axis=1)
        q=np.stack([Q[:,a+3:a+7] for a in self.objects.values()],axis=1)
        R=Rotation.from_quat(q.reshape(-1,4)[:,[1,2,3,0]]).as_matrix().reshape(len(Q),len(self.names),3,3)
        return p+np.einsum('toij,oj->toi',R,self.anchors),q

    def commands(self, x):
        x=np.asarray(x).reshape(2,28);Q=self.base.copy()
        for si,side in enumerate(('left','right')):
            c=self.cols[side];w=self.envelope[side]
            Q[:,c[:3]]+=self.rot[side].apply(np.tile(x[si,:3],(self.T,1)))*w[:,None]
            r=self.rot[side]*Rotation.from_rotvec(w[:,None]*x[si,3:6])
            angles=continuous_euler_xyz(r, initial_angles=Q[0,c[3:6]])
            # Avoid gratuitous Euler reparameterization for the identity proposal.
            if np.any(x[si,3:6]):Q[:,c[3:6]]=angles
            Q[:,c[6:]]=np.clip(Q[:,c[6:]]+w[:,None]*x[si,6:],self.limits[side][6:,0],self.limits[side][6:,1])
        Q[0]=self.target[0]
        return Q

    def evaluate(self, x):
        Q=self.commands(x)
        roll=servo_rollout(self.model,Q,self.fps,qpos_start=self.initial,record_substeps=False)
        if roll['unstable'] or len(roll['qpos']) != self.T:
            return dict(cost=1e9,unstable=True)
        p,q=self.object_poses(roll['qpos'])
        ep=np.linalg.norm(p-self.tp,axis=-1)
        er=angular_errors(q[:,None],self.tq[:,None],self.axes,self.kinds)[:,0]
        pos=ep.mean(0);ang=er.mean(0)
        ratio=np.maximum(pos/.03,ang/np.deg2rad(30))
        # Max-object loss prevents sacrificing the other hand's object.
        # No force-reduction term can trade away the task objective.
        cost=float(ratio.max()+.25*ratio.mean())
        hand_pen=float(np.max(roll['hand_pen_m']))
        cost+=100*max(0.,hand_pen-.01)
        cost+=.001*float(np.mean((np.asarray(x)/self.bound)**2))
        return dict(cost=cost,mean_position_cm=(100*pos).tolist(),mean_orientation_deg=np.rad2deg(ang).tolist(),
                    max_hand_pen_m=hand_pen,mp_sr=float(np.all((pos<.03)&(ang<np.deg2rad(30)))),unstable=False)


def worker_init(spec):
    global _WORKER
    _WORKER=Experiment(spec)


def worker_evaluate(x):
    return _WORKER.evaluate(x)


def optimize(episode, scene, reference, output, samples=96, iterations=10, workers=16, seed=44):
    name=f'episode_{int(episode):06d}';src=B/'episodes'/name
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    if (out/'result.json').exists():raise FileExistsError(out/'result.json')
    manifest=json.loads((src/'input_manifest.json').read_text())
    # Search and final replay must use the same compiled physics and exact
    # serialized initial keyframe; nanometre initial changes can bifurcate contact.
    frozen_dir=out/'search_physics';frozen_dir.mkdir(exist_ok=True)
    _,frozen=_freeze(scene,frozen_dir)
    spec=dict(scene=str(frozen.resolve()),reference=str(Path(reference).resolve()),
              original_reference=str(src/'tracking_reference.npz'),source=manifest['source'],geometry=str(src/'metric_geometry.json'))
    start=time.monotonic()
    exp=Experiment(spec);rng=np.random.default_rng(seed)
    seed_reports=[dict(reference=spec['reference'],report=exp.evaluate(np.zeros(56)))]
    # Geometry-only fits may worsen actual contact. Compare against the
    # orientation-equivalent original before allocating the search budget.
    fallback=B/'refinement_20261017/continuous_references'/f'{name}.npz'
    if str(fallback.resolve()) != spec['reference']:
        alt_spec=dict(spec,reference=str(fallback.resolve()))
        alt=Experiment(alt_spec);alt_report=alt.evaluate(np.zeros(56))
        seed_reports.append(dict(reference=str(fallback),report=alt_report))
        if alt_report['cost']<seed_reports[0]['report']['cost']:
            spec,exp=alt_spec,alt
    dump(out/'seed_comparison.json',seed_reports)
    mean=np.zeros(exp.dim);std=exp.scale.copy();best=mean.copy();history=[]
    initial=next(r['report'] for r in seed_reports if r['reference']==spec['reference']);best_report=initial
    dump(out/'settings.json',dict(episode=name,spec=spec,samples=samples,iterations=iterations,workers=workers,seed=seed,
        parameter_dimension=56,parameterization='One local-frame wrist +22 independent finger corrections per hand, contact-weighted over original timeline',
        physics='bounded servos, unactuated objects, unchanged model/initial state/timing',learned_policy=False,
        input_scene=str(Path(scene).resolve()),
        source_sha256={k:hashlib.sha256(Path(v).read_bytes()).hexdigest() for k,v in spec.items() if k!='source'}))
    dump(out/'initial.json',initial)
    with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn'),initializer=worker_init,initargs=(spec,)) as pool:
        for iteration in range(iterations):
            # Include incumbent, distribution mean and antithetic proposals.
            noise=rng.normal(size=((samples-2+1)//2,exp.dim))
            noise=np.concatenate([noise,-noise],axis=0)[:samples-2]
            proposals=np.vstack([best,mean,mean+noise*std])
            proposals=np.clip(proposals,-exp.bound,exp.bound)
            reports=list(pool.map(worker_evaluate,proposals))
            order=np.argsort([r['cost'] for r in reports]);elite=proposals[order[:max(4,samples//8)]]
            if reports[order[0]]['cost']<best_report['cost']:
                best=proposals[order[0]].copy();best_report=reports[order[0]]
            mean=.25*mean+.75*elite.mean(0)
            std=np.maximum(.25*std+.75*elite.std(0),.12*exp.scale)
            row=dict(iteration=iteration,best=best_report,elapsed_seconds=time.monotonic()-start,
                     std_fraction=float(np.mean(std/exp.scale)))
            history.append(row);dump(out/'progress.json',dict(history=history,best_parameters=best))
            print(json.dumps(row),flush=True)
            if best_report['mp_sr']==1.:break
    Q=exp.commands(best)
    path=TR.save_reference(out/'reference.npz',exp.model,Q,exp.ref['labels'],exp.fps,exp.ref['segment'],exp.ref['segments'],
        source=f'Full-finger physics CEM from {spec["reference"]}',arms_coupled=False,floating_runtime=True,no_retiming=True)
    execution=execute_reference(spec['scene'],path,out/'execution',expected_duration_s=(exp.T-1)/exp.fps)
    metrics=evaluate_files(manifest['source'],execution['trajectory'],src/'input_manifest.json',execution['scene'],src/'metric_geometry.json')
    dump(out/'mp_sr.json',metrics)
    result=dict(episode=name,initial=initial,best=best_report,parameters=best,history=history,reference=path,
                execution=execution,metrics=metrics,elapsed_seconds=time.monotonic()-start,learned_policy=False)
    dump(out/'result.json',result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--episode',type=int,required=True)
    p.add_argument('--scene',required=True);p.add_argument('--reference',required=True);p.add_argument('--output',required=True)
    p.add_argument('--samples',type=int,default=96);p.add_argument('--iterations',type=int,default=10)
    p.add_argument('--workers',type=int,default=16);p.add_argument('--seed',type=int,default=44)
    optimize(**vars(p.parse_args()))
