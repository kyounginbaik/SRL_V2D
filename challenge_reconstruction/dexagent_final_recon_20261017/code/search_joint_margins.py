"""Fit bounded pinky target margins with fresh, full-horizon policy rollouts."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import argparse,hashlib,json,multiprocessing as mp,time
import numpy as np
import mujoco
from joint_margin_policy import JointMarginPolicy
from refine_grasp_cem import Experiment,angular_errors,dump
from learned_feedback_policy import sha


class Search:
    def __init__(self,spec):
        self.spec=spec;inp=Path(spec['episode_dir']);manifest=json.loads((inp/'input_manifest.json').read_text())
        self.metric=Experiment(dict(scene=spec['scene'],reference=str(inp/'tracking_reference.npz'),
            original_reference=str(inp/'tracking_reference.npz'),source=manifest['source'],geometry=str(inp/'metric_geometry.json')))
        self.m=self.metric.model;self.policy=JointMarginPolicy(spec['augmented_checkpoint'])
        self.dt=float(self.m.opt.timestep);self.steps=round(1/self.metric.fps/self.dt)
        self.fj=np.array([j for j in range(self.m.njnt) if (self.m.joint(j).name or '').startswith('robot_')
                         and any(f'_{f}_' in self.m.joint(j).name for f in ['thumb','index','middle','ring','pinky'])])
        self.fq=self.m.jnt_qposadr[self.fj];self.frange=self.m.jnt_range[self.fj]
        self.dim=len(self.policy.data['margin_actuator_ids'])

    def evaluate(self,x):
        self.policy.data['margin_bias_rad']=np.asarray(x)
        d=mujoco.MjData(self.m);mujoco.mj_resetDataKeyframe(self.m,d,0);mujoco.mj_forward(self.m,d)
        Q=np.empty_like(self.metric.base);Q[0]=d.qpos;limit=0.
        for k in range(1,len(Q)):
            for _ in range(self.steps):
                d.ctrl[:]=self.policy(d.time+self.dt,d.qpos,d.qvel);mujoco.mj_step(self.m,d)
                limit=max(limit,float(np.max(np.maximum(self.frange[:,0]-d.qpos[self.fq],d.qpos[self.fq]-self.frange[:,1]))))
            if not np.isfinite(d.qpos).all() or d.warning[mujoco.mjtWarning.mjWARN_BADQACC].number:
                return dict(cost=1e6,unstable=True)
            Q[k]=d.qpos
        p,q=self.metric.object_poses(Q);ep=np.linalg.norm(p-self.metric.tp,axis=-1)
        er=angular_errors(q[:,None],self.metric.tq[:,None],self.metric.axes,self.metric.kinds)[:,0]
        pos=ep.mean(0);ang=er.mean(0);ratio=np.maximum(pos/.03,ang/np.deg2rad(30));limit_deg=float(np.degrees(limit))
        cost=float(ratio.max()+.25*ratio.mean()+100*max(0.,limit_deg-self.spec['tolerance_deg']))
        return dict(cost=cost,mean_position_cm=(100*pos).tolist(),mean_orientation_deg=np.degrees(ang).tolist(),
            mp_sr=float(np.all(ratio<1)),all_finger_limit_max_degrees=limit_deg,
            quality_pass=limit_deg<=self.spec['tolerance_deg'],unstable=False)

WORK=None
def init(spec):
    global WORK
    WORK=Search(spec)
def evaluate(x):return WORK.evaluate(x)


def run(checkpoint,base_module,base_class,scene,episode_dir,baseline_report,output,hands='both',
        samples=64,iterations=10,workers=16,seed=82,tolerance_deg=1.1):
    out=Path(output).resolve();out.mkdir(parents=True,exist_ok=True)
    if (out/'settings.json').exists():raise FileExistsError(out)
    inp=Path(episode_dir).resolve();manifest=json.loads((inp/'input_manifest.json').read_text());m=mujoco.MjModel.from_xml_path(scene)
    with np.load(checkpoint,allow_pickle=False) as z:d={k:z[k] for k in z.files}
    sides=['left','right'] if hands=='both' else [hands];ids=[]
    for side in sides:
        for suffix in ['pinky_CMC','pinky_MCP_AA']:
            joint=m.joint(f'robot_{side}_{side}_{suffix}').id
            hits=np.flatnonzero(m.actuator_trnid[:,0]==joint)
            if len(hits)!=1:raise ValueError('Expected one bounded actuator per pinky joint')
            ids.append(int(hits[0]))
    d.update(margin_base_module=np.array(base_module),margin_base_class=np.array(base_class),
        margin_actuator_ids=np.asarray(ids,dtype=int),margin_bias_rad=np.zeros(len(ids)))
    augmented=out/'initial_policy.npz';np.savez_compressed(augmented,**d)
    spec=dict(scene=str(Path(scene).resolve()),episode_dir=str(inp),augmented_checkpoint=str(augmented),tolerance_deg=tolerance_deg)
    search=Search(spec);best=np.zeros(len(ids));report=search.evaluate(best);original=json.loads(Path(baseline_report).read_text())
    expected=original['metrics']['declared_frame_reference_proxy']['objects']
    if not np.allclose(report['mean_position_cm'],[expected[n]['mean_position_error_cm'][0] for n in search.metric.names],atol=1e-8,rtol=0):
        raise RuntimeError('Zero-margin fast evaluator does not reproduce the original actor')
    rng=np.random.default_rng(seed);mean=np.tile([.14,.16],len(sides));std=np.tile([.07,.10],len(sides))
    lo=np.zeros(len(ids));hi=np.tile([.25,.35],len(sides));history=[];started=time.monotonic()
    dump(out/'initial.json',report);dump(out/'settings.json',dict(spec=spec,source_checkpoint=str(Path(checkpoint).resolve()),
        source_checkpoint_sha256=sha(checkpoint),baseline_report=str(Path(baseline_report).resolve()),
        bounds=[lo,hi],actuator_names=[m.actuator(a).name for a in ids],samples=samples,iterations=iterations,
        workers=workers,seed=seed,objective='Full original object tracking plus actual all-finger soft-limit excursion',
        source_joint_limits_and_effort_caps_unchanged=True,initial_state_unchanged=True,source_targets_unchanged=True))
    with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn'),initializer=init,initargs=(spec,)) as pool:
        for it in range(iterations):
            xs=np.vstack([best,mean,np.clip(mean+rng.normal(size=(samples-2,len(ids)))*std,lo,hi)])
            rs=list(pool.map(evaluate,xs,chunksize=1));order=np.argsort([r['cost'] for r in rs]);elite=xs[order[:max(6,samples//8)]]
            if rs[order[0]]['cost']<report['cost']:best=xs[order[0]].copy();report=rs[order[0]]
            mean=.25*mean+.75*elite.mean(0);std=np.maximum(.3*std+.7*elite.std(0),np.tile([.015,.025],len(sides)))
            row=dict(iteration=it,best=report,parameters=best,elapsed_s=time.monotonic()-started);history.append(row)
            dump(out/'progress.json',dict(history=history,best_parameters=best));dump(out/f'iteration_{it:03d}.json',dict(proposals=xs,reports=rs))
            print(json.dumps(row,default=lambda x:x.tolist()),flush=True)
            if report.get('mp_sr')==1. and report.get('quality_pass'):break
    d['margin_bias_rad']=best;selected=out/'policy.npz';np.savez_compressed(selected,**d)
    from evaluate_margin_feedback import evaluate as final_evaluate
    final=final_evaluate(int(manifest['episode'].split('_')[-1]),scene,str(selected),str(out/'evaluation'),str(inp))
    actual=final['metrics']['declared_frame_reference_proxy']['objects']
    if not np.allclose(report['mean_position_cm'],[actual[n]['mean_position_error_cm'][0] for n in search.metric.names],atol=1e-8,rtol=0):
        raise RuntimeError('Selected policy does not reproduce full independent evaluator')
    dump(out/'training_report.json',dict(learned_policy=True,method='Derivative-free physical policy fine-tuning of bounded pinky target margins',
        source_checkpoint=str(Path(checkpoint).resolve()),source_checkpoint_sha256=sha(checkpoint),
        checkpoint=str(selected),checkpoint_sha256=sha(selected),trainer_sha256=sha(__file__),
        actor_base=f'{base_module}.{base_class}',parameters=best,initial=search.evaluate(np.zeros(len(ids))),best=report,
        all_other_learned_weights_unchanged=True,policy_contains_command_table=False,iterations=len(history),
        acceptance_requires_actual_all_joint_limits=True))
    dump(out/'result.json',dict(best=report,parameters=best,history=history,checkpoint=str(selected),evaluation=final))

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ['checkpoint','scene','episode-dir','baseline-report','output']:p.add_argument('--'+k,required=True)
    p.add_argument('--base-module',default='learned_feedback_policy');p.add_argument('--base-class',default='LearnedPolicy')
    p.add_argument('--hands',choices=['left','right','both'],default='both')
    p.add_argument('--samples',type=int,default=64);p.add_argument('--iterations',type=int,default=10)
    p.add_argument('--workers',type=int,default=16);p.add_argument('--seed',type=int,default=82);p.add_argument('--tolerance-deg',type=float,default=1.1)
    run(**vars(p.parse_args()))
