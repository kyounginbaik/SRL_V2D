"""Full-horizon right-hand refinement after a preserved physical pickup prefix."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import argparse,json,multiprocessing as mp,time
import numpy as np
from scipy.spatial.transform import Rotation
from refine_grasp_cem import Experiment,B,TR,dump,_freeze,execute_reference,evaluate_files
from wrist_reference import continuous_euler_xyz


class RetentionExperiment(Experiment):
    def __init__(self,spec):
        super().__init__(spec);self.dim=28;self.scale=self.scale[28:]*.55;self.bound=self.bound[28:].copy()
        self.side=spec.get('side','right');self.start=float(spec['start_s']);self.stop=float(spec.get('stop_s',14.45));self.ramp=float(spec.get('ramp_s',.3))
        t=np.arange(self.T)/self.fps
        s=lambda u:np.clip(u,0,1)**3*(10-15*np.clip(u,0,1)+6*np.clip(u,0,1)**2)
        self.weight=s((t-self.start)/self.ramp)*(1-s((t-self.stop)/.35))
        self.protect=self.names.index(spec.get('protect_object','white_pot'))

    def commands(self,x):
        x=np.asarray(x);Q=self.base.copy();c=self.cols[self.side];w=self.weight;R=self.rot[self.side]
        Q[:,c[:3]]+=R.apply(w[:,None]*x[:3])
        if np.any(x[3:6]):Q[:,c[3:6]]=continuous_euler_xyz(R*Rotation.from_rotvec(w[:,None]*x[3:6]),Q[0,c[3:6]])
        Q[:,c[6:]]=np.clip(Q[:,c[6:]]+w[:,None]*x[6:],self.limits[self.side][6:,0],self.limits[self.side][6:,1])
        before=np.arange(self.T)/self.fps<=self.start;Q[before]=self.base[before]
        return Q

    def evaluate(self,x):
        r=super().evaluate(x)
        if r['unstable']:return r
        violation=max(r['mean_position_cm'][self.protect]/3.,r['mean_orientation_deg'][self.protect]/30.)-1.
        r['raw_cost']=r['cost'];r['protected_object_pass']=bool(violation<0.)
        r['cost']+=100*max(0.,violation)
        return r


WORK=None
def init(spec):
    global WORK;WORK=RetentionExperiment(spec)
def evaluate(x):return WORK.evaluate(x)


def optimize(source,output,samples=128,iterations=20,workers=32,seed=45,start_s=4.5):
    source=Path(source);out=Path(output);out.mkdir(parents=True,exist_ok=True)
    if (out/'result.json').exists():raise FileExistsError(out/'result.json')
    prev=json.loads((source/'result.json').read_text());inp=B/'episodes/episode_000003';manifest=json.loads((inp/'input_manifest.json').read_text())
    (out/'search_physics').mkdir(exist_ok=True);_,scene=_freeze(prev['execution']['scene'],out/'search_physics')
    spec=dict(scene=str(scene),reference=str(source/'reference.npz'),original_reference=str(inp/'tracking_reference.npz'),
        source=manifest['source'],geometry=str(inp/'metric_geometry.json'),start_s=start_s,stop_s=14.45,ramp_s=.3,side='right',protect_object='white_pot')
    exp=RetentionExperiment(spec);rng=np.random.default_rng(seed);best=np.zeros(28);mean=best.copy();std=exp.scale.copy();report=exp.evaluate(best);history=[];started=time.monotonic()
    dump(out/'settings.json',dict(spec=spec,samples=samples,iterations=iterations,workers=workers,seed=seed,scale=std,bound=exp.bound,dimension=28,
        objective='full episode proxy tracking for every object, with explicit white-pot success constraint; no target or time editing'))
    dump(out/'initial.json',report)
    with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn'),initializer=init,initargs=(spec,)) as pool:
        for iteration in range(iterations):
            # Explore around both the elite mean and the incumbent. Sparse
            # wrist/finger groups preserve successful dimensions in half trials.
            proposals=[best.copy(),mean.copy()]
            for j in range(samples-2):
                center=best if j%2==0 else mean
                noise=rng.normal(size=28)*std
                if j%4==0:noise[6:]=0
                elif j%4==1:noise[:6]=0
                proposals.append(np.clip(center+noise,-exp.bound,exp.bound))
            proposals=np.asarray(proposals);reports=list(pool.map(evaluate,proposals,chunksize=1));order=np.argsort([r['cost'] for r in reports]);elite=proposals[order[:max(6,samples//8)]]
            if reports[order[0]]['cost']<report['cost']:best=proposals[order[0]].copy();report=reports[order[0]]
            mean=.25*mean+.75*elite.mean(0);std=np.maximum(.3*std+.7*elite.std(0),.12*exp.scale)
            row=dict(iteration=iteration,best=report,elapsed_s=time.monotonic()-started);history.append(row)
            dump(out/'progress.json',dict(history=history,best_parameters=best));dump(out/f'iteration_{iteration:03d}.json',dict(proposals=proposals,reports=reports));print(json.dumps(row),flush=True)
            if report.get('mp_sr')==1.:break
    Q=exp.commands(best);unchanged=np.setdiff1d(np.arange(exp.model.nq),exp.cols['right'])
    assert np.array_equal(Q[:,unchanged],exp.base[:,unchanged]);assert np.array_equal(Q[:int(start_s*exp.fps)+1],exp.base[:int(start_s*exp.fps)+1])
    path=TR.save_reference(out/'reference.npz',exp.model,Q,exp.ref['labels'],exp.fps,exp.ref['segment'],exp.ref['segments'],source='Full-horizon physical retention refinement after preserved pickup; original left and object references unchanged',arms_coupled=False,floating_runtime=True)
    execution=execute_reference(spec['scene'],path,out/'execution');metrics=evaluate_files(manifest['source'],execution['trajectory'],inp/'input_manifest.json',execution['scene'],inp/'metric_geometry.json')
    dump(out/'result.json',dict(episode='episode_000003',initial=json.loads((out/'initial.json').read_text()),best=report,parameters=best,history=history,reference=str(path),execution=execution,metrics=metrics,spec=spec))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--samples',type=int,default=128);p.add_argument('--iterations',type=int,default=20);p.add_argument('--workers',type=int,default=32);p.add_argument('--seed',type=int,default=45);p.add_argument('--start-s',type=float,default=4.5)
    optimize(**vars(p.parse_args()))
