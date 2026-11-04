"""Fine robot-command search that preserves objects already passing MP-SR.

Mix independent wrist/finger/hand proposals with coupled proposals. Each is
scored in the same full episode physics; only robot reference commands change.
"""
from pathlib import Path
import argparse,json,hashlib,time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from scipy.ndimage import gaussian_filter1d
from refine_grasp_cem import B,Experiment,TR,dump,_freeze,execute_reference,evaluate_files

WORK=None
GROUPS=[np.arange(0,6),np.arange(6,28),np.arange(28,34),np.arange(34,56),np.arange(28),np.arange(28,56),np.arange(56)]

def experiment(spec):
    exp=Experiment(spec)
    if spec.get('fill_contact_gaps'):
        for side,w in exp.envelope.items():
            ids=np.flatnonzero(w>.2)
            if len(ids):
                span=np.zeros(len(w));span[ids[0]:ids[-1]+1]=1.
                span=gaussian_filter1d(span,max(1.,.12*exp.fps))
                exp.envelope[side]=span*np.minimum(np.arange(len(w))/max(1.,.3*exp.fps),1.)
    return exp

def init(spec):
    global WORK
    WORK=experiment(spec)

def evaluate(x):return WORK.evaluate(x)

def run(episode,samples=64,iterations=8,workers=8,seed=45,fill_contact_gaps=True,output_root=None,source_root=None):
    name=f'episode_{episode:06d}';inp=B/'episodes'/name;prev=Path(source_root or B/'refinement_20261017/cem_v2')/name
    out=Path(output_root or B/'refinement_20261017/block_polish')/name;out.mkdir(parents=True,exist_ok=True)
    if (out/'result.json').exists():raise FileExistsError(out/'result.json')
    previous=json.loads((prev/'result.json').read_text());manifest=json.loads((inp/'input_manifest.json').read_text())
    frozen_dir=out/'search_physics';frozen_dir.mkdir(exist_ok=True);_,frozen=_freeze(previous['execution']['scene'],frozen_dir)
    spec=dict(scene=str(frozen),reference=str(prev/'reference.npz'),original_reference=str(inp/'tracking_reference.npz'),
              source=manifest['source'],geometry=str(inp/'metric_geometry.json'),fill_contact_gaps=fill_contact_gaps)
    exp=experiment(spec);best=np.zeros(56);initial=exp.evaluate(best);best_report=initial
    protected=(np.array(initial['mean_position_cm'])<3.)&(np.array(initial['mean_orientation_deg'])<30.)
    scale=exp.scale*.35;bound=exp.bound*.5;rng=np.random.default_rng(seed);history=[];begin=time.monotonic()
    def admissible(r):
        return not r['unstable'] and bool(np.all((np.array(r['mean_position_cm'])[protected]<3.)&(np.array(r['mean_orientation_deg'])[protected]<30.)))
    dump(out/'settings.json',dict(episode=name,spec=spec,samples=samples,iterations=iterations,workers=workers,seed=seed,
        algorithm='Gaussian block proposals plus coupled proposals, retain strict cost improvements',
        protected_objects=[n for n,p in zip(exp.names,protected) if p],parameter_dimension=56,
        source_sha256={k:hashlib.sha256(Path(spec[k]).read_bytes()).hexdigest() for k in ('scene','reference','original_reference','geometry')},
        optimizer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),learned_policy=False))
    dump(out/'initial.json',initial)
    with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn'),initializer=init,initargs=(spec,)) as pool:
        for iteration in range(iterations):
            proposals=[best.copy()]
            for j in range(samples-1):
                group=GROUPS[(j//2+iteration)%len(GROUPS)]
                if j%2==0:noise=rng.normal(size=len(group))*scale[group]
                # Adjacent proposals use the same group and opposite direction.
                candidate=best.copy();candidate[group]+=(1 if j%2==0 else -1)*noise
                proposals.append(np.clip(candidate,-bound,bound))
            reports=list(pool.map(evaluate,proposals))
            order=sorted(range(samples),key=lambda j:reports[j]['cost'] if admissible(reports[j]) else 1e20)
            improved=False
            if admissible(reports[order[0]]) and reports[order[0]]['cost']<best_report['cost']:
                best=proposals[order[0]].copy();best_report=reports[order[0]];improved=True
                protected|=(np.array(best_report['mean_position_cm'])<3.)&(np.array(best_report['mean_orientation_deg'])<30.)
            scale*=.95 if improved else .85
            history.append(dict(iteration=iteration,best=best_report,elapsed_seconds=time.monotonic()-begin,
                                protected_rejections=sum(not admissible(r) for r in reports),improved=improved,
                                protected_objects=[n for n,p in zip(exp.names,protected) if p]))
            dump(out/'progress.json',dict(history=history,best_parameters=best));print(json.dumps(history[-1]),flush=True)
            if best_report['mp_sr']==1.:break
    Q=exp.commands(best)
    path=TR.save_reference(out/'reference.npz',exp.model,Q,exp.ref['labels'],exp.fps,exp.ref['segment'],exp.ref['segments'],
        source=f'Block grasp polish from {spec["reference"]}',floating_runtime=True,no_retiming=True)
    execution=execute_reference(spec['scene'],path,out/'execution',expected_duration_s=(exp.T-1)/exp.fps)
    metrics=evaluate_files(manifest['source'],execution['trajectory'],inp/'input_manifest.json',execution['scene'],inp/'metric_geometry.json')
    dump(out/'mp_sr.json',metrics);dump(out/'result.json',dict(episode=name,initial=initial,best=best_report,parameters=best,
        history=history,reference=path,execution=execution,metrics=metrics,elapsed_seconds=time.monotonic()-begin,learned_policy=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--episode',type=int,required=True);p.add_argument('--samples',type=int,default=64)
    p.add_argument('--iterations',type=int,default=8);p.add_argument('--workers',type=int,default=8);p.add_argument('--seed',type=int,default=45)
    p.add_argument('--output-root',type=Path);p.add_argument('--source-root',type=Path)
    run(**vars(p.parse_args()))
