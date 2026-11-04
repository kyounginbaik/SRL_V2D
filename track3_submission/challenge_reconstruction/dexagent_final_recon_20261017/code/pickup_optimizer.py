"""Reusable process-parallel physical CEM for an explicit experiment class."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import time,json,hashlib,inspect
import numpy as np
from physics_pickup_search_v2 import dump,_freeze,execute_reference,evaluate_files,TR

WORKER=None
def initialize(cls,spec):
    global WORKER
    WORKER=cls(spec)


def evaluate(x):return WORKER.evaluate(x)


def optimize(cls,spec,output,samples=128,iterations=20,workers=24,seed=44,resume=None):
    out=Path(output);out.mkdir(parents=True,exist_ok=True);spec=dict(spec)
    if (out/'result.json').exists():raise FileExistsError(out/'result.json')
    frozen_dir=out/'search_physics';frozen_dir.mkdir(parents=True,exist_ok=True)
    _,frozen=_freeze(spec['scene'],frozen_dir);spec['scene']=str(frozen.resolve())
    exp=cls(spec);rng=np.random.default_rng(seed);mean=np.zeros(exp.dim);std=exp.scale.copy();best=mean.copy()
    if resume:
        old=json.loads(Path(resume).read_text());best=np.asarray(old['best_parameters']);mean=best.copy()
    best_report=exp.evaluate(best);history=[]
    source=Path(inspect.getfile(cls));(out/'experiment_source.py').write_bytes(source.read_bytes())
    dump(out/'settings.json',dict(spec=spec,samples=samples,iterations=iterations,workers=workers,seed=seed,
         dimension=exp.dim,scale=exp.scale,bound=exp.bound,experiment_class=f'{cls.__module__}.{cls.__name__}',
         experiment_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest()))
    dump(out/'initial.json',best_report);started=time.monotonic()
    with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn'),initializer=initialize,initargs=(cls,spec)) as pool:
        for iteration in range(iterations):
            n=samples-2;centers=np.repeat(mean[None],n,axis=0);centers[:n//2]=best
            proposals=np.vstack([best,mean,centers+rng.normal(size=(n,exp.dim))*std]);proposals=np.clip(proposals,-exp.bound,exp.bound)
            reports=list(pool.map(evaluate,proposals,chunksize=1));order=np.argsort([r['cost'] for r in reports]);elite=proposals[order[:max(6,samples//8)]]
            if reports[order[0]]['cost']<best_report['cost']:best=proposals[order[0]].copy();best_report=reports[order[0]]
            mean=.2*mean+.8*elite.mean(0);std=np.maximum(.3*std+.7*elite.std(0),.15*exp.scale)
            row=dict(iteration=iteration,best=best_report,elapsed_s=time.monotonic()-started);history.append(row)
            dump(out/'progress.json',dict(history=history,best_parameters=best,mean=mean,std=std))
            dump(out/f'iteration_{iteration:03d}.json',dict(proposals=proposals,reports=reports));print(json.dumps(row),flush=True)
    Q,timing=exp.commands(best)
    refpath=TR.save_reference(out/'reference.npz',exp.m,Q,exp.ref['labels'],exp.fps,exp.ref['segment'],exp.ref['segments'],
        source=f'Physical CEM {cls.__name__}; immutable original object targets and clock',arms_coupled=False,floating_runtime=True)
    short=exp.evaluate(best,return_trace=True);trace=short.pop('trace',np.empty((0,14)));np.savez_compressed(out/'search_trace.npz',trace=trace)
    execution=execute_reference(spec['scene'],refpath,out/'execution')
    metrics=evaluate_files(spec['source'],execution['trajectory'],spec['manifest'],execution['scene'],spec['geometry'])
    result=dict(best=best_report,parameters=best,reference=str(refpath),execution=execution,metrics=metrics,history=history,spec=spec)
    dump(out/'result.json',result);return result
