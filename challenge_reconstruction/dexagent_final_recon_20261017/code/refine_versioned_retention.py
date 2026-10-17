"""Full-episode retention search for an explicitly versioned scene adapter.

Only one hand changes after a preserved pickup prefix; every trial replays from
the exact same scene home. Scoring uses every object's complete source timeline.
"""
import argparse, json, multiprocessing as mp, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from physics_retention_search import RetentionExperiment
from refine_grasp_cem import TR, dump, _freeze, execute_reference, evaluate_files

WORK=None
def initialize(spec):
    global WORK
    WORK=RetentionExperiment(spec)
def evaluate(x):return WORK.evaluate(x)


def run(source,episode_dir,output,start_s,stop_s,side='right',protect_object='white_pot',
        samples=128,iterations=18,workers=24,seed=63):
    source=Path(source).resolve();inp=Path(episode_dir).resolve();out=Path(output).resolve()
    out.mkdir(parents=True,exist_ok=True)
    if (out/'settings.json').exists():raise FileExistsError(out)
    prev=json.loads((source/'result.json').read_text());manifest=json.loads((inp/'input_manifest.json').read_text())
    frozen_dir=out/'search_physics';frozen_dir.mkdir()
    _,scene=_freeze(prev['execution']['scene'],frozen_dir)
    spec=dict(scene=str(scene),reference=str(source/'reference.npz'),original_reference=str(inp/'tracking_reference.npz'),
        source=manifest['source'],geometry=str(inp/'metric_geometry.json'),start_s=start_s,stop_s=stop_s,
        ramp_s=.3,side=side,protect_object=protect_object)
    exp=RetentionExperiment(spec);rng=np.random.default_rng(seed);best=np.zeros(28);mean=best.copy();std=exp.scale.copy()
    initial=exp.evaluate(best);report=initial;history=[];begin=time.monotonic()
    dump(out/'settings.json',dict(spec=spec,episode_dir=str(inp),samples=samples,iterations=iterations,workers=workers,seed=seed))
    dump(out/'initial.json',initial)
    with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn'),initializer=initialize,initargs=(spec,)) as pool:
        for iteration in range(iterations):
            proposals=[best.copy(),mean.copy()]
            for j in range(samples-2):
                center=best if j%2==0 else mean;noise=rng.normal(size=28)*std
                if j%4==0:noise[6:]=0
                elif j%4==1:noise[:6]=0
                proposals.append(np.clip(center+noise,-exp.bound,exp.bound))
            proposals=np.array(proposals);reports=list(pool.map(evaluate,proposals,chunksize=1))
            order=np.argsort([r['cost'] for r in reports]);elite=proposals[order[:max(6,samples//8)]]
            if reports[order[0]]['cost']<report['cost']:best=proposals[order[0]].copy();report=reports[order[0]]
            mean=.25*mean+.75*elite.mean(0);std=np.maximum(.3*std+.7*elite.std(0),.12*exp.scale)
            row=dict(iteration=iteration,best=report,elapsed_s=time.monotonic()-begin);history.append(row)
            dump(out/'progress.json',dict(history=history,best_parameters=best));print(json.dumps(row),flush=True)
            if report.get('mp_sr')==1.:break
    Q=exp.commands(best);unchanged=np.setdiff1d(np.arange(exp.model.nq),exp.cols[side])
    assert np.array_equal(Q[:,unchanged],exp.base[:,unchanged])
    assert np.array_equal(Q[:int(start_s*exp.fps)+1],exp.base[:int(start_s*exp.fps)+1])
    path=TR.save_reference(out/'reference.npz',exp.model,Q,exp.ref['labels'],exp.fps,exp.ref['segment'],exp.ref['segments'],
        source='Physical retention refinement; preserved pickup prefix and immutable object targets',floating_runtime=True)
    execution=execute_reference(spec['scene'],path,out/'execution')
    metrics=evaluate_files(manifest['source'],execution['trajectory'],inp/'input_manifest.json',execution['scene'],inp/'metric_geometry.json')
    dump(out/'result.json',dict(episode=manifest['episode'],input_dir=str(inp),initial=initial,best=report,parameters=best,
        history=history,reference=str(path),execution=execution,metrics=metrics,learned_policy=False))

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ['source','episode-dir','output']:p.add_argument('--'+k,required=True)
    for k in ['start-s','stop-s']:p.add_argument('--'+k,type=float,required=True)
    p.add_argument('--side',choices=['left','right'],default='right');p.add_argument('--protect-object',default='white_pot')
    p.add_argument('--samples',type=int,default=128);p.add_argument('--iterations',type=int,default=18)
    p.add_argument('--workers',type=int,default=24);p.add_argument('--seed',type=int,default=63)
    run(**vars(p.parse_args()))
