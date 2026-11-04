"""Small coordinate search around a physical full-horizon retention incumbent."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import argparse,json,multiprocessing as mp,time
import numpy as np
from physics_retention_search import RetentionExperiment
from refine_grasp_cem import TR,dump,_freeze,execute_reference,evaluate_files

WORK=None
def initialize(spec):
    global WORK
    WORK=RetentionExperiment(spec)
def evaluate(x):return WORK.evaluate(x)

def run(source,episode_dir,output,start_s,stop_s,protect_object,workers=16,iterations=12):
    prev=json.loads((source/'result.json').read_text());manifest=json.loads((episode_dir/'input_manifest.json').read_text())
    output.mkdir(parents=True,exist_ok=True)
    if (output/'settings.json').exists():raise FileExistsError(output)
    frozen_dir=output/'search_physics';frozen_dir.mkdir()
    _,scene=_freeze(prev['execution']['scene'],frozen_dir)
    spec=dict(scene=str(scene),reference=str(source/'reference.npz'),original_reference=str(episode_dir/'tracking_reference.npz'),
        source=manifest['source'],geometry=str(episode_dir/'metric_geometry.json'),start_s=start_s,stop_s=stop_s,
        ramp_s=.3,side='right',protect_object=protect_object)
    exp=RetentionExperiment(spec);best=np.zeros(28);report=exp.evaluate(best)
    steps=np.r_[[.00075]*3,[.006]*3,[.01]*22];history=[];begin=time.monotonic()
    dump(output/'settings.json',dict(spec=spec,workers=workers,iterations=iterations,initial_steps=steps,
        method='Explicit coordinate polling; actual full-horizon physics for every proposal, immutable object targets and original timing'))
    dump(output/'initial.json',report)
    with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn'),initializer=initialize,initargs=(spec,)) as pool:
        for iteration in range(iterations):
            proposals=np.vstack([best,best+np.diag(steps),best-np.diag(steps)])
            proposals=np.clip(proposals,-exp.bound,exp.bound)
            reports=list(pool.map(evaluate,proposals,chunksize=1));i=min(range(len(reports)),key=lambda k:reports[k]['cost'])
            improved=reports[i]['cost']<report['cost']
            if improved:best=proposals[i].copy();report=reports[i]
            else:steps*=.5
            history.append(dict(iteration=iteration,best=report,improved=improved,steps=steps.copy(),elapsed_s=time.monotonic()-begin))
            dump(output/'progress.json',dict(history=history,best_parameters=best));dump(output/f'iteration_{iteration:03d}.json',dict(proposals=proposals,reports=reports))
            print(json.dumps(history[-1],default=lambda x:x.tolist()),flush=True)
            if report.get('mp_sr')==1.:break
    Q=exp.commands(best);unchanged=np.setdiff1d(np.arange(exp.model.nq),exp.cols['right'])
    assert np.array_equal(Q[:,unchanged],exp.base[:,unchanged]);assert np.array_equal(Q[:int(start_s*exp.fps)+1],exp.base[:int(start_s*exp.fps)+1])
    path=TR.save_reference(output/'reference.npz',exp.model,Q,exp.ref['labels'],exp.fps,exp.ref['segment'],exp.ref['segments'],
        source='Small coordinate physical retention corrections with immutable object targets, prefix and timing',floating_runtime=True)
    execution=execute_reference(scene,path,output/'execution')
    metrics=evaluate_files(manifest['source'],execution['trajectory'],episode_dir/'input_manifest.json',execution['scene'],episode_dir/'metric_geometry.json')
    dump(output/'result.json',dict(episode=manifest['episode'],reference=str(path),execution=execution,metrics=metrics,
        spec=dict(spec,manifest=str(episode_dir/'input_manifest.json')),parameters=best,history=history,learned_policy=False))

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ['source','episode-dir','output']:p.add_argument('--'+k,type=Path,required=True)
    for k in ['start-s','stop-s']:p.add_argument('--'+k,type=float,required=True)
    p.add_argument('--protect-object',required=True);p.add_argument('--workers',type=int,default=16)
    p.add_argument('--iterations',type=int,default=12);run(**vars(p.parse_args()))
