#!/usr/bin/env python3
"""Refine left-hand holding commands while preserving a passing sweeper rollout."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
from pathlib import Path
import argparse,json,multiprocessing as mp,sys,time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from prepare_seven import BASE,ROOT,LEGACY,sha,write
sys.path[:0]=[str(LEGACY/'code'),str(ROOT/'third_party/DexAgent')]
from refine_grasp_cem import Experiment,TR,_freeze,execute_reference,evaluate_files

WORK=None
def init(spec):
    global WORK
    WORK=Experiment(spec)
def evaluate(x):return WORK.evaluate(x)


def run(source_run,output,samples,iterations,workers,seed):
    inp=BASE/'physics_collision_v2/episodes/episode_000030'
    source_run=source_run.resolve();output=output.resolve()
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True);frozen=output/'search_physics';frozen.mkdir()
    previous=json.loads((source_run/'result.json').read_text());manifest=json.loads((inp/'input_manifest.json').read_text())
    _,scene=_freeze(previous['execution']['scene'],frozen)
    spec=dict(scene=str(scene),reference=str(source_run/'reference.npz'),original_reference=str(inp/'tracking_reference.npz'),source=manifest['source'],geometry=str(inp/'metric_geometry.json'))
    exp=Experiment(spec);best=np.zeros(56);initial=exp.evaluate(best);best_report=initial
    protected=(np.array(initial['mean_position_cm'])<3)&(np.array(initial['mean_orientation_deg'])<30)
    if not protected[exp.names.index('mini_sweeper')]:raise ValueError('Source sweeper must already pass')
    rng=np.random.default_rng(seed);scale=exp.scale[:28].copy();bound=exp.bound[:28].copy();mean=np.zeros(28);std=scale.copy()
    history=[];start=time.monotonic()
    def admissible(r):
        return not r['unstable'] and bool(np.all((np.asarray(r['mean_position_cm'])[protected]<3)&(np.asarray(r['mean_orientation_deg'])[protected]<30)))
    write(output/'settings.json',dict(spec=spec,initial=initial,protected_objects=[n for n,p in zip(exp.names,protected) if p],
        parameter_dimension=28,parameterization='left wrist and22 fingers only, smooth original contact envelope; right commands unchanged',
        samples=samples,iterations=iterations,workers=workers,seed=seed,producer_script_sha256=sha(__file__),
        scene_sha256=sha(scene),reference_sha256=sha(source_run/'reference.npz'),original_objects_timing_unchanged=True))
    with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn'),initializer=init,initargs=(spec,)) as pool:
        for iteration in range(iterations):
            half=rng.normal(size=((samples-2+1)//2,28));noise=np.vstack([half,-half])[:samples-2]
            left=np.vstack([best[:28],mean,mean+noise*std]);left=np.clip(left,-bound,bound)
            proposals=np.c_[left,np.zeros((len(left),28))]
            reports=list(pool.map(evaluate,proposals,chunksize=1));valid=[i for i,r in enumerate(reports) if admissible(r)]
            order=sorted(valid,key=lambda i:reports[i]['cost']);improved=False
            if order and reports[order[0]]['cost']<best_report['cost']:
                best=proposals[order[0]].copy();best_report=reports[order[0]];improved=True
            if len(order)>=4:
                elite=left[order[:max(4,samples//8)]];mean=.3*mean+.7*elite.mean(0)
                std=np.maximum(.3*std+.7*elite.std(0),.15*scale)
            else:
                mean=best[:28].copy();std=np.maximum(.8*std,.15*scale)
            row=dict(iteration=iteration,best=best_report,improved=improved,admissible_candidates=len(valid),elapsed_s=time.monotonic()-start)
            history.append(row);write(output/'progress.json',dict(history=history,best_parameters=best.tolist(),mean=mean.tolist(),std=std.tolist()))
            print(json.dumps(row),flush=True)
            if best_report['mp_sr']==1:break
    commands=exp.commands(best)
    if not np.array_equal(commands[:,exp.cols['right']],exp.base[:,exp.cols['right']]):raise AssertionError('Right commands changed')
    reference=TR.save_reference(output/'reference.npz',exp.model,commands,exp.ref['labels'],exp.fps,exp.ref['segment'],exp.ref['segments'],
        source='Protected sweeper: physical search of left rack-holding commands only',floating_runtime=True,no_retiming=True)
    execution=execute_reference(scene,reference,output/'execution')
    metrics=evaluate_files(manifest['source'],execution['trajectory'],inp/'input_manifest.json',execution['scene'],inp/'metric_geometry.json')
    result=dict(episode='episode_000030',reference=str(reference),execution=execution,metrics=metrics,initial=initial,best=best_report,
        parameters=best.tolist(),history=history,learned_policy=False,right_robot_commands_unchanged=True,original_object_targets_unchanged=True)
    write(output/'result.json',result);print(json.dumps(dict(output=str(output),mp_sr=metrics['reference_proxy_mp_sr'],objects=metrics['official_style_reference_proxy']['objects'])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source-run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--samples',type=int,default=64);p.add_argument('--iterations',type=int,default=12);p.add_argument('--workers',type=int,default=8);p.add_argument('--seed',type=int,default=45)
    run(**vars(p.parse_args()))
