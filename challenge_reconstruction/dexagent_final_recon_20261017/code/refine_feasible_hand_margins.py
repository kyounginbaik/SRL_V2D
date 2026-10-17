"""Improve physical joint quality while retaining full-episode task feasibility.

Starts from an independently replayed learned-policy MP-SR pass. Unlike a
weighted search that can drift into failing demonstrations, task-feasible
proposals rank ahead of every failing proposal. No physical limits change.
"""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import argparse,json,multiprocessing as mp,time
import numpy as np
from search_pinky_margins_v2 import Search,init,evaluate
from refine_grasp_cem import dump
from learned_feedback_policy import sha
from evaluate_policy_component import evaluate as final_evaluate

def ranking(r):
    if r.get('unstable'):return 1e9
    ratio=max(max(r['mean_position_cm'])/3.,max(r['mean_orientation_deg'])/30.)
    if r.get('mp_sr')!=1.:return 1e4+1e3*max(0.,ratio-1.)+r['all_finger_limit_max_degrees']
    return r['all_finger_limit_max_degrees']+.03*ratio

def run(source,output,samples=96,iterations=16,workers=16,seed=112):
    source=Path(source).resolve();out=Path(output).resolve();out.mkdir(parents=True,exist_ok=True)
    if (out/'settings.json').exists():raise FileExistsError(out)
    previous=json.loads((source/'result.json').read_text());original_settings=json.loads((source/'settings.json').read_text())
    baseline=previous['evaluation']
    if not baseline['reference_proxy_success'] or not baseline['execution']['physics_verified']:raise ValueError('A genuine physical learned-policy pass is required')
    spec=original_settings['spec'];search=Search(spec)
    with np.load(previous['checkpoint'],allow_pickle=False) as z:d={k:z[k] for k in z.files}
    best=d['margin_bias_rad'].copy();report=search.evaluate(best);initial=report.copy()
    expected=baseline['metrics']['declared_frame_reference_proxy']['objects']
    if not np.allclose(report['mean_position_cm'],[expected[n]['mean_position_error_cm'][0] for n in search.metric.names],atol=1e-8,rtol=0):raise RuntimeError('Parent checkpoint evaluation mismatch')
    rows=original_settings['config']['joints'];lo=np.radians([r['lower_deg'] for r in rows]);hi=np.radians([r['upper_deg'] for r in rows])
    scale=np.radians([max(.35,min(2.,r['initial_std_deg']/4)) for r in rows]);std=scale.copy();mean=best.copy()
    rng=np.random.default_rng(seed);history=[];started=time.monotonic()
    dump(out/'settings.json',dict(spec=spec,config=original_settings['config'],source=str(source),source_checkpoint=previous['checkpoint'],
        source_checkpoint_sha256=sha(previous['checkpoint']),samples=samples,iterations=iterations,workers=workers,seed=seed,
        constraint='Every accepted actor must pass all object metrics on the full original trajectory',initial_std_rad=std))
    with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn'),initializer=init,initargs=(spec,)) as pool:
        for iteration in range(iterations):
            centers=np.repeat(mean[None],samples-2,axis=0);centers[:(samples-2)//2]=best
            noise=rng.normal(size=centers.shape)*std
            for k in range(0,len(noise),3):
                mask=np.zeros(len(best));mask[k%len(best)]=1;noise[k]*=mask
            xs=np.vstack([best,mean,np.clip(centers+noise,lo,hi)]);rs=list(pool.map(evaluate,xs,chunksize=1))
            order=np.argsort([ranking(r) for r in rs]);feasible=[i for i in order if rs[i].get('mp_sr')==1.]
            if not feasible:raise RuntimeError('Previously passing incumbent was not reproducible')
            elite=xs[feasible[:max(6,samples//8)]]
            if ranking(rs[order[0]])<ranking(report):best=xs[order[0]].copy();report=rs[order[0]]
            mean=.2*mean+.8*elite.mean(0);std=np.maximum(.4*std+.6*elite.std(0),scale*.2)
            row=dict(iteration=iteration,best=report,feasible_proposals=len(feasible),parameters=best,elapsed_s=time.monotonic()-started);history.append(row)
            dump(out/'progress.json',dict(history=history,best_parameters=best));dump(out/f'iteration_{iteration:03d}.json',dict(proposals=xs,reports=rs))
            print(json.dumps(row,default=lambda x:x.tolist()),flush=True)
            if report['quality_pass']:break
    d['margin_bias_rad']=best;checkpoint=out/'policy.npz';np.savez_compressed(checkpoint,**d)
    ep=int(baseline['episode'].split('_')[-1]);final=final_evaluate(ep,spec['scene'],str(checkpoint),str(out/'evaluation'),spec['episode_dir'])
    actual=final['metrics']['declared_frame_reference_proxy']['objects']
    if not np.allclose(report['mean_position_cm'],[actual[n]['mean_position_error_cm'][0] for n in search.metric.names],atol=1e-8,rtol=0):raise RuntimeError('Final actor disagrees with independent evaluator')
    dump(out/'training_report.json',dict(learned_policy=True,method='Task-constrained physical fine-tuning of learned finger margins',
        source_checkpoint=previous['checkpoint'],source_checkpoint_sha256=sha(previous['checkpoint']),
        checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),trainer_sha256=sha(__file__),parameters=best,
        initial=initial,best=report,iterations=len(history),policy_contains_command_table=False,
        all_other_learned_weights_unchanged=True,all44_fingers_checked_every_physics_substep=True))
    dump(out/'result.json',dict(best=report,parameters=best,history=history,checkpoint=str(checkpoint),evaluation=final))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True)
    p.add_argument('--samples',type=int,default=96);p.add_argument('--iterations',type=int,default=16)
    p.add_argument('--workers',type=int,default=16);p.add_argument('--seed',type=int,default=112);run(**vars(p.parse_args()))
