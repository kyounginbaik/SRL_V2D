"""Physically fine-tune configurable finger margins of a genuine learned actor.

All 44 physical finger joint limits are checked at every 500 Hz substep. The
episode's original objects, timestamps, initial state and physics remain fixed.
Task-passing incumbents are retained separately from minimum-penalty candidates.
"""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import argparse,json,multiprocessing as mp,time
import numpy as np,mujoco
from search_repaired_hand_margins_evaluator_v3 import Search,init,evaluate
from refine_grasp_cem import dump
from learned_feedback_policy import sha
from evaluate_policy_component import evaluate as evaluate_component


def run(checkpoint,scene,episode_dir,baseline_report,config,output,samples=96,iterations=16,workers=16,seed=93,tolerance_deg=1.):
    out=Path(output).resolve();out.mkdir(parents=True,exist_ok=True)
    if (out/'settings.json').exists():raise FileExistsError(out)
    inp=Path(episode_dir).resolve();manifest=json.loads((inp/'input_manifest.json').read_text())
    m=mujoco.MjModel.from_xml_path(scene);settings=json.loads(Path(config).read_text());joints=settings['joints']
    ids=[]
    for row in joints:
        j=m.joint(row['name']).id;matches=np.flatnonzero(m.actuator_trnid[:,0]==j)
        if len(matches)!=1:raise ValueError('Expected exactly one actuator for configured joint')
        if not any(f'_{f}_' in row['name'] for f in ['thumb','index','middle','ring','pinky']):raise ValueError('Finger margins only')
        ids.append(int(matches[0]))
    if len(set(ids))!=len(ids):raise ValueError('Duplicate actuator')
    with np.load(checkpoint,allow_pickle=False) as z:d={k:z[k] for k in z.files}
    if 'margin_actuator_ids' in d:raise ValueError('Do not silently overwrite an existing fitted margin wrapper')
    d.update(margin_base_module=np.array('learned_feedback_policy'),margin_base_class=np.array('LearnedPolicy'),
        margin_actuator_ids=np.array(ids,dtype=int),margin_bias_rad=np.zeros(len(ids)),
        margin_startup_only=np.zeros(len(ids),dtype=bool),margin_startup_end_s=np.array(.8))
    augmented=out/'initial_policy.npz';np.savez_compressed(augmented,**d)
    spec=dict(scene=str(Path(scene).resolve()),episode_dir=str(inp),augmented_checkpoint=str(augmented),tolerance_deg=tolerance_deg)
    search=Search(spec);best=np.zeros(len(ids));report=search.evaluate(best)
    original=json.loads(Path(baseline_report).read_text());expected=original['metrics']['declared_frame_reference_proxy']['objects']
    if not np.allclose(report['mean_position_cm'],[expected[n]['mean_position_error_cm'][0] for n in search.metric.names],atol=1e-8,rtol=0):
        raise RuntimeError('Zero-margin evaluator did not reproduce original actor')
    def array(key):return np.radians([r[key] for r in joints])
    mean=array('initial_mean_deg');std=array('initial_std_deg');low=array('lower_deg');high=array('upper_deg');floor=array('std_floor_deg')
    rng=np.random.default_rng(seed);history=[];started=time.monotonic()
    best_mp=(best.copy(),report.copy()) if report['mp_sr']==1. and report.get('preservation_pass') else None
    dump(out/'initial.json',report);dump(out/'settings.json',dict(spec=spec,config=settings,source_checkpoint=str(Path(checkpoint).resolve()),
        source_checkpoint_sha256=sha(checkpoint),baseline_report=str(Path(baseline_report).resolve()),samples=samples,iterations=iterations,workers=workers,seed=seed,
        objective='Original target tracking and actual all44-finger 500Hz joint-limit excursions; explicitly repaired fixed reconstruction scene, all physics and initial states immutable within search',
        trainer_sha256=sha(__file__),fast_evaluator_source_sha256=sha(Path(__file__).with_name('search_repaired_hand_margins_evaluator_v3.py'))))
    with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn'),initializer=init,initargs=(spec,)) as pool:
        for iteration in range(iterations):
            centers=np.repeat(mean[None],samples-2,axis=0);centers[:(samples-2)//2]=best
            xs=np.vstack([best,mean,np.clip(centers+rng.normal(size=centers.shape)*std,low,high)])
            rs=list(pool.map(evaluate,xs,chunksize=1))
            for x,r in zip(xs,rs):
                if r.get('mp_sr')==1. and r.get('preservation_pass') and (best_mp is None or r['all_finger_limit_max_degrees']<best_mp[1]['all_finger_limit_max_degrees']):
                    best_mp=(x.copy(),r.copy());dump(out/'best_mp.json',dict(parameters=x,report=r))
            order=np.argsort([r['cost'] for r in rs]);elite=xs[order[:max(6,samples//8)]]
            if rs[order[0]]['cost']<report['cost']:best=xs[order[0]].copy();report=rs[order[0]]
            mean=.25*mean+.75*elite.mean(0);std=np.maximum(.3*std+.7*elite.std(0),floor)
            row=dict(iteration=iteration,best=report,parameters=best,elapsed_s=time.monotonic()-started);history.append(row)
            dump(out/'progress.json',dict(history=history,best_parameters=best));dump(out/f'iteration_{iteration:03d}.json',dict(proposals=xs,reports=rs))
            print(json.dumps(row,default=lambda x:x.tolist()),flush=True)
            if report.get('mp_sr')==1. and report.get('quality_pass'):break
    if best_mp is not None and not (report.get('mp_sr')==1. and report.get('quality_pass')):best,report=best_mp
    d['margin_bias_rad']=best;selected=out/'policy.npz';np.savez_compressed(selected,**d)
    final=evaluate_component(int(manifest['episode'].split('_')[-1]),scene,str(selected),str(out/'evaluation'),str(inp))
    actual=final['metrics']['declared_frame_reference_proxy']['objects']
    if not np.allclose(report['mean_position_cm'],[actual[n]['mean_position_error_cm'][0] for n in search.metric.names],atol=1e-8,rtol=0):
        raise RuntimeError('Selected actor disagrees with independent full evaluator')
    dump(out/'training_report.json',dict(learned_policy=True,method='Derivative-free physical fine-tuning of configurable bounded finger margins',
        source_checkpoint=str(Path(checkpoint).resolve()),source_checkpoint_sha256=sha(checkpoint),checkpoint=str(selected),checkpoint_sha256=sha(selected),
        trainer_sha256=sha(__file__),base_architecture='learned_feedback_policy.LearnedPolicy',parameters=best,configured_joints=joints,
        initial=json.loads((out/'initial.json').read_text()),best=report,preservation_limits=dict(pot_mean_cm=1.6,lid_mean_cm=1.5,final_hand_force_n=.02,minimum_pot_table_force_n=.5,minimum_lid_pot_force_n=.1,maximum_object_speed_m_s=.02),all_other_learned_weights_unchanged=True,
        policy_contains_command_table=False,iterations=len(history),all44_fingers_checked_every_physics_substep=True))
    dump(out/'result.json',dict(best=report,parameters=best,history=history,checkpoint=str(selected),evaluation=final))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ['checkpoint','scene','episode-dir','baseline-report','config','output']:p.add_argument('--'+k,required=True)
    p.add_argument('--samples',type=int,default=96);p.add_argument('--iterations',type=int,default=16)
    p.add_argument('--workers',type=int,default=16);p.add_argument('--seed',type=int,default=93)
    p.add_argument('--tolerance-deg',type=float,default=1.);run(**vars(p.parse_args()))
