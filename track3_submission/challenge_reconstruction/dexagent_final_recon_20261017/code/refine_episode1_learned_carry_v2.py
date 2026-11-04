"""Six-parameter physical fine-tuning of learned phase weights, original goals."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import json,multiprocessing as mp,time
import numpy as np
from episode1_learned_carry_evaluator_v2 import Search,init,evaluate
from evaluate_policy_component import evaluate as final_evaluate
from audit_full_joint_limits import main as audit_joints
from audit_object_support import audit as audit_support
from audit_episode1_alignment_bias import audit as audit_alignment
from learned_feedback_policy import sha
from refine_grasp_cem import dump
B=Path(__file__).resolve().parents[1]
ROOT=B/'refinement_20261018/episode_000001/reconstruction_repair/patch_height094'


def main():
    source=ROOT/'learned_hand_margins/search_v3';out=ROOT/'learned_carry_refinement_v2';out.mkdir(exist_ok=True)
    if (out/'settings.json').exists():raise FileExistsError(out)
    checkpoint=source/'policy.npz';scene=ROOT/'hands_release_v2/execution/scene_physics.xml';inp=ROOT/'repaired_adapter'
    spec=dict(scene=str(scene),episode_dir=str(inp),augmented_checkpoint=str(checkpoint),tolerance_deg=1.5)
    search=Search(spec);best=np.zeros(6);report=search.evaluate(best)
    original=json.loads((source/'evaluation/result.json').read_text())['metrics']['official_style_reference_proxy']['objects']
    if not np.allclose(report['original_aligned_mean_position_cm'],[original[n]['mean_position_error_cm'][0] for n in search.metric.names],atol=1e-8,rtol=0):raise ValueError('Zero correction must reproduce unchanged original metric')
    dump(out/'settings.json',dict(spec=spec,source_checkpoint_sha256=sha(checkpoint),source_scene_sha256=sha(scene),
        initial=report,samples=64,iterations=12,workers=16,seed=1019,dim=6,
        scope='Physically optimize six smooth right-wrist carry offsets by modifying learned phase weights. Original reference, clock, meshes, initial state and all robot physical constraints unchanged.',
        windows_seconds=[[2.45,2.8,4.8,5.4],[12.65,13.,15.45,16.3]],translation_norm_bound_m=.005,
        preserved_source_frame_means_cm=[1.6,1.5],supported_clear_release_required=True,learned_feedback_weights_unchanged=True,
        ordinary_command_bounds_unchanged=True,soft_limit_baseline_deg=report['all_finger_limit_max_degrees'],soft_limit_review_ceiling_deg=1.8))
    rng=np.random.default_rng(1019);mean=np.zeros(6);std=np.full(6,.002);history=[];started=time.monotonic();passing=[]
    with ProcessPoolExecutor(max_workers=16,mp_context=mp.get_context('spawn'),initializer=init,initargs=(spec,)) as pool:
        for iteration in range(12):
            xs=np.vstack([best,mean,mean+rng.normal(size=(62,6))*std]);y=xs.reshape(-1,2,3);y*=np.minimum(1.,.005/np.maximum(np.linalg.norm(y,axis=-1,keepdims=True),1e-12));xs=y.reshape(-1,6)
            rs=list(pool.map(evaluate,xs,chunksize=1));order=np.argsort([r['cost'] for r in rs])
            for x,r in zip(xs,rs):
                if r.get('mp_sr')==1. and r.get('preservation_pass') and r.get('all_finger_limit_max_degrees',99)<=1.8:passing.append((x.copy(),r.copy()))
            if rs[order[0]]['cost']<report['cost']:best=xs[order[0]].copy();report=rs[order[0]]
            elite=xs[order[:8]];mean=.25*mean+.75*elite.mean(0);std=np.maximum(.3*std+.7*elite.std(0),.00025)
            row=dict(iteration=iteration,best=report,parameters=best,elapsed_s=time.monotonic()-started);history.append(row)
            dump(out/f'iteration_{iteration:03d}.json',dict(parameters=xs,reports=rs));dump(out/'progress.json',dict(history=history));print(json.dumps(row,default=lambda x:x.tolist()),flush=True)
            if passing:
                best,report=min(passing,key=lambda v:(v[1]['all_finger_limit_max_degrees']>1.5,v[1]['cost']))
                if report['quality_preserves_baseline']:break
    search.set_parameters(best);selected=out/'policy.npz';np.savez_compressed(selected,**search.policy.data)
    result=final_evaluate(1,str(scene),str(selected),str(out/'evaluation'),str(inp))
    audit_joints(out/'evaluation');audit_support(out/'evaluation');audit_alignment(out/'evaluation/result.json')
    dump(out/'training_report.json',dict(learned_policy=True,method='Derivative-free physical fine-tuning of a six-dimensional subspace of trained phase weights',source_checkpoint=str(checkpoint),source_checkpoint_sha256=sha(checkpoint),checkpoint=str(selected),checkpoint_sha256=sha(selected),trainer_sha256=sha(__file__),evaluator_source_sha256=sha(Path(__file__).with_name('episode1_learned_carry_evaluator_v2.py')),best=report,parameters=best,learned_feedback_weights_unchanged=True,policy_contains_command_table=False,original_object_reference_unchanged=True,original_physics_unchanged=True))
    dump(out/'result.json',dict(best=report,checkpoint=str(selected),evaluation=result,history=history))
    print(json.dumps(dict(best=report,original_proxy=result['metrics']['reference_proxy_mp_sr']),default=lambda x:x.tolist()),flush=True)
if __name__=='__main__':main()
