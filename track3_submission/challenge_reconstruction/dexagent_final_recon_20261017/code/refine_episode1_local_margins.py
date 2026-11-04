"""Measured four-coordinate margin neighborhood, preserving all physics."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import itertools,json,multiprocessing as mp,time
import numpy as np,mujoco
from search_repaired_hand_margins_evaluator_v3 import init,evaluate,Search
from evaluate_policy_component import evaluate as final_evaluate
from audit_full_joint_limits import main as audit_joints
from audit_object_support import audit as audit_support
from audit_episode1_alignment_bias import audit as audit_alignment
from learned_feedback_policy import sha
from refine_grasp_cem import dump
B=Path(__file__).resolve().parents[1]
ROOT=B/'refinement_20261018/episode_000001/reconstruction_repair/patch_height094'


def main():
    source=ROOT/'learned_hand_margins/search_v3';out=ROOT/'learned_hand_margins/search_local_v4'
    out.mkdir(parents=True,exist_ok=True)
    if (out/'settings.json').exists():raise FileExistsError(out)
    checkpoint=source/'policy.npz';scene=ROOT/'hands_release_v2/execution/scene_physics.xml';inp=ROOT/'repaired_adapter'
    with np.load(checkpoint,allow_pickle=False) as z:data={k:z[k] for k in z.files}
    m=mujoco.MjModel.from_xml_path(str(scene));base=data['margin_bias_rad'].copy()
    ids=data['margin_actuator_ids'];names=[m.actuator(int(a)).name for a in ids]
    lookup={name:i for i,name in enumerate(names)}
    axes=[('robot_right_right_index_PIP_position',[0,2,4]),('robot_left_left_pinky_CMC_position',[0,1,2]),
          ('robot_right_right_pinky_MCP_AA_position',[-3,0,3]),('robot_right_right_pinky_MCP_FE_position',[-3,0,3])]
    xs=[]
    for offsets in itertools.product(*(v for _,v in axes)):
        x=base.copy()
        for (name,_),deg in zip(axes,offsets):x[lookup[name]]+=np.radians(deg)
        xs.append(x)
    xs=np.vstack([base,np.unique(xs,axis=0)])
    spec=dict(scene=str(scene),episode_dir=str(inp),augmented_checkpoint=str(checkpoint),tolerance_deg=1.1)
    initial=Search(spec).evaluate(base)
    dump(out/'settings.json',dict(source_checkpoint=str(checkpoint),source_checkpoint_sha256=sha(checkpoint),
        source_scene_sha256=sha(scene),spec=spec,actuators=names,axes=axes,samples=len(xs),workers=16,
        initial=initial,scope='Measured local command-margin refinement; unchanged learned weights, gains, effort caps, state, meshes, objects and original timing'))
    start=time.monotonic()
    with ProcessPoolExecutor(max_workers=16,mp_context=mp.get_context('spawn'),initializer=init,initargs=(spec,)) as pool:
        reports=list(pool.map(evaluate,xs,chunksize=1))
    rows=[dict(parameters=x,report=r) for x,r in zip(xs,reports)]
    preserved=[(x,r) for x,r in zip(xs,reports) if r.get('preservation_pass') and r.get('mp_sr')==1.]
    if not preserved:raise RuntimeError('Even unchanged baseline lost preservation')
    best,report=min(preserved,key=lambda row:(row[1]['all_finger_limit_max_degrees'],row[1]['cost']))
    dump(out/'search.json',dict(rows=rows,selected_parameters=best,selected_report=report,elapsed_s=time.monotonic()-start))
    data['margin_bias_rad']=best;selected=out/'policy.npz';np.savez_compressed(selected,**data)
    result=final_evaluate(1,str(scene),str(selected),str(out/'evaluation'),str(inp))
    audit_joints(out/'evaluation');audit_support(out/'evaluation');audit_alignment(out/'evaluation/result.json')
    dump(out/'training_report.json',dict(learned_policy=True,method='Local physical fine-tuning of four bounded finger command margins',
        source_checkpoint=str(checkpoint),source_checkpoint_sha256=sha(checkpoint),checkpoint=str(selected),checkpoint_sha256=sha(selected),
        trainer_sha256=sha(__file__),best=report,parameters=best,all_other_learned_weights_unchanged=True,policy_contains_command_table=False,
        frozen_scene_sha256=sha(scene),original_object_reference_unchanged=True,actual_all44_fingers_checked_at_500hz=True))
    dump(out/'result.json',dict(best=report,checkpoint=str(selected),evaluation=result))
    print(json.dumps(dict(best=report,original_proxy=result['metrics']['reference_proxy_mp_sr']),default=lambda v:v.tolist()),flush=True)
if __name__=='__main__':main()
