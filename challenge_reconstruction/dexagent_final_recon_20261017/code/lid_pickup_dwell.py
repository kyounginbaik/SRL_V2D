"""Robot-only approach/closure dwell; preserve simulation time and object target."""
from pathlib import Path
import argparse,json,sys
import numpy as np,mujoco
B=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(B.parents[1]/'third_party/DexAgent'))
from tool_library.common import hand as H
from predict_settled_grasp import smooth
from floating_runtime import execute_reference
from evaluate_mp_sr import evaluate_files
from finger_gain_sweep import grasp_audit


def run(task):
    candidate=task//3;hold_end=(1.7,1.9,2.1)[task%3]
    inp=B/f'refinement_20261017/lid_normal/episode_000001/candidate_{candidate:02d}_depth3'
    ref=dict(np.load(inp/'reference.npz'));q=ref['qpos'].copy();fps=float(ref['fps'])
    scene=inp/'execution/scene_physics.xml';model=mujoco.MjModel.from_xml_path(str(scene))
    rig=H.HandRig.build(model,'robot_right_','right')
    cols=np.array([rig.wrist_qpos[k] for k in ('x','y','z','rx','ry','rz')]+list(rig.finger_qpos.values()))
    original=q.copy();times=np.arange(len(q))/fps
    start=1.;arrive=1.5;join=hold_end+.4;pose=original[round(1.7*fps),cols]
    initial=original[round(start*fps),cols];ending=original[round(join*fps),cols]
    for frame,t in enumerate(times):
        if t<start or t>join:continue
        if t<arrive:
            w=float(smooth((t-start)/(arrive-start)));q[frame,cols]=(1-w)*initial+w*pose
        elif t<=hold_end:q[frame,cols]=pose
        else:
            w=float(smooth((t-hold_end)/(join-hold_end)));q[frame,cols]=(1-w)*pose+w*ending
    untouched=np.setdiff1d(np.arange(model.nq),cols)
    assert np.array_equal(q[:,untouched],original[:,untouched]);assert np.array_equal(q[0],original[0])
    out=B/f'refinement_20261017/lid_dwell/episode_000001/candidate_{candidate:02d}_hold{hold_end:g}'
    out.mkdir(parents=True,exist_ok=True);reference=out/'reference.npz'
    np.savez_compressed(reference,**dict(ref,qpos=q))
    execution=execute_reference(scene,reference,out/'execution')
    metrics=evaluate_files(B.parent/'final_recon/episode_000001',execution['trajectory'],
        B/'episodes/episode_000001/input_manifest.json',execution['scene'])
    report=dict(candidate=candidate,robot_approach_start_s=start,robot_arrival_s=arrive,
        robot_hold_until_s=hold_end,robot_rejoins_original_command_s=join,
        source_reference=str(inp/'reference.npz'),object_initial_physics_and_timing_unchanged=True,
        execution=execution,metrics=metrics,grasp=grasp_audit(execution['scene'],execution['trajectory']))
    (out/'report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(metrics['official_style_reference_proxy'],indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--task',type=int,required=True)
    run(parser.parse_args().task)
