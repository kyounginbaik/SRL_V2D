"""Fit a brief unused-finger settling correction, then test the full learned actor."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import argparse
import json
import multiprocessing as mp
import numpy as np
import mujoco
from pinky_margin_policy_v2 import PinkyMarginPolicy
from evaluate_policy_component import evaluate
from audit_full_joint_limits import main as audit_joints
from learned_feedback_policy import sha

B = Path(__file__).resolve().parents[1]
BASE = B/'refinement_20261018/pinky_margin_v2/episode_000008'
WORK = None


def init():
    global WORK
    r = json.loads((BASE/'evaluation/result.json').read_text())
    m = mujoco.MjModel.from_xml_path(r['execution']['scene'])
    policy = PinkyMarginPolicy(BASE/'policy.npz')
    ids = np.array([j for j in range(m.njnt) if m.jnt_limited[j]
                    and (m.joint(j).name or '').startswith('robot_right_')
                    and any('_'+f+'_' in m.joint(j).name for f in ['thumb','index','middle','ring','pinky'])])
    WORK = m, policy, ids


def score(x):
    m, actor, ids = WORK
    actor.data['margin_bias_rad'][5:] = x
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m,d,0)
    mujoco.mj_forward(m,d)
    worst = 0.
    for _ in range(round(.15/m.opt.timestep)):
        d.ctrl[:] = actor(d.time+m.opt.timestep,d.qpos,d.qvel)
        mujoco.mj_step(m,d)
        q = d.qpos[m.jnt_qposadr[ids]]
        worst = max(worst, np.max(np.maximum(m.jnt_range[ids,0]-q,q-m.jnt_range[ids,1])))
    return float(np.degrees(worst))


def run(output,workers=16):
    out = Path(output);out.mkdir(parents=True,exist_ok=True)
    if (out/'search.json').exists():raise FileExistsError(out)
    init();m,actor,_ = WORK
    original = actor.data['margin_bias_rad'].copy()
    best=original[5:].copy();mean=best.copy();std=np.array([.07,.25,.3,.3,.3]);rng=np.random.default_rng(123)
    lo=np.array([0.,-.7,-.7,-.6,-.6]);hi=np.array([.2618,.7,.7,.6,.6])
    value=score(best);history=[]
    with ProcessPoolExecutor(workers,mp_context=mp.get_context('spawn'),initializer=init) as pool:
        for iteration in range(12):
            xs=np.vstack([best,mean,np.clip(mean+rng.normal(size=(94,5))*std,lo,hi)])
            values=np.array(list(pool.map(score,xs)));order=np.argsort(values)
            if values[order[0]]<value:best=xs[order[0]].copy();value=float(values[order[0]])
            elites=xs[order[:12]];mean=.2*mean+.8*elites.mean(0);std=np.maximum(.2*std+.8*elites.std(0),.015)
            history.append(dict(iteration=iteration,worst_startup_degrees=value,parameters=best.tolist()))
            print(json.dumps(history[-1]),flush=True)
    (out/'search.json').write_text(json.dumps(dict(history=history,source_checkpoint=str(BASE/'policy.npz'),
        source_sha256=sha(BASE/'policy.npz'),prefix_horizon_s=.15,initial_state_unchanged=True),indent=2)+'\n')
    r=json.loads((BASE/'evaluation/result.json').read_text())
    for i,left_flex in enumerate([original[2],-.03,-.07,-.12]):
        with np.load(BASE/'policy.npz',allow_pickle=False) as z:weights={k:z[k] for k in z.files}
        weights['margin_bias_rad'][5:]=best;weights['margin_bias_rad'][2]=left_flex
        trial=out/f'candidate_{i}';trial.mkdir(exist_ok=True);checkpoint=trial/'policy.npz';np.savez_compressed(checkpoint,**weights)
        report=dict(learned_policy=True,method='Physical fine-tuning of brief startup pinky correction and bounded left flexion margin',
            source_checkpoint=str(BASE/'policy.npz'),source_checkpoint_sha256=sha(BASE/'policy.npz'),
            checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),trainer_sha256=sha(__file__),
            parameters=weights['margin_bias_rad'].tolist(),startup_only_right=True,physics_and_initial_state_unchanged=True,
            prefix_horizon_s=.15,full_horizon_validation_required=True)
        (trial/'training_report.json').write_text(json.dumps(report,indent=2)+'\n')
        result=evaluate(8,r['execution']['scene'],checkpoint,trial/'evaluation',r['input_dir'])
        audit_joints(trial/'evaluation')
        print(json.dumps(dict(candidate=i,mp_sr=result['metrics']['reference_proxy_mp_sr'])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--workers',type=int,default=16)
    run(**vars(p.parse_args()))
