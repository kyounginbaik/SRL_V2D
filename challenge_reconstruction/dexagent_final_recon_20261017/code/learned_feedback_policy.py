"""Learn a phase/proprioception feedback policy from physical demonstrations.

Radial-basis behavior cloning is supervised learning, not FlashSAC. The policy
stores fitted regression weights and fixed features, never a command trajectory.
Recovery labels are a local stabilizing-teacher assumption; only a fresh online
physics rollout can establish whether the learned controller actually succeeds.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import scipy.linalg
import mujoco


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def features(phase, centers, width):
    p = np.asarray(phase).reshape(-1, 1)
    r = np.exp(-.5 * ((p-centers[None])/width)**2)
    r /= r.sum(axis=1, keepdims=True)
    return r


class LearnedPolicy:
    def __init__(self, checkpoint):
        with np.load(checkpoint, allow_pickle=False) as z:
            self.data = {k:z[k] for k in z.files}
        d = self.data
        if str(d['algorithm']) != 'radial_basis_behavior_cloning_with_learned_feedback':
            raise ValueError('Unexpected learned-policy format')
        self.duration = float(d['duration_s'])
        self.qadr = d['qadr']; self.vadr = d['vadr']

    def __call__(self, next_time_s, qpos, qvel):
        d = self.data
        state = np.concatenate([qpos[self.qadr],qvel[self.vadr]])
        state = (state-d['state_mean'])/d['state_scale']
        phi = features([next_time_s/self.duration],d['centers'],float(d['width']))[0]
        output = phi @ d['phase_weights'] + state @ d['feedback_weights']
        return np.clip(output*d['action_scale']+d['action_mean'],d['ctrl_lo'],d['ctrl_hi'])


def train(scene, trajectory, output, centers=512, feedback_gain=.03, velocity_gain=.0005, seed=43):
    out = Path(output).resolve(); out.mkdir(parents=True, exist_ok=True)
    checkpoint = out/'policy.npz'
    if checkpoint.exists(): raise FileExistsError(checkpoint)
    model = mujoco.MjModel.from_xml_path(str(scene))
    with np.load(trajectory, allow_pickle=False) as z:
        ctrl = z['substep_ctrl'].copy()
        q = np.vstack([z['initial_qpos'][None],z['substep_qpos'][:-1]])
        v = np.vstack([z['initial_qvel'][None],z['substep_qvel'][:-1]])
        times = z['substep_time_s'].copy()
        fps = float(z['fps']); frames = len(z['qpos'])
        if str(z['scene_sha256']) != sha(scene): raise ValueError('Teacher scene hash mismatch')
    if len(q) != len(ctrl) or len(v) != len(ctrl): raise ValueError('Causal alignment mismatch')
    js = model.actuator_trnid[:,0]
    qadr = model.jnt_qposadr[js].astype(int); vadr = model.jnt_dofadr[js].astype(int)
    slide = model.jnt_type[js] == mujoco.mjtJoint.mjJNT_SLIDE
    state = np.hstack([q[:,qadr],v[:,vadr]])
    action_scale = np.where(slide,.1,1.)
    state_scale = np.r_[action_scale,np.where(slide,1.,10.)]
    state_mean = state.mean(axis=0); action_mean = ctrl.mean(axis=0)
    norm_state = (state-state_mean)/state_scale
    norm_action = (ctrl-action_mean)/action_scale
    rng = np.random.default_rng(seed); begin=time.monotonic()

    # Learn feedback from paired local state perturbations. The teacher offers
    # small recovery target corrections, while all actual effort limits remain
    # the unchanged physical servos' limits. These samples are supervision, not
    # claimed successful physical rollouts.
    perturb = rng.normal(size=(8192,2*model.nu))
    perturb *= np.r_[np.where(slide,.002,.02),np.where(slide,.02,.2)]
    labels = -feedback_gain*perturb[:,:model.nu] - velocity_gain*perturb[:,model.nu:]
    feedback,_,_,_ = scipy.linalg.lstsq(perturb/state_scale,labels/action_scale,lapack_driver='gelsy')
    residual = norm_action-norm_state @ feedback
    phase_centers = np.linspace(0.,1.,centers)
    width = 1.15/max(1,centers-1)
    X = features(times/times[-1],phase_centers,width)
    heldout = np.arange(len(times)) % 7 == 3
    train_mask = ~heldout
    # Ridge regularization is numerical stabilization, not a physical metric.
    ridge = 1e-10
    A = X[train_mask].T @ X[train_mask] + ridge*np.eye(centers)
    weights = scipy.linalg.solve(A,X[train_mask].T @ residual[train_mask],assume_a='pos')
    prediction = (X @ weights + norm_state @ feedback)*action_scale+action_mean
    error = prediction-ctrl
    hold_error = error[heldout]
    recovery_prediction = (perturb/state_scale) @ feedback * action_scale
    report = dict(algorithm='radial_basis_behavior_cloning_with_learned_feedback',
        learned_policy=True, scene=str(Path(scene).resolve()), scene_sha256=sha(scene),
        demonstration=str(Path(trajectory).resolve()), demonstration_sha256=sha(trajectory),
        causal_alignment='action[k] paired with initial state if k=0, otherwise physical state[k-1]',
        training_samples=int(train_mask.sum()),heldout_samples=int(heldout.sum()),
        heldout_scope='Interleaved demonstration times; not a claim of task generalization',
        synthetic_recovery_samples=len(perturb),phase_basis_count=centers,
        feedback_teacher_position_gain=feedback_gain,feedback_teacher_velocity_gain=velocity_gain,
        action_rmse=float(np.sqrt(np.mean(error**2))),
        heldout_translation_max_error_m=float(np.max(np.abs(hold_error[:,slide]))),
        heldout_joint_max_error_rad=float(np.max(np.abs(hold_error[:,~slide]))),
        recovery_label_max_error=float(np.max(np.abs(recovery_prediction-labels))),
        feedback_weight_norm=float(np.linalg.norm(feedback)),
        training_seconds=time.monotonic()-begin,seed=seed,
        trainer_sha256=sha(__file__),policy_contains_command_table=False,
        actor_input='continuous phase and measured56joint positions +56velocities',
        actor_output='56 bounded actuator position commands',
        evaluation_required='Fresh full-horizon unassisted closed-loop physics from exact home')
    np.savez_compressed(checkpoint,algorithm=np.array(report['algorithm']),
        centers=phase_centers,width=width,phase_weights=weights,feedback_weights=feedback,
        state_mean=state_mean,state_scale=state_scale,action_mean=action_mean,action_scale=action_scale,
        qadr=qadr,vadr=vadr,ctrl_lo=model.actuator_ctrlrange[:,0],ctrl_hi=model.actuator_ctrlrange[:,1],
        actuator_names=np.asarray([model.actuator(i).name for i in range(model.nu)]),
        duration_s=times[-1],fps=fps,frames=frames,scene_sha256=np.array(sha(scene)))
    report['checkpoint']=str(checkpoint);report['checkpoint_sha256']=sha(checkpoint)
    (out/'training_report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)
    return report


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--trajectory',required=True)
    p.add_argument('--output',required=True);p.add_argument('--centers',type=int,default=512)
    p.add_argument('--feedback-gain',type=float,default=.03);p.add_argument('--velocity-gain',type=float,default=.0005)
    p.add_argument('--seed',type=int,default=43)
    train(**vars(p.parse_args()))
