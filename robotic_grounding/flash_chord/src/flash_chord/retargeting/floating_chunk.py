"""Bounded scalar and SO(3) corrections for floating Sharpa reference trajectories."""
import numpy as np
from scipy.spatial.transform import Rotation


def bounded_delta(delta, bound, velocity, fps):
    scale=min(1., bound/max(float(np.max(np.abs(delta))),1e-12))
    if len(delta)>1:
        scale=min(scale,velocity/max(float(np.max(np.abs(np.diff(delta,axis=0)))*fps),1e-12))
    return delta*scale


def floating_candidate(editor,action):
    base=editor.baseline.joint_q
    coefficients=np.asarray(action,dtype=np.float64).reshape(editor.basis.shape[1],editor.joint_basis.shape[0])
    if not np.isfinite(coefficients).all():raise ValueError('Nonfinite upper action')
    raw=editor.basis@np.clip(coefficients,-1,1)@editor.joint_basis
    q=base.copy();p=editor.projector;fps=editor.baseline.fps
    for i,(name,qid) in enumerate(zip(p.layout.scalar_joints.names,p.qids)):
        translation=name.endswith(('wrist_x','wrist_y','wrist_z'))
        bound,velocity=(.03,.10) if translation else (.12,.5)
        proposal=bounded_delta(raw[:,qid],bound,velocity,fps)
        delta=(1-editor.blend)*(editor.binding.joint_q[:,qid]-base[:,qid])+editor.blend*proposal
        delta=np.clip(base[:,qid]+delta,p.lower[i],p.upper[i])-base[:,qid]
        q[:,qid]=base[:,qid]+bounded_delta(delta,bound,velocity,fps)
    for start,_ in p.ball_joints:
        baseline=Rotation.from_quat(base[:,start:start+4])
        current=(Rotation.from_quat(editor.binding.joint_q[:,start:start+4])*baseline.inv()).as_rotvec()
        proposed=bounded_delta(raw[:,start:start+3],.12/np.sqrt(3),.5/np.sqrt(3),fps)
        delta=(1-editor.blend)*current+editor.blend*proposed
        delta=bounded_delta(delta,.12/np.sqrt(3),.5/np.sqrt(3),fps)
        q[:,start:start+4]=(Rotation.from_rotvec(delta)*baseline).as_quat()
    q[[0,-1]]=base[[0,-1]]
    if not np.isfinite(q).all():raise ValueError('Nonfinite floating reference')
    return q
