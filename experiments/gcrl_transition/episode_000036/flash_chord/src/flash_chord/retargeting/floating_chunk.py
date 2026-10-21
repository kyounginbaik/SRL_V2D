"""Bounded scalar and SO(3) corrections for floating Sharpa reference trajectories."""
import numpy as np
from scipy.spatial.transform import Rotation


def bounded_delta(delta, bound, velocity, fps):
    scale=min(1., bound/max(float(np.max(np.abs(delta))),1e-12))
    if len(delta)>1:
        scale=min(scale,velocity/max(float(np.max(np.abs(np.diff(delta,axis=0)))*fps),1e-12))
    return delta*scale


def floating_candidate(editor,action):
    # Bounds affect reference corrections, not the motor policy action range.
    finger_bound=float(editor.options.get("finger_offset_rad", .52))
    finger_speed=float(editor.options.get("finger_correction_speed_rad_s", 2.0))
    rotation_bound=float(editor.options.get("wrist_rotation_offset_rad", .52))
    rotation_speed=float(editor.options.get("wrist_rotation_correction_speed_rad_s", 1.5))
    translation_bound=float(editor.options.get("wrist_translation_offset_m", .03))
    translation_speed=float(editor.options.get("wrist_translation_correction_speed_m_s", .10))
    if min(finger_bound,finger_speed,rotation_bound,rotation_speed,
           translation_bound,translation_speed)<=0:
        raise ValueError("Correction limits must be positive")
    base=editor.baseline.joint_q
    action=np.asarray(action,dtype=np.float64)
    wrist_dim=editor.basis.shape[1]*editor.joint_basis.shape[0]
    coefficients=action[:wrist_dim].reshape(editor.basis.shape[1],editor.joint_basis.shape[0])
    if not np.isfinite(coefficients).all():raise ValueError('Nonfinite upper action')
    raw=editor.basis@np.clip(coefficients,-1,1)@editor.joint_basis
    q=base.copy();p=editor.projector;fps=editor.baseline.fps
    for i,(name,qid) in enumerate(zip(p.layout.scalar_joints.names,p.qids)):
        translation=name.endswith(('wrist_x','wrist_y','wrist_z'))
        bound,velocity=(translation_bound,translation_speed) if translation else (finger_bound,finger_speed)
        proposal=bounded_delta(raw[:,qid],bound,velocity,fps)
        delta=(1-editor.blend)*(editor.binding.joint_q[:,qid]-base[:,qid])+editor.blend*proposal
        delta=np.clip(base[:,qid]+delta,p.lower[i],p.upper[i])-base[:,qid]
        q[:,qid]=base[:,qid]+bounded_delta(delta,bound,velocity,fps)
    for start,_ in p.ball_joints:
        baseline=Rotation.from_quat(base[:,start:start+4])
        current=(Rotation.from_quat(editor.binding.joint_q[:,start:start+4])*baseline.inv()).as_rotvec()
        proposed=bounded_delta(raw[:,start:start+3],rotation_bound/np.sqrt(3),rotation_speed/np.sqrt(3),fps)
        delta=(1-editor.blend)*current+editor.blend*proposed
        delta=bounded_delta(delta,rotation_bound/np.sqrt(3),rotation_speed/np.sqrt(3),fps)
        q[:,start:start+4]=(Rotation.from_rotvec(delta)*baseline).as_quat()
    if editor.options.get("edit_scope") in ("wrists_only", "wrists_and_grasp_prototypes", "wrists_and_oc_grasps"):
        for hand in p.layout.hands:
            q[:,hand.finger_q_ids]=base[:,hand.finger_q_ids]
    if editor.options.get("edit_scope") == "wrists_and_grasp_prototypes":
        # Finger references move toward grasp prototypes inside contact windows (same blend as wrists).
        proposal=editor.grasp_proto.finger_delta(action[wrist_dim:],base)
        qlist=list(p.qids)
        for hand in p.layout.hands:
            for qid in hand.finger_q_ids:
                i=qlist.index(qid)
                delta=(1-editor.blend)*(editor.binding.joint_q[:,qid]-base[:,qid])+editor.blend*proposal[:,qid]
                q[:,qid]=np.clip(base[:,qid]+delta,p.lower[i],p.upper[i])
    if editor.options.get("edit_scope") == "wrists_and_oc_grasps":
        # Absolute choice: grasps are not averaged with the previous reference (a grasp works at its full pose).
        q=editor.oc.proposal(action[wrist_dim:],q)
        qlist=list(p.qids)
        for hand in p.layout.hands:
            for qid in hand.finger_q_ids:
                i=qlist.index(qid)
                q[:,qid]=np.clip(q[:,qid],p.lower[i],p.upper[i])
    q[[0,-1]]=base[[0,-1]]
    if not np.isfinite(q).all():raise ValueError('Nonfinite floating reference')
    return q
