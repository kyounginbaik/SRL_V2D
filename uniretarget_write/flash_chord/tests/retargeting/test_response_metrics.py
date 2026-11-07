from types import SimpleNamespace as NS
import numpy as np
from flash_chord.retargeting.response_editor import joint_object_metrics,geometric_edit_penalty


def test_joint_success_cannot_hide_failed_lid():
    assert joint_object_metrics(np.array([[0.,.18]]),np.zeros((1,2)))['object_success_robust_fraction']==0.
    assert joint_object_metrics(np.array([[.01,.02]]),np.zeros((1,2)))['object_success_robust_fraction']==1.


def test_reference_penalty_is_quaternion_sign_invariant():
    q=np.zeros((20,58));q[:,[6,35]]=1.
    edited=q.copy();edited[5:15,3:7]*=-1
    layout=NS(hands=[NS(finger_q_ids=np.arange(7,29)),NS(finger_q_ids=np.arange(36,58))])
    assert geometric_edit_penalty(edited,q,layout,20.)==0.
    shifted=q.copy();shifted[5:15,0]=.01
    assert geometric_edit_penalty(shifted,q,layout,20.)>0.
