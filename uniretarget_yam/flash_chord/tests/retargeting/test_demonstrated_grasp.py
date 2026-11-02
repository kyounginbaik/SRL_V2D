import numpy as np
from scipy.spatial.transform import Rotation

from flash_chord.retargeting.contact_refinement import ContactSegment,HandChannels,RefinementConfig
from flash_chord.retargeting.demonstrated_grasp import capture_steady_grasp,render_demonstrated_grasp,wrist_from_root
from flash_chord.retargeting.grasp_feedback import ObjectRelativeGrasp


def test_capture_preserves_native_finger_preload_and_separates_wrist_command_geometry():
    h=np.zeros((40,7));h[:,6]=1;h[:,0]=.1
    o=h.copy();o[:,0]=0
    c=np.zeros((40,9));c[:,:3]=h[:,:3]+[0,0,.01];c[:,6:]=[1.8,-.2,.7]
    schema=dict(wrist_pos_dof_ids=[0,1,2],wrist_orient_dof_ids=[3,4,5],finger_dof_ids=[6,7,8])
    good=np.ones(40,bool);good[:8]=False
    grasp,report=capture_steady_grasp(h,o,c,good,schema,interval=(0,40),width_frames=10)
    assert grasp is not None and report['window'][0]>=8
    assert grasp.command.fingers==(1.8,-.2,.7)
    assert grasp.command.wrist_position_o[2]==.01
    assert grasp.geometry.wrist_position_o[2]==0.
    assert not report['source_is_reacquisition_certificate']
    assert capture_steady_grasp(h,o,c,np.zeros(40,bool),schema,interval=(0,40),width_frames=10)[0] is None


def fixture():
    frames=200;q=np.zeros((frames,12));q[:,2]=.3;q[:,6]=1;q[:,7:9]=.2
    q[:,9:]=np.arange(frames)[:,None]*.001
    hand=HandChannels('left',(0,1,2),(3,4,5,6),(7,8),(1.,1.))
    offset=np.tile([.02,0,0],(frames,1));wrist=np.c_[q[:,:3]+offset,q[:,3:7]]
    obj=np.zeros((frames,7));obj[:,2]=.3;obj[:,6]=1
    segment=ContactSegment('left',0,60,130,1)
    phases=[ContactSegment('left',0,0,15,0),segment,ContactSegment('left',1,180,190,2)]
    grasp=ObjectRelativeGrasp((.08,0,0),tuple(Rotation.from_rotvec([0,0,.4]).as_quat()),(.9,.6))
    return q,hand,wrist,obj,segment,phases,grasp


def test_full_grasp_realization_preserves_root_wrist_offset_and_preload_separately():
    q,hand,wrist,obj,segment,phases,grasp=fixture();before=obj.copy()
    out,commands,info=render_demonstrated_grasp(q,hand,wrist,obj,grasp,segment,phases,[0,0],[.7,.5],fps=20.,
        return_config=RefinementConfig(max_translation_speed_m_s=10.,max_rotation_speed_rad_s=10.))
    assert out is not None,info
    actual=wrist_from_root(out,hand,q,wrist)
    np.testing.assert_allclose(actual[80,:3],[.08,0,.3],atol=1e-12)
    np.testing.assert_allclose(out[80,7:9],[.7,.5])
    np.testing.assert_allclose(commands['fingers'][80],[.9,.6])
    np.testing.assert_array_equal(out[:,9:],q[:,9:]);np.testing.assert_array_equal(obj,before)
    a,b=info['window'];np.testing.assert_array_equal(out[:a+1],q[:a+1]);np.testing.assert_array_equal(out[b:],q[b:])
    assert info['finger_goal_error_rad']==0.


def test_insufficient_time_rejects_complete_grasp_instead_of_shrinking_it():
    q,hand,wrist,obj,segment,_,grasp=fixture()
    phases=[ContactSegment('left',1,0,58,0),segment]
    out,commands,info=render_demonstrated_grasp(q,hand,wrist,obj,grasp,segment,phases,[0,0],[.7,.5],fps=20.)
    assert out is None and commands is None
    assert info['reason']=='insufficient_full_grasp_transition_time'


def test_actual_return_slew_can_fit_when_inherited_motion_cancels_conservative_offset():
    q,hand,wrist,obj,segment,_,_=fixture()
    # The original finger is already returning from +1 to 0. Replacing the
    # contact posture by -1 needs a two-radian correction at contact, but only
    # one radian of actual return motion. Check actual slew, not its bound.
    q[:,7]=1.;q[131:151,7]=np.linspace(.95,0.,20);q[151:,7]=0.
    phases=[segment,ContactSegment('left',1,152,190,2)]
    grasp=ObjectRelativeGrasp((.02,0,0),(0.,0.,0.,1.),(-1.,.2))
    out,commands,info=render_demonstrated_grasp(q,hand,wrist,obj,grasp,segment,phases,[-1.,0.],[1.,1.],fps=20.)
    assert out is not None,info
    assert info['required_return_frames']>info['return_frames']
    assert np.abs(np.diff(out[130:152,7])).max()*20.<=3.+1e-5
    np.testing.assert_allclose(out[80,7],-1.)
