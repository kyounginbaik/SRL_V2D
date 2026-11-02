"""Actual native-PD tracking and actuation contracts for the feedforward tape."""
import unittest
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from flash_chord.retargeting.native_inverse_dynamics import NativeScalarPDInverse

XML='''<mujoco><option timestep=".001" gravity="0 0 -9.81"/>
<worldbody><body pos="0 0 .5"><joint type="slide" axis="0 0 1" damping="1"/>
<geom type="sphere" size=".01" mass="1"/></body></worldbody>
<actuator><position joint="0" kp="100" kv="20" forcerange="-50 50"/></actuator></mujoco>'''.replace('<joint type=','<joint name="slide" type=').replace('joint="0"','joint="slide"')


class InverseDynamicsTests(unittest.TestCase):
    def test_native_pd_feedforward_removes_gravity_and_dynamic_tracking_offset(self):
        model=mujoco.MjModel.from_xml_string(XML);inverse=NativeScalarPDInverse(model)
        feedforward=mujoco.MjData(model);baseline=mujoco.MjData(model);errors=[];original=model.qpos0.copy()
        for step in range(1800):
            t=step*model.opt.timestep;goal=.05*np.sin(2*t);vel=.1*np.cos(2*t);acc=-.2*np.sin(2*t)
            targets=inverse.targets([goal],[vel],[acc])
            self.assertFalse(targets.violations)
            feedforward.ctrl[0]=targets.position_controls[0];baseline.ctrl[0]=goal
            mujoco.mj_step(model,feedforward);mujoco.mj_step(model,baseline)
            if step>600:errors.append([abs(feedforward.qpos[0]-goal),abs(baseline.qpos[0]-goal)])
        errors=np.asarray(errors)
        self.assertLess(errors[:,0].mean(),.002)
        self.assertGreater(errors[:,1].mean(),.06)
        np.testing.assert_array_equal(model.qpos0,original)

    def test_separate_velocity_actuator_is_accounted_for(self):
        xml=XML.replace('kv="20"','kv="0"').replace('</actuator>','<velocity joint="slide" kv="20"/></actuator>')
        model=mujoco.MjModel.from_xml_string(xml);inverse=NativeScalarPDInverse(model)
        target=inverse.targets([.02],[.1],[0])
        self.assertAlmostEqual(target.position_controls[0],.02+(9.81+2.1)/100,places=7)

    def test_actuator_routed_gravity_compensation_is_not_counted_twice(self):
        # jnt_actgravcomp routes qfrc_gravcomp through qfrc_actuator, so it is
        # absent from qfrc_passive. The floating Sharpa wrist slides use this.
        xml=XML.replace('<body pos="0 0 .5">','<body pos="0 0 .5" gravcomp="1">').replace('damping="1"/>','damping="1" actuatorgravcomp="true"/>')
        model=mujoco.MjModel.from_xml_string(xml);data=mujoco.MjData(model)
        control=NativeScalarPDInverse(model).targets([.02]).position_controls[0]
        self.assertAlmostEqual(control,.02,places=7)
        data.qpos[0]=.02;data.ctrl[0]=control
        for _ in range(200):mujoco.mj_step(model,data)
        self.assertAlmostEqual(data.qpos[0],.02,places=5)

    def test_passive_gravity_compensation_is_not_counted_twice(self):
        xml=XML.replace('<body pos="0 0 .5">','<body pos="0 0 .5" gravcomp="1">')
        model=mujoco.MjModel.from_xml_string(xml)
        self.assertAlmostEqual(NativeScalarPDInverse(model).targets([.02]).position_controls[0],.02,places=7)

    def test_infeasible_effort_is_reported_without_changing_physics(self):
        model=mujoco.MjModel.from_xml_string(XML.replace('-50 50','-1 1'))
        target=NativeScalarPDInverse(model).targets([0])
        self.assertTrue(target.violations)
        self.assertEqual(target.violations[0]['reason'],'native_position_effort_limit')
        np.testing.assert_array_equal(model.actuator_forcerange,[[-1,1]])

    def test_ball_drives_are_never_silently_treated_as_scalar_position_targets(self):
        model=mujoco.MjModel.from_xml_string('<mujoco><worldbody><body><joint type="ball"/><geom size=".01" mass="1"/></body></worldbody></mujoco>')
        target=NativeScalarPDInverse(model).targets(model.qpos0)
        self.assertEqual(target.position_controls,{})
        np.testing.assert_array_equal(target.unhandled_native_dofs,[0,1,2])

class BallInverseTests(unittest.TestCase):
    def test_child_frame_ball_target_holds_a_nonidentity_pose_against_gravity(self):
        xml='<mujoco><option timestep=".001"/><worldbody><body pos="0 0 1"><joint name="wrist" type="ball" damping="1"/>' \
            '<geom type="box" pos=".08 0 0" size=".04 .02 .02" mass="1"/></body></worldbody><actuator>' \
            '<motor joint="wrist" gear="1 0 0"/><motor joint="wrist" gear="0 1 0"/><motor joint="wrist" gear="0 0 1"/>' \
            '</actuator></mujoco>'
        model=mujoco.MjModel.from_xml_string(xml);inverse=NativeScalarPDInverse(model,ball_drives={0:(25.,45.)})
        goal=Rotation.from_rotvec([.4,-.3,.2]);q=goal.as_quat()[[3,0,1,2]]
        result=inverse.targets(q);self.assertEqual(len(result.unhandled_native_dofs),0)
        self.assertFalse(result.violations)
        target=Rotation.from_rotvec(result.native_ball_rotvec_targets[0])
        corrected=mujoco.MjData(model);baseline=mujoco.MjData(model)
        corrected.qpos[:]=q;baseline.qpos[:]=q
        for _ in range(1500):
            for data,rotation in [(corrected,target),(baseline,goal)]:
                current=Rotation.from_quat(data.qpos[[1,2,3,0]])
                torque=np.clip(25*(current.inv()*rotation).as_rotvec(),-45,45)
                data.ctrl[:]=torque;mujoco.mj_step(model,data)
                np.testing.assert_allclose(data.qfrc_actuator,torque,atol=1e-8)
        error=lambda data:(goal.inv()*Rotation.from_quat(data.qpos[[1,2,3,0]])).magnitude()
        self.assertLess(error(corrected),1e-4)
        self.assertGreater(error(baseline),.01)

if __name__=='__main__':unittest.main()
