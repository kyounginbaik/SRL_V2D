"""Native geometry/force fit and unassisted two-finger acquisition/lift test."""
import unittest
import numpy as np
import mujoco
from flash_chord.retargeting.contact_invariant_fit import fit_native_contact_pose,scalar_force_preload,calibrate_native_normal_preload

# Match the canonical SimConfig contact impedance ratio. The earlier default-1
# diagnostic slipped despite a static fit; that failure remains in v1/v2 logs.
XML='''<mujoco><option timestep="0.001" gravity="0 0 -9.81" cone="pyramidal" iterations="100" impratio="20"/>
<default><geom friction="1 0.005 0.0001" condim="3" solref="0.005 1" solimp="0.95 0.99 0.001"/><joint damping="1"/></default>
<worldbody><geom name="table" type="plane" pos="0 0 0.04" size=".2 .2 .01"/>
<body name="left" pos="-.025 0 .05"><joint name="lx" type="slide" axis="1 0 0" range="-.01 .02"/>
<joint name="lz" type="slide" axis="0 0 1" range="-.01 .1"/><geom name="leftpad" type="sphere" size=".005" mass=".01"/></body>
<body name="right" pos=".025 0 .05"><joint name="rx" type="slide" axis="1 0 0" range="-.02 .01"/>
<joint name="rz" type="slide" axis="0 0 1" range="-.01 .1"/><geom name="rightpad" type="sphere" size=".005" mass=".01"/></body>
<body name="object" pos="0 0 .05"><freejoint/><geom name="objectshape" type="box" size=".01 .01 .01" mass=".1"/></body></worldbody>
<actuator><position joint="lx" kp="5000" kv="10" forcerange="-20 20"/><position joint="lz" kp="5000" kv="10" forcerange="-20 20"/>
<position joint="rx" kp="5000" kv="10" forcerange="-20 20"/><position joint="rz" kp="5000" kv="10" forcerange="-20 20"/></actuator></mujoco>'''


def fit_model():
    model=mujoco.MjModel.from_xml_string(XML)
    result=fit_native_contact_pose(model,model.qpos0.copy(),[0,1,2,3],[-.01,-.005,-.02,-.005],[.02,.005,.01,.005],
        [.01,.005,.01,.005],[[model.geom('leftpad').id],[model.geom('rightpad').id]],
        [model.body('left').id,model.body('right').id],model.body('object').id,[0,0,.981,0,0,0],
        normal_force_budget=4.,max_evaluations=200)
    return model,result


def assert_lift_from_fresh_reset(case,model,fit):
    ctrl,load,missing=scalar_force_preload(model,fit)
    case.assertGreater(ctrl[0],fit.qpos[0])
    case.assertLess(ctrl[2],fit.qpos[2])
    # The analytic preload is a first proposal. Identify the remaining
    # native contact compliance through fresh reset AND loaded-motion prefixes.
    # A resting-table force measurement did not transfer to moving friction.
    def measure(controls):
        state=mujoco.MjData(model)
        for step in range(600):
            lift=.04*np.clip((step-400)/600,0,1)
            for actuator,value in controls.items():
                state.ctrl[actuator]=value+(lift if actuator in (1,3) else 0.)
            mujoco.mj_step(model,state)
        forces=np.zeros_like(fit.forces_on_object)
        object_body=model.body('object').id
        body_to_patch={int(model.geom_bodyid[g]):i for i,g in enumerate(fit.hand_geom_ids)}
        for contact_id,c in enumerate(state.contact[:state.ncon]):
            body0,body1=model.geom_bodyid[c.geom]
            if body1==object_body and body0 in body_to_patch:side,body=1,body0
            elif body0==object_body and body1 in body_to_patch:side,body=-1,body1
            else:continue
            wrench=np.zeros(6);mujoco.mj_contactForce(model,state,contact_id,wrench)
            forces[body_to_patch[body]]+=side*c.frame.reshape(3,3).T@wrench[:3]
        return forces
    ctrl,calibration=calibrate_native_normal_preload(model,fit,ctrl,measure)
    case.assertTrue(calibration['converged'],msg=str(calibration))
    data=mujoco.MjData(model)  # a NEW original open-finger task reset
    for step in range(1200):
        lift=.04*np.clip((step-400)/600,0,1)
        for actuator,value in ctrl.items(): data.ctrl[actuator]=value+(lift if actuator in (1,3) else 0.)
        mujoco.mj_step(model,data)
    height=float(data.xpos[model.body('object').id,2])
    case.assertGreater(height,.08,msg=str((height,fit.contact_gaps,ctrl)))



class InvariantFitTests(unittest.TestCase):
    def test_joint_geometry_and_force_fit_forms_a_supported_contact_pair(self):
        model,fit=fit_model()
        self.assertTrue(fit.model_feasible,msg=str((fit.message,fit.contact_gaps,fit.wrench_residual,fit.cost)))
        self.assertLess(np.linalg.norm(fit.wrench_residual),.05)
        np.testing.assert_array_equal(fit.qpos[4:],model.qpos0[4:])
        self.assertGreater(fit.qpos[0],.005)
        self.assertLess(fit.qpos[2],-.005)

    def test_force_preload_closes_both_sides_and_lifts_from_actual_reset(self):
        model,fit=fit_model()
        assert_lift_from_fresh_reset(self,model,fit)

    def test_valid_normal_force_margin_is_retained(self):
        model,fit=fit_model();ctrl,_,_=scalar_force_preload(model,fit)
        def measure(_):return 1.25*fit.forces_on_object
        adjusted,result=calibrate_native_normal_preload(model,fit,ctrl,measure)
        self.assertTrue(result['converged'])
        self.assertEqual(adjusted,ctrl)
        self.assertEqual(len(result['history']),1)

    def test_excess_force_budget_is_not_reported_as_calibrated(self):
        model,fit=fit_model();ctrl,_,_=scalar_force_preload(model,fit)
        def measure(_):return np.array([p.normal for p in fit.patches])*(fit.normal_force_budget+1.)
        _,result=calibrate_native_normal_preload(model,fit,ctrl,measure)
        self.assertFalse(result['converged'])
        self.assertTrue(result['history'][0]['normal_force_budget_exceeded'])

    def test_ball_reference_edits_preserve_unit_quaternions_and_object_coordinates(self):
        xml=XML.replace('<body name="left"', '<body name="wrist"><joint name="wristball" type="ball"/>'
            '<inertial pos="0 0 0" mass=".1" diaginertia=".001 .001 .001"/><body name="left"')
        xml=xml.replace('<body name="object"','</body><body name="object"')
        model=mujoco.MjModel.from_xml_string(xml)
        ids=[int(model.jnt_qposadr[model.joint(name).id]) for name in ('lx','lz','rx','rz')]
        result=fit_native_contact_pose(model,model.qpos0.copy(),ids,[-.01,-.005,-.02,-.005],[.02,.005,.01,.005],
            [.01,.005,.01,.005],[[model.geom('leftpad').id],[model.geom('rightpad').id]],
            [model.body('left').id,model.body('right').id],model.body('object').id,[0,0,.981,0,0,0],
            normal_force_budget=4.,max_evaluations=100,rotation_qpos_starts=[0])
        self.assertAlmostEqual(float(np.linalg.norm(result.qpos[:4])),1.,places=10)
        np.testing.assert_array_equal(result.qpos[8:],model.qpos0[8:])

    def test_object_coordinate_edits_are_rejected(self):
        model=mujoco.MjModel.from_xml_string(XML)
        with self.assertRaisesRegex(ValueError,'object coordinates'):
            fit_native_contact_pose(model,model.qpos0.copy(),[4],[-1],[1],[.1],[[1],[2]],[1,2],3,[0,0,1,0,0,0],normal_force_budget=4)

if __name__=='__main__': unittest.main()
