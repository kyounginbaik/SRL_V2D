"""Surface selection across independent chains with native mesh geometry."""
import unittest
import numpy as np
import mujoco
from flash_chord.retargeting.contact_wrench import ContactPatch
from flash_chord.retargeting.surface_contact_search import demand_coverage,search_native_surface_contacts


def cube(size):
    return ' '.join(str(v) for x in (-size,size) for y in (-size,size) for z in (-size,size) for v in (x,y,z))


def scene():
    parts=[]
    for i,pos in enumerate([(-.025,0,.05),(.025,0,.05),(0,.025,.05)]):
        joints=''.join(f'<joint name="j{i}{k}" type="slide" axis="{int(k==0)} {int(k==1)} {int(k==2)}" range="-.02 .02"/>' for k in range(3))
        parts.append(f'<body name="finger{i}" pos="{pos[0]} {pos[1]} {pos[2]}">{joints}<geom name="pad{i}" type="mesh" mesh="pad" mass=".01"/></body>')
    xml=f'''<mujoco><option gravity="0 0 -9.81" impratio="20"/>
    <default><geom condim="3" friction="1 .005 .0001"/></default>
    <asset><mesh name="pad" vertex="{cube(.005)}"/><mesh name="box" vertex="{cube(.01)}"/></asset>
    <worldbody><geom type="plane" pos="0 0 .04" size=".2 .2 .01"/>{''.join(parts)}
    <body name="object" pos="0 0 .05"><freejoint/><geom type="mesh" mesh="box" mass=".1"/></body></worldbody></mujoco>'''
    return mujoco.MjModel.from_xml_string(xml)


class SurfaceSearchTests(unittest.TestCase):
    def test_three_chain_surface_grasp_is_force_feasible_and_reachable(self):
        model=scene();groups=[[model.geom('pad'+str(i)).id] for i in range(3)]
        fit,report=search_native_surface_contacts(model,model.qpos0.copy(),np.arange(9),np.full(9,-.02),np.full(9,.02),
            np.full(9,.01),groups,[model.body('finger'+str(i)).id for i in range(3)],model.body('object').id,
            [0,0,.981,0,0,0],maximum_fits=8,max_evaluations=100)
        self.assertTrue(fit.model_feasible,msg=str(report))
        self.assertGreater(report['nominally_force_feasible_arrangements'],0)
        self.assertGreaterEqual(len(set(model.geom_bodyid[fit.hand_geom_ids])),2)
        np.testing.assert_allclose(fit.wrench_residual,0,atol=1e-6)
        np.testing.assert_array_equal(fit.qpos[9:],model.qpos0[9:])

    def test_faces_under_a_resting_object_are_not_proposed(self):
        model=scene();groups=[[model.geom('pad'+str(i)).id] for i in range(3)]
        seed=model.qpos0.copy();seed[11]-=1e-4  # A settled soft contact sinks slightly into its support.
        fit,report=search_native_surface_contacts(model,seed,np.arange(9),np.full(9,-.02),np.full(9,.02),
            np.full(9,.01),groups,[model.body('finger'+str(i)).id for i in range(3)],model.body('object').id,
            [0,0,.981,0,0,0],maximum_fits=8,max_evaluations=100,faces_per_direction=4)
        self.assertGreater(report['faces_blocked_by_support'],0)
        self.assertLess(max(patch.normal[2] for patch in fit.patches),.5)

    def test_demand_coverage_counts_only_supported_motion_wrenches(self):
        def patch(position,normal,group):
            n=np.asarray(normal,float);t1=np.cross(n,[0.,0.,1.]);t1/=np.linalg.norm(t1)
            return ContactPatch(tuple(position),tuple(n),tuple(t1),tuple(np.cross(n,t1)),(.7,.7),group)
        pinch=[patch((-.01,0,0),(1,0,0),0),patch((.01,0,0),(-1,0,0),1)]
        tripod=[patch((-.01,-.03,0),(1,0,0),0),patch((-.01,.03,0),(1,0,0),1),patch((.01,0,0),(-1,0,0),2)]
        demands=np.array([[0,0,.98,0,0,0],[0,0,.98,.02,0,0],[0,0,.98,0,.02,0]])
        limits=lambda c:{i:8. for i in range(c)};minima=lambda c:{i:1. for i in range(c)}
        # Two collinear points have no free torsion about their axis; a tripod resists it.
        self.assertLess(demand_coverage(pinch,limits(2),minima(2),demands,np.zeros(3)),1.)
        self.assertEqual(demand_coverage(tripod,limits(3),minima(3),demands,np.zeros(3)),1.)
        self.assertEqual(demand_coverage(tripod,limits(3),minima(3),np.empty((0,6)),np.zeros(3)),1.)

if __name__=='__main__':unittest.main()
