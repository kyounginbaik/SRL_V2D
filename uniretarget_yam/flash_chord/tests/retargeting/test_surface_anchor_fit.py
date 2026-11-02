"""Assigned material contacts provide a gradient across a nearest-face plateau."""
import unittest
import numpy as np
import mujoco
from flash_chord.retargeting.contact_wrench import ContactPatch
from flash_chord.retargeting.surface_anchor_fit import SurfaceAnchor,fit_native_surface_anchors,closest_triangle_point
from test_contact_invariant_fit import XML,assert_lift_from_fresh_reset


def fit_anchors():
    model=mujoco.MjModel.from_xml_string(XML)
    anchors=[]
    for name,sign in [('leftpad',1.),('rightpad',-1.)]:
        point=(-sign*.01,0.,.05);normal=(sign,0.,0.)
        patch=ContactPatch(point,normal,(0.,1.,0.),(0.,0.,1.),(.7,.7),name)
        anchors.append(SurfaceAnchor(model.geom(name).id,(sign*.005,0.,0.),normal,patch))
    result=fit_native_surface_anchors(model,model.qpos0.copy(),[0,1,2,3],[-.01,-.005,-.02,-.005],[.02,.005,.01,.005],
        [.01,.005,.01,.005],anchors,[model.body('left').id,model.body('right').id],model.body('object').id,
        [0,0,.981,0,0,0],normal_force_limits={'leftpad':4.,'rightpad':4.})
    return model,result


class AnchorFitTests(unittest.TestCase):
    def test_actual_material_points_reach_opposing_object_surfaces(self):
        model,fit=fit_anchors()
        self.assertTrue(fit.model_feasible)
        self.assertLess(max(fit.contact_gaps),1e-4)
        np.testing.assert_array_equal(fit.qpos[4:],model.qpos0[4:])

    def test_anchor_fit_then_force_calibration_lifts_in_fresh_physics(self):
        model,fit=fit_anchors();assert_lift_from_fresh_reset(self,model,fit)

class TriangleContactTests(unittest.TestCase):
    def test_projection_stays_inside_the_finite_face(self):
        tri=np.array([[0,0,0],[1,0,0],[0,1,0]],float)
        np.testing.assert_allclose(closest_triangle_point([.2,.2,1],tri),[.2,.2,0])
        np.testing.assert_allclose(closest_triangle_point([2,2,1],tri),[.5,.5,0])
        np.testing.assert_allclose(closest_triangle_point([-1,-1,0],tri),[0,0,0])

    def test_surface_area_contact_does_not_require_unactuated_tangent_motion(self):
        model=mujoco.MjModel.from_xml_string(XML);anchors=[]
        for name,sign in [('leftpad',1.),('rightpad',-1.)]:
            x=-sign*.01;normal=(sign,0.,0.)
            patch=ContactPatch((x,sign*.004,.05),normal,(0.,1.,0.),(0.,0.,1.),(.7,.7),name)
            triangle=((x,-.01,.04),(x,.01,.04),(x,0.,.06))
            anchors.append(SurfaceAnchor(model.geom(name).id,(sign*.005,0.,0.),normal,patch,triangle))
        result=fit_native_surface_anchors(model,model.qpos0.copy(),[0,1,2,3],[-.01,-.005,-.02,-.005],[.02,.005,.01,.005],
            [.01,.005,.01,.005],anchors,[model.body('left').id,model.body('right').id],model.body('object').id,
            [0,0,.981,0,0,0],normal_force_limits={'leftpad':4.,'rightpad':4.})
        self.assertTrue(result.model_feasible)
        self.assertLess(max(result.contact_gaps),1e-4)
        np.testing.assert_allclose([p.position[1] for p in result.patches],0,atol=1e-6)
        np.testing.assert_allclose(result.wrench_residual,0,atol=1e-6)

class MaterialSupportTests(unittest.TestCase):
    def test_vertex_normal_cone_allows_contact_without_face_alignment(self):
        xml=XML.replace('type="sphere" size=".005"','type="box" size=".005 .005 .005"')
        model=mujoco.MjModel.from_xml_string(xml);anchors=[]
        for name,sign in [('leftpad',1.),('rightpad',-1.)]:
            hint=np.array([sign,sign,0.])/np.sqrt(2)
            patch=ContactPatch((-sign*.01,sign*.005,.05),(sign,0.,0.),(0.,1.,0.),(0.,0.,1.),(.7,.7),name)
            anchors.append(SurfaceAnchor(model.geom(name).id,(sign*.005,sign*.005,0.),tuple(hint),patch))
        result=fit_native_surface_anchors(model,model.qpos0.copy(),[0,1,2,3],[-.01,-.005,-.02,-.005],[.02,.005,.01,.005],
            [.01,.005,.01,.005],anchors,[model.body('left').id,model.body('right').id],model.body('object').id,
            [0,0,.981,0,0,0],normal_force_limits={'leftpad':4.,'rightpad':4.})
        self.assertTrue(result.model_feasible)
        self.assertLess(max(result.contact_gaps),1e-4)

    def test_fictitious_material_point_outside_a_sphere_is_rejected(self):
        model=mujoco.MjModel.from_xml_string(XML)
        patch=ContactPatch((-.01,0.,.05),(1.,0.,0.),(0.,1.,0.),(0.,0.,1.),(.7,.7),'left')
        anchor=SurfaceAnchor(model.geom('leftpad').id,(.005,.005,0.),(1.,0.,0.),patch)
        with self.assertRaisesRegex(ValueError,'actual hand collider'):
            fit_native_surface_anchors(model,model.qpos0.copy(),[0],[-.01],[.02],[.01],[anchor],
                [model.body('left').id],model.body('object').id,[1,0,0,0,0,0],normal_force_limits={'left':4.})

if __name__=='__main__':unittest.main()
