"""Distance witness validation on unchanged native convex collision meshes."""
import unittest
import numpy as np
import mujoco
from flash_chord.retargeting.convex_distance import ConvexDistanceQuery

VERTICES=' '.join(str(v) for x in (-.01,.01) for y in (-.01,.01) for z in (-.01,.01) for v in (x,y,z))
XML=f'''<mujoco><asset><mesh name="cube" vertex="{VERTICES}"/></asset><worldbody>
<body pos="0 0 0"><freejoint/><geom type="mesh" mesh="cube"/></body>
<body pos=".03 0 0"><freejoint/><geom type="mesh" mesh="cube"/></body></worldbody></mujoco>'''


class DistanceTests(unittest.TestCase):
    def scene(self):
        model=mujoco.MjModel.from_xml_string(XML);data=mujoco.MjData(model);mujoco.mj_forward(model,data)
        return model,data

    def test_false_zero_is_repaired_only_with_supporting_witnesses(self):
        model,data=self.scene()
        def false_zero(m,d,a,b,maxdist,points):points[:]=[.01,0,0,.02,0,0];return 0.
        query=ConvexDistanceQuery(model,false_zero);query.update(data)
        distance,points=query.distance(0,1)
        self.assertAlmostEqual(distance,.01,places=7)
        self.assertEqual(query.repairs,1);self.assertEqual(query.fallbacks,0)

    def test_contradictory_witnesses_use_independent_convex_qp(self):
        model,data=self.scene()
        def wrong(m,d,a,b,maxdist,points):points[:]=[.01,.004,0,.02,-.004,0];return 0.
        query=ConvexDistanceQuery(model,wrong);query.update(data)
        distance,points=query.distance(0,1)
        self.assertAlmostEqual(distance,.01,places=7)
        self.assertEqual(query.fallbacks,1)
        self.assertAlmostEqual(np.linalg.norm(points[3:]-points[:3]),distance,places=10)

    def test_real_native_mesh_distance_and_moving_pose(self):
        model,data=self.scene();query=ConvexDistanceQuery(model);query.update(data)
        self.assertAlmostEqual(query.distance(0,1)[0],.01,places=6)
        data.qpos[7]=.05;mujoco.mj_forward(model,data);query.update(data)
        self.assertAlmostEqual(query.distance(0,1)[0],.03,places=6)

    def test_geometry_penetration_is_bounded_even_without_native_pair_records(self):
        model,data=self.scene();model.geom_contype[:]=0;model.geom_conaffinity[:]=0
        data.qpos[7]=.015;mujoco.mj_forward(model,data)
        def missing(m,d,a,b,maxdist,points):points[:]=0.;return 0.
        query=ConvexDistanceQuery(model,missing);query.update(data)
        distance,points=query.distance(0,1)
        self.assertAlmostEqual(distance,-.005,places=6)
        self.assertEqual(query.penetration_bounds,1)
        self.assertAlmostEqual(np.linalg.norm(points[3:]-points[:3]),abs(distance),places=9)

    def test_lower_bound_is_safe_and_updates_after_motion(self):
        model,data=self.scene();query=ConvexDistanceQuery(model);query.update(data)
        self.assertLessEqual(query.lower_bound(0,1),query.distance(0,1)[0]+1e-7)
        old=query.lower_bound(0,1)
        data.qpos[7]=.05;mujoco.mj_forward(model,data);query.update(data)
        self.assertGreater(query.lower_bound(0,1),old)
        self.assertLessEqual(query.lower_bound(0,1),query.distance(0,1)[0]+1e-7)

    def test_penetration_is_never_repaired_into_positive_clearance(self):
        model,data=self.scene();data.qpos[7]=.015;mujoco.mj_forward(model,data)
        query=ConvexDistanceQuery(model);query.update(data)
        distance,_=query.distance(0,1)
        self.assertLess(distance,0)
        self.assertAlmostEqual(distance,-.005,places=5)

if __name__=='__main__':unittest.main()
