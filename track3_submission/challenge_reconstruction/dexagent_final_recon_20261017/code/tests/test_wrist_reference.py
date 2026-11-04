import sys
from pathlib import Path
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from wrist_reference import continuous_euler_xyz


class ContinuousWristTests(unittest.TestCase):
    def test_pitch_crosses_pi_without_false_half_turn(self):
        expected=np.deg2rad(np.array([[-216.,177.,-80.],[-217.,181.,-81.],[-219.,185.,-82.]]))
        rotations=Rotation.from_euler('XYZ',expected)
        result=continuous_euler_xyz(rotations,expected[0])
        self.assertLess(np.abs(result-expected).max(),1e-10)
        self.assertLess(np.rad2deg(np.abs(np.diff(result,axis=0))).max(),5.)

    def test_all_axes_wrap_and_preserve_orientations(self):
        rng=np.random.default_rng(43)
        angles=np.cumsum(rng.normal(0,.03,(100,3)),axis=0)+np.array([3*np.pi,np.pi,-3*np.pi])
        rotations=Rotation.from_euler('XYZ',angles)
        result=continuous_euler_xyz(rotations,angles[0])
        self.assertLess((rotations.inv()*Rotation.from_euler('XYZ',result)).magnitude().max(),1e-10)
        self.assertLess(np.abs(np.diff(result,axis=0)).max(),.2)

    def test_invalid_anchor_rejected(self):
        with self.assertRaises(ValueError):
            continuous_euler_xyz(Rotation.identity(2),[0,float('nan'),0])


if __name__=='__main__':unittest.main()
