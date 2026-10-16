import importlib.util
from pathlib import Path
import sys
import unittest
p=Path(__file__).parents[2]/'src/flash_chord/lifecycle/performance_voc.py'
spec=importlib.util.spec_from_file_location('performance_voc',p);mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
Gate=mod.PerformanceVOCGate

def good():
 return dict(per_object_gate_success_fraction=[.99,.99],per_object_position_p95_m=[.02,.02],
             per_object_orientation_p95_rad=[.1,.1],robot_position_rms_m=.02,full_horizon=True,all_errors_finite=True)

class TestGate(unittest.TestCase):
 def test_no_timer_promotion_and_sustained_success(self):
  g=Gate(2)
  self.assertEqual(g.scale,1.)
  for _ in range(2):self.assertFalse(g.observe(good(),0)['lowered'])
  self.assertTrue(g.observe(good(),0)['lowered']);self.assertEqual(g.scale,.75)
 def test_one_bad_object_blocks_and_resets(self):
  g=Gate(2);g.observe(good(),0);d=good();d['per_object_position_p95_m']=[0.,.08]
  self.assertFalse(g.observe(d,0)['passed']);self.assertEqual(g.streak,0);self.assertEqual(g.scale,1.)
 def test_missing_nan_and_short_rollout_block(self):
  for change in [{'per_object_position_p95_m':[.01]}, {'per_object_orientation_p95_rad':[float('nan'),.01]}, {'full_horizon':False}, {'all_errors_finite':False}]:
   g=Gate(2);d=good();d.update(change)
   for _ in range(5):self.assertFalse(g.observe(d,0)['lowered'])
 def test_hand_tracking_does_not_override_object_success(self):
  g=Gate(2);d=good();d['robot_position_rms_m']=.2
  for _ in range(3):result=g.observe(d,0)
  self.assertTrue(result['lowered'])
 def test_reference_change_resets_streak(self):
  g=Gate(2);g.observe(good(),0);g.observe(good(),0)
  self.assertFalse(g.observe(good(),1)['lowered']);self.assertEqual(g.streak,1)
 def test_zero_requires_all_stages(self):
  g=Gate(2)
  for i in range(24):g.observe(good(),0)
  self.assertEqual(g.scale,0.)
  self.assertFalse(g.observe(good(),0)['lowered'])

if __name__=='__main__':unittest.main()
