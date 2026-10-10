import ast,unittest
from pathlib import Path
import numpy as np
p=Path(__file__).parents[2]/'src/flash_chord/retargeting/ttsa_editor.py'
node=next(n for n in ast.parse(p.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='upper_cost')
ns={'np':np};exec(compile(ast.Module(body=[node],type_ignores=[]),str(p),'exec'),ns)
class TestUpperObjectCost(unittest.TestCase):
 def test_robot_tracking_does_not_change_upper_cost(self):
  m=dict(object_position_robust_m=.02,object_orientation_robust_rad=.1,object_success_robust_fraction=.9,wrench_robust_mse=.2,unstable_world_fraction=0,robot_position_robust_mse=0)
  cost=lambda:ns['upper_cost'](m,np.zeros((20,2)),20,1,robust=True)
  before=cost();m['robot_position_robust_mse']=100
  self.assertEqual(before,cost())
 def test_worse_object_position_increases_cost(self):
  m=dict(object_position_robust_m=.02,object_orientation_robust_rad=.1,object_success_robust_fraction=.9,wrench_robust_mse=.2,unstable_world_fraction=0)
  before=ns['upper_cost'](m,np.zeros((20,2)),20,1,robust=True);m['object_position_robust_m']=.03
  self.assertGreater(ns['upper_cost'](m,np.zeros((20,2)),20,1,robust=True),before)
