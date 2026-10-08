import ast,importlib.util,unittest
from pathlib import Path
from types import SimpleNamespace as S
import numpy as np
from scipy.spatial.transform import Rotation
SRC=Path('/data1/SRL_V2D/robotic_grounding/flash_chord/src/flash_chord/retargeting')
spec=importlib.util.spec_from_file_location('floating_chunk',SRC/'floating_chunk.py');fc=importlib.util.module_from_spec(spec);spec.loader.exec_module(fc)
class FloatingEditorTests(unittest.TestCase):
 def test_repeated_publications_preserve_manifold_and_bounds(self):
  n=101;base=np.zeros((n,6));base[:,5]=1;base[:,1]=.2
  basis=np.sin(np.linspace(0,np.pi,n))[:,None];jb=np.zeros((5,6));jb[:2,:2]=np.eye(2);jb[2:,2:5]=np.eye(3)
  p=S(ball_joints=[(2,2)],qids=np.array([0,1]),lower=np.array([-1,-1]),upper=np.array([1,1]),layout=S(scalar_joints=S(names=['left_wrist_x','left_finger'])))
  e=S(baseline=S(joint_q=base,fps=20),binding=S(joint_q=base),projector=p,basis=basis,joint_basis=jb,blend=.25)
  for i in range(20):
   q=fc.floating_candidate(e,np.random.default_rng(i).normal(size=5)*10)
   np.testing.assert_allclose(np.linalg.norm(q[:,2:6],axis=1),1,atol=1e-6)
   np.testing.assert_allclose(q[[0,-1]],base[[0,-1]],atol=1e-6)
   self.assertLessEqual(abs(q[:,0]).max(),.030001);self.assertLessEqual(abs(q[:,1]-.2).max(),.120001)
   self.assertLessEqual(Rotation.from_quat(q[:,2:6]).magnitude().max(),.120001)
   self.assertLessEqual(np.max(np.linalg.norm(np.diff(Rotation.from_quat(q[:,2:6]).as_rotvec(),axis=0),axis=1))*20,.50001)
   e.binding=S(joint_q=q)
 def test_cannot_sacrifice_second_object_for_better_mean(self):
  tree=ast.parse((SRC/'ttsa_editor.py').read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='passes_study_gate');ns={'np':np};exec(compile(ast.Module(body=[fn],type_ignores=[]),'gate','exec'),ns)
  base=dict(object_position_robust_m=.1,object_orientation_robust_rad=.1,object_success_robust_fraction=.5,unstable_world_fraction=0,per_object_position_mean_m=[.1,.1],per_object_orientation_mean_rad=[.1,.1])
  candidate={**base,'object_position_robust_m':.08,'per_object_position_mean_m':[.04,.12]}
  self.assertFalse(ns['passes_study_gate'](candidate,base));self.assertTrue(ns['passes_study_gate'](base,base))
if __name__=='__main__':unittest.main()
