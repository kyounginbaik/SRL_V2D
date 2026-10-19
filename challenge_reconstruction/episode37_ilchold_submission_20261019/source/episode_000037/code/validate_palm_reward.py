from pathlib import Path
import numpy as np,warp as wp,json
from flash_chord.objectives.keypoints import hand_keypoints_objective
from flash_chord.objectives.shaping import ObjectiveShape
b=Path(__file__).resolve().parents[1]
with wp.ScopedDevice('cpu'):
 poses=np.zeros((12,7),np.float32);poses[:,6]=1
 bodies=wp.array(poses,dtype=wp.transform);ids=wp.array(np.arange(12),dtype=wp.int32);local=wp.zeros(12,dtype=wp.vec3)
 def reward(palm,displaced=None):
  target=np.zeros((12,3),np.float32)
  if displaced is not None:target[displaced,0]=.1
  targets=wp.array(target,dtype=wp.vec3);out=wp.zeros(2,dtype=wp.float32)
  wp.launch(hand_keypoints_objective,dim=12,inputs=[bodies,ids,local,targets,6,2,12,.1,int(ObjectiveShape.LAPLACIAN),0.,palm],outputs=[out])
  return out.numpy()/6
 for palm in [1.,10.]:np.testing.assert_allclose(reward(palm),1.,atol=1e-6)
 palm_loss=1-reward(10.,0)[0];tip_loss=1-reward(10.,1)[0]
 np.testing.assert_allclose(palm_loss/tip_loss,10.,rtol=1e-4)
 np.testing.assert_allclose(reward(1.,0)[0],reward(1.,1)[0],atol=1e-7)
 report={'passed':True,'unit_maximum_preserved':True,'palm_to_single_finger_influence_ratio':float(palm_loss/tip_loss),'default_equal_weight_compatible':True}
 (b/'palm_reward_validation.json').write_text(json.dumps(report,indent=2));print(report)
