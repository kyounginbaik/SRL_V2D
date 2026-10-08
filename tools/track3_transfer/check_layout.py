import newton,numpy as np
from flash_chord.embodiments.sharpa_hands import SharpaHands
from flash_chord.embodiments.binding import build_robot_reference_binding
from flash_chord.retargeting.manipulation_bilevel import LocalProjector
robot=SharpaHands();builder=newton.ModelBuilder();layout=robot.build(builder)
q=np.tile(np.asarray(builder.joint_q,dtype=np.float32),(4,1));target=np.tile(np.asarray(builder.joint_target_pos,dtype=np.float32),(4,1));binding=build_robot_reference_binding(builder,layout,q,target,20)
p=LocalProjector(robot,binding)
assert len(p.ball_joints)==2
print('Floating layout validated',len(p.qids),'scalar joints,',len(p.ball_joints),'ball wrists')
