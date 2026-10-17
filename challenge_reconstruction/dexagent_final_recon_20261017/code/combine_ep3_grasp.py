from pathlib import Path
import argparse,json
import numpy as np,mujoco
from grasp_geometry_seeds_v6 import B,TR
p=argparse.ArgumentParser();p.add_argument('--right-reference',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
left=B/'refinement_20261017/cem_v2/episode_000003/reference.npz'
scene=B/'refinement_20261017/cem_v2/episode_000003/execution/scene_physics.xml'
m=mujoco.MjModel.from_xml_path(str(scene));r=TR.load_reference(a.right_reference,m);l=TR.load_reference(left,m)
q=r['qpos'].copy();cols=[int(m.jnt_qposadr[j]) for j in range(m.njnt) if m.joint(j).name.startswith('robot_left_')]
assert len(cols)==28 and q.shape==l['qpos'].shape and np.array_equal(q[0],l['qpos'][0])
q[:,cols]=l['qpos'][:,cols]
for name in ['white_pot','wooden_spoon']:
 c=int(m.joint(name+'_free').qposadr[0]);assert np.array_equal(q[:,c:c+7],r['qpos'][:,c:c+7]);assert np.array_equal(q[:,c:c+7],l['qpos'][:,c:c+7])
a.output.parent.mkdir(parents=True,exist_ok=True)
TR.save_reference(a.output,m,q,r['labels'],r['fps'],r['segment'],r['segments'],source='CEM left-hand commands plus separately proposed right-hand grasp; full new physical execution required',arms_coupled=False)
(a.output.parent/'combination_manifest.json').write_text(json.dumps({'left_reference':str(left),'right_reference':str(a.right_reference),'scene':str(scene),'left_columns':cols,'object_targets_unchanged':True,'initial_state_unchanged':True,'timing_unchanged':True},indent=2))
print(a.output)
