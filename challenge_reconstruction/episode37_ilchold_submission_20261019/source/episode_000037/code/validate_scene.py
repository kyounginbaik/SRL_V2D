from pathlib import Path
import faulthandler
faulthandler.dump_traceback_later(45, repeat=True)
import json,numpy as np
import warp as wp
wp.config.kernel_cache_dir="/tmp/uniretarget_mixed_warp"
import flash_chord.embodiments.sharpa_hands
from flash_chord.embodiments.base import get_embodiment
from flash_chord.scene.setup import setup_scene
from flash_chord.scene.collision import CollisionPolicy
B=Path(__file__).resolve().parents[1]
s=setup_scene(parquet=B/'frozen/episode_000001/reference.parquet',control_fps=20.,motion_speed=1.,embodiment=get_embodiment('sharpa_hands')(),collision=CollisionPolicy(),world_count=1,include_support=False,decompose_objects=False)
q=np.load(B/'frozen/episode_000001/completed_ik.npz');ref=s.scene.robot_reference;report={'scene_built':True,'physics_rollout_tested':False,'frames':ref.num_frames,'sides':{}}
for side in ['left','right']:
 expected=q[side+'_qpos'][:,:3]
 actual=np.asarray(ref.hand(side).wrist_pos_w)
 report['sides'][side]={'max_wrist_error_m':float(np.abs(actual-expected).max())}
 assert np.allclose(expected,actual,atol=1e-5)
(B/'scene_validation.json').write_text(json.dumps(report,indent=2)+'\n');print(report,flush=True)
