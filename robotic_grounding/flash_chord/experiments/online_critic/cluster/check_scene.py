"""Assemble one CPU world to check the complete portable asset closure."""
import json
from pathlib import Path
import warp as wp
import flash_chord.embodiments.sharpa_hands
from flash_chord.embodiments.base import get_embodiment
from flash_chord.scene.setup import setup_scene
from flash_chord.scene.collision import CollisionPolicy

root = Path(__file__).resolve().parents[1]
with wp.ScopedDevice('cpu'):
    setup = setup_scene(parquet=root/'frozen/episode_000001/reference.parquet',
        control_fps=20., motion_speed=1., embodiment=get_embodiment('sharpa_hands')(),
        collision=CollisionPolicy(), world_count=1, include_support=False, decompose_objects=False)
    assert setup.reference.num_frames == 335
    assert setup.scene.robot_reference.joint_q.shape[0] == 335
report = dict(passed=True, frames=335, scope='One-world CPU scene and all referenced robot/object assets')
(root/'reports/scene_preflight.json').write_text(json.dumps(report, indent=2)+'\n')
print('CPU scene and portable assets verified', flush=True)
