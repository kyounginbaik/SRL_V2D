"""Frozen initialization evidence must bind the exact geometry and reference."""
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

import pytest

spec = importlib.util.spec_from_file_location('recipe_initial_state', Path(__file__).with_name('initial_state.py'))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.fixture
def evidence(tmp_path):
    assets = tmp_path/'assets'
    assets.mkdir()
    hashes = {}
    for name in ('support.usda', 'support_report.json', 'vessel.obj', 'lid.obj'):
        path = assets/name
        path.write_text(name)
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    for name in ('vessel', 'lid'):
        path = assets/(name+'.urdf')
        path.write_text(f'<robot><link name="body"><collision><geometry><mesh filename="{name}.obj"/></geometry></collision></link></robot>')
        hashes[name+'.urdf'] = hashlib.sha256(path.read_bytes()).hexdigest()
    pose = {'position': [0.,0.,.1], 'quaternion_wxyz': [1.,0.,0.,0.]}
    report = dict(schema_version=1, episode='episode_000001', source_reference_sha256='a'*64,
        source_and_frozen_object_trajectories_identical=True, support_sha256=hashes['support.usda'],
        support_report_sha256=hashes['support_report.json'],
        object_initial_pose_overrides={'vessel': dict(pose), 'lid': dict(pose)},
        object_asset_closure_sha256={name: {'assets/'+name+suffix: hashes[name+suffix]
                                          for suffix in ('.obj','.urdf')} for name in ('vessel','lid')},
        cpu_physics_passed=True, gpu_physics_passed=None, detachment_validation={'passed': True},
        passive_validation=dict(duration_seconds=2., physics={'assistance': 0},
            exact_command_index_reference_gate=dict(target_unchanged=True, pass_every_sample_first2s=True,
                max_position_error_first2s_m=[.01,.02], max_orientation_error_first2s_rad=[.1,.2],
                limits=dict(position_m=.05, orientation_rad=math.radians(20), persistence_frames=3))))
    path = tmp_path/'initial_state_report.json'
    def validate():
        path.write_text(json.dumps(report))
        return module.validate_initial_state_report(path, episode='episode_000001', source_sha256='a'*64,
            frozen_directory=tmp_path, object_names=['vessel','lid'],
            object_urdf_paths=['assets/vessel.urdf','assets/lid.urdf'])
    return report, validate, assets


def test_valid_report_preserves_poses_and_does_not_claim_gpu_pass(evidence):
    report, validate, _ = evidence
    result = validate()
    assert result['object_initial_pose_overrides'] == report['object_initial_pose_overrides']
    assert result['initial_state_scope'] == 'frame_zero_only'
    assert result['initial_state_cpu_physics_passed']
    assert not result['initial_state_gpu_physics_passed']
    assert not result['reference_commands_modified']


@pytest.mark.parametrize('tamper', ['source', 'support', 'object', 'missing_name', 'quat', 'cpu',
    'detachment', 'target', 'tracking', 'numeric_error', 'threshold', 'missing_mesh'])
def test_changed_or_failed_evidence_is_rejected(evidence, tamper):
    report, validate, assets = evidence
    gate = report['passive_validation']['exact_command_index_reference_gate']
    if tamper == 'source': report['source_reference_sha256'] = 'b'*64
    elif tamper == 'support': (assets/'support.usda').write_text('changed')
    elif tamper == 'object': (assets/'lid.obj').write_text('changed')
    elif tamper == 'missing_name': del report['object_initial_pose_overrides']['lid']
    elif tamper == 'quat': report['object_initial_pose_overrides']['lid']['quaternion_wxyz'] = [0,0,0,0]
    elif tamper == 'cpu': report['cpu_physics_passed'] = False
    elif tamper == 'detachment': report['detachment_validation']['passed'] = False
    elif tamper == 'target': gate['target_unchanged'] = False
    elif tamper == 'tracking': gate['pass_every_sample_first2s'] = False
    elif tamper == 'numeric_error': gate['max_position_error_first2s_m'] = [.15,.15]
    elif tamper == 'missing_mesh': del report['object_asset_closure_sha256']['lid']['assets/lid.obj']
    else: gate['limits']['position_m'] = .15
    with pytest.raises(ValueError): validate()


def test_recipe_hydra_override_roundtrips_named_full_poses(evidence):
    from hydra.core.override_parser.overrides_parser import OverridesParser
    report, validate, _ = evidence
    recipe = Path(__file__).resolve().parents[1]/'code/hypergradient_launch.py'
    if not recipe.exists():
        recipe = Path(__file__).resolve().parents[1]/'episode_000001/code/hypergradient_launch.py'
    spec = importlib.util.spec_from_file_location('test_hypergradient_launch', recipe)
    launch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launch)
    config = {'episode': validate(), 'wrist_control': {}}
    command = []
    launch.configure_hypergradient(command, config, 'smoke', 2048, recipe.parent)
    override, = [x for x in command if x.startswith('scene.object_initial_pose_overrides=')]
    assert OverridesParser.create().parse_override(override).value() == report['object_initial_pose_overrides']
