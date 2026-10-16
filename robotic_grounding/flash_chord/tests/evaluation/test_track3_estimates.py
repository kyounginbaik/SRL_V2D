import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from flash_chord.evaluation.track3_estimates import estimate_track3, score_recording


def fixture():
    r = np.zeros((10, 2, 7)); r[..., 6] = 1; r[:, 1, 0] = .3
    vertices = [np.eye(3)*.05, np.eye(3)*.05]
    return r[:, None].copy(), r, vertices


def test_identity_and_rigid_world_alignment():
    a, r, v = fixture()
    rot = Rotation.from_rotvec([.2, -.4, .1])
    a[..., :3] = a[..., :3] @ rot.as_matrix().T + [.3, -.2, .5]
    a[..., 3:] = rot.as_quat()
    result = estimate_track3(a, r, v)
    assert result['auc'] == result['mp_sr'] == result['sp_sr'] == 1
    assert result['rpe_cm'] < 1e-12 and result['mppe_cm'] < 1e-12


def test_one_bad_object_fails_maniptrans_but_spider_can_pass():
    a, r, v = fixture(); a[1:, :, 1, 0] += .08
    result = estimate_track3(a, r, v)
    assert result['mp_sr'] == 0 and result['sp_sr'] == 1
    assert result['rpe_cm'] == pytest.approx(7.2)
    assert result['mppe_cm'] == pytest.approx(3.6)
    assert 0 < result['auc'] < 1


def test_lid_half_turn_equivalence_and_tilt_penalty():
    a, r, v = fixture()
    a[1:, :, 1, 3:] = Rotation.from_rotvec([0, 0, np.pi]).as_quat()
    axes = [[0,0,0], [0,0,1]]
    result = estimate_track3(a, r, v, axes=axes, orders=[1,2])
    assert result['mp_sr'] == 1 and result['mppe_cm'] < 1e-12
    a[1:, :, 1, 3:] = Rotation.from_rotvec([np.pi/2, 0, 0]).as_quat()
    assert estimate_track3(a, r, v, axes=axes, orders=[1,2])['mp_sr'] == 0


def test_single_object_rpe_and_multi_world_success_rate():
    a, r, v = fixture(); a = np.repeat(a[:, :, :1], 2, axis=1)
    a[1:, 1, 0, 0] += .2
    result = estimate_track3(a, r[:, :1], v[:1])
    assert result['rpe_cm'] == 0
    assert result['mp_sr'] == result['sp_sr'] == .5


@pytest.mark.parametrize('bad', [np.nan, np.inf])
def test_invalid_trajectory_is_not_reported_as_success(bad):
    a, r, v = fixture(); a[3, 0, 0, 0] = bad
    with pytest.raises(ValueError): estimate_track3(a, r, v)


@pytest.mark.parametrize('relocated', [False, True])
def test_recording_pipeline_and_full_horizon_guard(tmp_path, relocated):
    import pyarrow as pa
    import pyarrow.parquet as pq
    import trimesh
    (tmp_path/'assets').mkdir()
    mesh_path = tmp_path/'assets/box.ply'
    trimesh.creation.box(extents=[.1,.2,.3]).export(mesh_path)
    a, r, _ = fixture(); a, r = a[:, :, :1], r[:, :1]
    parquet = tmp_path/'reference.parquet'
    pq.write_table(pa.Table.from_pylist([dict(object_body_position=r[..., :3].tolist(),
        object_body_wxyz=r[..., [6,3,4,5]].tolist(), object_body_names=['wooden_spoon'],
        object_mesh_paths=['/missing/producer/assets/box.ply' if relocated else str(mesh_path)])]), parquet)
    recording = tmp_path/'poses.npz'
    np.savez(recording, achieved_pose_w=a, frame_indices=np.arange(10), voc_scale=0.)
    report = score_recording(recording, parquet, tmp_path/'scores.json')
    assert set(report['metrics']) == {'auc','sp_sr','mp_sr','mppe_cm','rpe_cm','cd_o_cm'}
    assert report['metrics']['cd_o_cm'] == 0 and not report['rpe_applicable']
    np.savez(recording, achieved_pose_w=a, frame_indices=np.arange(10)+1, voc_scale=0.)
    with pytest.raises(ValueError, match='exactly once'):
        score_recording(recording, parquet, tmp_path/'bad.json')
