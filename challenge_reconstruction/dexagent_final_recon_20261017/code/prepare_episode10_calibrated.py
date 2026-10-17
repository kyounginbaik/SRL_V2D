"""Versioned measured-gravity adapter for episode10; original targets stay fixed."""
from pathlib import Path
import hashlib,json
import numpy as np
from scipy.spatial.transform import Rotation
import prepare_episodes as adapter
from evaluate_mp_sr import geometry_from_reference

B=Path(__file__).resolve().parents[1]


def main():
    out=B/'refinement_20261018/episode_000010/calibrated_inputs'
    target=out/'episodes/episode_000010'
    if target.exists():raise FileExistsError(target)
    report_path=B/'refinement_20261018/expanded_batch/source_review/episode_000010/table_support.json'
    report=json.loads(report_path.read_text())
    if not report['tilted_plane_quality_pass']:raise ValueError('Table plane fit rejected')
    source=B.parents[1]/'challenge_reconstruction/final_recon/episode_000010'
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    before={n:sha(source/n) for n in ['object_poses.npz','reference.parquet','manifest.json']}
    assert before['object_poses.npz']==report['source_pose_sha256']
    evidence=dict(path=str(report_path),sha256=sha(report_path),source_normal=report['fitted_plane_normal_source'],normal_spread_degrees=report['normal_spread_degrees'],median_plane_residual_m=report['plane_median_absolute_residual_m'],initial_mesh_bottoms_in_setup_m=report['aligned_initial_mesh_bottoms_m'])
    adapter.WORKSPACE=out
    manifest=adapter.prepare(10,np.array(report['proposed_setup_from_source']),evidence)
    (target/'metric_geometry.json').write_text(json.dumps(geometry_from_reference(source),indent=2)+'\n')
    raw=np.load(source/'object_poses.npz')['world_from_object']
    ref=np.load(target/'reference_trajectory.npz');T=np.broadcast_to(np.eye(4),raw.shape).copy()
    T[:,:,:3,:3]=Rotation.from_quat(ref['object_quaternions_wxyz'].reshape(-1,4),scalar_first=True).as_matrix().reshape(raw.shape[:2]+(3,3));T[:,:,:3,3]=ref['object_positions']
    recovered=np.array(manifest['source_from_setup'])@T
    p_error=float(abs(recovered[:,:,:3,3]-raw[:,:,:3,3]).max());full_error=float(abs(recovered-raw).max())
    assert p_error<1e-10 and full_error<1e-6
    assert before=={n:sha(source/n) for n in before}
    result=dict(source_hashes=before,original_object_targets_unchanged=True,position_roundtrip_max_error_m=p_error,pose_roundtrip_max_error=full_error,
        source_float32_rotation_projection_tolerance=1e-6,gravity_alignment='Full rigid scene transform aligns independently measured table normal with setup+Z; objects, hands and cameras transformed together.',
        support_evidence=evidence,original_five_adapters_untouched=True)
    (target/'calibrated_frame_validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(input_manifest=str(target/'input_manifest.json'),validation=result),indent=2))


if __name__=='__main__':main()
