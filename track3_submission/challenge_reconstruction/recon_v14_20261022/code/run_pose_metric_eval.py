"""Stereo FoundationPose rerun for an EVAL object with the metric submission mesh (instead of provided_mesh_scaled.obj).
usage: run_pose_metric_eval.py <stageA_object_dir> <depth_dir> <mesh> <out_dir> <frames>"""
from pathlib import Path
import json, sys
import v2d.docker.container as container
old, depth, mesh, out, n = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]), int(sys.argv[5])
repair = Path('/data1/SRL_V2D/challenge_reconstruction/public_gt_check_20261020/code')
original = container.run_in_container
def wrapped(**kw):
    kw['extra_volumes'] = (kw.get('extra_volumes') or []) + ['/data1:/data1:ro', str(repair / 'run_video_to_poses_visible.py') + ':/workspace/v2d_foundation_pose/lib/run_video_to_poses.py:ro']; return original(**kw)
container.run_in_container = wrapped
from v2d.foundation_pose.docker.run_video_to_poses import run_video_to_poses
assert len(list(depth.glob('*.png'))) == n, (len(list(depth.glob('*.png'))), n)
out.mkdir(parents=True, exist_ok=True)
run_video_to_poses(video_path=str((old / 'anycalib/undistorted.mp4').resolve()), depth_folder=str(depth.resolve()), masks_folder=str(old / 'masks/1'), camera_intrinsics_path=str(old / 'intrinsics_stable.json'), mesh_path=str(mesh), poses_dir=str(out), weights_dir='/data1/SRL_V2D/reconstruction/data/weights/foundation_pose', reference_frame=0, target_width=1014, target_height=760, reregister_iou_thresh=.3, mask_depth=True, dev=True)
assert len(list(out.glob('*.json'))) == n
(out.parent / (out.name + '_complete.json')).write_text(json.dumps(dict(stageA=str(old), depth=str(depth), mesh=str(mesh), frames=n, note='metric submission mesh; otherwise identical to the eval stereo FoundationPose run'), indent=2))
