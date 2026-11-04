"""G15 variant of tools/reconstruction_audit/camera.py (unchanged algorithm) with clock-consistent stereo lifting:
the stereo depth of frame t is on the ego_cam_b/c clock, i.e. at RGB time t+delta (camsync.py, GT-free). Each SIFT
keypoint is therefore lifted with the depth sampled at its flow-predicted position at t+delta (dense Farneback flow
between consecutive RGB frames) instead of at its RGB-time pixel. Output: OUT_DIR/camera.npz (never the episode folder).
usage: camera_sync.py EPISODE_ROOT OUT_DIR DELTA_KEY"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path

import cv2
import numpy as np


@dataclass
class Keyframe:
    index: int
    descriptors: np.ndarray
    points: np.ndarray
    valid: np.ndarray
    world_from_camera: np.ndarray


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path, nargs='?')
    parser.add_argument('out', type=Path)
    parser.add_argument('key')
    parser.add_argument('--index', type=int)
    args = parser.parse_args()
    root = args.input
    delta = np.asarray(json.loads(Path('/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamG15/out/sync_delta_t.json').read_text())[args.key], float)
    if args.index is not None:
        manifest = Path('/data1/SRL_V2D/challenge_reconstruction/reconstruction_recovery_20261026/canonical_manifest.json')
        root = Path(json.loads(manifest.read_text())[args.index]['output'])
    if root is None:
        parser.error('Supply an input directory or --index')
    args.out.mkdir(parents=True, exist_ok=True); output = args.out / 'camera.npz'
    if output.exists():
        print('Already estimated', output, flush=True)
        return
    prepared = json.loads((root / 'prepared.json').read_text())
    record = prepared['record']
    intrinsics = json.loads((root / 'intrinsics.json').read_text())
    k = np.array([[intrinsics['fx'], 0., intrinsics['cx']],
                  [0., intrinsics['fy'], intrinsics['cy']], [0., 0., 1.]])
    width, height = intrinsics['width'], intrinsics['height']
    cv2.setNumThreads(4)
    cv2.setRNGSeed(260126)
    sift = cv2.SIFT_create(nfeatures=6000)
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    cap = cv2.VideoCapture(str(root / 'rgb.mp4'))
    grays = []
    while True:
        ok_, im_ = cap.read()
        if not ok_: break
        grays.append(cv2.cvtColor(im_, cv2.COLOR_BGR2GRAY))
    cap.release(); cap = cv2.VideoCapture(str(root / 'rgb.mp4'))
    flows = {}
    def flow(t):
        t = min(max(t, 0), len(grays) - 2)
        if t not in flows:
            a = cv2.resize(grays[t], (width // 2, height // 2)); b = cv2.resize(grays[t + 1], (width // 2, height // 2))
            flows[t] = cv2.resize(cv2.calcOpticalFlowFarneback(a, b, None, .5, 4, 21, 3, 5, 1.2, 0), (width, height)) * 2
        return flows[t]
    def advance(pixels, frame, d):
        p = pixels.copy(); t = frame
        while d > 1e-6:
            step = min(1., d); f = flow(t); xy = np.clip(np.rint(p).astype(int), 0, [width - 1, height - 1])
            p = p + step * f[xy[:, 1], xy[:, 0]]; d -= step; t += 1
        return p

    def features(frame):
        success, image = cap.read()
        if not success:
            raise ValueError(f'Missing RGB frame {frame}')
        excluded = np.zeros((height, width), np.uint8)
        for name in record['names']:
            for label in ('1', '2', '3'):
                mask = cv2.imread(str(root / 'objects' / name / 'masks' / label / f'{frame:06d}.png'), 0)
                if mask is not None:
                    excluded |= mask > 0
        excluded = cv2.dilate(excluded, np.ones((31, 31), np.uint8))
        keypoints, descriptors = sift.detectAndCompute(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (1 - excluded) * 255)
        pixels = np.array([value.pt for value in keypoints], float).reshape(-1, 2)
        encoded = cv2.imread(str(root / 'depth' / f'{frame:06d}.png'), -1)
        z = 65535. / np.maximum(encoded.astype(float), 1.) - 1.
        z[(encoded >= 65535) | (encoded == 0)] = np.nan
        xy = np.rint(advance(pixels, frame, float(delta[min(frame, len(delta) - 1)]))).astype(int)
        depths = z[np.clip(xy[:, 1], 0, height - 1), np.clip(xy[:, 0], 0, width - 1)]
        points = np.column_stack([(pixels[:, 0] - k[0, 2]) * depths / k[0, 0],
                                  (pixels[:, 1] - k[1, 2]) * depths / k[1, 1], depths])
        valid = np.isfinite(depths) & (depths > .15) & (depths < 4.)
        return pixels, descriptors, points, valid

    def fit(keyframe, pixels, descriptors):
        if descriptors is None or keyframe.descriptors is None or len(descriptors) < 12:
            return None
        matches = matcher.knnMatch(descriptors, keyframe.descriptors, k=2)
        selected = [pair[0] for pair in matches if len(pair) == 2 and pair[0].distance < .75 * pair[1].distance
                    and keyframe.valid[pair[0].trainIdx]]
        # A landmark can vote only once in an otherwise ambiguous repeated texture.
        unique = {}
        for match in selected:
            if match.trainIdx not in unique or match.distance < unique[match.trainIdx].distance:
                unique[match.trainIdx] = match
        selected = list(unique.values())
        if len(selected) < 16:
            return None
        points = keyframe.points[[value.trainIdx for value in selected]]
        observations = pixels[[value.queryIdx for value in selected]]
        success, rotation, translation, inliers = cv2.solvePnPRansac(points, observations, k, None,
            reprojectionError=2., iterationsCount=2000, confidence=.999, flags=cv2.SOLVEPNP_EPNP)
        if not success or inliers is None or len(inliers) < 16:
            return None
        indices = inliers[:, 0]
        rotation, translation = cv2.solvePnPRefineLM(points[indices], observations[indices], k, None,
                                                  rotation, translation)
        camera_from_keyframe = np.eye(4)
        camera_from_keyframe[:3, :3] = cv2.Rodrigues(rotation)[0]
        camera_from_keyframe[:3, 3] = translation[:, 0]
        projection = cv2.projectPoints(points[indices], rotation, translation, k, None)[0][:, 0]
        error = np.linalg.norm(projection - observations[indices], axis=1)
        return (keyframe.world_from_camera @ np.linalg.inv(camera_from_keyframe), len(indices),
                float(np.median(error)), float(np.percentile(error, 95)))

    frames = record['frames']
    poses = np.tile(np.eye(4), (frames, 1, 1))
    good = np.zeros(frames, bool)
    inliers = np.zeros(frames, int)
    errors = np.full((frames, 2), np.nan)
    pixels, descriptors, points, valid = features(0)
    keyframes = [Keyframe(0, descriptors, points, valid, np.eye(4))]
    good[0] = True
    inliers[0] = int(valid.sum())
    history = []
    try:
        for frame in range(1, frames):
            pixels, descriptors, points, valid = features(frame)
            hypotheses = [(key, fit(key, pixels, descriptors)) for key in keyframes]
            hypotheses = [(key, value) for key, value in hypotheses if value is not None]
            if not hypotheses:
                poses[frame] = poses[frame - 1]
                history.append(dict(frame=frame, tracked=False, camera_held=True))
                continue
            key, chosen = max(hypotheses, key=lambda pair: pair[1][1])
            poses[frame], inliers[frame], errors[frame, 0], errors[frame, 1] = chosen
            good[frame] = chosen[1] >= 40
            if frame - keyframes[-1].index >= 15 and (chosen[1] < 200 or frame - keyframes[-1].index >= 60):
                keyframes.append(Keyframe(frame, descriptors, points, valid, poses[frame].copy()))
                if len(keyframes) > 3:
                    keyframes.pop(1)  # Retain frame zero plus the two most recent keyframes.
            row = dict(frame=frame, tracked=bool(good[frame]), inliers=int(chosen[1]),
                       reprojection_median_px=chosen[2], reprojection_p95_px=chosen[3], reference_frame=key.index)
            history.append(row)
            if frame % 50 == 0:
                print(json.dumps(row), flush=True)
    finally:
        cap.release()
    np.savez_compressed(output, world_from_camera=poses, cam0_from_camera=poses, valid=good,
                        inliers=inliers, reprojection_error_px=errors, fps=np.array(20.))
    report = dict(episode=record['episode'], split=record['split'], frames=frames,
                  source='static-background SIFT + metric stereo + PnP-RANSAC/refinement',
                  ground_truth_used=False, g15_delta_key=args.key, valid_fraction=float(good.mean()),
                  low_inlier_frames=np.flatnonzero(inliers < 40).tolist(),
                  median_inliers=float(np.median(inliers[1:])), history=history,
                  limitations='Missing estimates are flagged and held; camera motion is not certified by algebraic roundtrips.')
    (args.out / 'camera_report.json').write_text(json.dumps(report, indent=2) + '\n')
    print('CAMERA_COMPLETE', report['valid_fraction'], report['median_inliers'], flush=True)


if __name__ == '__main__':
    main()
