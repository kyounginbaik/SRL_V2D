"""CPU-render aligned source/reference/baseline/refined comparison videos (titles configurable; 2026-10-20 copy).

Uses calibrated moving camera and original timestamps. Kinematic reference is
explicitly labelled; physics panes only show measured qpos from recordings.
"""
from __future__ import annotations
import os
import sys
from pathlib import Path
_LIB = Path(__file__).resolve().parents[1] / ".software_gl/usr/lib64"
if __name__ == "__main__" and _LIB.is_dir() and str(_LIB) not in os.environ.get("LD_LIBRARY_PATH", "").split(":"):
    env = dict(os.environ, LD_LIBRARY_PATH=str(_LIB) + ":" + os.environ.get("LD_LIBRARY_PATH", ""))
    os.execve(sys.executable, [sys.executable, *sys.argv], env)
os.environ["MUJOCO_GL"] = "osmesa"
os.environ["PYOPENGL_PLATFORM"] = "osmesa"
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("LP_NUM_THREADS", "2")
import argparse
import json
import subprocess
import shutil
import cv2
import mujoco
import numpy as np


def _load_report(value):
    if isinstance(value, dict):
        return value
    return json.loads(Path(value).read_text())


def _render_camera(renderer, model, data, camera, frame):
    cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "video_cam")
    if cid < 0:
        cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "ego")
    if cid < 0:
        raise ValueError("scene has no calibrated video_cam or ego camera")
    model.cam_pos[cid] = camera["positions"][frame]
    model.cam_quat[cid] = camera["quaternions_wxyz"][frame]
    # Rendering measured states needs poses/lights, not collision solving.
    mujoco.mj_kinematics(model, data)
    mujoco.mj_camlight(model, data)
    option = mujoco.MjvOption()
    option.geomgroup[:] = [1, 1, 1, 0, 0, 0]
    renderer.update_scene(data, camera=cid, scene_option=option)
    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
    return cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)


def _label(frame, title, timestamp, subtitle=""):
    w = frame.shape[1]
    bar = np.full((62, w, 3), 24, np.uint8)
    cv2.putText(bar, title, (9, 21), cv2.FONT_HERSHEY_SIMPLEX, .49, (245,245,245), 1, cv2.LINE_AA)
    cv2.putText(bar, f"{timestamp:.2f}s  {subtitle}", (9, 45), cv2.FONT_HERSHEY_SIMPLEX, .4, (180,210,235), 1, cv2.LINE_AA)
    return np.vstack([bar, frame])


def render_comparison(episode_dir, baseline_report, refined_report, out_dir,
                      width=384, height=288, stride=2, include_source=True,
                      reference_scene=None, titles=("Reference (kinematic)", "Baseline (physics)", "Refined (physics)"), video_name="reference_baseline_refined.mp4"):
    episode = Path(episode_dir).resolve(); out = Path(out_dir).resolve(); out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((episode / "input_manifest.json").read_text())
    base, refined = _load_report(baseline_report), _load_report(refined_report)
    camera = dict(np.load(episode / "camera_trajectory.npz", allow_pickle=False))
    ref = dict(np.load(episode / "tracking_reference.npz", allow_pickle=False))
    tracks = [ref, dict(np.load(base["trajectory"], allow_pickle=False)), dict(np.load(refined["trajectory"], allow_pickle=False))]
    # Reference joint coordinates belong to the adapter's robot frame. A
    # baseline can use an older wrist frame, so callers can supply that scene
    # explicitly instead of rendering recharted angles in the old frame.
    scenes = [reference_scene or base["scene"], base["scene"], refined["scene"]]
    fps = float(ref["fps"]); frames = len(ref["qpos"])
    if any(len(t["qpos"]) != frames for t in tracks):
        raise ValueError("full reference and physics timelines must have the same frame count")
    times = np.arange(frames) / fps
    for t in tracks[1:]:
        if not np.allclose(t["time_s"], times, atol=.0021, rtol=0):
            raise ValueError("physics timing does not match reference; no video time warping is allowed")
    models = [mujoco.MjModel.from_xml_path(str(s)) for s in scenes]
    data = [mujoco.MjData(m) for m in models]
    renderers = [mujoco.Renderer(m, height=height, width=width) for m in models]
    cap = cv2.VideoCapture(manifest["source_video"]) if include_source else None
    if cap is not None and not cap.isOpened():
        raise ValueError("original source video cannot be opened")
    panels = 4 if cap is not None else 3
    target = out / video_name
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        import imageio_ffmpeg
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
               "-s", f"{width*panels}x{height+62}", "-r", str(fps/stride), "-i", "-", "-an",
               "-c:v", "libx264", "-threads", "2", "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p", str(target)]
    proc = subprocess.Popen(command, stdin=subprocess.PIPE)
    snapshots = sorted(set([0, frames//3, 2*frames//3, frames-1]))
    frame_indices = list(range(0, frames, stride))
    if frame_indices[-1] != frames - 1:
        frame_indices.append(frames - 1)
    rendered = 0
    source_next = 0
    try:
        for frame in frame_indices:
            row = []
            if cap is not None:
                # Decode sequentially; seeking for every sampled frame repeatedly
                # decodes the source's full GOP and is unnecessarily expensive.
                while source_next <= frame:
                    ok, rgb = cap.read(); source_next += 1
                    if not ok:
                        raise ValueError(f"source video missing frame {frame}")
                rgb = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)
                row.append(_label(rgb, "Original human video", times[frame]))
            for k, title in enumerate(titles):
                data[k].qpos[:] = tracks[k]["qpos"][frame]
                img = _render_camera(renderers[k], models[k], data[k], camera, frame)
                row.append(_label(img, title, times[frame], "not physical execution" if k == 0 else "recorded simulation"))
            picture = np.hstack(row)
            proc.stdin.write(picture.tobytes()); rendered += 1
            if any(abs(frame - t) < stride for t in snapshots):
                cv2.imwrite(str(out / f"comparison_{frame:05d}.jpg"), picture, [cv2.IMWRITE_JPEG_QUALITY, 93])
    finally:
        proc.stdin.close(); status = proc.wait()
        for renderer in renderers:
            renderer.close()
        if cap is not None:
            cap.release()
    if status:
        raise RuntimeError(f"ffmpeg exited {status}")
    report = {"video": str(target), "frames_rendered": rendered, "fps": fps/stride,
              "source_frames": frames, "duration_s": (frames-1)/fps,
              "moving_camera": True, "time_warping": False,
              "panels": (["original human video"] if include_source else []) + ["kinematic reference", "baseline physics", "refined physics"],
              "baseline_scene": base["scene"], "refined_scene": refined["scene"],
              "reference_scene": scenes[0],
              "baseline_trajectory": base["trajectory"], "refined_trajectory": refined["trajectory"],
              "screenshots": [str(p) for p in sorted(out.glob("comparison_*.jpg"))]}
    (out / "render_report.json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--args", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(render_comparison(**json.loads(args.args.read_text())), indent=2))
