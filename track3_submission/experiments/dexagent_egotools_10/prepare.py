"""Verify downloaded media, cut task clips, and configure native DexAgent runs."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys

import cv2
import imageio_ffmpeg
import yaml

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1] / 'third_party/DexAgent'
tasks = json.loads((ROOT / 'tasks.json').read_text())
manifest = json.loads((ROOT / 'data/manifest.json').read_text())
record_path=ROOT/'media_validation.json'
records = json.loads(record_path.read_text()) if record_path.exists() else []
for task in tasks:
    if any(r['id']==task['id'] for r in records):
        continue
    episode = next(e for e in manifest['episodes'] if e['uuid'] == task['uuid'])
    entry = next(f for f in episode['files'] if f['path'].endswith('/stereo_left_synced.mp4'))
    source = ROOT / 'data/raw' / entry['path']
    if source.stat().st_size != entry['bytes']:
        print(f'Skipping incomplete download: {task["id"]}', flush=True)
        continue
    cap = cv2.VideoCapture(str(source))
    assert cap.isOpened(), str(source)
    fps, width, height = cap.get(5), int(cap.get(3)), int(cap.get(4))
    frames = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames += 1
    advertised = int(cap.get(7))
    cap.release()
    assert frames > 0 and abs(frames-advertised) <= 1, (source, frames, advertised)
    dest = ROOT / 'clips' / (task['id'] + '.mp4')
    dest.parent.mkdir(exist_ok=True)
    check=cv2.VideoCapture(str(dest)) if dest.exists() else None
    valid=bool(check is not None and check.isOpened() and check.read()[0])
    if check is not None:check.release()
    if not valid:
        temporary=dest.with_name(dest.stem+'.partial.mp4')
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-hide_banner', '-loglevel', 'error',
                        '-ss', str(task['start_s']), '-i', str(source), '-t', str(task['end_s']-task['start_s']),
                        '-an', '-c:v', 'libx264', '-threads', '2', '-crf', '18', '-preset', 'fast',
                        '-movflags', '+faststart', '-y', str(temporary)], check=True)
        check=cv2.VideoCapture(str(temporary))
        assert check.isOpened() and check.read()[0], str(temporary)
        check.release()
        temporary.replace(dest)
    calib = json.loads((source.parent / 'calibration.json').read_text())
    k = calib['cam01']['K']
    # This is a calibrated rectified camera on a MOVING head rig. Its static pose
    # in the default robot bench is unknown: never substitute the shipped ego pose.
    cfg = {'agent': {'provider': 'external', 'model': 'codex-current-session',
                     'transport_dir': str(ROOT / 'transport' / task['id']), 'response_timeout_s': 3600},
           'run': {'name': task['id'], 'output_dir': str(ROOT / 'runs')},
           'examples': {'reuse': False, 'save_assets': False},
           'inputs': {'video': str(dest), 'task': task['task'], 'setup': str(REPO / 'setup'),
                      'library': str(REPO / 'tool_library')},
           'camera': {'source': 'estimate', 'width': width, 'height': height,
                      'intrinsics': dict(zip(('fx','fy','cx','cy'), k))},
           'stage1': {'examples': [], 'max_rounds': 12, 'max_wall_minutes': 60},
           'stage2': {'max_rounds': 12, 'max_wall_minutes': 30},
           'stage3': {'max_rounds': 12, 'max_wall_minutes': 30, 'methods': []},
           'compute': {'workers': 2}}
    config_path = ROOT / 'configs' / (task['id']+'.yaml')
    config_path.parent.mkdir(exist_ok=True)
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    rec = dict(task, source_file=str(source.relative_to(ROOT)), source_bytes=entry['bytes'],
               source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
               source_decoded_frames=frames, source_advertised_frames=advertised,
               source_manifest_frames=episode['frame_count'],
               source_fps=fps, width=width, height=height,
               clip=str(dest.relative_to(ROOT)), clip_sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),
               camera_intrinsics=cfg['camera']['intrinsics'], config=str(config_path.relative_to(ROOT)))
    records.append(rec)
    record_path.write_text(json.dumps(sorted(records,key=lambda r:r['id']),indent=2))
    print(json.dumps({'task':task['id'], 'decoded_frames':frames,'source_fps':fps}), flush=True)
(ROOT / 'media_validation.json').write_text(json.dumps(records, indent=2))
(ROOT / 'runtime.json').write_text(json.dumps({'python':sys.executable, 'python_version':sys.version,
    'dexagent_commit':subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip(),
    'data_revision':json.loads((ROOT/'data/hub_info.json').read_text())['sha'],
    'packages':{d.metadata['Name']:d.version for d in __import__('importlib.metadata',fromlist=['distributions']).distributions()}},indent=2))
