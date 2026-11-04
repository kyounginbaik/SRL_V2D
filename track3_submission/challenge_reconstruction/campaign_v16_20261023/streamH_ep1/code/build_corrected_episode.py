"""Stream H, episode 1: build (a) a lid-on corrected copy of the source reconstruction (same method/args as
refinement_20261021_ep1/v13/ref_2.4_4.0_12.9_12.3, via the unchanged lid_on_correction.py) and (b) a corrected-track episode dir
(copy of the repaired adapter whose manifest points at the corrected source). New directories only."""
import os, sys, json, shutil, hashlib, subprocess
from pathlib import Path
H = Path('/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamH_ep1')
LEG = Path('/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017')
SRC = Path('/data1/SRL_V2D/challenge_reconstruction/final_recon/episode_000001')
ADAPTER = LEG / 'refinement_20261018/episode_000001/reconstruction_repair/patch_height094/repaired_adapter'
V13 = LEG / 'refinement_20261021_ep1/v13/ref_2.4_4.0_12.9_12.3'
SCRIPT = LEG / 'refinement_20261021_ep1/v13/code/lid_on_correction.py'
ARGS = ['2.4', '4.0', '12.9', '12.3', str(ADAPTER / 'metric_geometry.json')]
OUT_SRC = H / 'staging/source_corrected/episode_000001'; OUT_EP = H / 'staging/episode_corrected/episode_000001'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
if OUT_SRC.exists() or OUT_EP.exists(): raise SystemExit('exists')
subprocess.run([sys.executable, str(SCRIPT), str(SRC), str(OUT_SRC), *ARGS], check=True)
for d in ('assets', 'stereo_calibration', 'audit_after'):
    if (SRC / d).is_dir(): shutil.copytree(SRC / d, OUT_SRC / d)
same = sha(OUT_SRC / 'object_poses.npz') == sha(V13 / 'object_poses.npz')
print('object_poses identical to v13/ref_2.4_4.0_12.9_12.3:', same)
OUT_EP.mkdir(parents=True)
for f in ('camera_trajectory.npz', 'tracking_reference.npz', 'mesh_export_audit.json', 'metric_geometry.json'):
    shutil.copy2(ADAPTER / f, OUT_EP / f)
shutil.copytree(ADAPTER / 'setup', OUT_EP / 'setup')
m = json.loads((ADAPTER / 'input_manifest.json').read_text())
assert m['source'] == str(SRC) and m['source_object_poses_sha256'] == sha(SRC / 'object_poses.npz') and m['source_reference_sha256'] == sha(SRC / 'reference.parquet')
m['source'] = str(OUT_SRC); m['source_reference'] = str(OUT_SRC / 'reference.parquet')
m['source_reference_sha256'] = sha(OUT_SRC / 'reference.parquet'); m['source_object_poses_sha256'] = sha(OUT_SRC / 'object_poses.npz')
rec = json.loads((OUT_SRC / 'lid_on_correction.json').read_text())
m['lid_on_correction'] = dict(record=str(OUT_SRC / 'lid_on_correction.json'), record_sha256=sha(OUT_SRC / 'lid_on_correction.json'), script=str(SCRIPT), script_sha256=sha(SCRIPT), arguments=ARGS,
    original_manifest=str(ADAPTER / 'input_manifest.json'), original_manifest_sha256=sha(ADAPTER / 'input_manifest.json'),
    original_source=str(SRC), original_source_object_poses_sha256=sha(SRC / 'object_poses.npz'), identical_to=str(V13) if same else None,
    frames_changed=rec['frames_changed'], max_pot_rotation_change_deg=rec['max_pot_rotation_change_deg'], frame0_rotation_change_deg=rec['frame0_rotation_change_deg'],
    evidence=[str(LEG / 'refinement_20261021_ep1/stereo_icp_check.json'), str(LEG / 'refinement_20261021_ep1/stereo_icp_check_nolid.json'), str(LEG / 'refinement_20261021_ep1/stereo_height_check.json')],
    note='Pot orientation while the lid sits on it (0-2.4 s, 12.9 s-end) replaced by its nearest lid-off on-table orientation; the lid is carried rigidly; positions of the pot anchor unchanged. reference.parquet is the unchanged original (as in the ep36 corrected track).')
(OUT_EP / 'input_manifest.json').write_text(json.dumps(m, indent=2) + '\n')
print(json.dumps(dict(source=str(OUT_SRC), episode_dir=str(OUT_EP), object_poses_sha256=m['source_object_poses_sha256'], identical_to_v13=same)))
