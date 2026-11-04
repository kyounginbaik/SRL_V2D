"""Re-execute a teacher reference (open-loop servo) in the compose-home scene and evaluate against the compose-corrected episode
dir; writes a fresh teacher dir (result.json + reference.npz + execution/) usable by ilc_carry_generic.py. ILC tables dropped.
usage: make_source.py TEACHER_DIR OUT_DIR [EPISODE_DIR] [SCENE]"""
import json, shutil, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, '/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamV_C19/code')
from common19 import *
ROOT = Path('/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017'); sys.path[:0] = [str(ROOT / 'code'), '/data1/SRL_V2D/third_party/DexAgent']
from floating_runtime import execute_reference
from evaluate_mp_sr import evaluate_files
src, out = Path(sys.argv[1]), Path(sys.argv[2])
epd = Path(sys.argv[3]) if len(sys.argv) > 3 else V / 'staging_C/episode_compose/episode_000019'
scene = Path(sys.argv[4]) if len(sys.argv) > 4 else V / 'staging_C/scene/scene_physics_compose_home.xml'
out.mkdir(parents=True, exist_ok=False); prev = json.loads((src / 'result.json').read_text())
shutil.copy2(src / 'reference.npz', out / 'reference.npz')
ex = execute_reference(str(scene), out / 'reference.npz', out / 'execution'); man = json.loads((epd / 'input_manifest.json').read_text())
met = evaluate_files(man['source'], ex['trajectory'], epd / 'input_manifest.json', ex['scene'], epd / 'metric_geometry.json')
res = {k: v for k, v in prev.items() if k != 'ilc'}; res.update(reference=str(out / 'reference.npz'), execution=ex, metrics=met, input_dir=str(epd), source=str(src), note='teacher reference re-executed in the compose-home scene (stream V)')
(out / 'result.json').write_text(json.dumps(res, indent=1, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else str(x)) + '\n')
o = met['official_style_reference_proxy']['objects']; cs, _ = compose_score_traj(ex['trajectory'])
print(json.dumps(dict(out=str(out), mp_sr=met['reference_proxy_mp_sr'], vs_ref={k: (round(v['mean_position_error_cm'][0], 2), round(v['mean_orientation_error_degrees'][0], 1)) for k, v in o.items()}, vs_compose=cs, completed=ex['execution_completed'])))
