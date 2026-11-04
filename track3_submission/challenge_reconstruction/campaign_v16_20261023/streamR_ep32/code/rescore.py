"""Score an executed trajectory against the original and the corrected references (evaluate_mp_sr.evaluate_files, unchanged).
usage: rescore.py TRAJ_NPZ SCENE_XML EPDIR[:LABEL] ..."""
import sys, json
sys.path[:0] = ['/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/code', '/data1/SRL_V2D/third_party/DexAgent']
from pathlib import Path
from evaluate_mp_sr import evaluate_files
traj, scene = sys.argv[1], sys.argv[2]; out = {}
for arg in sys.argv[3:]:
    ed, _, lab = arg.partition(':'); ed = Path(ed); man = json.loads((ed / 'input_manifest.json').read_text())
    mt = evaluate_files(man['source'], traj, ed / 'input_manifest.json', scene, ed / 'metric_geometry.json'); o = mt['official_style_reference_proxy']
    out[lab or str(ed)] = {k: (round(v['mean_position_error_cm'][0], 3), round(v['mean_orientation_error_degrees'][0], 2), round(v['max_position_error_m'][0] * 100, 2)) for k, v in o['objects'].items()} | dict(mp_sr=o['mp_sr'])
print(json.dumps(out))
