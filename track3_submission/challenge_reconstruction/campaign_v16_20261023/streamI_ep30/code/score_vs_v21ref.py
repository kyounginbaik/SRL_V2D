"""Re-score a verified rollout (common_v2_candidates/TAG/.../trajectory) against another world's reference (e.g. the v21 world_6prong_cut).
usage: score_vs_v21ref.py TRAJECTORY_NPZ SCENE_XML EPISODE_DIR"""
import sys, json
sys.path[:0] = ['/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/code', '/data1/SRL_V2D/third_party/DexAgent']
from pathlib import Path
from evaluate_mp_sr import evaluate_files
tr, sc, ep = sys.argv[1:4]; ep = Path(ep); man = json.loads((ep / 'input_manifest.json').read_text())
m = evaluate_files(man['source'], tr, ep / 'input_manifest.json', sc, ep / 'metric_geometry.json')
print(json.dumps({k: (round(v['mean_position_error_cm'][0], 3), round(v['mean_orientation_error_degrees'][0], 2)) for k, v in m['official_style_reference_proxy']['objects'].items()}))
