"""Re-score an executed episode-14 trajectory against OUR v21 reference (world_6prong episode dir = ZIP object poses) with the same
evaluate_mp_sr.evaluate_files the build uses. usage: score_vs_v21ref.py TRAJECTORY_NPZ SCENE_XML"""
import sys, json
from pathlib import Path
sys.path[:0] = ['/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/code', '/data1/SRL_V2D/third_party/DexAgent']
from evaluate_mp_sr import evaluate_files
ep = Path('/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamM_ep14/world_6prong/episodes/episode_000014')
man = json.loads((ep / 'input_manifest.json').read_text())
m = evaluate_files(man['source'], sys.argv[1], ep / 'input_manifest.json', sys.argv[2], ep / 'metric_geometry.json')
print(json.dumps(dict(mp_sr=m['reference_proxy_mp_sr'], objects={k: (round(v['mean_position_error_cm'][0], 3), round(v['mean_orientation_error_degrees'][0], 2)) for k, v in m['official_style_reference_proxy']['objects'].items()})))
