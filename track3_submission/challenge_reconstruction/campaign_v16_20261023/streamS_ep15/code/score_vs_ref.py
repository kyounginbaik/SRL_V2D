"""Score an executed trajectory (result dir with result.json/execution or an evaluation dir) against ANOTHER world's reference
(e.g. a world_6prong_nt rollout vs the OLD-track reference of world_6prong). Same evaluator (evaluate_mp_sr.evaluate_files).
usage: score_vs_ref.py RESULT_JSON REF_WORLD"""
import json, sys
from pathlib import Path
sys.path[:0] = ['/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/code', '/data1/SRL_V2D/third_party/DexAgent']
from evaluate_mp_sr import evaluate_files
r = json.loads(Path(sys.argv[1]).read_text()); ep = Path(sys.argv[2]) / 'episodes/episode_000015'; man = json.loads((ep / 'input_manifest.json').read_text())
met = evaluate_files(man['source'], r['execution']['trajectory'], ep / 'input_manifest.json', r['execution']['scene'], ep / 'metric_geometry.json')
o = met['official_style_reference_proxy']['objects']
print(json.dumps(dict(result=sys.argv[1], ref_world=sys.argv[2], mp_sr=met['reference_proxy_mp_sr'], objects={k: (round(v['mean_position_error_cm'][0], 3), round(v['mean_orientation_error_degrees'][0], 2), round(v['max_position_error_m'][0] * 100, 2)) for k, v in o.items()})))
