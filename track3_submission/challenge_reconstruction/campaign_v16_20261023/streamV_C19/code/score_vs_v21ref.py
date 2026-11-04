"""Score a rollout trajectory against the v21 (our original final_recon) reference with the unchanged legacy evaluator."""
import sys, json
from pathlib import Path
sys.path[:0] = ['/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/code', '/data1/SRL_V2D/third_party/DexAgent']
from evaluate_mp_sr import evaluate_files
E = Path('/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261018/expanded_batch/repaired_export_inputs/episode_000019')
man = json.loads((E / 'input_manifest.json').read_text())
for t in sys.argv[1:]:
    met = evaluate_files(man['source'], t, E / 'input_manifest.json', None, E / 'metric_geometry.json')
    print(t.split('/')[-6], met['reference_proxy_mp_sr'], {k: (round(v['mean_position_error_cm'][0], 3), round(v['mean_orientation_error_degrees'][0], 2)) for k, v in met['official_style_reference_proxy']['objects'].items()})
