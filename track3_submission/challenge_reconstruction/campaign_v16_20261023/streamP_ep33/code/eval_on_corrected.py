"""Stream P ep33: evaluate an existing actor checkpoint (unchanged weights) with the unchanged evaluate_policy_component in the
same frozen scene against the roll-corrected episode dir; copies training files into roll/<name>/training (distilled layout).
usage: eval_on_corrected.py ACTOR_DIR NAME"""
import os
for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ[k] = '1'
import sys, json, shutil
from pathlib import Path
LEG = Path('/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017'); sys.path[:0] = [str(LEG / 'code'), '/data1/SRL_V2D/third_party/DexAgent']
P = Path('/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamP_ep33'); OEP = P / 'staging/episode_corrected/episode_000033'
from evaluate_policy_component import evaluate
src, name = Path(sys.argv[1]), sys.argv[2]; dest = P / 'roll' / name
if (dest / 'evaluation/result.json').exists(): raise SystemExit('done')
(dest / 'training').mkdir(parents=True, exist_ok=True)
for f in ('policy.npz', 'training_report.json'): shutil.copy2(src / 'training' / f, dest / 'training' / f)
(dest / 'origin.json').write_text(json.dumps(dict(copied_from=str(src)), indent=1))
scene = json.loads((src / 'evaluation/result.json').read_text())['execution']['scene']
r = evaluate(33, scene, str(dest / 'training/policy.npz'), str(dest / 'evaluation'), str(OEP))
o = r['metrics']['official_style_reference_proxy']['objects']
print('DONE', name, json.dumps({k: (round(v['mean_position_error_cm'][0], 3), round(v['mean_orientation_error_degrees'][0], 2), round(v['max_position_error_m'][0] * 100, 2)) for k, v in o.items()}), flush=True)
