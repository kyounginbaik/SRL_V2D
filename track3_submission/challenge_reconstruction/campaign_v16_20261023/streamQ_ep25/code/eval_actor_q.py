"""Evaluate an existing plain LearnedPolicy checkpoint with the unchanged evaluate_policy_component in a given scene against a given
episode dir. usage: eval_actor_q.py CHECKPOINT SCENE EPISODE_DIR OUT_DIR"""
import os
for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ[k] = '1'
import sys, json
sys.path[:0] = ['/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/code', '/data1/SRL_V2D/third_party/DexAgent']
from evaluate_policy_component import evaluate
ck, scene, epdir, out = sys.argv[1:5]
evaluate(25, scene, ck, out, epdir)
ev = json.load(open(out + '/result.json')); o = ev['metrics']['official_style_reference_proxy']['objects']
print(json.dumps(dict(mp_sr=ev['metrics']['reference_proxy_mp_sr'], verified=ev['execution']['physics_verified'], replay=ev['execution'].get('independent_control_replay', {}).get('ok'),
                      objects={k: (round(v['mean_position_error_cm'][0], 3), round(v['mean_orientation_error_degrees'][0], 2), round(v['max_position_error_m'][0] * 100, 1)) for k, v in o.items()})))
