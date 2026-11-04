"""Stream N: evaluate an existing learned actor checkpoint closed-loop in a Stream-N world (unchanged evaluate_policy_component).
usage: eval_actor.py CHECKPOINT_NPZ WORLD_DIR OUT_DIR"""
import os
for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ[k] = '1'
import sys, json
from pathlib import Path
LEG = Path('/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017')
sys.path[:0] = [str(LEG / 'code'), '/data1/SRL_V2D/third_party/DexAgent']
from evaluate_policy_component import evaluate
ck, world, out = (Path(x).resolve() for x in sys.argv[1:4]); epdir = world / 'episodes/episode_000016'; scene = world / 'scene/controller_scene.xml'
evaluate(16, str(scene), str(ck), str(out), str(epdir))
ev = json.loads((out / 'result.json').read_text()); obs = ev['metrics']['official_style_reference_proxy']['objects']
print(json.dumps(dict(out=str(out), checkpoint=str(ck), mp_sr=ev['metrics']['reference_proxy_mp_sr'], verified=ev['execution']['physics_verified'],
      objects={k: (round(v['mean_position_error_cm'][0], 2), round(v['mean_orientation_error_degrees'][0], 1), round(v['max_position_error_m'][0] * 100, 1)) for k, v in obs.items()})), flush=True)
