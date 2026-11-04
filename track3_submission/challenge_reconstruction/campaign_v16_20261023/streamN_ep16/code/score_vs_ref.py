"""Score an executed trajectory against another world's reference (same world frame, same meshes): e.g. a newtrack rollout vs the OLD
spoon reference of world_6prong. usage: score_vs_ref.py TRAJECTORY_NPZ SCENE_XML REF_WORLD_DIR"""
import sys, json
from pathlib import Path
sys.path[:0]=['/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/code','/data1/SRL_V2D/third_party/DexAgent']
from evaluate_mp_sr import evaluate_files
tr, scene, w = sys.argv[1], sys.argv[2], Path(sys.argv[3]); e = w / 'episodes/episode_000016'; man = json.loads((e / 'input_manifest.json').read_text())
m = evaluate_files(man['source'], tr, e / 'input_manifest.json', scene, e / 'metric_geometry.json')
print(json.dumps(dict(ref_world=str(w), mp_sr=m['reference_proxy_mp_sr'], objects={k: (round(v['mean_position_error_cm'][0], 3), round(v['mean_orientation_error_degrees'][0], 2), round(v['max_position_error_m'][0] * 100, 2)) for k, v in m['official_style_reference_proxy']['objects'].items()})))
