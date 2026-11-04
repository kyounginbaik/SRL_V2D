"""Finish a time-limited refine_versioned_retention_full.py run from its progress.json (best_parameters): rebuild the reference
exactly as the tool's tail does (exp.commands(best), same asserts), execute, evaluate, write result.json (same schema + finished_from_progress).
usage: ret_finish.py RET_OUT_DIR"""
import os
for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ[k] = '1'
import sys, json
from pathlib import Path
import numpy as np
sys.path[:0] = ['/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/code', '/data1/SRL_V2D/third_party/DexAgent']
from physics_retention_search import RetentionExperiment
from refine_grasp_cem import TR, dump, execute_reference, evaluate_files
out = Path(sys.argv[1]).resolve(); st = json.loads((out / 'settings.json').read_text()); spec = st['spec']; inp = Path(st['episode_dir'])
pr = json.loads((out / 'progress.json').read_text()); best = np.asarray(pr['best_parameters'], float); hist = pr['history']
if (out / 'result.json').exists(): raise SystemExit('exists')
exp = RetentionExperiment(spec); side = spec['side']; manifest = json.loads((inp / 'input_manifest.json').read_text())
Q = exp.commands(best); unchanged = np.setdiff1d(np.arange(exp.model.nq), exp.cols[side])
assert np.array_equal(Q[:, unchanged], exp.base[:, unchanged]); assert np.array_equal(Q[:int(spec['start_s'] * exp.fps) + 1], exp.base[:int(spec['start_s'] * exp.fps) + 1])
path = TR.save_reference(out / 'reference.npz', exp.model, Q, exp.ref['labels'], exp.fps, exp.ref['segment'], exp.ref['segments'],
    source='Physical retention refinement; preserved pickup prefix and immutable object targets', floating_runtime=True)
execution = execute_reference(spec['scene'], path, out / 'execution')
metrics = evaluate_files(manifest['source'], execution['trajectory'], inp / 'input_manifest.json', execution['scene'], inp / 'metric_geometry.json')
initial = json.loads((out / 'initial.json').read_text())
dump(out / 'result.json', dict(episode=manifest['episode'], input_dir=str(inp), initial=initial, best=hist[-1]['best'], parameters=best, history=hist, reference=str(path),
    execution=execution, metrics=metrics, learned_policy=False, finished_from_progress=dict(iterations_done=len(hist), note='Slurm time limit; finished by streamA ret_finish.py')))
fmt = {k: (round(v['mean_position_error_cm'][0], 2), round(v['mean_orientation_error_degrees'][0], 1)) for k, v in metrics['official_style_reference_proxy']['objects'].items()}
print(json.dumps(dict(out=str(out), mp_sr=metrics['reference_proxy_mp_sr'], objects={k: list(v) + [0] for k, v in fmt.items()}, search_best=hist[-1]['best']['mean_position_cm'])))
