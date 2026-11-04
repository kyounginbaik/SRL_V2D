"""Stream R ep32: execute a pickup spec with zero search offsets (or given parameters from a progress/result json) over the full horizon,
without CEM: physics_pickup_search_v2 commands -> floating_runtime.execute_reference -> metrics vs the spec's (corrected) source and vs the original.
Prints the right-hand grasp survival per window. usage: zero_eval32.py SPEC_JSON OUT_DIR [PARAMS_JSON]"""
import sys, json
from pathlib import Path
import numpy as np
LEG = Path('/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017'); sys.path[:0] = ['/data1/SRL_V2D/third_party/DexAgent', str(LEG / 'code')]
import physics_pickup_search_v2 as P
from tool_library.common import tracking as TR
from floating_runtime import execute_reference, _freeze
from evaluate_mp_sr import evaluate_files
spec = json.loads(Path(sys.argv[1]).read_text()); out = Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
(out / 'search_physics').mkdir(exist_ok=True); _, frozen = _freeze(spec['scene'], out / 'search_physics'); spec['scene'] = str(frozen.resolve())
exp = P.PickupExperiment(spec); x = np.zeros(exp.dim)
if len(sys.argv) > 3: j = json.loads(Path(sys.argv[3]).read_text()); x = np.asarray(j.get('best_parameters', j.get('parameters')), float)
Q, timing = exp.commands(x)
refp = TR.save_reference(out / 'reference.npz', exp.m, Q, exp.ref['labels'], exp.fps, exp.ref['segment'], exp.ref['segments'], source='pickup spec commands (zero or given offsets), full horizon', arms_coupled=False, floating_runtime=True)
ex = execute_reference(spec['scene'], refp, out / 'execution'); res = dict(spec=spec, parameters=x, timing=timing, reference=str(refp), execution=ex)
ORIG = LEG / 'refinement_20261018/expanded_batch/repaired_export_inputs/episode_000032'
for tag, man in (('corrected', Path(spec['manifest'])), ('original', ORIG / 'input_manifest.json')):
    mm = json.loads(man.read_text()); mt = evaluate_files(mm['source'], ex['trajectory'], man, ex['scene'], man.parent / 'metric_geometry.json')
    res['metrics' if tag == 'corrected' else 'metrics_original'] = mt; o = mt['official_style_reference_proxy']
    print(tag, o['mp_sr'], {k: (round(v['mean_position_error_cm'][0], 2), round(v['mean_orientation_error_degrees'][0], 1)) for k, v in o['objects'].items()}, flush=True)
tr = np.load(ex['trajectory']); F = tr['hand_object_force_n']; on = [str(s) for s in tr['object_names']]; si = on.index('wooden_spoon')
ref = TR.load_reference(spec['original_reference'], exp.m)['qpos']; oc = exp.oc
err = np.linalg.norm(tr['object_positions'][:, si] - ref[:len(tr['object_positions']), oc:oc + 3], axis=1) * 100
print('t: ' + ' '.join(f'{s/20:.1f}:{err[s:s+10].mean():.1f}cm/F{F[s:s+10,1].mean():.1f}' for s in range(40, 230, 10)), flush=True)
(out / 'result.json').write_text(json.dumps(res, indent=1, default=lambda v: v.tolist() if isinstance(v, np.ndarray) else str(v)) + '\n')
