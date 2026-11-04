"""Episode 33: re-time the LEFT-hand release primitive of the shipped v13 actor (MultiReleasePolicy over PinkyMarginPolicy over
LearnedPolicy; weights/margins untouched) and evaluate the new checkpoint with the unchanged evaluate_policy_component in the
same frozen scene. Only release0_* (left hand) scalar/array parameters are changed, as given on the command line.
usage: rewrap33.py BASE_ACTOR_DIR OUT_DIR key=value [key=value ...]   (value: float, or 'x1.5' to scale an array)"""
import os, sys, json, hashlib, shutil
for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ[k] = '1'
from pathlib import Path
import numpy as np
B = Path('/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017'); sys.path[:0] = [str(B / 'code'), '/data1/SRL_V2D/third_party/DexAgent']
from evaluate_policy_component import evaluate
base, out = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve(); sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
ev = json.loads((base / 'evaluation/result.json').read_text()); scene = ev['execution']['scene']; epd = ev['input_dir']
with np.load(base / 'training/policy.npz', allow_pickle=False) as z: d = {k: z[k] for k in z.files}
changes = {}
for kv in sys.argv[3:]:
    k, v = kv.split('=', 1); assert k.startswith('release0_') or k.startswith('release1_'), k
    if v.startswith('x'): d[k] = d[k] * float(v[1:])
    else: d[k] = np.array(float(v))
    changes[k] = d[k].tolist()
(out / 'training').mkdir(parents=True, exist_ok=True); ck = out / 'training/policy.npz'; np.savez_compressed(ck, **d)
tr = json.loads((base / 'training/training_report.json').read_text())
tr.update(release_retiming=dict(base_actor=str(base / 'training/policy.npz'), base_sha256=sha(base / 'training/policy.npz'), changes=changes, script=str(Path(__file__).resolve()), script_sha256=sha(__file__)), checkpoint_sha256=sha(ck))
(out / 'training/training_report.json').write_text(json.dumps(tr, indent=2, default=str))
evaluate(33, scene, str(ck), str(out / 'evaluation'), epd)
r = json.loads((out / 'evaluation/result.json').read_text()); m = r['metrics']['official_style_reference_proxy']
print('DONE', json.dumps(dict(changes=changes, mp_sr=m['mp_sr'], objects={k: (round(v['mean_position_error_cm'][0], 2), round(v['mean_orientation_error_degrees'][0], 1), round(v['max_position_error_m'][0] * 100, 1)) for k, v in m['objects'].items()})), flush=True)
