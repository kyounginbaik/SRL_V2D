"""J8 ep19: superpose phase-weight residuals of actors fine-tuned from a common parent (W = W_base + sum_i (W_i - W_parent_i)).
Writes OUT/training/policy.npz + training_report.json and copies BASE/evaluation/result.json (scene / input dir only) so actor_res19.py
'save' can run the unchanged official evaluation.  usage: merge_actor.py OUT BASE_ACTOR PARENT:CHILD [PARENT:CHILD ...]"""
import sys, json, shutil, hashlib, numpy as np
from pathlib import Path
out, base = Path(sys.argv[1]), Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=False); (out / 'training').mkdir(); (out / 'evaluation').mkdir()
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
with np.load(base / 'training/policy.npz', allow_pickle=False) as z: d = {k: z[k] for k in z.files}
parts = []
for pc in sys.argv[3:]:
    p, c = map(Path, pc.split(':'))
    with np.load(p / 'training/policy.npz') as zp, np.load(c / 'training/policy.npz') as zc:
        for k in zp.files:
            if k != 'phase_weights': assert np.array_equal(zp[k], zc[k]) and np.array_equal(zp[k], d[k]), k
        d['phase_weights'] = d['phase_weights'] + (zc['phase_weights'] - zp['phase_weights'])
    parts.append(dict(parent=str(p), child=str(c), parent_sha256=sha(p / 'training/policy.npz'), child_sha256=sha(c / 'training/policy.npz')))
np.savez_compressed(out / 'training/policy.npz', **d)
tr = json.loads((base / 'training/training_report.json').read_text())
tr.update(phase_residual_merge=dict(base=str(base), base_sha256=sha(base / 'training/policy.npz'), added=parts, script=str(Path(__file__).resolve()), script_sha256=sha(__file__),
          note='phase-weight residuals found by actor_res19.py CEMs on disjoint windows / sides, superposed; feedback weights and everything else identical'), checkpoint_sha256=sha(out / 'training/policy.npz'))
(out / 'training/training_report.json').write_text(json.dumps(tr, indent=2, default=str))
shutil.copy2(base / 'evaluation/result.json', out / 'evaluation/result.json'); print(out)
