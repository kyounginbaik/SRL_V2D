"""Re-apply the shipped actor's wrapper keys (margin_* PinkyMarginPolicy, release_* ReleasePolicy, component_output_decimals)
onto a freshly distilled LearnedPolicy, optionally with zero margins, and evaluate with the unchanged evaluate_policy_component.
usage: wrap_donor_keys.py EPISODE DISTILLED_DIR OUT_DIR [--zero-margins]"""
import os
for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ[k] = '1'
import sys, json, hashlib
from pathlib import Path
import numpy as np
B = Path('/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017'); sys.path[:0] = [str(B / 'code'), '/data1/SRL_V2D/third_party/DexAgent']
ep, actor, out = int(sys.argv[1]), Path(sys.argv[2]).resolve(), Path(sys.argv[3]).resolve(); zero = '--zero-margins' in sys.argv
DONOR = B / f'refinement_20261018/shared_policy_handoff/episode_{ep:06d}/portable/original/policy.npz'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
ev = json.loads((actor / 'evaluation/result.json').read_text()); scene = ev['execution']['scene']
with np.load(actor / 'training/policy.npz', allow_pickle=False) as z: d = {k: z[k] for k in z.files}
with np.load(DONOR, allow_pickle=False) as z:
    for k in z.files:
        if k.startswith(('margin_', 'release_', 'multi_release')) or k == 'component_output_decimals': d[k] = z[k]
if zero and 'margin_bias_rad' in d: d['margin_bias_rad'] = np.zeros_like(d['margin_bias_rad'])
(out / 'training').mkdir(parents=True, exist_ok=True); ck = out / 'training/policy.npz'; np.savez_compressed(ck, **d)
tr = json.loads((actor / 'training/training_report.json').read_text())
tr.update(wrapper=dict(recipe='shipped wrapper keys re-applied' + (' with zero margins' if zero else ''), donor=str(DONOR), donor_sha256=sha(DONOR), base_actor=str(actor / 'training/policy.npz'),
          base_actor_sha256=sha(actor / 'training/policy.npz'), wrapper_script=str(Path(__file__).resolve()), wrapper_sha256=sha(__file__)), checkpoint_sha256=sha(ck))
(out / 'training/training_report.json').write_text(json.dumps(tr, indent=2, default=str))
from evaluate_policy_component import evaluate
evaluate(ep, scene, str(ck), str(out / 'evaluation'), ev['input_dir'])
r = json.loads((out / 'evaluation/result.json').read_text()); mm = r['metrics']['official_style_reference_proxy']['objects']
print(json.dumps(dict(out=str(out), passed=r['metrics']['reference_proxy_mp_sr'] == 1 and r['execution']['physics_verified'], objects={k: (round(v['mean_position_error_cm'][0], 2), round(v['mean_orientation_error_degrees'][0], 1)) for k, v in mm.items()})), flush=True)
