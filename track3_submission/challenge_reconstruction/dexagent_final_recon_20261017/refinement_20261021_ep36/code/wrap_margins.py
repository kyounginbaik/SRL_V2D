"""Wrap a plain LearnedPolicy checkpoint with pinky_margin_policy_v2.PinkyMarginPolicy using the margin_* parameters of the
shipped episode-36 checkpoint (optionally scaled), then evaluate with the unchanged policy evaluator on the corrected episode dir.
usage: wrap_margins.py BASE_POLICY.npz OUT_DIR [SCALE]"""
import os
for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ[k] = '1'
import sys, json
from pathlib import Path
import numpy as np
LEG = Path('/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017'); sys.path[:0] = [str(LEG / 'code'), '/data1/SRL_V2D/third_party/DexAgent']
from evaluate_policy_component import evaluate
base, out = Path(sys.argv[1]), Path(sys.argv[2]); scale = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
donor = np.load(LEG / 'refinement_20261018/shared_policy_handoff/episode_000036/portable/original/policy.npz', allow_pickle=False)
data = {k: v for k, v in np.load(base, allow_pickle=False).items() if not k.startswith('margin')}
for k in donor.files:
    if k.startswith('margin'): data[k] = donor[k].copy()
data['margin_bias_rad'] = data['margin_bias_rad'] * scale
(out / 'training').mkdir(parents=True, exist_ok=True); ck = out / 'training/policy.npz'; np.savez_compressed(ck, **data)
rep = json.loads((base.parent / 'training_report.json').read_text()); rep.update(wrapped_with='pinky_margin_policy_v2.PinkyMarginPolicy', margin_source='shipped episode-36 checkpoint', margin_scale=scale, base_checkpoint=str(base))
(out / 'training/training_report.json').write_text(json.dumps(rep, indent=2))
src = json.loads((base.parent.parent / 'evaluation/result.json').read_text()); scene = src['execution']['scene'] if 'execution' in src else None
teacher = json.loads((base.parent / 'training_report.json').read_text())['scene']
evaluate(36, teacher, ck, out / 'evaluation', LEG / 'refinement_20261021_ep36/episode_corrected/episode_000036')
r = json.loads((out / 'evaluation/result.json').read_text()); o = r['metrics']['official_style_reference_proxy']['objects']['wooden_spoon']
print(json.dumps(dict(out=str(out), scale=scale, pass_=r['metrics']['reference_proxy_mp_sr'] == 1, cm=round(o['mean_position_error_cm'][0], 2), deg=round(o['mean_orientation_error_degrees'][0], 1))), flush=True)
