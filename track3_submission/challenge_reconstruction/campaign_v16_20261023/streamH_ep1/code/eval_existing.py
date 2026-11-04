"""Stream H ep1: evaluate an existing registered actor checkpoint (unchanged weights) with the unchanged policy evaluator against
the corrected-track episode dir. Copies the training files into distilled/<name>/training so the dir has the distilled layout.
usage: eval_existing.py NAME TRAINING_DIR"""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'): os.environ[k]='1'
import sys, json, shutil
from pathlib import Path
LEG=Path('/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017'); sys.path[:0]=[str(LEG/'code'),'/data1/SRL_V2D/third_party/DexAgent']
H=Path('/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamH_ep1'); EPDIR=H/'staging/episode_corrected/episode_000001'
from evaluate_policy_component import evaluate
name, tdir = sys.argv[1], Path(sys.argv[2]); dest=H/'distilled'/name
if (dest/'evaluation/result.json').exists(): raise SystemExit('done '+str(dest))
(dest/'training').mkdir(parents=True, exist_ok=True)
for f in ('policy.npz','training_report.json'):
    if not (dest/'training'/f).exists(): shutil.copy2(tdir/f, dest/'training'/f)
rep=json.loads((dest/'training/training_report.json').read_text())
(dest/'origin.json').write_text(json.dumps(dict(copied_from=str(tdir), checkpoint_sha256=rep['checkpoint_sha256']), indent=1))
r=evaluate(1, rep['scene'], str(dest/'training/policy.npz'), str(dest/'evaluation'), str(EPDIR))
o=r['metrics']['official_style_reference_proxy']['objects']
print(json.dumps(dict(name=name, mp_sr=r['metrics']['reference_proxy_mp_sr'], objects={k:(round(v['mean_position_error_cm'][0],2),round(v['mean_orientation_error_degrees'][0],1)) for k,v in o.items()})), flush=True)
