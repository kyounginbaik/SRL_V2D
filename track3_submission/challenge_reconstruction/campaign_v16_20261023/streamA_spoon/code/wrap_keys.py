"""Re-apply selected wrapper keys of an episode's shipped (donor) actor onto a freshly distilled LearnedPolicy and evaluate.
usage: wrap_keys.py EPISODE DISTILLED_DIR OUT_DIR --keys margin|release|all [--donor POLICY_NPZ]"""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'): os.environ[k]='1'
import sys, json, hashlib, argparse
from pathlib import Path
import numpy as np
B=Path('/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017'); sys.path[:0]=[str(B/'code'),'/data1/SRL_V2D/third_party/DexAgent']
p=argparse.ArgumentParser(); p.add_argument('episode',type=int); p.add_argument('distilled'); p.add_argument('out'); p.add_argument('--keys',default='margin'); p.add_argument('--donor')
a=p.parse_args(); actor=Path(a.distilled).resolve(); out=Path(a.out).resolve()
DONOR=Path(a.donor) if a.donor else B/f'refinement_20261018/shared_policy_handoff/episode_{a.episode:06d}/portable/original/policy.npz'
sha=lambda q: hashlib.sha256(Path(q).read_bytes()).hexdigest()
ev=json.loads((actor/'evaluation/result.json').read_text()); scene=ev['execution']['scene']
with np.load(actor/'training/policy.npz',allow_pickle=False) as z: d={k:z[k] for k in z.files}
want={'margin':('margin_',),'release':('release_','multi_release'),'all':('margin_','release_','multi_release')}[a.keys]
with np.load(DONOR,allow_pickle=False) as z:
    for k in z.files:
        if k.startswith(want) or k=='component_output_decimals': d[k]=z[k]
if a.keys=='margin' and 'release_base_class' in d:
    pass
(out/'training').mkdir(parents=True,exist_ok=True); ck=out/'training/policy.npz'; np.savez_compressed(ck,**d)
tr=json.loads((actor/'training/training_report.json').read_text())
tr.update(wrapper=dict(recipe=f'donor wrapper keys ({a.keys}) re-applied',donor=str(DONOR),donor_sha256=sha(DONOR),base_actor=str(actor/'training/policy.npz'),base_actor_sha256=sha(actor/'training/policy.npz'),wrapper_script=str(Path(__file__).resolve()),wrapper_sha256=sha(__file__)),checkpoint_sha256=sha(ck))
(out/'training/training_report.json').write_text(json.dumps(tr,indent=2,default=str))
from evaluate_policy_component import evaluate
evaluate(a.episode,scene,str(ck),str(out/'evaluation'),ev['input_dir'])
r=json.loads((out/'evaluation/result.json').read_text()); mm=r['metrics']['official_style_reference_proxy']['objects']
print(json.dumps(dict(out=str(out),passed=r['metrics']['reference_proxy_mp_sr']==1 and r['execution']['physics_verified'],objects={k:(round(v['mean_position_error_cm'][0],2),round(v['mean_orientation_error_degrees'][0],1)) for k,v in mm.items()})),flush=True)
