"""Fresh common-registry evaluation and independent audits of selected actors."""
from pathlib import Path
import argparse,json,subprocess,sys
import numpy as np
from evaluate_policy_component import evaluate
B=Path(__file__).resolve().parents[1]

def run(episode):
    source=B/f'refinement_20261018/expanded_batch/feasible_hand_margin_policy/episode_{episode:06d}'
    parent=json.loads((source/'evaluation/result.json').read_text());out=source/'shared_evaluation'
    if not (out/'result.json').exists():
        fresh=evaluate(episode,parent['execution']['scene'],source/'policy.npz',out,parent['input_dir'])
        with np.load(parent['execution']['trajectory']) as a,np.load(fresh['execution']['trajectory']) as b:
            checks={k:bool(np.array_equal(a[k],b[k])) for k in ['qpos','qvel','ctrl','substep_qpos','substep_qvel','substep_ctrl']}
        (out/'unchanged_registry_replay.json').write_text(json.dumps(checks,indent=2)+'\n')
        if not all(checks.values()):raise RuntimeError('Registry change affected the selected actor')
    subprocess.run([sys.executable,str(B/'code/audit_expanded_pickup.py'),'--episode',str(episode),'--run',str(out)],check=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--episode',type=int,required=True);run(**vars(p.parse_args()))
