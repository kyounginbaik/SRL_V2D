"""Refresh selected33 metadata under the frozen shared actor registry."""
from pathlib import Path
import json,subprocess,sys
import numpy as np
from evaluate_policy_component import evaluate
B=Path(__file__).resolve().parents[1]
source=B/'refinement_20261018/expanded_batch/bimanual_release/episode_000033/case_0'
old=json.loads((source/'evaluation/result.json').read_text());out=source/'shared_evaluation'
if not (out/'result.json').exists():
    result=evaluate(33,old['execution']['scene'],source/'policy.npz',out,old['input_dir'])
    with np.load(old['execution']['trajectory']) as a,np.load(result['execution']['trajectory']) as b:
        checks={k:bool(np.array_equal(a[k],b[k])) for k in ['qpos','qvel','ctrl','substep_qpos','substep_qvel','substep_ctrl']}
    (out/'shared_registry_equivalence.json').write_text(json.dumps(checks,indent=2)+'\n')
    if not all(checks.values()):raise RuntimeError('Shared registry changed the selected actor')
subprocess.run([sys.executable,str(B/'code/audit_expanded_pickup.py'),'--episode','33','--run',str(out)],check=True)
