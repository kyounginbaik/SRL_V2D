"""write + officially evaluate (unchanged evaluate_policy_component) the planK re-plan of BASE with PARAMS (JSON) into OUT (cemK.write_checkpoint).
usage: STREAMC_EP=EP export_plan.py BASE PARAMS_JSON OUT"""
import sys, json, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import cemK, planK, hashlib
from pathlib import Path
base, params, out = sys.argv[1], json.loads(sys.argv[2]), sys.argv[3]
p = dict(planK.DEFAULT); p.update(params)
if (Path(out) / 'evaluation/result.json').exists(): raise SystemExit('exists')
cemK.write_checkpoint(base, p, out, dict(exported_by=str(Path(__file__).resolve()), export_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
