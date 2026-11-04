"""make a staging base dir (input_manifest.json copied from the v22b staging base + actor_report.json for a phase-residual actor).
usage: mkbase.py ACTOR_DIR STAGING_DIR "selection note" """
import sys, json, shutil
from pathlib import Path
a, s, note = Path(sys.argv[1]).resolve(), Path(sys.argv[2]), sys.argv[3]
E = Path('/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamE_ep25_30/ep25')
s.mkdir(parents=True, exist_ok=False); shutil.copy2(E / 'staging_R103s62/input_manifest.json', s / 'input_manifest.json')
base = json.loads((E / 'staging_R103s62/actor_report.json').read_text())
r = json.loads((a / 'evaluation/result.json').read_text()); o = r['metrics']['official_style_reference_proxy']['objects']
errs = {k: dict(cm=round(v['mean_position_error_cm'][0], 3), deg=round(v['mean_orientation_error_degrees'][0], 2)) for k, v in o.items()}
base.update(checkpoint=str(a / 'training/policy.npz'), training_report=str(a / 'training/training_report.json'), evaluation=str(a / 'evaluation'),
            passed=bool(r['reference_proxy_success']), object_errors=errs, maximum_ratio=max(max(e['cm'] / 3, e['deg'] / 30) for e in errs.values()),
            physical_replay_verified=bool(r['execution']['physics_verified']), reference_proxy_mp_sr=r['metrics']['reference_proxy_mp_sr'], selection=note)
(s / 'actor_report.json').write_text(json.dumps(base, indent=1)); print(errs)
