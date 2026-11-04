"""Controlled retention ablation of the observed contact-only lid acquisition."""
from pathlib import Path
import argparse,json,sys
B=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(B.parents[1]/'third_party/DexAgent'))
from contact_retention_runtime import execute_retention
from evaluate_mp_sr import evaluate_files
from audit_acquisition import audit


def run(require_geometry):
    source=B/'refinement_20261017/lid_acquisition_v2/episode_000001/candidate_02_open10'
    config=json.loads((source/'config.json').read_text())
    config.update(require_opposed_contact_geometry=bool(require_geometry),release_start_s=4.45,release_blend_s=.25)
    name='geometry_gate' if require_geometry else 'contact_only_gate'
    out=B/f'refinement_20261017/lid_retention/episode_000001/{name}'
    out.mkdir(parents=True,exist_ok=True);(out/'config.json').write_text(json.dumps(config,indent=2))
    execution=execute_retention(source/'execution/scene_physics.xml',config['source_reference'],out/'execution',config)
    metrics=evaluate_files(B.parent/'final_recon/episode_000001',execution['trajectory'],
        B/'episodes/episode_000001/input_manifest.json',execution['scene'])
    retention=audit(out/'execution')
    report=dict(require_geometry=bool(require_geometry),config=config,execution=execution,metrics=metrics,retention=retention)
    (out/'report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(metrics['official_style_reference_proxy'],indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--task',type=int,required=True)
    run(parser.parse_args().task)
