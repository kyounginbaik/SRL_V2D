"""Full-physics evaluation of grasp seed commands with genuine replay checks."""
from pathlib import Path
import argparse,json,sys
B=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(B.parents[1]/'third_party/DexAgent'))
from floating_runtime import execute_reference
from evaluate_mp_sr import evaluate_files

def run(episode,variant,reference,output_root):
    name=f'episode_{episode:06d}'; inp=B/'episodes'/name
    out=Path(output_root)/name/variant;out.mkdir(parents=True,exist_ok=True)
    scene=B/'refinement_20261017/controller_sweep_continuous'/name/variant/'controller_scene.xml'
    manifest=json.loads((inp/'input_manifest.json').read_text())
    execution=execute_reference(scene,reference,out/'execution')
    metrics=evaluate_files(manifest['source'],execution['trajectory'],inp/'input_manifest.json',execution['scene'],inp/'metric_geometry.json')
    report=dict(episode=name,variant=variant,reference=str(reference),execution=execution,metrics=metrics,learned_policy=False)
    (out/'report.json').write_text(json.dumps(report,indent=2,default=str))
    print(json.dumps(report,default=str))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--episode',type=int,required=True);p.add_argument('--variant',required=True)
    p.add_argument('--reference',type=Path,required=True);p.add_argument('--output-root',type=Path,required=True)
    run(**vars(p.parse_args()))
