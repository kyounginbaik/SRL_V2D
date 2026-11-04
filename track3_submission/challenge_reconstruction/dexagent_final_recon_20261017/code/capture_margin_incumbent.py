"""Evaluate an immutable snapshot of a real physical-search incumbent."""
from pathlib import Path
import argparse,json,shutil
import numpy as np
from learned_feedback_policy import sha
from evaluate_policy_component import evaluate
from refine_grasp_cem import dump

def run(source,output):
 source=Path(source).resolve();out=Path(output).resolve();out.mkdir(parents=True,exist_ok=True)
 if (out/'settings.json').exists():raise FileExistsError(out)
 snapshot=json.loads((source/'best_mp.json').read_text());settings=json.loads((source/'settings.json').read_text());spec=settings['spec']
 dump(out/'search_incumbent_snapshot.json',snapshot);dump(out/'settings.json',settings)
 with np.load(spec['augmented_checkpoint']) as z:w={k:z[k] for k in z.files}
 w['margin_bias_rad']=np.asarray(snapshot['parameters']);checkpoint=out/'policy.npz';np.savez_compressed(checkpoint,**w)
 inp=Path(spec['episode_dir']);ep=int(json.loads((inp/'input_manifest.json').read_text())['episode'].split('_')[-1]);result=evaluate(ep,spec['scene'],checkpoint,out/'evaluation',inp)
 actual=result['metrics']['declared_frame_reference_proxy']['objects'];assert np.allclose([v['mean_position_error_cm'][0] for v in actual.values()],snapshot['report']['mean_position_cm'],atol=1e-8,rtol=0)
 report=dict(method='Immutable physically optimized margin incumbent with fresh full evaluation',source_search=str(source),source_checkpoint=settings['source_checkpoint'],source_checkpoint_sha256=settings['source_checkpoint_sha256'],checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),trainer_sha256=settings['trainer_sha256'],snapshot_evaluator_sha256=sha(__file__),parameters=snapshot['parameters'],best=snapshot['report'],reference_proxy_success=result['reference_proxy_success'],original_learned_weights_unchanged=True)
 dump(out/'training_report.json',report);dump(out/'result.json',dict(best=snapshot['report'],parameters=snapshot['parameters'],checkpoint=str(checkpoint),evaluation=result))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True);run(**vars(p.parse_args()))
