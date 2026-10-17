"""Freeze and physically evaluate a saved temporal-search incumbent."""
import argparse,json,shutil
from pathlib import Path
from refine_grasp_cem import B,execute_reference,evaluate_files,dump

def run(source,output):
    source=Path(source).resolve();out=Path(output).resolve();out.mkdir(parents=True,exist_ok=True)
    if (out/'reference.npz').exists():raise FileExistsError(out/'reference.npz')
    settings=json.loads((source/'settings.json').read_text());name=settings['episode']
    inp=B/'episodes'/name;manifest=json.loads((inp/'input_manifest.json').read_text())
    shutil.copy2(source/'incumbent.npz',out/'reference.npz')
    shutil.copy2(source/'progress.json',out/'progress_at_snapshot.json')
    ex=execute_reference(settings['spec']['scene'],out/'reference.npz',out/'execution')
    metrics=evaluate_files(manifest['source'],ex['trajectory'],inp/'input_manifest.json',ex['scene'],inp/'metric_geometry.json')
    dump(out/'report.json',dict(episode=name,reference=str(out/'reference.npz'),execution=ex,metrics=metrics,learned_policy=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True)
    run(**vars(p.parse_args()))
