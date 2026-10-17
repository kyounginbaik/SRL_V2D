"""Combine robot commands, then execute both objects together in full physics."""
from pathlib import Path
import argparse,json,sys,hashlib
import numpy as np,mujoco
B=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(B.parents[1]/'third_party/DexAgent'))
from tool_library.common import tracking as TR
from floating_runtime import execute_reference
from evaluate_mp_sr import evaluate_files

def run(episode,left,right,scene,output):
    name=f'episode_{episode:06d}';inp=B/'episodes'/name;m=mujoco.MjModel.from_xml_path(str(scene))
    ref=TR.load_reference(inp/'tracking_reference.npz',m);Q=ref['qpos'].copy();provenance={}
    for side,path in (('left',left),('right',right)):
        seed=TR.load_reference(path,m)
        if seed['qpos'].shape!=Q.shape or seed['fps']!=ref['fps']:raise ValueError('Timeline mismatch')
        cols=m.jnt_qposadr[[j for j in range(m.njnt) if m.joint(j).name.startswith(f'robot_{side}_')]]
        assert len(cols)==28
        Q[:,cols]=seed['qpos'][:,cols];provenance[side]=dict(path=str(path),sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest())
    assert np.array_equal(Q[0],ref['qpos'][0])
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    path=TR.save_reference(out/'reference.npz',m,Q,ref['labels'],ref['fps'],ref['segment'],ref['segments'],source='Combined robot command candidates; objects must be resimulated',floating_runtime=True)
    ex=execute_reference(scene,path,out/'execution');manifest=json.loads((inp/'input_manifest.json').read_text())
    metrics=evaluate_files(manifest['source'],ex['trajectory'],inp/'input_manifest.json',ex['scene'],inp/'metric_geometry.json')
    r=dict(episode=name,reference=path,provenance=provenance,execution=ex,metrics=metrics,learned_policy=False)
    (out/'report.json').write_text(json.dumps(r,indent=2));print(json.dumps(r))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--episode',type=int,required=True)
    for key in ('left','right','scene','output'):p.add_argument('--'+key,type=Path,required=True)
    run(**vars(p.parse_args()))
