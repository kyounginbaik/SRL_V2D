"""Versioned source-evidence cup shape candidate; original targets stay fixed."""
from pathlib import Path
import argparse,json,sys,xml.etree.ElementTree as ET
import numpy as np,trimesh
from floating_runtime import execute_reference
from evaluate_mp_sr import evaluate_files
B=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(B.parents[1]/'third_party/DexAgent'))

def run(scale):
    inp=B/'episodes/episode_000032';source=B.parent/'final_recon/episode_000032';root=B/'refinement_20261018/expanded_batch'
    old=json.loads((root/'native_mpc/episode_000032/result.json').read_text());man=json.loads((inp/'input_manifest.json').read_text())
    A=np.array(man['setup_from_source']);p=np.load(source/'object_poses.npz');k=list(p['names']).index('blue_cup');initial=A@p['world_from_object'][0,k]
    axis=np.array(json.loads((inp/'metric_geometry.json').read_text())['objects']['blue_cup']['axis_local']);axis/=np.linalg.norm(axis)
    if (initial[:3,:3]@axis)[2]<0:axis=-axis
    original=trimesh.load(source/'assets/blue_cup_visual.obj',process=False);top=float(np.max(original.vertices@axis))
    out=root/f'cup32_height_candidate/scale_{scale:.3f}';out.mkdir(parents=True,exist_ok=True)
    if (out/'result.json').exists():raise FileExistsError(out)
    tree=ET.parse(old['execution']['scene']);changed=[]
    for e in tree.findall('.//asset/mesh'):
        if not e.get('name','').startswith('blue_cup_'):continue
        path=Path(e.get('file'));mesh=trimesh.load(path,process=False)
        mesh.vertices+=np.outer((scale-1)*(mesh.vertices@axis-top),axis)
        target=out/'assets'/path.name;target.parent.mkdir(exist_ok=True);mesh.export(target);e.set('file',str(target.resolve()));changed.append(str(target))
    scene=out/'scene.xml';tree.write(scene,encoding='unicode')
    results={}
    for label,reference in [('native',old['search']['reference']),('original',str(inp/'tracking_reference.npz'))]:
        execution=execute_reference(scene,reference,out/label/'execution')
        metrics=evaluate_files(man['source'],execution['trajectory'],inp/'input_manifest.json',execution['scene'],inp/'metric_geometry.json')
        result=dict(episode='episode_000032',execution=execution,metrics=metrics,provisional_reconstruction_candidate=True,
            original_inputs_and_metric_geometry_unchanged=True,robot_commands_unchanged=True,physical_mass_inertia_gains_limits_unchanged=True)
        (out/label/'result.json').write_text(json.dumps(result,indent=2)+'\n');results[label]=result
    report=dict(scale=scale,axis_local_upward=axis.tolist(),preserved_top_local=top,changed_meshes=changed,
        source_evidence=str(root/'source_review/episode_000032/height_audit_v2/report.json'),results=results,submission_ready=False)
    (out/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v['metrics']['official_style_reference_proxy'] for k,v in results.items()}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--scale',type=float,default=.9);run(**vars(p.parse_args()))
