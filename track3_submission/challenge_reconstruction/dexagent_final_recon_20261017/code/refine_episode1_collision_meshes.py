"""Versioned fine convex decomposition of RGB/depth-supported object geometry."""
from pathlib import Path
import argparse,hashlib,json,time
import coacd,numpy as np,trimesh
from repair_episode1_registration import B,OUT

def run(task, version='pot_geometry_rgb_depth', max_parts=256):
    name=['white_pot','white_pot_lid'][task]
    source=OUT/version/'white_pot_visual.obj' if task==0 else B.parent/'final_recon/episode_000001/assets/white_pot_lid_visual.obj'
    suffix=version.removeprefix('pot_geometry_rgb_depth')
    dest=OUT/('fine_collision'+suffix)/name;dest.mkdir(parents=True,exist_ok=True)
    if (dest/'report.json').exists():raise FileExistsError('Preserve completed collision candidates: '+str(dest))
    mesh=trimesh.load(source,force='mesh',process=True)
    params=dict(threshold=.0015,real_metric=True,max_convex_hull=max_parts,preprocess_mode='auto',preprocess_resolution=100,
        resolution=4000,mcts_nodes=20,mcts_iterations=150,mcts_max_depth=3,merge=True,decimate=False,extrude=False,seed=42)
    coacd.set_log_level('info');start=time.monotonic()
    parts=coacd.run_coacd(coacd.Mesh(np.asarray(mesh.vertices),np.asarray(mesh.faces)),**params)
    paths=[];volumes=[]
    for i,(v,f) in enumerate(parts):
        part=trimesh.Trimesh(v,f,process=True).convex_hull;path=dest/f'part_{i:03}.obj';part.export(path);paths.append(str(path));volumes.append(float(part.volume))
    report=dict(source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),parts=len(parts),files=paths,
        sum_convex_volumes_m3=sum(volumes),elapsed_s=time.monotonic()-start,parameters=params,version='coacd1.0.14',
        threshold_documentation='https://github.com/SarahWeiii/CoACD/releases',
        scope='collision approximation audit; not proof of task success; original input meshes unchanged')
    (dest/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--task',type=int,choices=[0,1],required=True)
    p.add_argument('--version',default='pot_geometry_rgb_depth');p.add_argument('--max-parts',type=int,default=256);run(**vars(p.parse_args()))
