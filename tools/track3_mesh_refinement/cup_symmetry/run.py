"""Refine generated cup meshes using a rotation-symmetry prior, then score separately."""
import csv,json,os,sys,hashlib
from pathlib import Path
import numpy as np
import trimesh
from scipy.ndimage import gaussian_filter1d
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent))
from score import reference_path,object_chamfer_cm,object_seed
from v2dlb.mesh_budget import budget_mesh

def main():
 name=['blue_cup','beige_cup'][int(os.environ['SLURM_ARRAY_TASK_ID'])]
 selection=json.loads((ROOT.parent/'selected_video_only/selection_manifest.json').read_text())
 source=Path(selection['objects'][name]['mesh'])
 mesh=trimesh.load(source,force='mesh',process=True)
 samples,_=trimesh.sample.sample_surface(mesh,30000,seed=42)
 center=samples.mean(axis=0)
 values,vectors=np.linalg.eigh(np.cov((samples-center).T))
 trials=[]
 for k in range(3):
  axis=vectors[:,k];t=(samples-center)@axis
  radial=np.linalg.norm(samples-center-t[:,None]*axis,axis=1)
  edges=np.linspace(*np.quantile(t,[.08,.92]),30)
  dispersion=[]
  for low,high in zip(edges[:-1],edges[1:]):
   r=radial[(t>=low)&(t<high)]
   if len(r)>40: dispersion.append((np.quantile(r,.85)-np.quantile(r,.15))/np.median(r))
  trials.append(float(np.median(dispersion)))
 axis=vectors[:,int(np.argmin(trials))]
 local=np.asarray(mesh.vertices)-center
 z=local@axis
 radial_vec=local-z[:,None]*axis
 radius=np.linalg.norm(radial_vec,axis=1)
 t=(samples-center)@axis
 sampled_r=np.linalg.norm(samples-center-t[:,None]*axis,axis=1)
 edges=np.linspace(t.min(),t.max(),65);mid=(edges[:-1]+edges[1:])/2
 profiles=[]
 for q in [.2,.8]:
  profile=[]
  for low,high in zip(edges[:-1],edges[1:]):
   r=sampled_r[(t>=low)&(t<=high)]
   profile.append(np.quantile(r,q) if len(r)>10 else np.nan)
  profile=np.array(profile);valid=np.isfinite(profile)
  profile=np.interp(mid,mid[valid],profile[valid])
  profiles.append(gaussian_filter1d(profile,1))
 inner,outer=[np.interp(z,mid,p) for p in profiles]
 target=np.where(radius<(inner+outer)/2,inner,outer)
 # Preserve horizontal end surfaces (including the cup floor), where radial symmetry
 # does not mean collapsing every radius to one circle.
 normals=np.abs(np.asarray(mesh.vertex_normals)@axis)
 weight=np.clip((.8-normals)/.6,0,1)
 out=ROOT/name;out.mkdir(exist_ok=True)
 variants={}
 for strength in [.5,1.0]:
  refined=mesh.copy();r=radius+strength*weight*(target-radius)
  refined.vertices=center+z[:,None]*axis+radial_vec*(r/np.maximum(radius,1e-10))[:,None]
  path=out/f'symmetry_{strength}.glb';refined.export(path);variants[str(strength)]=path
 (out/'provenance.json').write_text(json.dumps({'source':str(source),'sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'generation_uses_gt':False,'axis_dispersion':trials,'axis':axis.tolist()},indent=2))
 reference=budget_mesh(reference_path(name),4096,4096)
 with (ROOT.parent/'cd_o_per_episode_object.csv').open() as f: roster=[r for r in csv.DictReader(f) if r['object_name']==name]
 result=[]
 for strength,path in variants.items():
  candidate=budget_mesh(path,4096,4096)
  scores=[object_chamfer_cm(reference,candidate,object_seed(int(r['episode']),int(r['object_slot']))) for r in roster]
  result.append({'strength':strength,'mesh':str(path),'mean_cd_o_cm':float(np.mean(scores)),'scores':scores})
  print(result[-1],flush=True)
 (out/'scores.json').write_text(json.dumps(result,indent=2))

if __name__=='__main__':main()
