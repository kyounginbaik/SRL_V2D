# Puts each public mesh into the body frame of each episode's poses. The poses stay exactly as they are.
import numpy as np,trimesh,glob,os,json
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R
Z='/scratch2/exaFLOPs/latest_recon/v2/best_reconstruction_20261103/targets/';M='/scratch2/exaFLOPs/srl_yam_run/source_meshes/public/';O='/scratch2/exaFLOPs/srl_yam_run/source_meshes/registered/'
def load(p):
    s=trimesh.load(p);m=s.to_geometry() if hasattr(s,'to_geometry') else (s.dump(concatenate=True) if isinstance(s,trimesh.Scene) else s);m=trimesh.Trimesh(m.vertices,m.faces,process=False);m.merge_vertices(digits_vertex=6);m.remove_unreferenced_vertices();return m
def icp(a,b,Ri,it):
    c=cKDTree(b);Rm=Ri;t=b.mean(0)-(Rm@a.T).T.mean(0)
    for _ in range(it):
        x=(Rm@a.T).T+t;d,i=c.query(x);q=d<np.percentile(d,90);y=b[i][q];xa=a[q];ca,cy=xa.mean(0),y.mean(0);U,_,Vt=np.linalg.svd((xa-ca).T@(y-cy));D=np.diag([1,1,np.sign(np.linalg.det(Vt.T@U.T))]);Rm=Vt.T@D@U.T;t=cy-Rm@ca
    x=(Rm@a.T).T+t;d1=cKDTree(b).query(x)[0];d2=cKDTree(x).query(b)[0];return Rm,t,float((d1.mean()+d2.mean())/2),float(max(np.percentile(d1,99),np.percentile(d2,99)))
rep=[];cache={}
for f in sorted(glob.glob(Z+'*/*_mesh_metric.obj')):
    e=f.split('/')[-2];n=os.path.basename(f)[:-len('_mesh_metric.obj')];vis=M+n+'_visual.glb' if os.path.exists(M+n+'_visual.glb') else M+n+'.glb'
    if not os.path.exists(vis):rep.append(dict(episode=e,object=n,public=False));print(e,n,'no public mesh',flush=True);continue
    if n not in cache:cache[n]=load(vis)
    m=cache[n];a=m.sample(6000);t0=trimesh.load(f,force='mesh');b=t0.sample(6000);best=None
    for Ri in [np.eye(3)]+[R.random(random_state=s).as_matrix() for s in range(48)]:
        r=icp(a,b,Ri,25)
        if best is None or r[2]<best[2]:best=r
    Rm,t,mean,p99=icp(a,b,best[0],60);T=np.eye(4);T[:3,:3]=Rm;T[:3,3]=t
    os.makedirs(O+e,exist_ok=True);np.save(O+e+f'/{n}_T.npy',T);out=m.copy();out.apply_transform(T);out.export(O+e+f'/{n}.obj')
    col=M+n+'_collision.glb'
    if os.path.exists(col):c=load(col);c.apply_transform(T);c.export(O+e+f'/{n}_collision.obj')
    row=dict(episode=e,object=n,public=True,mean_cm=round(100*mean,2),p99_cm=round(100*p99,2),ext_public_cm=(100*m.extents).round(1).tolist(),ext_target_cm=(100*t0.extents).round(1).tolist());rep.append(row);print(row,flush=True)
json.dump(rep,open('/scratch2/exaFLOPs/srl_yam_run/source_meshes/registration_all.json','w'),indent=1);print('REGISTER_ALL_DONE')
