import numpy as np,trimesh,glob,os,json
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R
Z='/scratch2/exaFLOPs/latest_recon/v2/best_reconstruction_20261103/targets/';M='/scratch2/exaFLOPs/srl_yam_run/source_meshes/public/'
def load(p):
    s=trimesh.load(p);m=s.to_geometry() if hasattr(s,'to_geometry') else (s.dump(concatenate=True) if isinstance(s,trimesh.Scene) else s);return m
def icp(a,b,Ri,it=40):
    c=cKDTree(b);Rm=Ri;t=b.mean(0)-(Rm@a.T).T.mean(0);err=9
    for _ in range(it):
        x=(Rm@a.T).T+t;d,i=c.query(x);q=d<np.percentile(d,90);y=b[i][q];xa=a[q];ca,cy=xa.mean(0),y.mean(0);H=(xa-ca).T@(y-cy);U,_,Vt=np.linalg.svd(H);D=np.diag([1,1,np.sign(np.linalg.det(Vt.T@U.T))]);Rm=Vt.T@D@U.T;t=cy-Rm@ca
    x=(Rm@a.T).T+t;d1=cKDTree(b).query(x)[0];d2=cKDTree(x).query(b)[0];return Rm,t,float((d1.mean()+d2.mean())/2),float(max(np.percentile(d1,99),np.percentile(d2,99)))
out={}
for name in ['beige_cup','blue_cup','mini_dust_pan','mini_sweeper','plastic_dish_rack','water_pitcher','white_pot','white_pot_lid','wooden_piece_1','wooden_piece_2']:
    pv=M+(name+'_visual.glb' if os.path.exists(M+name+'_visual.glb') else name+'.glb');m=load(pv);tg=sorted(glob.glob(Z+f'*/{name}_mesh_metric.obj'))
    a=m.sample(6000);ext=m.extents
    best=None;t0=trimesh.load(tg[0],force='mesh');b=t0.sample(6000);scale=1.0
    if np.max(ext)>5:scale=.001;a=a*scale;ext=ext*scale
    R0=[np.eye(3)]+[R.random(random_state=s).as_matrix() for s in range(48)]
    for Ri in R0:
        r=icp(a,b,Ri,it=25)
        if best is None or r[2]<best[2]:best=r
    res=icp(a,b,best[0],it=60)
    out[name]=dict(file=os.path.basename(pv),verts=len(m.vertices),faces=len(m.faces),watertight=bool(m.is_watertight),scale=scale,ext_public_cm=(100*ext).round(1).tolist(),ext_target_cm=(100*t0.extents).round(1).tolist(),mean_cm=round(100*res[2],2),p99_cm=round(100*res[3],2),rot_from_identity_deg=round(float(np.degrees(R.from_matrix(res[0]).magnitude())),1),R=res[0].tolist(),t=res[1].tolist(),n_episodes=len(tg))
    print(name,{k:v for k,v in out[name].items() if k not in('R','t')},flush=True)
json.dump(out,open('/scratch2/exaFLOPs/srl_yam_run/source_meshes/registration.json','w'),indent=1)
print('REGISTER_DONE')
