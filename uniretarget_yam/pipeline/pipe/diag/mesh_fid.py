import glob,json,os,time,numpy as np,trimesh,coacd
from scipy.spatial import ConvexHull,cKDTree
from multiprocessing import Pool
REG='/scratch2/exaFLOPs/srl_yam_run/source_meshes/registered/'
pick={'white_pot':'episode_000008','beige_cup':'episode_000008','blue_cup':'episode_000020','water_pitcher':'episode_000020','plastic_dish_rack':'episode_000016','mini_sweeper':'episode_000030','mini_dust_pan':'episode_000038','white_pot_lid':'episode_000001','wooden_piece_1':'episode_000025','wooden_piece_2':'episode_000025'}
def fid(hulls,gt):
    g=gt.sample(15000);hp=[];eq=[ConvexHull(h.vertices).equations for h in hulls]
    def outside(p,skip):
        ok=np.ones(len(p),bool)
        for j,E in enumerate(eq):
            if j==skip:continue
            ok&=~(((E[:,:3]@p.T).T+E[:,3]).max(1)<-1e-4)
        return ok
    tot=sum(h.area for h in hulls)
    for j,h in enumerate(hulls):
        n=max(30,int(20000*h.area/tot));p=h.sample(n);hp.append(p[outside(p,j)])
    hp=np.vstack(hp);A=cKDTree(hp).query(g)[0];B=cKDTree(g).query(hp)[0]
    return dict(gt_to_shape_mean=round(100*float(A.mean()),2),gt_to_shape_p95=round(100*float(np.percentile(A,95)),2),shape_to_gt_mean=round(100*float(B.mean()),2),shape_to_gt_p95=round(100*float(np.percentile(B,95)),2),frac_shape_over_1cm=round(float((B>.01).mean()),3))
def job(a):
    name,th,mx=a;e=pick[name];src=next(p for p in (REG+f'{e}/{name}_decomp.obj',) if os.path.exists(p));gt=trimesh.load(REG+f'{e}/{name}.obj',force='mesh',process=False);m=trimesh.load(src,force='mesh');t=time.time();parts=coacd.run_coacd(coacd.Mesh(m.vertices,m.faces),threshold=th,max_convex_hull=mx,preprocess_mode='auto');H=[trimesh.Trimesh(x,y).convex_hull for x,y in parts];r=dict(object=name,threshold=th,max=mx,parts=len(H),sec=round(time.time()-t),**fid(H,gt));print(json.dumps(r),flush=True);return r
if __name__=='__main__':
    J=[(n,th,mx) for n in pick for th,mx in ((.05,16),(.02,64))]
    with Pool(8) as p:R=p.map(job,J,chunksize=1)
    json.dump(R,open('/scratch2/exaFLOPs/srl_yam_run/source_meshes/fidelity.json','w'),indent=1);print('FID_DONE')
