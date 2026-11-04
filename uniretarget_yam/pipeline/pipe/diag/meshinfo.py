import mujoco,numpy as np,json,os,trimesh,glob
for n in (8,37,25,20):
    e=f'/uniretarget_yam/episodes/recon_{n:06d}';m=mujoco.MjModel.from_binary_path(e+'/yam/yam_model.mjb');print('ep',n)
    Z='/scratch2/exaFLOPs/latest_recon/v2/best_reconstruction_20261103'
    for o in json.load(open(e+'/recon_meta.json'))['objects']:
        b=m.body(o['name']).id;G=np.flatnonzero(m.geom_bodyid==b);tv=0;info=[]
        for g in G:
            mid=m.geom_dataid[g];nv=int(m.mesh_vertnum[mid]) if m.geom_type[g]==mujoco.mjtGeom.mjGEOM_MESH else 0;tv+=nv;info.append((nv,int(m.geom_contype[g]),int(m.geom_conaffinity[g]),int(m.geom_group[g])))
        print(' ',o['name'],'parts in meta',o['parts'],'geoms',len(G),'total verts',tv,info[:14])
        for lab,p in (('targets',f"{Z}/targets/episode_{n:06d}/{o['name']}_mesh_metric.obj"),('submitted',f"{Z}/submitted_meshes/episode_{n:06d}/{o['name']}.obj"),('bundle',f"{Z}/bundles/episode_{n:06d}/meshes/{o['name']}.obj")):
            if os.path.exists(p):
                t=trimesh.load(p,force='mesh');print('    ',lab,'verts',len(t.vertices),'faces',len(t.faces),'watertight',t.is_watertight,'extent cm',(100*t.extents).round(1),'vol L',round(abs(t.volume)*1000,3))
