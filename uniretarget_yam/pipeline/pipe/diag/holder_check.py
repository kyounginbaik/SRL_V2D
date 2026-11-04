import glob,numpy as np,trimesh,matplotlib
matplotlib.use('Agg');import matplotlib.pyplot as plt
D='/scratch2/exaFLOPs/srl_yam_run/videos/diagnosis/';REG='/scratch2/exaFLOPs/srl_yam_run/source_meshes/registered/episode_000016/';O='/scratch2/exaFLOPs/SRL_V2D/uniretarget_yam/objects/.solidrack_recon_000016/plastic_dish_rack_collision/'
v=np.load(D+'.rack_slice.npz')['v'];full=trimesh.load(REG+'plastic_dish_rack.obj',force='mesh',process=False);H16=[trimesh.load(p,force='mesh') for p in sorted(glob.glob(O+'part_*.obj'))];H64=[trimesh.load(p,force='mesh') for p in sorted(glob.glob(D+'.rack_0.02/part_*.obj'))]
low=v[v[:,1]<np.percentile(v[:,1],25)];print('rack bounds',full.bounds.round(3).tolist());print('spoon lowest part x,y,z ranges',low.min(0).round(3),low.max(0).round(3))
heights=[-.05,-.03,-.01,.01,.03,.05];cz,cx=-.2,.03
fig,axs=plt.subplots(3,6,figsize=(17,8.6))
for c,y0 in enumerate(heights):
    for r,(t,ms) in enumerate((('GT mesh',[full]),('16 parts',H16),('64 parts',H64))):
        a=axs[r,c]
        for m in ms:
            sec=m.section(plane_origin=[0,y0,0],plane_normal=[0,1,0])
            if sec is None:continue
            for d in sec.discrete:
                if len(ms)>1:a.fill(d[:,2],d[:,0],alpha=.8,color='#e08a1e',lw=0)
                a.plot(d[:,2],d[:,0],color='#8a4a00' if len(ms)>1 else '#222',lw=.7)
        k=np.abs(v[:,1]-y0)<.012;a.scatter(v[k,2],v[k,0],s=9,color='#1f77c9',zorder=5);a.set_title(f'{t}, y={100*y0:.0f} cm',fontsize=9);a.set_aspect('equal');a.set_xlim(cz-.075,cz+.075);a.set_ylim(cx-.09,cx+.09);a.tick_params(labelsize=6)
plt.tight_layout();plt.savefig(D+'ep16_holder_slices.png',dpi=95)
# Straight down the holder from above, along the spoon's axis region: how far until the first GT surface?
top=full.bounds[1,1]+.02;grid=[(x,z) for x in np.arange(-.09,.1,.01) for z in np.arange(-.255,-.14,.01)]
org=np.array([[x,top,z] for x,z in grid]);dirs=np.tile([0,-1.,0],(len(org),1));loc,ray,_=full.ray.intersects_location(org,dirs,multiple_hits=False)
first=np.full(len(org),np.nan);first[ray]=top-loc[:,1];M=first.reshape(19,-1)*100;print('depth below top of mesh to first GT surface, cm; rows x, cols z');zs=np.arange(-.255,-.14,.01)
print('z   ',' '.join(f'{z*100:5.0f}' for z in zs))
for i,x in enumerate(np.arange(-.09,.1,.01)):print(f'{x*100:4.0f}',' '.join('    .' if np.isnan(q) else f'{q:5.1f}' for q in M[i]))
