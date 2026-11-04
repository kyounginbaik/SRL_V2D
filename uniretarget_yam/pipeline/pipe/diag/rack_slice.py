import glob,numpy as np,trimesh,matplotlib
matplotlib.use('Agg');import matplotlib.pyplot as plt
D='/scratch2/exaFLOPs/srl_yam_run/videos/diagnosis/';REG='/scratch2/exaFLOPs/srl_yam_run/source_meshes/registered/episode_000016/';O='/scratch2/exaFLOPs/SRL_V2D/uniretarget_yam/objects/recon_000016/plastic_dish_rack_collision/'
s=np.load(D+'.rack_slice.npz');v,c=s['v'],s['c'];full=trimesh.load(REG+'plastic_dish_rack.obj',force='mesh',process=False)
print('bounds',full.bounds.round(3).tolist(),'spoon y range',v[:,1].min().round(3),v[:,1].max().round(3),'x',v[:,0].min().round(3),v[:,0].max().round(3),'z',v[:,2].min().round(3),v[:,2].max().round(3))
sets=[('Public GT mesh',[full]),('Current: 16 convex parts',[trimesh.load(p,force='mesh') for p in sorted(glob.glob(O+'part_*.obj'))]),('Finer: 64 convex parts',[trimesh.load(p,force='mesh') for p in sorted(glob.glob(D+'.rack_0.02/part_*.obj'))])]
heights=[float(np.percentile(v[:,1],q)) for q in (8,22)]
fig,axs=plt.subplots(2,3,figsize=(12,6.6))
for r,y0 in enumerate(heights):
    for a,(t,ms) in zip(axs[r],sets):
        for m in ms:
            sec=m.section(plane_origin=[0,y0,0],plane_normal=[0,1,0])
            if sec is None:continue
            for d in sec.discrete:
                if len(ms)>1:a.fill(d[:,2],d[:,0],alpha=.75,color='#e08a1e',lw=0)
                a.plot(d[:,2],d[:,0],color='#8a4a00' if len(ms)>1 else '#333',lw=.8)
        k=np.abs(v[:,1]-y0)<.01;a.scatter(v[k,2],v[k,0],s=7,color='#1f77c9',zorder=5,label='spoon goal (frame 150)');a.set_title(f'{t}, height {y0*100:.1f} cm',fontsize=10);a.set_aspect('equal');cz=np.median(v[:,2]);cx=np.median(v[:,0]);a.set_xlim(cz-.1,cz+.1);a.set_ylim(cx-.1,cx+.1);a.legend(fontsize=7,loc='upper right');a.set_xlabel('m');a.set_ylabel('m')
plt.tight_layout();plt.savefig(D+'ep16_rack_holder_cross_section.png',dpi=110);print('saved')
