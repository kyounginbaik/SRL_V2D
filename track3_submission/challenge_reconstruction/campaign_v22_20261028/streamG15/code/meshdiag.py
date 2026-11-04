import numpy as np, kitx, trimesh
M=kitx.M
seen={}
for ep in kitx.TUNE+kitx.HELD:
    X=kitx.load_export(ep,'r8_base')
    for n,p in X['meshes'].items():
        if p in seen: continue
        cv,cf=kitx.sub_mesh(p); v,f,anc,ax,od=kitx.ref_geom(n,1)
        cc,cr=M._surface_moments(cv,cf); rc,rr=M._surface_moments(v,f)
        ext_c=np.ptp(cv,0); ext_r=np.ptp(v,0)
        # registration + chamfer
        ca,rot=M._register_object_body((cv,cf),(v,f),anc,1)
        cp=(M._sample_mesh_surface(cv,cf,3000,1)-cc)*(rr/cr); rp=M._sample_mesh_surface(v,f,3000,1)-rc
        from scipy.spatial import cKDTree
        R_,off=M._align_rigid(rp,cp); q=cp@R_.T+off
        d1=cKDTree(rp).query(q)[0];d2=cKDTree(q).query(rp)[0]
        seen[p]=1
        print(f"{n:18s} ep{ep:2d} scale(ref/ours) {rr/cr:.3f} ext ours {np.round(np.sort(ext_c)*100,1)} ref {np.round(np.sort(ext_r)*100,1)} chamfer(scaled) {d1.mean()*100:.2f}/{d2.mean()*100:.2f} cm  centroid-shift {np.linalg.norm(ca-cc)*100:.2f}cm")
