"""Every object's mesh drawn at its recorded pose over the original video: orange = public GT mesh, magenta = no GT mesh (reconstruction)."""
import json,os,glob,numpy as np,cv2,trimesh
Z='/scratch2/exaFLOPs/latest_recon/v2/best_reconstruction_20261103/';REG='/scratch2/exaFLOPs/srl_yam_run/source_meshes/registered/';V='/scratch2/exaFLOPs/srl_yam_run/source_videos/ego_cam_a/';O='/scratch2/exaFLOPs/srl_yam_run/videos/mesh_check/';rep=json.load(open('/scratch2/exaFLOPs/srl_yam_run/source_meshes/registration_all.json'));err={(r['episode'],r['object']):r.get('mean_cm') for r in rep}
cells=[]
for e in sorted(os.listdir(Z+'bundles')):
    b=np.load(Z+f'bundles/{e}/object_poses.npz');I=json.load(open(Z+f'bundles/{e}/intrinsics.json'));W=b['world_from_rgb'];names=[str(x) for x in b['names']];T={n:np.load(Z+f'targets/{e}/{n}_trajectory.npz')['T_world_obj'] for n in names}
    mv=np.array([np.linalg.norm(T[n][:,:3,3]-T[n][0,:3,3],axis=1) for n in names]).sum(0);peak=int(mv.argmax());cap=cv2.VideoCapture(V+e+'.mp4');pair=[]
    meshes={}
    for n in names:
        p=REG+f'{e}/{n}_decomp.obj';gt=os.path.exists(p);m=trimesh.load(p if gt else Z+f'targets/{e}/{n}_mesh_metric.obj',force='mesh',process=False);F=np.asarray(m.faces);F=F[::max(1,len(F)//6000)];meshes[n]=(np.asarray(m.vertices),F,gt)
    for f in (0,peak):
        cap.set(cv2.CAP_PROP_POS_FRAMES,f);ok,im=cap.read();im=cv2.resize(im,(I['width'],I['height']));ov=im.copy();edge=im.copy()
        for n in names:
            v,F,gt=meshes[n];P=(T[n][f,:3,:3]@v.T).T+T[n][f,:3,3];c=(P-W[f,:3,3])@W[f,:3,:3];vis=c[:,2]>.05;uv=np.stack([I['fx']*c[:,0]/np.maximum(c[:,2],.05)+I['cx'],I['fy']*c[:,1]/np.maximum(c[:,2],.05)+I['cy']],1);col=(30,140,255) if gt else (220,60,220);good=F[vis[F].all(1)];z=c[good].mean(1) if False else c[:,2][good].mean(1)
            for t in good[np.argsort(-z)]:cv2.fillConvexPoly(ov,np.round(uv[t]).astype(np.int32),col,lineType=cv2.LINE_AA)
        out=cv2.addWeighted(ov,.55,im,.45,0)
        cv2.putText(out,f'{e[-3:].lstrip("0")} f{f}',(14,44),0,1.2,(255,255,255),3,cv2.LINE_AA);y=80
        for n in names:
            g=err.get((e,n));cv2.putText(out,f"{n}: {'GT %.2f cm'%g if g is not None else 'NO GT'}",(14,y),0,.72,(255,255,255) if g is not None else (220,160,255),2,cv2.LINE_AA);y+=30
        pair.append(out)
    cv2.imwrite(O+f'ep{int(e[-6:]):02d}_mesh_overlay.png',np.hstack([cv2.resize(x,(760,570)) for x in pair]));cells.append(cv2.resize(pair[1],(507,380)));print(e,[(n,meshes[n][2]) for n in names],peak,flush=True)
for i,a in enumerate((0,10)):
    S=cells[a:a+10];cv2.imwrite(O+f'all_episodes_{i+1}.png',np.vstack([np.hstack(S[j:j+5]) for j in range(0,len(S),5)]))
print('OVERLAY_DONE')
