import json,numpy as np,trimesh
C='/data1/SRL_V2D/challenge_reconstruction/reconstruction_recovery_20261026/canonical/evaluation/episode_000032'
K=json.load(open(f'{C}/intrinsics.json'));fx,fy,cx,cy=K['fx'],K['fy'],K['cx'],K['cy']
d=np.load('cmp_tracks.npz');KD='../bundles/K32_R10/episode_000032'
Bs=np.asarray(json.load(open(f'{C}/ball_models_v1.json'))['wooden_spoon'])
def proj(p):return np.c_[fx*p[:,0]/p[:,2]+cx,fy*p[:,1]/p[:,2]+cy]
def cl(P,r=.012,minn=12,maxk=10):
    left=np.ones(len(P),bool);C_=[]
    while left.sum()>minn and len(C_)<maxk:
        q=P[left];cnt=[(np.linalg.norm(q-x,axis=1)<r).sum() for x in q];c=q[int(np.argmax(cnt))]
        for _ in range(5):c=np.median(q[np.linalg.norm(q-c,axis=1)<r],axis=0)
        n=np.linalg.norm(P-c,axis=1)<r*1.5
        if (n&left).sum()<minn:break
        C_.append(c);left&=~n
    return np.array(C_)
out={}
for obj,mesh,TR in (('wooden_spoon','wooden_spoon',{'old':d['T0'],'hansol':d['Ths'],'ours':d['Ns']}),('blue_cup','blue_cup',{'old':d['Oc'],'hansol':d['Thc'],'ours':d['Nc']})):
    DET=json.load(open(f'{C}/ball_detections_{obj}.json'))['frames'];m=trimesh.load(f'{KD}/meshes/{mesh}.obj',force='mesh');com=np.asarray(m.vertices).mean(0)
    print('==',obj)
    for nm,T in TR.items():
        P=[];F=[];UV=[]
        for p in range(258):
            for x in (DET[p+1] if p+1<len(DET) else []):
                if x.get('depth_mad_m',1)>.01:continue
                b=T[p][:3,:3].T@(np.asarray(x['camera_point'])-T[p][:3,3])
                if np.linalg.norm(b-com)<.2:P.append(b);F.append(p);UV.append(x['uv'])
        P=np.array(P);F=np.array(F);UV=np.array(UV);Cn=cl(P);D=np.linalg.norm(P[:,None]-Cn[None],axis=2);dm=D.min(1);a=D.argmin(1)
        # pixel error against the track's own spike model, per detection
        px=np.array([np.linalg.norm(proj((Cn@T[f][:3,:3].T+T[f][:3,3]))[a[i]]-UV[i]) for i,f in enumerate(F)])
        w=lambda A,B:(F>=A)&(F<B)
        print(f'{nm:7s} det {len(P)} spikes {len(Cn)} within1cm {100*(dm<.01).mean():.0f}% median {100*np.median(dm):.2f}cm | px median {np.median(px):.1f} p90 {np.percentile(px,90):.1f} | f85-95 {np.median(px[w(85,96)]) if w(85,96).any() else -1:.1f} f125-165 {np.median(px[w(125,165)]):.1f} f170-190 {np.median(px[w(170,190)]):.1f} | per-spike n',np.bincount(a[dm<.015],minlength=len(Cn)).tolist())
        out[f'{obj}_{nm}']=Cn
    dist=trimesh.proximity.closest_point(m,out[f'{obj}_ours'])[1]*100;print('ours spikes to mesh surface cm',np.round(dist,2).tolist())
    if obj=='wooden_spoon':print('model v1 balls',len(Bs),' to mesh cm',np.round(trimesh.proximity.closest_point(m,Bs)[1]*100,2).tolist(),' ours-to-model cm',np.round(np.linalg.norm(out[f'{obj}_ours'][:,None]-Bs[None],axis=2).min(1)*100,2).tolist())
np.savez('spikes.npz',**out)
