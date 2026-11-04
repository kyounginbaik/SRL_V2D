import numpy as np
for n in (25,37,8):
    d=f'/scratch2/exaFLOPs/hawor_rerun/out/episode_{n:06d}/episode_{n:06d}/tracks_0_{dict(zip((25,37,8),(356,151,219)))[n]}/model_tracks.npy'
    t=np.load(d,allow_pickle=True).item();print(n,'tids',list(t.keys()))
    for k,v in t.items():
        print(' tid',k,type(v),len(v),(v[0].keys() if isinstance(v,(list,np.ndarray)) and hasattr(v[0],'keys') else type(v[0])))
        fr=[x['frame'] for x in v] if hasattr(v[0],'keys') else []
        b=np.array([x['det_box'] for x in v]).reshape(len(v),-1) if hasattr(v[0],'keys') and 'det_box' in v[0] else None
        if b is not None:
            area=(b[:,2]-b[:,0])*(b[:,3]-b[:,1]);print('   frames',fr[0],fr[-1],'count',len(fr),'hand',v[0].get('det_handedness'),'box',b[0].round(0),'area p5/50/95',np.percentile(area,[5,50,95]).round(0),'min',area.min().round(0));cx=(b[:,0]+b[:,2])/2;cy=(b[:,1]+b[:,3])/2;st=np.hypot(np.diff(cx),np.diff(cy));print('   center jump px p99/max',np.percentile(st,99).round(0),st.max().round(0),'at',fr[int(st.argmax())],'small-area frames',[fr[i] for i in np.flatnonzero(area<.35*np.median(area))][:30])
