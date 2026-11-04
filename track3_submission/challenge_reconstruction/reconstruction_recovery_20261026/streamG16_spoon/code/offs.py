"""per-episode roll offset of the adopted track vs the ball model: early static window f0-23 vs the whole clip (GT-free)"""
import glob, json, numpy as np
G='/data1/SRL_V2D/challenge_reconstruction/reconstruction_recovery_20261026/streamG16_spoon'
wrap=lambda a:(np.asarray(a)+180)%360-180
res={}
for EP in [3,5,6,15,16,32,33,35,36,37]:
    Z=[np.load(f) for f in sorted(glob.glob(f'{G}/out/scan_ep{EP}_*.npz'))]; ball=np.concatenate([z['ball'][:,0] for z in Z]); nb=np.concatenate([z['nb'] for z in Z]); thb=Z[0]['thb']
    A=np.load(f'{G}/out/ana_ep{EP}_l0.05_p20_b2_r1.npz'); inf=A['inf']
    best=wrap(thb[np.nanargmin(np.where(np.isnan(ball),np.inf,ball),1)])
    # adopted ep3/ep32 tracks contain deliberate rolls; best is relative to adopted already
    def stats(m):
        if m.sum()==0: return None
        b=best[m]; cur=np.nanmean(np.minimum(ball[m],40),0); j=int(np.argmin(cur))
        # curve-min with bootstrap over frames
        rng=np.random.default_rng(0); bs=[]
        idx=np.where(m)[0]
        for _ in range(300):
            s=rng.choice(idx,len(idx)); bs.append(wrap(thb[np.argmin(np.nanmean(np.minimum(ball[s],40),0))]))
        bs=np.array(bs)
        return dict(n=int(m.sum()), median_best=float(np.median(b)), iqr=[float(np.percentile(b,25)),float(np.percentile(b,75))], curve_min=float(wrap(thb[j])),
                    curve_px_at0=round(float(cur[0]),2), curve_px_min=round(float(cur[j]),2), boot_ci=[float(np.percentile(bs,5)),float(np.percentile(bs,95))])
    early=inf&(np.arange(len(inf))<24); 
    res[EP]=dict(early=stats(early), first60=stats(inf&(np.arange(len(inf))<60)), all=stats(inf))
    print(EP, 'early', res[EP]['early'], '\n    first60', res[EP]['first60'], '\n    all', res[EP]['all'])
json.dump(res, open(f'{G}/out/roll_offsets.json','w'), indent=1)
