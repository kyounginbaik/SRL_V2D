import numpy as np, trimesh
F='/data1/SRL_V2D/challenge_reconstruction/final_recon/episode_000010/assets'
prof={}
for n in ('blue_cup','beige_cup'):
    m=trimesh.load(f'{F}/{n}_visual.obj',force='mesh',process=False)
    P=trimesh.sample.sample_surface(m,300000,seed=0)[0]; c=P.mean(0); w,U=np.linalg.eigh(np.cov((P-c).T)); a=U[:,-1]
    h=(P-c)@a; rr=np.linalg.norm((P-c)-np.outer(h,a),axis=1)
    if rr[h>np.percentile(h,85)].mean()<rr[h<np.percentile(h,15)].mean(): a=-a; h=-h
    h=h-h.min(); ang=np.arctan2(*(np.c_[(P-c)@np.cross(a,[1,0,0]), (P-c)@np.cross(a,np.cross(a,[1,0,0]))].T))
    zs=np.arange(0.002,h.max()-0.001,0.0025); rin=[];rout=[];rinmin=[];routmax=[]
    for z in zs:
        s=(h>=z-0.0015)&(h<z+0.0015)&(rr>0.015)
        ins=[];outs=[]
        for k in range(36):
            q=s&(ang>=-np.pi+k*np.pi/18)&(ang<-np.pi+(k+1)*np.pi/18)
            if q.sum()>3: ins.append(rr[q].min()); outs.append(rr[q].max())
        rin.append(np.median(ins)); rout.append(np.median(outs)); rinmin.append(np.percentile(ins,10)); routmax.append(np.percentile(outs,90))
    prof[n]=dict(z=zs,rin=np.array(rin),rout=np.array(rout),rinmin=np.array(rinmin),routmax=np.array(routmax),H=h.max())
    print(n,'H',round(h.max()*100,2)); 
    for i in range(0,len(zs),6): print(f'  z {zs[i]*100:5.1f} rin {rin[i]*100:.2f} rout {rout[i]*100:.2f} (p10 in {rinmin[i]*100:.2f} p90 out {routmax[i]*100:.2f})')
b,e=prof['blue_cup'],prof['beige_cup']
def minoff(bin_, eout, clear=0.0):
    for off in np.arange(0.0,0.12,0.0005):
        hh=e['z']+off; ok=hh<b['z'][-1]
        if np.all(eout[ok]+clear<=np.interp(hh[ok],b['z'],bin_)): return off
for lab,bi,eo in (('median',b['rin'],e['rout']),('p10/p90',b['rinmin'],e['routmax'])):
    for cl in (0,0.0005,0.001,0.0015):
        print(lab,'clear',cl*1000,'mm -> min bottom offset cm',round(minoff(bi,eo,cl)*100,2))
