import numpy as _n
for _a, _t in dict(bool=bool, int=int, float=float, complex=complex, object=object, str=str, unicode=str, nan=float("nan"), inf=float("inf")).items():
    if _a not in _n.__dict__: setattr(_n, _a, _t)
import pickle, numpy as np, torch, trimesh
from manotorch.manolayer import ManoLayer
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
RT = "/lustre/exaflops/retarget_bench/data/oakink2_root"; MR="/scratch2/exaFLOPs/daid_oakink/mano_root"
T = "scene_02__A002++seq__e9aab4c73ce9d2c16b85__2023-04-22-09-37-57"; PEN, BOARD = "O02@0029@00011", "O02@0054@00001"
a = pickle.load(open(f"{RT}/anno_preview/{T}.pkl", "rb"))
d = np.load("tip.npz"); fs=d["fs"]; L=d["local_plus"]; bi=int(d["thin"]); face=float(d["face"][1])
h = L[:,bi]-face
other=[i for i in range(3) if i!=bi]
touch = np.abs(h)<0.008
# runs of contact
runs=[];s=None
for i,t in enumerate(touch):
    if t and s is None: s=i
    if (not t) and s is not None: runs.append((s,i));s=None
if s is not None: runs.append((s,len(touch)))
print("segment frames",len(fs),"sec",len(fs)/120)
stroke_len=0
for r in runs:
    if r[1]-r[0]<12: continue
    p=L[r[0]:r[1]][:,other]; ln=np.linalg.norm(np.diff(p,axis=0),axis=1).sum()
    stroke_len+=ln
    print(f"stroke t {r[0]/120:.2f}-{r[1]/120:.2f}s frames {fs[r[0]]}-{fs[r[1]-1]} len {ln*100:.1f}cm h mean {h[r[0]:r[1]].mean()*100:.2f}cm std {h[r[0]:r[1]].std()*100:.2f}")
print("total stroke cm",stroke_len*100)
tp=L[touch]; print("touch fraction",touch.mean(),"area cm",(np.ptp(tp[:,other],0)*100).round(1))
sp=np.linalg.norm(np.diff(L,axis=0),axis=1)*120; print("touch tip speed cm/s median",np.median(sp[touch[:-1]])*100,"p95",np.percentile(sp[touch[:-1]],95)*100)
fig,ax=plt.subplots(1,2,figsize=(12,4))
ax[0].plot(np.arange(len(h))/120,h*100); ax[0].axhline(.8,c='k',ls=':');ax[0].axhline(-.8,c='k',ls=':');ax[0].set_ylim(-2,10);ax[0].set_xlabel('s');ax[0].set_ylabel('tip above board cm')
ax[1].plot(L[:,other[0]]*100,L[:,other[1]]*100,c='.8',lw=.5); ax[1].scatter(tp[:,other[0]]*100,tp[:,other[1]]*100,s=1,c=np.flatnonzero(touch));ax[1].set_aspect('equal');ax[1].set_title('tip on board')
plt.tight_layout();plt.savefig("tip_timeline.png",dpi=90)
# Pen history: motion and hand distance across the whole sequence
allf=sorted(a["obj_transf"][PEN]); 
for side,key in (("right","rh"),("left","lh")):
    ml=ManoLayer(mano_assets_root=MR,rot_mode="quat",side=side,center_idx=0,use_pca=False,flat_hand_mean=True)
    use=[f for f in allf[::30] if f in a["raw_mano"]]
    M=[a["raw_mano"][f] for f in use]
    with torch.no_grad(): out=ml(pose_coeffs=torch.cat([m[f"{key}__pose_coeffs"] for m in M]).float(),betas=torch.cat([m[f"{key}__betas"] for m in M]).float())
    v=(out.verts+torch.cat([m[f"{key}__tsl"] for m in M]).float()[:,None]).numpy()
    pen=trimesh.load(f"{RT}/object_repair/align_ds/{PEN}/model.obj",force="mesh"); pv=pen.sample(600)
    dd=[]; cnt=[]
    for i,f in enumerate(use):
        Tf=np.asarray(a["obj_transf"][PEN][f]); w=pv@Tf[:3,:3].T+Tf[:3,3]
        D=np.linalg.norm(v[i][None]-w[:,None],axis=2); dd.append(D.min()); cnt.append((D.min(0)<.008).sum())
    dd=np.array(dd);cnt=np.array(cnt)
    print(side,"dist to pen cm / verts within 8mm, every ~2.5 s:")
    print(" ",[(int(use[i]),round(float(dd[i]*100),1),int(cnt[i])) for i in range(0,len(use),10)])
    seg=[i for i,f in enumerate(use) if 2109<=f<3474]; print(side,"during writing: min dist cm mean",dd[seg].mean()*100,"verts mean",cnt[seg].mean(),"frac touching",(dd[seg]<.008).mean())
p=np.stack([np.asarray(a["obj_transf"][PEN][f])[:3,3] for f in allf]); 
mv=np.linalg.norm(p-p[0],axis=1); first=allf[int(np.argmax(mv>.02))]; print("pen first moves >2cm at frame",first,"frames from",allf[0],"to",allf[-1])
