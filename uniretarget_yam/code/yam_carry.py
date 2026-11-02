"""For reachable grasps: can the arm follow a rigid carry along the goal? Kinematics only, no physics."""
import sys,json,numpy as np
from multiprocessing import Pool
from pathlib import Path
from scipy.spatial.transform import Rotation as R
sys.path.insert(0,str(Path(__file__).parent))
from yam_loop import YamTask,FPS,ARM_SPEED
T=None
def init(e,side,k):
    global T;T=YamTask(e,side,int(k))
def carry(s,limit):
    wp=np.asarray(s['p']);wr=R.from_rotvec(s['r']);q=np.asarray(s['arm'],float);pe=[];re=[];step=ARM_SPEED/FPS if limit else None;fr=range(T.onset,T.frames,1 if limit else 4)
    for f in fr:
        rel=T.gr[f]*T.r0.inv();tp=T.gp[f]+rel.apply(wp-T.o0);tr=rel*wr
        q,a,b=T.arm_ik(q,tp,tr,steps=10 if limit else 18,limit=(step if limit else None));pe.append(a);re.append(b)
    pe=np.array(pe);re=np.degrees(re);return float(((pe<.03)&(re<30)).mean()),float(100*pe.mean()),float(re.mean())
def job(seed):
    rng=np.random.default_rng(seed)
    for _ in range(25):
        g=T.propose(rng)
        if g is None:continue
        s=T.solve_hand(g,rng)
        if s is None or s['residual_m']>=.008 or s['penetration_m']>=.008:continue
        a=carry(s,False);b=carry(s,True);palm=R.from_rotvec(s['r']).apply(T.local_normal)@(-T.up);return dict(free=a,limited=b,palm_down=float(palm),coverage=s['coverage'],sample=s)
    return None
for spec in sys.argv[1:]:
    e,side,k,n=spec.split(',')
    with Pool(4,init,(e,side,k)) as p:r=[x for x in p.map(job,range(int(n)),chunksize=1) if x]
    fr=np.array([x['free'][0] for x in r]);li=np.array([x['limited'][0] for x in r]);print(Path(e).name[-2:],side,'grasps',len(r),'| path fraction within 3cm/30deg: no-speed-limit p50/p90/max',np.percentile(fr,[50,90,100]).round(2).tolist(),'| 3 rad/s p50/p90/max',np.percentile(li,[50,90,100]).round(2).tolist());b=max(r,key=lambda x:x['limited'][0]);print('  best limited: frac',round(b['limited'][0],2),'mean miss',round(b['limited'][1],1),'cm',round(b['limited'][2],0),'deg | palm-down cos',round(b['palm_down'],2),'| same grasp no-limit',round(b['free'][0],2));print('  palm-down cos vs limited frac corr',round(float(np.corrcoef([x['palm_down'] for x in r],li)[0,1]),2) if len(r)>3 else None);Path(e,'yam/carry_scores.json').write_text(json.dumps([{k_:v for k_,v in x.items()} for x in sorted(r,key=lambda x:-x['limited'][0])],indent=0))
