"""After a lenient pass, push the object while it is held: failures refine the reference toward a grip that survives."""
import argparse,json,sys,time
from multiprocessing import Pool
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
import yam_loop as Y
GATE=.9;GRASP_BIAS=.3  # Pushes are random, so edits fitted to one push's error matter less than the grasp itself: most children mutate the grasp.
def main():
    p=argparse.ArgumentParser();p.add_argument('--episode',type=Path,required=True);p.add_argument('--side',required=True);p.add_argument('--object',type=int,default=0);p.add_argument('--tape',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--levels',type=float,nargs='+',required=True);p.add_argument('--workers',type=int,default=16);p.add_argument('--population',type=int,default=64);p.add_argument('--generations',type=int,default=40);p.add_argument('--stall',type=int,default=10);p.add_argument('--trials',type=int,default=24);p.add_argument('--holdout',type=int,default=96);p.add_argument('--seed',type=int,default=0);p.add_argument('--measure',action='store_true');p.add_argument('--onset-floor',type=float,default=Y.ONSET_FLOOR);p.add_argument('--wide-caps',action='store_true')
    a=p.parse_args();Y.ONSET_FLOOR=a.onset_floor;a.wide_caps and (setattr(Y,'VMAX',(.15,1.,1.5)),setattr(Y,'WMAX',(.8,6.,8.)));a.output.mkdir(parents=True,exist_ok=True);log=(a.output/'harden.jsonl').open('a');t0=time.monotonic();rng=np.random.default_rng(a.seed);c=a.seed*1000000+1;args=(a.episode,a.side,a.object,(),(),(),None);base=Y.YamTask(*args[:6]);h=base.frames-1;original=json.loads(a.tape.read_text());original=original[0] if isinstance(original,list) else original;original=original['sample']
    def say(**row):
        row['seconds']=round(time.monotonic()-t0);print(json.dumps(row),flush=True);log.write(json.dumps(row)+'\n');log.flush()
    def ids(n):
        nonlocal c;r=range(c,c+n);c+=n;return r
    def good(t):return t['frontier']>=h-1 and t['mean_cm']<100*Y.TOL_P and t['mean_deg']<np.degrees(Y.TOL_R)
    with Pool(a.workers,Y._init,(args,)) as pool:
        def trials(sample,level,seeds):return pool.map(Y._job,[(sample,h,'real',int(i),level) for i in seeds],chunksize=1)
        def rate(sample,level,seeds):
            tr=trials(sample,level,seeds);ok=[good(t) for t in tr];return float(np.mean(ok)),float(np.mean([t['mean_cm'] for t in tr if t['frames']>=h-1] or [99.])),float(np.mean([t['mean_deg'] for t in tr if t['frames']>=h-1] or [999.]))
        # The same held-out seeds judge every tape, so pushes and physics match between before and after.
        held=[900000000+i for i in range(a.holdout)]
        def report(name,sample):
            for lv in [0.]+a.levels:
                x=rate(sample,lv,held);say(report=name,level=lv,pass_rate=x[0],cm=round(x[1],2),deg=round(x[2],1))
        say(stage='start',frames=base.frames,onset=base.onset,object_weight_N=round(base.mg,3));report('original',original)
        if a.measure:return
        current=original;done=[]
        for level in a.levels:
            # The clean gate keeps hardening from trading the original task for push resistance.
            r0=rate(current,level,ids(a.trials))[0];say(level=level,stage='begin',rate=r0)
            if r0>=GATE:done.append(level);say(level=level,stage='already_passes');continue
            hist=trials(current,level,ids(a.population));best_rate,best=r0,current;stalled=0;passed=False
            for g in range(a.generations):
                hist.sort(key=Y.rank);hist=hist[:300];top=hist[:12];chk=pool.map(Y._job,[(r['sample'],h,'real',int(i),level) for r in top for i in ids(3)],chunksize=1);ver=[]
                for k,r in enumerate(top):
                    w=min(chk[k*3:(k+1)*3],key=lambda x:(x['frontier'],x['lift_mm']));w['sample']=r['sample'];w['verified']=True;ver.append(w)
                hist=ver+hist[12:];hist.sort(key=Y.rank);gain=False
                for r in [x for x in hist if x.get('verified')][:3]:
                    # A retest of the starting tape can pass by luck: only a changed tape counts, and a second batch must confirm it.
                    if r['sample']==current:continue
                    x=rate(r['sample'],level,ids(a.trials))
                    if x[0]>best_rate:
                        x2=rate(r['sample'],level,ids(a.trials));pooled=.5*(x[0]+x2[0])
                        if pooled>best_rate:best_rate,best=pooled,r['sample'];gain=True
                b=hist[0];say(level=level,generation=g,best_rate=best_rate,frontier=b['frontier'],cause=b['cause'],grip_lost=b.get('grip_lost'),cm=round(b['mean_cm'],2),deg=round(b['mean_deg'],1),press=round(b['sample']['press'],4),edits=len(b['sample'].get('edits',())))
                stalled=0 if gain else stalled+1
                if best_rate>=GATE:
                    clean=rate(best,0.,ids(a.trials))[0];say(level=level,stage='clean_gate',rate=clean)
                    if clean>=GATE:passed=True;break
                    best_rate,best=r0,current  # It lost the original task: not an improvement.
                if stalled>=a.stall:break
                el=hist[:max(6,len(hist)//20)];kids=pool.map(Y._child,[(el[int(rng.integers(len(el)))],int(i),'real',GRASP_BIAS) for i in ids(a.population)],chunksize=4);hist+=pool.map(Y._job,[(s,h,'real',int(i),level) for s,i in zip(kids,ids(len(kids)))],chunksize=2)
            if not passed:say(level=level,stage='stopped',best_rate=best_rate);break
            current=best;done.append(level);(a.output/f'hardened_{level:g}.json').write_text(json.dumps(dict(level=level,sample=current),indent=1)+'\n');say(level=level,stage='hardened',rate=best_rate)
        say(stage='finished',levels_passed=done);report('hardened',current);(a.output/'hardened.json').write_text(json.dumps(dict(levels=done,sample=current),indent=1)+'\n')
if __name__=='__main__':main()
