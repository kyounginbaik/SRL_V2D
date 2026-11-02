"""Automatic frontier refinement: propose, test in physics, trace the cause, edit there.

Fixed: object goal and physics. Everything the hand does is searched. There is no
gradient through the simulator, so edits are guided samples: the trace chooses the
frame, measured object error supplies a direction while the grip holds, and physics
on freshly jittered trials decides. The canonical simulator judges milestones later.
"""
import argparse,json,sys,time
from multiprocessing import Pool
from pathlib import Path
import numpy as np,mujoco
from scipy.spatial.transform import Rotation as R
sys.path.insert(0,str(Path(__file__).parent))
from antipodal_search import Antipodal,mutate
from grasp_search import TOL_P,TOL_R

SHAKE_FRAMES=140  # One lift second, then two seconds per axis at 20 fps.
ROBUST_TRIALS,ROBUST_MINIMUM=24,.9  # Calibrated on tapes with known official MP-SR.
SWEEP_LIMIT=.006  # The model's fingers overlap about 4.8 mm at rest; deeper paths jam the hand.
LOST_P,LOST_R=.15,np.radians(150.)  # Beyond these the object is lost, not lagging.


def spent_over(ep,er,total):
    """Official success is an episode mean, so excursions are allowed while budget remains.

    Spending stays on a line ending exactly at the mean limit after the last frame.
    """
    f=len(ep);slack=.3*total  # A late-starting object may spend early, as long as the final mean can still pass.
    return sum(ep)>TOL_P*(.7*f+slack) or sum(er)>TOL_R*(.7*f+slack)


class Loop(Antipodal):
    def __init__(self,*a,**k):
        super().__init__(*a,**k);m=self.m
        other='left_' if self.side=='right' else 'right_';self.left=[j for j in range(m.njnt) if m.body(m.jnt_bodyid[j]).name.startswith(other)]
        # Other objects must not be disturbed while this one is refined.
        # Objects not yet refined are judged on their own later step; all others count now.
        self.pending=set();self.future=set();self.park_other=False;self.other_onset=self.goal_onsets();self.others=[(k,qi) for k,qi in enumerate(self.s.object_qpos) if qi!=self.qo];self.allow_park=True;self.rigid=False

    def limited_follow(self,vmax,wmax,n):
        ps,rs=super().limited_follow(vmax,wmax,n);edits=getattr(self,'_edits',())
        if edits:
            f=np.arange(n);dp=np.zeros((n,3));dr=np.zeros((n,3))
            for e in edits:
                g=np.exp(-.5*((f-e['frame'])/e['width'])**2);dp+=g[:,None]*np.asarray(e['dp']);dr+=g[:,None]*np.asarray(e['dr'])
            ps=ps+dp;rs=[R.from_rotvec(dr[i])*rs[i] for i in range(n)]
        return ps,rs

    def goal_onsets(self):
        out={}
        for k in range(self.s.poses.shape[1]):
            o=self.s.poses[:,k];i=o[0];al=R.from_quat(i[3:7])*R.from_quat(i[10:14]).inv();gp=al.apply(o[:,7:10])+i[:3]-al.apply(i[7:10]);gr=al*R.from_quat(o[:,10:14]);moved=(np.linalg.norm(gp-gp[0],axis=1)>.02)|((gr*gr[0].inv()).magnitude()>.35);out[k]=int(np.flatnonzero(moved)[0]) if moved.any() else len(gp)
        return out

    def compile_tape(self,s,horizon):
        if self.rigid:s=dict(s,vmax=max(s.get('vmax',.25),.3),wmax=max(s.get('wmax',1.5),2.))  # Speed limits may not collapse into a pivot.
        self._edits=s.get('edits',());Q,T=self.compile_grasp(s,horizon);self._edits=()
        if s.get('left_park') or self.park_other:
            # Parked receiver: only taken when physics shows the source left hand interferes.
            n=len(Q);nq=self.s.qs;nd=self.s.ds;jm=self.s.jmap;f0=int(s.get('left_park') or 1)  # An unrefined hand must not disturb the object waiting for it.
            for j in self.left:
                w=4 if self.m.jnt_type[j]==1 else 1;wd=3 if self.m.jnt_type[j]==1 else 1;qi=nq[jm[j]];di=nd[jm[j]];Q[f0:n,qi:qi+w]=self.Q[f0,qi:qi+w];T[f0:n,di:di+wd]=self.T[f0,di:di+wd]
        return Q,T

    def randomize(self,rng):
        """Physics a tape must tolerate: canonical and fast solvers are not identical."""
        m=self.m
        if not hasattr(self,'nominal'):self.nominal=(m.geom_friction.copy(),m.body_mass.copy(),m.body_inertia.copy(),float(m.opt.timestep))
        friction,mass,inertia,dt=self.nominal;g=np.flatnonzero(m.geom_bodyid==self.obj);scale=float(rng.uniform(1.,1.25));m.geom_friction[:]=friction;m.geom_friction[g,0]=friction[g,0]*rng.uniform(.8,1.);m.body_mass[:]=mass;m.body_inertia[:]=inertia;m.body_mass[self.obj]=mass[self.obj]*scale;m.body_inertia[self.obj]=inertia[self.obj]*scale;m.opt.timestep=dt*float(rng.choice([1.,.8,1.25]));self.s.substeps=int(round(1/(self.fps*m.opt.timestep)));mujoco.mj_setConst(m,self.s.d)

    def trial(self,s,horizon,jitter):
        """Run one fresh trial and report frontier, cause and the object's error near it."""
        Qn,Tn=self.compile_tape(s,horizon);sim=self.s;m=self.m;o=self.reset_pose.copy();o[:3]+=jitter;sim.reset(Qn[0],o);disturbed=False;other_spent={};ep=[];er=[];ev=[];lift=0.;held=[];f6=np.zeros(6);peak=0.;grip_ref=None;slips=[];grip_break=None;shake=getattr(self,'stage',None) in ('shake','slow')
        for f in range(min(horizon,len(Tn))):
            sim.step(Tn[f]);d=sim.d;po=d.qpos[self.qo:self.qo+3].copy();ro=R.from_quat(d.qpos[self.qo+3:self.qo+7][[1,2,3,0]]);goal=self.gp_raw[f]
            if not np.isfinite(d.qpos).all():return dict(frontier=f,exploded=True,mean_cm=99.,mean_deg=999.,lift_mm=0.,hold=0,peak_up=0.,cause=f,frames=f)
            e1=float(np.linalg.norm(po-goal));rv=(self.gr_raw[f]*ro.inv()).as_rotvec();e2=float(np.linalg.norm(rv));ep.append(e1);er.append(e2);ev.append(np.r_[goal-po,rv]);lift=max(lift,float((po-self.o0)@self.up));up=0.;touch=False
            if f>=self.onset-8:
                for i in range(d.ncon):
                    c=d.contact[i];a,b=int(m.geom_bodyid[c.geom[0]]),int(m.geom_bodyid[c.geom[1]])
                    if self.obj in (a,b) and (a in self.right or b in self.right):
                        touch=True;mujoco.mj_contactForce(m,d,i,f6);fw=c.frame.reshape(3,3).T@f6[:3];up+=float((fw if b==self.obj else -fw)@self.up)
                peak=max(peak,up/self.mg)
            held.append(touch)
            if (shake or (self.rigid and self.stage=='real')) and f>=self.onset+10:
                # Rigidity rung: judged by object-to-hand slip, never by luck near the goal.
                wr=R.from_matrix(d.xmat[self.body].reshape(3,3).copy());wp=d.xpos[self.body];local=wr.inv().apply(po-wp)
                if grip_ref is None:grip_ref=(ro,wr,local)
                angle=float(np.degrees(((grip_ref[1].inv()*wr)*(grip_ref[0].inv()*ro).inv()).magnitude()));shift=float(np.linalg.norm(local-grip_ref[2]));slips.append(angle)
                other=any(self.obj in (int(m.geom_bodyid[c.geom[0]]),int(m.geom_bodyid[c.geom[1]])) and not ({int(m.geom_bodyid[c.geom[0]]),int(m.geom_bodyid[c.geom[1]])}&self.right) for c in d.contact[:d.ncon])
                if (shake and (angle>25. or shift>.02 or other)) or (not shake and (angle>45. or shift>.03)):
                    grip_break=f;break
            if f>self.onset and self.stage!='real' and (e1>2.5*TOL_P or e2>2.5*TOL_R):break
            if f>self.onset and self.stage=='real' and (e1>LOST_P or e2>LOST_R or spent_over(ep,er,self.frames)):break
            if self.stage=='real' and self.others and f%5==0:
                for k,qi in self.others:
                    if k in self.pending or (k in self.future and f>=self.other_onset[k]):continue
                    e_p,e_r=self.other_error(k,qi,f,d);spent=other_spent.setdefault(k,[0.,0.]);spent[0]+=5*e_p;spent[1]+=5*e_r;limit=.7*(f+1)+.3*self.frames
                    if spent[0]>TOL_P*limit or spent[1]>TOL_R*limit:disturbed=True
            if disturbed:break
        ep=np.asarray(ep);er=np.asarray(er);ev=np.asarray(ev);held=np.asarray(held);over=np.flatnonzero((ep>2*TOL_P)|(er>2*TOL_R));frontier=int(over[0]) if len(over) else len(ep)
        if self.stage=='real':frontier=len(ep)-1 if (ep[-1]>LOST_P or er[-1]>LOST_R or disturbed or spent_over(ep,er,self.frames)) else len(ep)
        if shake:frontier=grip_break if grip_break is not None else len(ep)
        elif grip_break is not None:frontier=min(frontier,grip_break)
        # Trace: the last frame the hand still touched the object before the break.
        before=np.flatnonzero(held[:frontier+1]);cause=int(before[-1]) if len(before) else max(0,frontier-1);grip_lost=frontier-cause>2
        window=np.arange(max(0,cause-12),min(len(ev),frontier+1),3);dense=np.arange(min(self.onset,len(ev)-1),len(ev),5)
        frontier=min(frontier,int(s.get('cap',frontier)))
        return dict(frontier=frontier,cause=cause,grip_lost=bool(grip_lost),mean_cm=float(100*ep.mean()),mean_deg=float(np.mean(slips)) if slips else float(np.degrees(er.mean())),lift_mm=lift*1e3,hold=int(held.sum()),peak_up=float(peak),frames=len(ep),error_frames=window.tolist(),error=ev[window].tolist(),dense_frames=dense.tolist(),dense_error=ev[dense].tolist())

    def other_error(self,k,qi,f,d):
        pose=self.s.poses[f if f<len(self.s.poses) else -1,k];init=self.s.poses[0,k];al=R.from_quat(init[3:7])*R.from_quat(init[10:14]).inv();goal=al.apply(pose[7:10])+init[:3]-al.apply(init[7:10]);turn=(R.from_quat(d.qpos[qi+3:qi+7][[1,2,3,0]]).inv()*(al*R.from_quat(pose[10:14]))).magnitude();return float(np.linalg.norm(d.qpos[qi:qi+3]-goal)),float(turn)

    def edit(self,parent,rng):
        """Trace-guided child: a local path bump at the cause, directed by measured error if held."""
        o=dict(parent['sample']);edits=list(o.get('edits',()))
        if parent.get('dense_frames') and rng.random()<.4:
            # Whole-path correction: while the grip holds, object error says how the hand should move.
            gain=float(rng.uniform(.2,.8));edits+=[dict(frame=int(f),width=3.,dp=(gain*np.asarray(e[:3])).tolist(),dr=(gain*np.asarray(e[3:])).tolist()) for f,e in zip(parent['dense_frames'],parent['dense_error'])];o['edits']=edits[-400:];return o
        if self.allow_park and rng.random()<.05:o['left_park']=int(self.onset) if not o.get('left_park') else 0
        cause=parent.get('cause',self.onset);frames=parent.get('error_frames') or [cause];error=parent.get('error') or [[0.]*6]
        i=int(rng.integers(len(frames)));frame=int(frames[i]);vec=np.asarray(error[i]);gain=0. if parent.get('grip_lost') and rng.random()<.5 else float(rng.uniform(.2,1.));width=float(rng.uniform(2.,10.))
        edits.append(dict(frame=frame+int(rng.integers(-4,5)),width=width,dp=(gain*vec[:3]+rng.normal(0,.004,3)).tolist(),dr=(gain*vec[3:]+rng.normal(0,.05,3)).tolist()));o['edits']=edits[-400:]
        return o


T_=None
def _init(args):
    global T_;T_=Loop(*args[:5]);T_.stage=None;T_.allow_park=args[5];T_.rigid=args[6];T_.pending=set(args[7]);T_.no_trace=args[8];T_.future=set(args[9]);T_.park_other=args[10]
def _job(job):
    s,h,stage,seed=job;rng=np.random.default_rng(seed)
    if T_.stage!=stage:T_.set_rung(stage);T_.stage=stage
    try:
        T_.randomize(rng);r=T_.trial(s,h,rng.uniform(-.003,.003,3)*np.array([1,1,0]))
    except Exception as err:r=dict(frontier=-1,mean_cm=99.,mean_deg=999.,lift_mm=0.,hold=0,peak_up=0.,cause=0,frames=0,failure=repr(err)[:160])
    r['sample']=s;r['rung']=stage;return r
def _proposal(seed):
    rng=np.random.default_rng(seed)
    for _ in range(200):
        g=T_.propose(rng)
        if g is None:continue
        s=T_.solve_hand(g,rng)
        if s['residual_m']<.006 and s['penetration_m']<.006:
            s['sweep_m']=T_.sweep(s)
            if s['sweep_m']<SWEEP_LIMIT:return s
    return None
def _child(job):
    e,seed,stage=job;rng=np.random.default_rng(seed)
    if T_.stage!=stage:T_.set_rung(stage);T_.stage=stage
    # Grasp rungs test the grasp alone; local path edits only exist on the real task.
    child=mutate(e['sample'],rng,.6) if stage!='real' or T_.no_trace or rng.random()>=.7 else T_.edit(e,rng)
    child.pop('cap',None);return child
def canonical_frontiers(base,manifest,samples,horizon):
    """The real judge: canonical physics from the real reset on the compiled tapes."""
    from flash_chord.retargeting.physical_reference_evaluator import evaluate_native_references
    from flash_chord.training.checkpoint import read_checkpoint_metadata
    from flash_chord.training.checkpoint_metadata import CONFIG_METADATA_KEY
    qs=[];ts=[]
    for s in samples:
        Q,T=base.compile_tape(s,base.frames-1);fq=base.Q.copy();ft=base.T.copy();fq[:len(Q)]=Q;ft[:len(T)]=T;fq[-1]=base.Q[-1];ft[-1]=base.T[-1];qs.append(fq);ts.append(ft)
    config=json.loads(read_checkpoint_metadata(manifest['checkpoint'])[CONFIG_METADATA_KEY]);steps=min(horizon,base.frames-1)
    result,data=evaluate_native_references(config,manifest,np.asarray(qs),np.asarray(ts),replicas=1,device='cpu',steps=steps);k=base.s.object_index;out=[];P=data['object_pose']
    def means(o):
        i=o[0];al=R.from_quat(i[3:7])*R.from_quat(i[10:14]).inv();gp=al.apply(o[:,7:10])+i[:3]-al.apply(i[7:10]);gr=al*R.from_quat(o[:,10:14]);return np.linalg.norm(o[:,:3]-gp,axis=1).mean(),(R.from_quat(o[:,3:7]).inv()*gr).magnitude().mean()
    for w in range(len(samples)):
        o=P[:,w,k];i=o[0];al=R.from_quat(i[3:7])*R.from_quat(i[10:14]).inv();gp=al.apply(o[:,7:10])+i[:3]-al.apply(i[7:10]);gr=al*R.from_quat(o[:,10:14]);ep=np.linalg.norm(o[:,:3]-gp,axis=1);er=(R.from_quat(o[:,3:7]).inv()*gr).magnitude();front=len(ep)
        for f in range(len(ep)):
            if f>base.onset and (ep[f]>LOST_P or er[f]>LOST_R or spent_over(ep[:f+1],er[:f+1],base.frames)):front=f;break
        out.append(dict(frontier=front,cm=float(100*ep.mean()),deg=float(np.degrees(er.mean())),official=result['official_mp_sr'] if steps==base.frames-1 else None,others_ok=bool(all(means(P[:,w,j])[0]<TOL_P and means(P[:,w,j])[1]<TOL_R for j in range(P.shape[2]) if j!=k and j not in base.pending and j not in base.future)),passed=bool(steps==base.frames-1 and front>=len(ep) and ep.mean()<TOL_P and er.mean()<TOL_R and all(means(P[:,w,j])[0]<TOL_P and means(P[:,w,j])[1]<TOL_R for j in range(P.shape[2]) if j!=k and j not in base.pending and j not in base.future))))
    return out


def rank(r):
    # A motionless object has low error, so the lift rung must reward lifting first.
    if r.get('rung')=='lift':return (-min(r['lift_mm'],60.),-r['hold'],-r['frontier'],max(r['mean_cm']/3,r['mean_deg']/30))
    return (-r['frontier'],max(r['mean_cm']/3,r['mean_deg']/30),-r['lift_mm'])


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--physics',type=Path,required=True);p.add_argument('--plan',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=4);p.add_argument('--proposals',type=int,default=300);p.add_argument('--population',type=int,default=160);p.add_argument('--generations',type=int,default=200);p.add_argument('--seed',type=int,default=0);p.add_argument('--side',default='right');p.add_argument('--stall',type=int,default=40);p.add_argument('--seeds',type=Path);p.add_argument('--verify',type=int,default=3);p.add_argument('--object',type=int,default=0);p.add_argument('--no-park',action='store_true');p.add_argument('--future',type=int,nargs='*',default=[]);p.add_argument('--park-other',action='store_true');p.add_argument('--adaptive',action='store_true');p.add_argument('--no-ladder',action='store_true');p.add_argument('--no-trace',action='store_true');p.add_argument('--pending',type=int,nargs='*',default=[]);p.add_argument('--start-stage',default='real');p.add_argument('--rigid',action='store_true');p.add_argument('--manifest',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);manifest=json.loads(a.manifest.read_text());log=(a.output/'generations.jsonl').open('a');(a.output/'milestones').mkdir(exist_ok=True);rng=np.random.default_rng(a.seed);t0=time.monotonic();counter=a.seed*1000000+1;base=Loop(a.model,a.physics,a.plan,a.side,a.object);base.rigid=a.rigid;base.pending=set(a.pending);base.future=set(a.future);base.park_other=a.park_other;frames=base.frames
    def ids(n):
        nonlocal counter;out=range(counter,counter+n);counter+=n;return out
    with Pool(a.workers,_init,((a.model,a.physics,a.plan,a.side,a.object,not a.no_park,a.rigid,a.pending,a.no_trace,a.future,a.park_other),),maxtasksperchild=400) as pool:
        pop=[r['sample'] for r in json.loads(a.seeds.read_text())] if a.seeds else [s for s in pool.imap_unordered(_proposal,ids(a.proposals),chunksize=2) if s is not None]
        print(json.dumps(dict(stage='proposals',kept=len(pop),seconds=round(time.monotonic()-t0),frames=frames,onset=base.onset)),flush=True);stage=a.start_stage if a.seeds else ('real' if a.no_ladder else 'lift');horizon={'lift':min(frames-1,80),'shake':min(frames-1,base.onset+SHAKE_FRAMES),'slow':min(frames-1,base.onset+SHAKE_FRAMES),'real':min(frames-1,base.onset+60)}[stage]
        hist=pool.map(_job,[(s,horizon,stage,i) for s,i in zip(pop,ids(len(pop)))],chunksize=2);best=-1;stalled=0;milestone=-1;escalated=False
        for g in range(a.generations):
            hist.sort(key=rank);hist=hist[:800];top=hist[:16]
            # Fresh trials: an elite keeps only the frontier it reproduces on all of them.
            jobs=[(r['sample'],horizon,stage,i) for r in top for i in ids(a.verify)];checked=pool.map(_job,jobs,chunksize=1);verified=[]
            for k,r in enumerate(top):
                runs=checked[k*a.verify:(k+1)*a.verify];w=min(runs,key=lambda x:x['frontier']);w['sample']=r['sample'];w['verified']=True;verified.append(w)
            hist=verified+hist[16:];hist.sort(key=rank);b=hist[0];stage_done=(stage=='lift' and sum(r['lift_mm']>40 for r in hist)>=40) or (stage in ('shake','slow') and sum(r['frontier']>=horizon-1 for r in hist)>=20)
            row=dict(generation=g,stage=stage,horizon=horizon,evaluated=counter-a.seed*1000000-1,seconds=round(time.monotonic()-t0),frontier=b['frontier'],cause=b['cause'],grip_lost=b.get('grip_lost'),hold=b['hold'],lift_mm=round(b['lift_mm'],1),cm=round(b['mean_cm'],2),deg=round(b['mean_deg'],1),edits=len(b['sample'].get('edits',())),lifting=sum(r['lift_mm']>50 for r in hist),q90=float(np.percentile([r['frontier'] for r in hist],90)),failures=sum('failure' in r for r in hist))
            print(json.dumps(row),flush=True);log.write(json.dumps(row)+'\n');log.flush()
            if stage=='real' and b['frontier']>=milestone+15:
                milestone=b['frontier'];(a.output/'milestones'/f'frontier_{milestone:04d}.json').write_text(json.dumps(dict(row=row,result={k:v for k,v in b.items() if k not in('error','error_frames','dense_frames','dense_error')}),indent=1)+'\n')
            improved=b['frontier']>best
            best=max(best,b['frontier']) if improved else best;stalled=0 if improved else stalled+1
            if stage in ('lift','shake','slow') and not stage_done and stalled>=a.stall:
                print(json.dumps(dict(stage=stage+'_stalled',frontier=best,passing=sum(r['frontier']>=horizon-1 for r in hist))),flush=True);stage_done=True
            if stage_done:
                seeds=[r['sample'] for r in (hist if stage!='lift' else [x for x in hist if x['lift_mm']>40])][:80]
                print(json.dumps(dict(stage='promoted',from_stage=stage,seeds=len(seeds),passing=sum(r['frontier']>=horizon-1 for r in hist) if stage!='lift' else None)),flush=True)
                if stage!='lift':(a.output/(stage+'_elites.json')).write_text(json.dumps([{k:v for k,v in r.items() if k not in('error','error_frames','dense_frames','dense_error')} for r in hist[:40]],indent=1)+'\n')
                stage,horizon={'lift':(('real',min(frames-1,base.onset+60)) if a.adaptive else ('shake',min(frames-1,base.onset+SHAKE_FRAMES))),'shake':('slow',min(frames-1,base.onset+SHAKE_FRAMES)),'slow':('real',min(frames-1,base.onset+60))}[stage];best=-1;stalled=0;hist=pool.map(_job,[(s,horizon,stage,i) for s,i in zip(seeds,ids(len(seeds)))],chunksize=2);continue
            ready=[r for r in hist if r['frontier']>=horizon-1 and r.get('verified')]
            if stage=='real' and ready:
                # Horizon and success both wait for canonical physics, which caps what it refutes.
                judged=ready[:4];base.stage='real';base.set_rung('real');verdicts=canonical_frontiers(base,manifest,[r['sample'] for r in judged],horizon)
                for r,v in zip(judged,verdicts):
                    r['canonical']=v
                    if v['frontier']<horizon-1:r['sample']=dict(r['sample'],cap=v['frontier']);r['frontier']=min(r['frontier'],v['frontier'])
                gate=dict(stage='canonical_gate',horizon=horizon,verdicts=[{k:(round(x,2) if isinstance(x,float) else x) for k,x in v.items()} for v in verdicts]);print(json.dumps(gate),flush=True);log.write(json.dumps(gate)+'\n');log.flush()
                good=[r for r,v in zip(judged,verdicts) if v['frontier']>=horizon-1]
                if good and horizon>=frames-1 and any(v['passed'] for v in verdicts):
                    # One canonical replay misses thin margins; randomized trials predicted official MP-SR.
                    winner=None
                    for r,v in zip(judged,verdicts):
                        if not v['passed']:continue
                        trials=pool.map(_job,[(r['sample'],horizon,stage,i) for i in ids(ROBUST_TRIALS)],chunksize=1);rate=float(np.mean([t['frontier']>=horizon-1 and t['mean_cm']<100*TOL_P and t['mean_deg']<np.degrees(TOL_R) for t in trials]));r['robust_pass_rate']=rate;line=dict(stage='robustness',rate=rate,canonical=v);print(json.dumps(line),flush=True);log.write(json.dumps(line)+'\n');log.flush()
                        if rate>=ROBUST_MINIMUM:winner=r;break
                        r['sample']=dict(r['sample'],cap=horizon-2);r['frontier']=min(r['frontier'],horizon-2)
                    if winner is not None:
                        hist.remove(winner);hist.insert(0,winner);(a.output/'canonical_pass.json').write_text(json.dumps({k:x for k,x in winner.items() if k not in('error','error_frames','dense_frames','dense_error')},indent=1)+'\n');print(json.dumps(dict(stage='canonical_passed',canonical=winner['canonical'],robust_pass_rate=winner['robust_pass_rate'])),flush=True);break
                if good and horizon<frames-1:
                    horizon=min(frames-1,horizon+60);hist=pool.map(_job,[(r['sample'],horizon,stage,i) for r,i in zip(good+[x for x in hist if x not in good][:80-len(good)],ids(80))],chunksize=2);best=-1;stalled=0;continue
            if stage=='real' and stalled>=a.stall:
                print(json.dumps(dict(stage='stalled',frontier=best,generations=stalled)),flush=True)
                if not (a.adaptive and not escalated):break
                # Easy objects never reach this. A stalled one earns the costlier grip rungs, once.
                escalated=True;(a.output/'first_real_best.json').write_text(json.dumps({k:v for k,v in hist[0].items() if k not in('error','error_frames','dense_frames','dense_error')},indent=1)+'\n');seeds=[{k:v for k,v in r['sample'].items() if k not in('edits','cap')} for r in hist[:80]];stage='shake';horizon=min(frames-1,base.onset+SHAKE_FRAMES);best=-1;stalled=0;hist=pool.map(_job,[(s,horizon,stage,i) for s,i in zip(seeds,ids(len(seeds)))],chunksize=2);print(json.dumps(dict(stage='escalated',seeds=len(seeds))),flush=True);continue
            elite=hist[:max(8,len(hist)//20)];kids=pool.map(_child,[(elite[int(rng.integers(len(elite)))],i,stage) for i in ids(a.population)],chunksize=4)
            hist+=pool.map(_job,[(s,horizon,stage,i) for s,i in zip(kids,ids(len(kids)))],chunksize=2)
    hist.sort(key=rank);base.stage='real';base.set_rung('real');Q,T=base.compile_tape(hist[0]['sample'],frames-1);fq=base.Q.copy();ft=base.T.copy();fq[:len(Q)]=Q;ft[:len(T)]=T;fq[-1]=base.Q[-1];ft[-1]=base.T[-1];np.savez_compressed(a.output/'best_plan.npz',joint_q=fq[None].astype(np.float32),joint_target=ft[None].astype(np.float32));(a.output/'top.json').write_text(json.dumps([{k:v for k,v in r.items() if k not in('error','error_frames','dense_frames','dense_error')} for r in hist[:40]],indent=1)+'\n');print('failures',[r['failure'] for r in hist if 'failure' in r][:3],flush=True)

if __name__=='__main__':main()
