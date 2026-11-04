"""Small wrist-command corrections inferred only during observed opposing contact.

The object-pose discrepancy is a robot-command proposal, never an object
actuation. Every proposal is re-executed from the unchanged initial state and
judged on the entire episode. This is an iterative-learning-control diagnostic.
"""
from pathlib import Path
import argparse,json,sys
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import numpy as np,mujoco
from scipy.spatial.transform import Rotation
from scipy.ndimage import gaussian_filter1d,binary_closing
B=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(B.parents[1]/'third_party/DexAgent'))
from tool_library.common import tracking as TR,episode as E
from floating_runtime import execute_reference
from wrist_reference import continuous_euler_xyz
from evaluate_mp_sr import evaluate_files

def score(metrics):
    objects=metrics['declared_frame_reference_proxy']['objects'];ratios=[];passed=[]
    for o in objects.values():
        p=o['mean_position_error_cm'][0];r=o['mean_orientation_error_degrees'][0]
        ratios.append(max(p/3,r/30));passed.append(p<3 and r<30)
    return max(ratios)+.25*np.mean(ratios),np.array(passed)

def evaluate(job):
    name,scene,reference,out=job;out=Path(out);inp=B/'episodes'/name
    ex=execute_reference(scene,reference,out/'execution')
    manifest=json.loads((inp/'input_manifest.json').read_text())
    m=evaluate_files(manifest['source'],ex['trajectory'],inp/'input_manifest.json',ex['scene'],inp/'metric_geometry.json')
    r=dict(episode=name,reference=reference,execution=ex,metrics=m,learned_policy=False)
    (out/'report.json').write_text(json.dumps(r,indent=2));return r

def run(episode,workers=6,source_run=None,output=None,local_windows=False):
    name=f'episode_{episode:06d}';src=Path(source_run).resolve() if source_run else B/'refinement_20261017/cem_v2'/name
    rp=src/'result.json' if (src/'result.json').exists() else src/'report.json'
    result=json.loads(rp.read_text());scene=result['execution']['scene']
    m=mujoco.MjModel.from_xml_path(scene);d=mujoco.MjData(m)
    ref=TR.load_reference(result['reference'],m);Q=ref['qpos'];fps=ref['fps'];N=len(Q)
    z=dict(np.load(result['execution']['trajectory']));objects=E.object_free_joints(m)
    active={s:{o:np.zeros(N,bool) for o in objects} for s in ('left','right')}
    for t in range(N):
        d.qpos[:]=z['qpos'][t];mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d);mujoco.mj_collision(m,d)
        touches={s:{o:set() for o in objects} for s in active}
        for c in d.contact:
            if c.dist>0:continue
            a=m.body(int(m.geom_bodyid[c.geom1])).name;b=m.body(int(m.geom_bodyid[c.geom2])).name
            for hand,obj in ((a,b),(b,a)):
                if obj not in objects:continue
                for side in active:
                    if hand.startswith(f'robot_{side}_'):
                        for f in ('thumb','index','middle','ring','pinky'):
                            if f in hand:touches[side][obj].add(f)
        for s in active:
            for o in objects:active[s][o][t]='thumb' in touches[s][o] and bool(touches[s][o]-{'thumb'})
    corrections={};details={}
    for side in active:
        obj=max(objects,key=lambda o:active[side][o].sum());mask=active[side][obj];ids=np.flatnonzero(mask)
        if len(ids)<4:continue
        mask=binary_closing(mask,structure=np.ones(3));w=gaussian_filter1d(mask.astype(float),max(1.,fps*.15))
        # Preserve pickup exactly: corrections start only after observed contact.
        w*=np.clip((np.arange(N)-ids[0])/(fps*.25),0.,1.)
        a=objects[obj];p=z['qpos'][:,a:a+3];targetp=Q[:,a:a+3]
        r=Rotation.from_quat(z['qpos'][:,a+3:a+7][:,[1,2,3,0]])
        targetr=Rotation.from_quat(Q[:,a+3:a+7][:,[1,2,3,0]])
        dp=gaussian_filter1d(targetp-p,max(1.,fps*.15),axis=0)
        dr=gaussian_filter1d((targetr*r.inv()).as_rotvec(),max(1.,fps*.15),axis=0)
        dp*=np.minimum(1.,.03/np.maximum(np.linalg.norm(dp,axis=1),1e-12))[:,None]
        dr*=np.minimum(1.,np.deg2rad(15)/np.maximum(np.linalg.norm(dr,axis=1),1e-12))[:,None]
        cols=m.jnt_qposadr[[m.joint(f'robot_{side}_{side}_{s}').id for s in ('pos_x','pos_y','pos_z','rot_x','rot_y','rot_z')]]
        corrections[side]=(cols,p,dp,dr,w)
        details[side]=dict(object=obj,opposing_contact_seconds=float(mask.sum()/fps),first_contact_s=float(ids[0]/fps),
                           max_translation_correction_m=float(np.linalg.norm(dp*w[:,None],axis=1).max()))
    out=Path(output).resolve() if output else B/'refinement_20261017/feedback_polish'/name;out.mkdir(parents=True,exist_ok=True)
    if (out/'summary.json').exists():raise FileExistsError(out/'summary.json')
    (out/'proposal_details.json').write_text(json.dumps(details,indent=2))
    jobs=[]
    times=np.arange(N)/fps
    variants=[(gain,0.,times[-1],False) for gain in (.5,1.)]
    if local_windows:
        variants=[(gain,start,stop,rot_only) for gain in (.15,.35,.65)
                  for start,stop in ((0.,times[-1]),(1.5,6.),(6.,times[-1])) for rot_only in (False,True)]
    for group in ('left','right','both'):
        if group!='both' and group not in corrections:continue
        if not corrections:continue
        for gain,start,stop,rot_only in variants:
            candidate=Q.copy()
            for side,(c,op,dp,dr,w) in corrections.items():
                if group not in (side,'both'):continue
                w=w.copy()
                if local_windows:
                    w*=np.clip((times-start)/.35,0,1)*np.clip((stop-times)/.35,0,1)
                delta=Rotation.from_rotvec(gain*w[:,None]*dr)
                candidate[:,c[:3]]=op+delta.apply(Q[:,c[:3]]-op)+(0. if rot_only else gain)*w[:,None]*dp
                angles=continuous_euler_xyz(delta*Rotation.from_euler('XYZ',Q[:,c[3:6]]),initial_angles=Q[0,c[3:6]])
                nochange=w==0;angles[nochange]=Q[nochange][:,c[3:6]]
                candidate[:,c[3:6]]=angles
                candidate[nochange[:,None]&np.isin(np.arange(m.nq),c)[None,:]]=Q[nochange[:,None]&np.isin(np.arange(m.nq),c)[None,:]]
            trial=out/(f'{group}_{gain:.2f}_{start:.1f}_{stop:.1f}_rot{int(rot_only)}' if local_windows else f'{group}_{gain:.1f}');trial.mkdir(exist_ok=True)
            path=TR.save_reference(trial/'reference.npz',m,candidate,ref['labels'],fps,ref['segment'],ref['segments'],
                source=f'Contact-gated wrist feedback proposal from {src}',floating_runtime=True,no_retiming=True)
            jobs.append((name,scene,path,str(trial)))
    with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn')) as pool:reports=list(pool.map(evaluate,jobs))
    base_score,protected=score(result['metrics']);best=None;best_score=base_score
    for r in reports:
        cost,passed=score(r['metrics'])
        if np.all(passed[protected]) and r['execution']['physics_verified'] and r['execution']['max_hand_pen_m']<.005 and cost<best_score:
            best,best_score=r,cost
    summary=dict(episode=name,source=str(src),initial_score=float(base_score),best_score=float(best_score),accepted=best is not None,
                 selected=best,proposals=reports,learned_policy=False)
    (out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps({k:v for k,v in summary.items() if k not in ('selected','proposals')}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--episode',type=int,required=True);p.add_argument('--workers',type=int,default=6)
    p.add_argument('--source-run');p.add_argument('--output');p.add_argument('--local-windows',action='store_true');run(**vars(p.parse_args()))
