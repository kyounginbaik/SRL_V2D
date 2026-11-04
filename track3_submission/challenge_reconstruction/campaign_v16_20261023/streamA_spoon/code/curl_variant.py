"""Finger-curl variant of a teacher reference: during [t_on, t_off] blend the chosen right/left-hand finger flexion joints
(MCP_FE, PIP, DIP) toward a closure fraction of their range (power grasp around the handle), execute in the teacher's frozen
scene and score with the unchanged MP-SR proxy. Object targets, clock, wrist and the other hand are untouched.
usage: curl_variant.py --teacher DIR --episode-dir DIR --out DIR [--side right --object wooden_spoon --fingers middle,ring,pinky
       --curl 0.7 --index-pip 0.0 --index-dip 0.0 --thumb-ip 0.0 --t-on 1.5 --t-off 18.8 --ramp 0.5]"""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'): os.environ[k]='1'
import argparse, json, sys
from pathlib import Path
import numpy as np, mujoco
from scipy.spatial.transform import Rotation as R
ROOT=Path('/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017'); sys.path[:0]=[str(ROOT/'code'),'/data1/SRL_V2D/third_party/DexAgent']
from tool_library.common import tracking as TR
from floating_runtime import execute_reference
from evaluate_mp_sr import evaluate_files
sm=lambda u:np.clip(u,0,1)**3*(10-15*np.clip(u,0,1)+6*np.clip(u,0,1)**2)
def dump(p,o): Path(p).write_text(json.dumps(o,indent=1,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else str(x))+'\n')

def build(m, Q0, fps, a):
    T=len(Q0); t=np.arange(T)/fps; w=sm((t-a.t_on)/a.ramp)*(1-sm((t-a.t_off)/a.ramp)); Q=Q0.copy(); changed=[]
    def blend(jname, frac):
        j=m.joint(jname).id; adr=int(m.jnt_qposadr[j]); lo,hi=m.jnt_range[j]; target=lo+frac*(hi-lo)
        Q[:,adr]=(1-w)*Q0[:,adr]+w*np.maximum(Q0[:,adr],target) if a.only_close else (1-w)*Q0[:,adr]+w*target; changed.append(jname)
    for f in [x for x in a.fingers.split(',') if x]:
        for jn in ('MCP_FE','PIP','DIP'): blend(f'robot_{a.side}_{a.side}_{f}_{jn}', a.curl)
    if a.index_pip>0: blend(f'robot_{a.side}_{a.side}_index_PIP', a.index_pip)
    if a.index_dip>0: blend(f'robot_{a.side}_{a.side}_index_DIP', a.index_dip)
    if a.thumb_ip>0: blend(f'robot_{a.side}_{a.side}_thumb_IP', a.thumb_ip)
    if a.thumb_mcp>0: blend(f'robot_{a.side}_{a.side}_thumb_MCP_FE', a.thumb_mcp)
    for item in [x for x in a.add.split(',') if x]:   # additive margins: joint=+rad, e.g. thumb_IP=0.2
        jn,val=item.split('='); j=m.joint(f'robot_{a.side}_{a.side}_{jn}').id; adr=int(m.jnt_qposadr[j]); Q[:,adr]=Q0[:,adr]+w*float(val); changed.append(jn+'+'+val)
    return Q, changed

def inhand(m, traj, obj, side):
    d=mujoco.MjData(m); Q=traj['qpos']; ob=m.body(obj).id; palm=m.body(f'robot_{side}_{side}_base_yaw').id; rel=[]; ncon=[]
    og={g for g in range(m.ngeom) if m.geom_bodyid[g]==ob}
    for k in range(len(Q)):
        d.qpos[:]=Q[k]; mujoco.mj_forward(m,d)
        Tw_o=np.eye(4); Tw_o[:3,:3]=d.xmat[ob].reshape(3,3); Tw_o[:3,3]=d.xpos[ob]; Tw_p=np.eye(4); Tw_p[:3,:3]=d.xmat[palm].reshape(3,3); Tw_p[:3,3]=d.xpos[palm]
        rel.append(np.linalg.inv(Tw_p)@Tw_o)
        parts=set()
        for i in range(d.ncon):
            c=d.contact[i]
            if c.geom1 in og or c.geom2 in og:
                o=c.geom2 if c.geom1 in og else c.geom1; b=m.body(m.geom_bodyid[o]).name
                if b.startswith(f'robot_{side}'): parts.add('_'.join(b.split('_')[2:4]))
        ncon.append(len(parts))
    rel=np.array(rel); ncon=np.array(ncon); held=np.flatnonzero(ncon>=2)
    if len(held)==0: return dict(held_frames=0)
    k0=held[0]; dT=np.einsum('ij,njk->nik',np.linalg.inv(rel[k0]),rel); dp=np.linalg.norm(dT[:,:3,3],axis=1)*100; dr=np.degrees(R.from_matrix(dT[:,:3,:3]).magnitude())
    last=held[-1]
    return dict(held_frames=int(len(held)),first_hold_s=k0/20.,last_hold_s=last/20.,drift_cm_at_last_hold=float(dp[last]),drift_deg_at_last_hold=float(dr[last]),drift_cm_max_in_hold=float(dp[k0:last+1].max()),drift_deg_max_in_hold=float(dr[k0:last+1].max()),contact_parts_median=float(np.median(ncon[k0:last+1])))

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--teacher',required=True); p.add_argument('--episode-dir',required=True); p.add_argument('--out',required=True)
    p.add_argument('--side',default='right'); p.add_argument('--object',default='wooden_spoon'); p.add_argument('--fingers',default='middle,ring,pinky')
    for k,v in (('curl',.7),('index_pip',0.),('index_dip',0.),('thumb_ip',0.),('thumb_mcp',0.),('t_on',1.5),('t_off',18.8),('ramp',.5)): p.add_argument('--'+k.replace('_','-'),type=float,default=v)
    p.add_argument('--add',default='',help='comma list joint=rad additive margins during the window')
    p.add_argument('--only-close',action='store_true',help='never open a joint below its base command')
    a=p.parse_args(); src=Path(a.teacher); out=Path(a.out); out.mkdir(parents=True,exist_ok=True); epd=Path(a.episode_dir)
    prev=json.loads((src/'result.json').read_text()); scene=prev['execution']['scene']; m=mujoco.MjModel.from_xml_path(scene)
    ref=TR.load_reference(src/'reference.npz',m); Q0=ref['qpos']; fps=ref['fps']
    Q,changed=build(m,Q0,fps,a); lim=m.jnt_range[m.actuator_trnid[:,0]]; qa=m.jnt_qposadr[m.actuator_trnid[:,0]]; Q[:,qa]=np.clip(Q[:,qa],lim[:,0],lim[:,1]); Q[0]=Q0[0]
    refp=TR.save_reference(out/'reference.npz',m,Q,ref['labels'],fps,ref['segment'],ref['segments'],source=f'finger curl variant {vars(a)} on {src}; immutable object targets/clock',arms_coupled=False,floating_runtime=True)
    ex=execute_reference(scene,refp,out/'execution'); man=json.loads((epd/'input_manifest.json').read_text())
    met=evaluate_files(man['source'],ex['trajectory'],epd/'input_manifest.json',ex['scene'],epd/'metric_geometry.json')
    ih=inhand(m,np.load(ex['trajectory']),a.object,a.side)
    res=dict(episode=prev.get('episode',man.get('episode')),input_dir=str(epd),reference=str(refp),execution=ex,metrics=met,inhand=ih,variant=vars(a),changed_joints=changed,source=str(src),learned_policy=False)
    dump(out/'result.json',res); o=met['official_style_reference_proxy']['objects']
    print(json.dumps(dict(out=str(out),mp_sr=met['reference_proxy_mp_sr'],objects={k:(round(v['mean_position_error_cm'][0],2),round(v['mean_orientation_error_degrees'][0],1)) for k,v in o.items()},inhand=ih,completed=ex['execution_completed'])),flush=True)
