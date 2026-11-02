"""What is the right hand physically touching while it fails to reach the fitted pose?"""
import sys,numpy as np,mujoco
from pathlib import Path
run,model,world,star=Path(sys.argv[1]),sys.argv[2],int(sys.argv[3]),np.load(sys.argv[4])['qpos']
m=mujoco.MjModel.from_binary_path(model);z=np.load(run/'physics.npz');r=np.load(run/'references.npz');cap=int(z['native_capture_world_ids'][world]);grp=int(z['world_reference_group'][cap])
Q=z['native_qpos'][:,world];off=z['native_contact_offsets'];w=z['native_contact_worldid'];g=z['native_contact_geom'];F=z['native_contact_newton_force'];D=z['native_contact_dist']
name=lambda b:m.body(b).name.replace('right_sharpa_wave_right_','R_').replace('left_sharpa_wave_left_','L_').replace('wooden_spoon_object','SPOON')
for f in (14,20,24,27,30,33):
    s,e=off[f],off[f+1];agg={}
    for i in np.flatnonzero(w[s:e]==cap)+s:
        a,b=name(m.geom_bodyid[g[i,0]]),name(m.geom_bodyid[g[i,1]])
        if not (a.startswith('R_') or b.startswith('R_')):continue
        key=' | '.join(sorted((a,b)));n,dep=agg.get(key,(0.,0.));agg[key]=(n+float(np.linalg.norm(F[i,:3])),min(dep,float(D[i])))
    print(f,{k:(round(v[0],2),round(1e3*v[1],1)) for k,v in sorted(agg.items(),key=lambda kv:-kv[1][0])})
k=30;dq=Q[k,36:58]-star[36:58];order=np.argsort(-np.abs(dq))[:6];nm=[m.joint(j).name for j in range(30,52)]
print('largest finger errors vs fit at f30:',[(nm[i].replace('right_sharpa_wave_right_','').replace('right_',''),round(float(dq[i]),3)) for i in order])
rq=r['joint_q'][grp][k,36:58];rt=r['joint_target'][grp][k,34:56]
print('same joints: q_ref',np.round(rq[order],3).tolist(),'target',np.round(rt[order],3).tolist(),'achieved',np.round(Q[k,36:58][order],3).tolist(),'limits',np.round(m.jnt_range[30:52][order],2).tolist())
print('wrist ach-fit (m)',np.round(Q[k,29:32]-star[29:32],3).tolist())
