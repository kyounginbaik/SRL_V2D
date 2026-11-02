"""Why does the real carry fail? Kinematic clearance of a rigid carry, and time-scale ablation."""
import sys,json,numpy as np,mujoco
from pathlib import Path
from scipy.spatial.transform import Rotation as R
sys.path.insert(0,str(Path(__file__).parent))
from grasp_lab import Lab
root=Path(__file__).parent.parent/'runs';fit=root/'surface_anchor_fit_v7'
lab=Lab(fit/'native_model.mjb',root/'cpu_baseline_jacobians_v1/physics.npz',19);m,d=lab.m,lab.d
attempts=json.load(open(fit/'attempts.json'))
print('ranked arrangements',attempts['nominally_force_feasible_arrangements'],'attempt 18 coverage',[a.get('demand_coverage') for a in attempts['attempts'] if a.get('index')==18])
z=np.load(fit/'candidate_018.npz');star=z['qpos'];P,N,G=z['patch_positions'],z['patch_normals'],z['hand_geom_ids']
print('geoms',[m.body(m.geom_bodyid[g]).name[23:] for g in G],'normals',np.round(N,2).tolist(),'dots',np.round(N@N.T,2)[np.triu_indices(3,1)].tolist())
op=star[lab.qo:lab.qo+3].copy();o0=R.from_quat(star[lab.qo+3:lab.qo+7][[1,2,3,0]])
d.qpos[:]=star;mujoco.mj_kinematics(m,d);wp0=d.xpos[lab.body].copy();wr0=R.from_matrix(d.xmat[lab.body].reshape(3,3).copy())
print('palm-from-object vector at grasp (world m)',np.round(wp0-op,3).tolist(),'angle from up deg',round(float(np.degrees(np.arccos((wp0-op)@lab.up/np.linalg.norm(wp0-op)))),1))
slide=[int(m.jnt_qposadr[j]) for j in lab.slides];bq=int(m.jnt_qposadr[lab.ball]);fixed=wr0*R.from_quat(star[bq:bq+4][[1,2,3,0]]).inv()
right=[g for g in range(m.ngeom) if m.geom_bodyid[g] in {int(m.jnt_bodyid[j]) for j in lab.joints} and m.geom_contype[g]+m.geom_conaffinity[g]>0]
stat=[g for g in range(m.ngeom) if m.geom_bodyid[g]==0 or m.body(m.geom_bodyid[g]).name=='support']
print('rigid carry: frame goalz turn_deg wristz wristspeed_m/s wristTurn_rad/s handTableDist(mm)')
prev=None
for k in list(range(lab.onset,lab.onset+90,5)):
    rel=lab.gr[k]*o0.inv();p=lab.gp[k]+rel.apply(wp0-op);rw=rel*wr0
    q=star.copy();q[slide]=p-(wp0-star[slide]);q[bq:bq+4]=(fixed.inv()*rw).as_quat()[[3,0,1,2]];q[lab.qo:lab.qo+3]=lab.gp[k];q[lab.qo+3:lab.qo+7]=lab.gr[k].as_quat()[[3,0,1,2]]
    d.qpos[:]=q;mujoco.mj_kinematics(m,d);dist=min(mujoco.mj_geomDistance(m,d,int(h),int(s),.3,None) for h in right for s in stat)
    sp=None if prev is None else (round(float(np.linalg.norm(p-prev[0])*4),2),round(float((rw*prev[1].inv()).magnitude()*4),2));prev=(p,rw)
    print(k,round(float(lab.gp[k][2]),3),round(float(np.degrees(rel.magnitude())),1),round(float(p[2]),3),sp,round(1e3*dist,1))
# Time-scale ablation: replay the same goal path slower (diagnostic only; real timing stays fixed).
base=(lab.gp.copy(),lab.gr,lab.gp_raw.copy(),lab.gr_raw)
for scale in (1,2,4,8):
    idx=np.minimum(lab.onset+np.arange(0,len(base[0])*scale)//scale,len(base[0])-1);idx=np.r_[np.arange(lab.onset),idx][:len(base[0])*scale]
    lab.gp,lab.gr,lab.gp_raw,lab.gr_raw=base[0][idx],base[1][idx],base[2][idx],base[3][idx]
    r=lab.run(star,P,N,G,8.,horizon=86*scale)
    print('time x',scale,{k:(round(v,2) if isinstance(v,float) else v) for k,v in r.items()},'first-break goal turn deg',round(float(np.degrees((lab.gr[lab.onset+max(r.get('first_break',0),0)]*o0.inv()).magnitude())),1))
