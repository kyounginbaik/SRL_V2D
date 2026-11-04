"""Which workspace shifts leave the robot's starting pose and its parked arm clear of the scene.
A shift moves the robot against everything else, exactly as rebuilding the scene with that offset does."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
import numpy as np,mujoco,yam_loop as Y
CLEAR_START,CLEAR_PATH=.015,.005
p=argparse.ArgumentParser();p.add_argument('--episode',required=True);p.add_argument('--side',required=True);p.add_argument('--object',type=int,required=True);p.add_argument('--output',default='');p.add_argument('--report',action='store_true');p.add_argument('--start-only',action='store_true');a=p.parse_args()
Y.ONSET_FLOOR=1.4;t=Y.YamTask(a.episode,a.side,a.object);t.set_stage('real');m=t.m;d=mujoco.MjData(m);m.geom_margin[:]=.02;other='left' if a.side=='right' else 'right'
robot=np.array([m.body(b).name.startswith(('left_','right_')) for b in range(m.nbody)]);roots=[b for b in range(m.nbody) if robot[b] and not robot[m.body_parentid[b]]];rigs=[g for g in range(m.ngeom) if m.geom_bodyid[g]==0 and (m.geom(g).name or '').startswith('yam_')];bp=m.body_pos.copy();gp=m.geom_pos.copy();frames=range(0,t.frames-1,5)
def hits(limit,only=None,ignore_mover=False):
    mujoco.mj_kinematics(m,d);mujoco.mj_collision(m,d);o={}
    for i in range(d.ncon):
        c=d.contact[i];x,y=int(m.geom_bodyid[c.geom[0]]),int(m.geom_bodyid[c.geom[1]])
        if robot[x]==robot[y]:continue
        r,s=(x,y) if robot[x] else (y,x)
        if only and not m.body(r).name.startswith(only):continue
        if ignore_mover and s==t.obj:continue
        if c.dist<limit:k=f'{m.body(r).name}|{m.body(s).name}';o[k]=min(o.get(k,9.),float(c.dist))
    return o
def check(off):
    m.body_pos[:]=bp;m.geom_pos[:]=gp
    for b in roots:m.body_pos[b]=bp[b]-off
    for g in rigs:m.geom_pos[g]=gp[g]-off
    d.qpos[:]=t.base_q;d.qpos[t.qo:t.qo+7]=t.reset_pose;start=hits(CLEAR_START);path={}
    for f in (() if a.start_only else frames):
        d.qpos[:]=t.base_q;d.qpos[t.qo:t.qo+3]=t.gp_raw[f];d.qpos[t.qo+3:t.qo+7]=t.gr_raw[f].as_quat()[[3,0,1,2]];h=hits(CLEAR_PATH,only=other)
        for k,v in h.items():path[k]=min(path.get(k,9.),v)
    return start,path
if a.report:
    s,c=check(np.zeros(3));print(json.dumps(dict(start_clear=not s,start={k:round(100*v,2) for k,v in s.items()},parked_path={k:round(100*v,2) for k,v in c.items()})),flush=True);sys.exit(0)
ok=[];why={}
for x in np.arange(-.25,.251,.05):
    for y in np.arange(-.15,.301,.05):
        s,c=check(np.array([x,y,0.]))
        if not s and not c:ok.append([round(float(x),2),round(float(y),2)])
        else:why[f'{x:.2f},{y:.2f}']=dict(start=len(s),path=len(c))
Path(a.output).write_text(json.dumps(dict(allowed=ok,rejected=why)));print(json.dumps(dict(allowed=len(ok),rejected=len(why),zero_ok=[0.,0.] in ok or [0.0,-0.0] in ok)),flush=True)
