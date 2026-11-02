"""Ready pose for the shorter tool: palm where the setup's documented ready pose put it, pointing down the same way."""
import numpy as np,mujoco,json,re,sys,tempfile
from pathlib import Path
from scipy.spatial.transform import Rotation as R
def load(path):
    t=Path(path).read_text();t=re.sub(r'(<robot[^>]*>)',r'\1<mujoco><compiler discardvisual="true" fusestatic="false" balanceinertia="true"/></mujoco>',t,count=1);f=tempfile.NamedTemporaryFile('w',suffix='.urdf',delete=False);f.write(t);f.close();return mujoco.MjModel.from_xml_path(f.name)
old=load('/data5/exaFLOPs/proj/uniretarget/furniturebench_yam/src/furniturebench_yam/assets/data/urdf/yam_sharpa_right.urdf');new=load(str(Path(__file__).parent/'yam_assets/yam_sharpa_right_nogripper.urdf'));Q=[0.,1.3,1.3,-1.2,0.,0.];name='right_hand_C_MC'
def fk(m,q):
    d=mujoco.MjData(m);d.qpos[[m.jnt_qposadr[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,f'joint{i+1}')] for i in range(6)]]=q;mujoco.mj_kinematics(m,d);b=m.body(name).id;return d,b
d,b=fk(old,Q);tp=d.xpos[b].copy();tr=R.from_matrix(d.xmat[b].reshape(3,3).copy());print('old ready palm',tp.round(3).tolist())
m=new;d=mujoco.MjData(m);ids=[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,f'joint{i+1}') for i in range(6)];qa=[m.jnt_qposadr[j] for j in ids];va=[m.jnt_dofadr[j] for j in ids];b=m.body(name).id;lo,hi=m.jnt_range[ids,0],m.jnt_range[ids,1];rng=np.random.default_rng(0);best=None;jp=np.zeros((3,m.nv));jr=np.zeros_like(jp)
for t in range(40):
    q=np.array(Q)+rng.normal(0,.5,6)*(t>0);q=np.clip(q,lo,hi)
    for _ in range(120):
        d.qpos[qa]=q;mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d);e=np.r_[tp-d.xpos[b],.3*(tr*R.from_matrix(d.xmat[b].reshape(3,3).copy()).inv()).as_rotvec()];mujoco.mj_jacBody(m,d,jp,jr,b);J=np.r_[jp[:,va],.3*jr[:,va]];q=np.clip(q+.6*J.T@np.linalg.solve(J@J.T+1e-3*np.eye(6),e),lo,hi)
    d.qpos[qa]=q;mujoco.mj_kinematics(m,d);ep=float(np.linalg.norm(tp-d.xpos[b]));er=float(np.degrees((tr*R.from_matrix(d.xmat[b].reshape(3,3).copy()).inv()).magnitude()));margin=float(np.min(np.minimum(q-lo,hi-q)));key=ep+.002*er-.005*min(margin,.3)
    if best is None or key<best[0]:best=(key,q.copy(),ep,er,margin)
print(json.dumps(dict(ready=best[1].round(3).tolist(),pos_err_cm=round(100*best[2],2),rot_err_deg=round(best[3],1),limit_margin_rad=round(best[4],2))))
