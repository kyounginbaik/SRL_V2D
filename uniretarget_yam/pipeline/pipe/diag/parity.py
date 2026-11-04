"""Print the MuJoCo model Newton actually simulates, next to the one the search uses."""
import os,sys,json
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE","false")
src=open('/cmp/ref_check.py').read().split("options = dict(")[0]
sys.argv=['x','A',os.environ['RECON_PARQUET']];exec(src)
import mujoco,numpy as np
from flash_chord.runtime.sim import build_solver
model=setup.scene.model;solver=build_solver(model,env_cfg.sim);m=solver.mj_model
print('sim',env_cfg.sim);print('newton opt dt',m.opt.timestep,'cone',m.opt.cone,'impratio',m.opt.impratio,'integrator',m.opt.integrator,'noslip',m.opt.noslip_iterations,'o_margin',m.opt.o_margin)
y=mujoco.MjModel.from_binary_path(os.environ['RECON_EP']+'/yam/yam_model.mjb');print('search opt dt',y.opt.timestep,'cone',y.opt.cone,'impratio',y.opt.impratio,'integrator',y.opt.integrator)
def dump(tag,m):
    names=lambda i:mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,i) or ''
    bn=lambda i:mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,int(m.geom_bodyid[i])) or ''
    seen=set()
    for i in range(m.ngeom):
        if m.geom_contype[i]==0 and m.geom_conaffinity[i]==0:continue
        k=(bn(i).split('/')[-1][:18],int(m.geom_condim[i]),tuple(np.round(m.geom_friction[i],4)),tuple(np.round(m.geom_solref[i],4)),tuple(np.round(m.geom_solimp[i],3)),round(float(m.geom_margin[i]),4),round(float(m.geom_gap[i]),4))
        key=k[1:]
        if key in seen:continue
        seen.add(key);print(tag,k)
dump('newton',m);dump('search',y)
for tag,mm in (('newton',m),('search',y)):
    fing=[i for i in range(mm.nv) if mm.dof_damping[i]>0][:60];print(tag,'nu',mm.nu,'damping range',float(mm.dof_damping.min()),float(mm.dof_damping.max()),'armature',float(mm.dof_armature.max()))
    a=[mm.actuator_gainprm[i,0] for i in range(mm.nu)];print(tag,'gain unique',sorted(set(np.round(a,2)))[:12],'force',sorted(set(np.round(mm.actuator_forcerange[:,1],2)))[:12])
    print(tag,'object mass',[(mujoco.mj_id2name(mm,mujoco.mjtObj.mjOBJ_BODY,b),round(float(mm.body_mass[b]),3)) for b in range(mm.nbody) if any(s in (mujoco.mj_id2name(mm,mujoco.mjtObj.mjOBJ_BODY,b) or '') for s in('cup','pitcher','object'))][:6])
