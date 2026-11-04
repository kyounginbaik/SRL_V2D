import json,sys,numpy as np,mujoco
for n in map(int,sys.argv[1:]):
    e=f'/uniretarget_yam/episodes/recon_{n:06d}';m=mujoco.MjModel.from_binary_path(e+'/yam/yam_model.mjb');M=json.load(open(e+'/recon_meta.json'));d=mujoco.MjData(m);mujoco.mj_forward(m,d)
    # Only the objects and the table: hands parked far away by turning robot collisions off.
    for g in range(m.ngeom):
        if m.body(int(m.geom_bodyid[g])).name.startswith(('left_','right_')):m.geom_contype[g]=0;m.geom_conaffinity[g]=0
    names=[o['name'] for o in M['objects']];p0={x:d.xpos[m.body(x).id].copy() for x in names};sup=m.body('support').pos.copy();r=float(m.geom_size[np.flatnonzero(m.geom_bodyid==m.body('support').id)[0],0]);top=sup[2]+.02
    mujoco.mj_resetData(m,d)
    for _ in range(int(2/m.opt.timestep)):mujoco.mj_step(m,d)
    print(n,'table xy',sup[:2].round(3),'R',round(r,3),{x:dict(start_dist_from_centre=round(float(np.linalg.norm(p0[x][:2]-sup[:2])),3),low_above_top=None,moved_cm=round(float(100*np.linalg.norm(d.xpos[m.body(x).id]-p0[x])),1),z_drop_cm=round(float(100*(p0[x][2]-d.xpos[m.body(x).id][2])),1)) for x in names},flush=True)
