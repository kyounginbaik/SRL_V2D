import sys,json,time,numpy as np
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from fastsim import FastSim
root=Path(__file__).parent.parent/'runs';M=root/'surface_anchor_fit_v7/native_model.mjb'
print('checks: solver',)
for run in ('loop_r2_eval','grasp_exec_eval_v2'):
    z=np.load(root/run/'physics.npz');r=np.load(root/run/'references.npz');s=FastSim(M,root/run/'physics.npz')
    print(run,'opt iter',s.m.opt.iterations,'ls',s.m.opt.ls_iterations,'integ',s.m.opt.integrator,'cone',s.m.opt.cone,'impratio',s.m.opt.impratio,'solver',s.m.opt.solver,'tol',s.m.opt.tolerance)
    for w in (0,1,2):
        cap=int(z['native_capture_world_ids'][w]);grp=int(z['world_reference_group'][cap]);C=z['native_qpos'][:,w];t=time.monotonic();F=s.replay(r['joint_q'][grp],r['joint_target'][grp],min(90,len(C)));dt=time.monotonic()-t
        print(' world',w,'sec',round(dt,1))
        for f in (0,5,10,15,20,25,30,35,40,60,85):
            if f>=len(F):continue
            print('  f',f,'R wrist cm',round(100*float(np.linalg.norm(F[f,29:32]-C[f,29:32])),2),'L wrist cm',round(100*float(np.linalg.norm(F[f,0:3]-C[f,0:3])),2),'R finger max',round(float(np.abs(F[f,36:58]-C[f,36:58]).max()),3),'obj mm',round(1e3*float(np.linalg.norm(F[f,58:61]-C[f,58:61])),1))
