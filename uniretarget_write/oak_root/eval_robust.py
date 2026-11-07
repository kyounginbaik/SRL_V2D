"""Stress-test one checkpoint in Newton: the model is changed before the solver is built, then the normal eval runs and records the rollout.
Usage: eval_robust.py CKPT OUT.npz WORLDS KIND FACTOR   KIND in friction | mass | kp | none.  friction scales every shape; mass scales the pen (mass and inertia); kp scales finger drive stiffness and damping."""
import sys, runpy, numpy as np
sys.path.insert(0, '/workspace/src')
ckpt, out, W, kind, fac = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], float(sys.argv[5])
import flash_chord.envs.base as B
_orig = B.build_solver
def scale(arr, idx, f):
    a = arr.numpy(); a[idx] = a[idx] * f; arr.assign(a)
def build_solver(model, cfg):
    bl = list(model.body_label); jl = list(model.joint_label)
    if kind == 'friction':
        scale(model.shape_material_mu, slice(None), fac)
    elif kind == 'mass':
        ids = [i for i, n in enumerate(bl) if n.startswith('object/')]
        scale(model.body_mass, ids, fac); scale(model.body_inertia, ids, fac); scale(model.body_inv_mass, ids, 1 / fac); scale(model.body_inv_inertia, ids, 1 / fac)
    elif kind == 'kp':
        qd = model.joint_qd_start.numpy(); ids = [d for j, n in enumerate(jl) if 'sharpa_wave/' in n for d in range(qd[j], qd[j + 1])]
        scale(model.joint_target_ke, ids, fac); scale(model.joint_target_kd, ids, fac)
    print('PERTURB', kind, fac, flush=True)
    return _orig(model, cfg)
B.build_solver = build_solver
sys.argv = ['eval.py', ckpt, out, W]
runpy.run_path('/oak/eval.py', run_name='__main__')
