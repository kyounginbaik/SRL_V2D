"""Make a copy of a teacher dir (result.json + reference.npz, execution reused by path) whose reference object columns equal the
episode dir's tracking_reference.npz object columns (the corrected spoon track). Objects are unactuated, so the robot commands and
the execution are unchanged; this only satisfies refine_grasp_cem's 'seed modified object targets' guard for corrected-track
episodes. usage: rebase_reference_targets.py SRC_DIR OUT_DIR EPISODE_DIR"""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'): os.environ[k]='1'
import sys, json, shutil, numpy as np, mujoco
from pathlib import Path
sys.path[:0]=['/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/code','/data1/SRL_V2D/third_party/DexAgent']
from tool_library.common import tracking as TR
src,out,epd=map(Path,sys.argv[1:4]); out.mkdir(parents=True,exist_ok=False)
prev=json.loads((src/'result.json').read_text()); m=mujoco.MjModel.from_xml_path(prev['execution']['scene'])
ref=TR.load_reference(src/'reference.npz',m); Q=ref['qpos'].copy(); orig=TR.load_reference(epd/'tracking_reference.npz',m)['qpos']
assert len(Q)==len(orig), (len(Q),len(orig))
changed=[]
for j in range(m.njnt):
    if m.jnt_type[j]==mujoco.mjtJoint.mjJNT_FREE and m.joint(j).name.startswith(tuple(n for n in ['wooden_spoon','mini_sweeper','plastic_dish_rack'])):
        a=int(m.jnt_qposadr[j]); d=float(np.abs(Q[:,a:a+7]-orig[:,a:a+7]).max()); Q[:,a:a+7]=orig[:,a:a+7]; changed.append((m.joint(j).name,d))
robot=np.setdiff1d(np.arange(m.nq),np.concatenate([np.arange(int(m.jnt_qposadr[j]),int(m.jnt_qposadr[j])+7) for j in range(m.njnt) if m.jnt_type[j]==mujoco.mjtJoint.mjJNT_FREE]))
assert np.array_equal(Q[:,robot],ref['qpos'][:,robot])
p=TR.save_reference(out/'reference.npz',m,Q,ref['labels'],ref['fps'],ref['segment'],ref['segments'],source=f'object target columns rebased onto {epd}/tracking_reference.npz (robot commands identical to {src})',arms_coupled=False,floating_runtime=True)
res=dict(prev); res.update(reference=str(p),rebased_from=str(src),object_columns_replaced=changed,robot_commands_unchanged=True)
(out/'result.json').write_text(json.dumps(res,indent=1,default=str)+'\n'); print(json.dumps(dict(out=str(out),changed=changed)))
