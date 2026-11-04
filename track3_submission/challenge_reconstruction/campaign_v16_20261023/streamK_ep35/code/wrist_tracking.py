"""Wrist tracking error timeline: executed floating-wrist pose vs the commanded reference (reference.npz of the teacher dir).
usage: wrist_tracking.py TEACHER_DIR [--step 10]"""
import sys, json, numpy as np, mujoco
from pathlib import Path
from scipy.spatial.transform import Rotation as R
sys.path[:0]=['/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/code','/data1/SRL_V2D/third_party/DexAgent']
from tool_library.common import tracking as TR
src=Path(sys.argv[1]); step=int(sys.argv[sys.argv.index('--step')+1]) if '--step' in sys.argv else 10
prev=json.loads((src/'result.json').read_text()); scene=prev['execution']['scene']; m=mujoco.MjModel.from_xml_path(scene)
ref=TR.load_reference(src/'reference.npz',m)['qpos']; tr=np.load(prev['execution']['trajectory']); Q=tr['qpos']; n=min(len(ref),len(Q)); fps=20.
for side in ('right','left'):
    pre=f'robot_{side}_{side}_'; wc=[int(m.joint(pre+'pos_'+x).qposadr[0]) for x in 'xyz']; wr=[int(m.joint(pre+'rot_'+x).qposadr[0]) for x in 'xyz']
    ep=np.linalg.norm(Q[:n,wc]-ref[:n,wc],axis=1)*100; er=np.degrees((R.from_euler('XYZ',Q[:n,wr])*R.from_euler('XYZ',ref[:n,wr]).inv()).magnitude())
    print(f'{side} wrist tracking: mean {ep.mean():.2f} cm / {er.mean():.1f} deg, max {ep.max():.2f} cm / {er.max():.1f} deg')
    if side=='right':
        for s in range(0,n,step):
            sl=slice(s,min(n,s+step)); print(f'  {s/fps:5.1f}s  wrist pos err {ep[sl].mean():5.2f} cm (max {ep[sl].max():5.2f})  rot err {er[sl].mean():5.1f} deg (max {er[sl].max():5.1f})')
