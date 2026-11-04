"""Seed reference for physics_pickup_search_v2 in a re-targeted world: robot columns of SEED_NPZ unchanged, object free-joint
columns replaced by the world's tracking_reference (the search requires seed object columns == original reference).
usage: make_seed.py SEED_NPZ WORLD OUT_NPZ"""
import sys, numpy as np, mujoco
from pathlib import Path
sys.path[:0]=['/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/code','/data1/SRL_V2D/third_party/DexAgent']
from tool_library.common import tracking as TR
seed,world,out=Path(sys.argv[1]),Path(sys.argv[2]),Path(sys.argv[3]); m=mujoco.MjModel.from_xml_path(str(world/'scene/controller_scene.xml'))
s=TR.load_reference(seed,m); o=TR.load_reference(world/'episodes/episode_000016/tracking_reference.npz',m); Q=s['qpos'].copy()
for n in ('plastic_dish_rack','wooden_spoon'):
    c=int(m.joint(n+'_free').qposadr[0]); Q[:,c:c+7]=o['qpos'][:,c:c+7]
TR.save_reference(out,m,Q,s['labels'],s['fps'],s['segment'],s['segments'],source=f'{seed} robot columns + object columns of {world} tracking_reference (re-targeted spoon track)',floating_runtime=True,no_retiming=True)
print('wrote',out,'max robot change',0.0)
