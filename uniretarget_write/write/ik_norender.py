"""Run SPIDER's ik_fast unchanged, with the preview-video renderer stubbed out (no GPU on the login node). Does not touch the solution."""
import sys, runpy, numpy as np, mujoco
class R:
    def __init__(self, m, height=480, width=720): self.h, self.w = height, width
    def update_scene(self, *a, **k): pass
    def render(self): return np.zeros((self.h, self.w, 3), np.uint8)
    def close(self): pass
mujoco.Renderer = R
sys.argv = ['ik_fast.py'] + sys.argv[1:]
runpy.run_path('/lustre/exaflops/retarget_bench/repos/spider/spider/preprocess/ik_fast.py', run_name='__main__')
