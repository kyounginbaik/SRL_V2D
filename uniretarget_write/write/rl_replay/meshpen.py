"""Finger-pen overlap against the true finger meshes (visual geoms, convex hull) instead of SPIDER's capsules. Usage: meshpen.py QPOS.npz"""
import sys, numpy as np, mujoco
sys.path.insert(0,'.'); from planner import XML
Q=np.load(sys.argv[1]); Q=Q['q'] if hasattr(Q,'files') else Q
m=mujoco.MjModel.from_xml_path(XML); d=mujoco.MjData(m)
nm=lambda g: mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,g) or ''
PG=[g for g in range(m.ngeom) if nm(g).startswith('right_object_') and 'visual' not in nm(g)]
HV=[g for g in range(m.ngeom) if m.geom_type[g]==mujoco.mjtGeom.mjGEOM_MESH and m.geom_group[g]==1 and 'right_' in nm(g) and 'object' not in nm(g) and 'C_MC' not in nm(g)]
fr=np.zeros(6); D=[]; who={}
for t in range(0,len(Q),5):
    d.qpos[:]=Q[t]; mujoco.mj_forward(m,d); best=1.
    for h in HV:
        for p in PG:
            x=mujoco.mj_geomDistance(m,d,h,p,.02,fr)
            if x<best: best=x; bw=nm(h)
    D.append(best)
    if best<-.002: who[bw]=who.get(bw,0)+1
D=np.array(D); print(sys.argv[1].split('/')[-1], dict(n_hand_meshes=len(HV), n_pen=len(PG), overlap_p5_cm=round(float(-np.percentile(D,5))*100,2), overlap_med_cm=round(float(-np.median(D))*100,2)), sorted(who.items(),key=lambda x:-x[1])[:4])
