"""A stand under the arm bases for videos. Drawn into the render scene only: physics and the search never see it."""
import json
from pathlib import Path
import numpy as np,mujoco
from scipy.spatial.transform import Rotation as R
HALF_X,HALF_Y,BACK,FLOOR_BELOW_TABLE=.085,.12,.03,.72
def boxes(model_dir):
    p=json.loads((Path(model_dir)/'yam_placement.json').read_text());L,Rg=[np.array(p['bases'][s]['pos'],float) for s in('left','right')];M=R.from_quat(p['bases']['left']['quat'],scalar_first=True).as_matrix();up=np.array(p['up'],float);top=float(L@up);floor=float(p['table_height'])-FLOOR_BELOW_TABLE;mid=.5*(L+Rg);half_y=float(abs((L-Rg)@M[:,1]))/2+HALF_Y;c=mid-BACK*M[:,0];c=c-(c@up)*up+.5*(top+floor)*up
    return [(c,np.array([HALF_X,half_y,.5*(top-floor)]),M,np.array([.86,.87,.89,1.],np.float32))]
def add(scene,shapes,objid=None):
    for c,h,M,rgba in shapes:
        if scene.ngeom>=scene.maxgeom:return
        g=scene.geoms[scene.ngeom];mujoco.mjv_initGeom(g,mujoco.mjtGeom.mjGEOM_BOX,h,c,M.ravel(),rgba)
        g.segid=scene.ngeom  # mjv_initGeom leaves this unset, and segmentation rendering indexes by it.
        if objid is not None:g.objtype=mujoco.mjtObj.mjOBJ_GEOM;g.objid=objid
        scene.ngeom+=1
