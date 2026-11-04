from pathlib import Path
import tempfile
import mujoco
import numpy as np
from tool_library.skills.construct_round_test_tube import construct_round_test_tube

for label,radius,length,wall,mass in [('small',.004,.055,.0004,.003),('medium',.006,.085,.0006,.006),('large',.009,.12,.0008,.015)]:
    with tempfile.TemporaryDirectory() as directory:
        built=construct_round_test_tube(directory,radius=radius,length=length,wall=wall,mass=mass)
        model=mujoco.MjModel.from_xml_path(built['xml']);data=mujoco.MjData(model);mujoco.mj_forward(model,data)
        assert model.nq==7 and model.nv==6 and model.neq==0
        bid=model.body('test_tube').id
        assert abs(model.body_mass[bid]-mass)<1e-10
        assert np.all(model.body_inertia[bid]>0)
        geom=np.array([-1],dtype=np.int32)
        distance=mujoco.mj_ray(model,data,np.array([0.,0.,length+.01]),np.array([0.,0.,-1.]),None,1,-1,geom)
        assert model.geom(int(geom[0])).name=='test_tube_bottom'
        assert abs(distance-(length+.01-2*radius))<1e-6
        assert built['cavity_depth_m']>0
        print('[PASS] '+label+' open cavity, closed round bottom, free joint, positive mass and inertia')
