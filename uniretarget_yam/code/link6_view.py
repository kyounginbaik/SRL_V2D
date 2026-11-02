import os;os.environ['MUJOCO_GL']='osmesa'
import sys,numpy as np,mujoco,cv2
A='/data5/exaFLOPs/proj/uniretarget/furniturebench_yam/assets_src/yam/assets/'
xml=f"""<mujoco><asset><mesh name="l6" file="{A}link_6_visual.stl"/><mesh name="l5" file="{A}link_5_visual.stl"/></asset><visual><global offwidth="900" offheight="700"/><headlight ambient=".5 .5 .5" diffuse=".5 .5 .5"/></visual><worldbody>
<geom type="mesh" mesh="l6" pos="-5.10177e-05 -0.00269952 0.0481778" rgba=".55 .57 .62 1"/>
{''.join(f'<geom type="box" size=".09 .09 .0004" pos="0 0 {z}" rgba="{c} .55" contype="0" conaffinity="0"/>' for z,c in ((.014,'1 0 0'),(.022,'0 .8 0'),(.05,'0 0 1'),(.08,'1 .6 0')))}
</worldbody></mujoco>"""
m=mujoco.MjModel.from_xml_string(xml);d=mujoco.MjData(m);mujoco.mj_forward(m,d);r=mujoco.Renderer(m,700,900);T=[]
for az,el in ((90,-5),(0,-5),(45,-30),(225,20)):
    c=mujoco.MjvCamera();c.lookat[:]=[0.02,0,.05];c.distance=.33;c.azimuth=az;c.elevation=el;r.update_scene(d,c);im=r.render().copy();cv2.putText(im,f'az{az} el{el}  red z.014 green z.022 blue z.05 orange z.08',(10,28),0,.6,(255,255,0),2);T.append(im)
cv2.imwrite(sys.argv[1],cv2.cvtColor(np.vstack([np.hstack(T[:2]),np.hstack(T[2:])]),cv2.COLOR_RGB2BGR))
v=m.mesh_vert[m.mesh_vertadr[0]:m.mesh_vertadr[0]+m.mesh_vertnum[0]]+m.geom_pos[0];print('verts',len(v),'mesh z range in link frame',v[:,2].min().round(4),v[:,2].max().round(4))
for z0,z1 in zip(np.arange(0,.105,.008),np.arange(.008,.113,.008)):
    s=(v[:,2]>=z0)&(v[:,2]<z1)
    if s.any():print(round(float(z0),3),'x',v[s,0].min().round(3),v[s,0].max().round(3),'y',v[s,1].min().round(3),v[s,1].max().round(3),'n',int(s.sum()))
