"""Write YAM+Sharpa URDFs without the gripper: link_6's mesh is the parallel gripper; the hand goes on the wrist."""
import xml.etree.ElementTree as ET,sys,json,numpy as np,mujoco
from pathlib import Path
SRC='/data5/exaFLOPs/proj/uniretarget/furniturebench_yam/src/furniturebench_yam/assets/data/urdf/yam_sharpa_%s.urdf';OUT=Path(__file__).parent/'yam_assets';OUT.mkdir(exist_ok=True)
def rpy(r,p,y):
    cr,sr,cp,sp,cy,sy=np.cos(r),np.sin(r),np.cos(p),np.sin(p),np.cos(y),np.sin(y);return np.array([[cy*cp,cy*sp*sr-sy*cr,cy*sp*cr+sy*sr],[sy*cp,sy*sp*sr+cy*cr,sy*sp*cr-cy*sr],[-sp,cp*sr,cp*cr]])
report={}
for side in ('right','left'):
    t=ET.parse(SRC%side);r=t.getroot();links={l.get('name'):l for l in r.findall('link')};J={j.get('name'):j for j in r.findall('joint')}
    # Where link_5's own geometry ends along the wrist axis: that is the output face.
    l5=links['link_5'];mesh=l5.find('visual/geometry/mesh').get('filename');o=l5.find('visual/origin');off=np.array([float(x) for x in o.get('xyz').split()]) if o is not None else np.zeros(3);rp=rpy(*[float(x) for x in o.get('rpy').split()]) if o is not None and o.get('rpy') else np.eye(3)
    m=mujoco.MjModel.from_xml_string(f'<mujoco><compiler balanceinertia="true"/><asset><mesh name="a" file="{mesh}" refpos="0 0 0"/></asset><worldbody><geom type="mesh" mesh="a"/></worldbody></mujoco>');v=m.mesh_vert[:m.mesh_vertnum[0]];v=(R:=__import__('scipy.spatial.transform',fromlist=['Rotation']).Rotation).from_quat(m.mesh_quat[0][[1,2,3,0]]).apply(v)+m.mesh_pos[0];v=off+v@rp.T
    jo=J['joint6'].find('origin');jp=np.array([float(x) for x in jo.get('xyz').split()]);jr=rpy(*[float(x) for x in jo.get('rpy').split()]);in6=(v-jp)@jr;face=float(in6[:,2].max())
    l6=links['link_6'];[l6.remove(x) for x in l6.findall('visual')+l6.findall('collision')];iner=l6.find('inertial');old=float(iner.find('mass').get('value'));iner.find('mass').set('value','0.05');iner.find('origin').set('xyz','0 0 0.005');i=iner.find('inertia');[i.set(k,'0.00002' if k in('ixx','iyy','izz') else '0') for k in i.attrib]
    mount=J[f'{side}_hand_mount_joint'].find('origin');was=mount.get('xyz');mount.set('xyz',f'0 0 {face:.4f}');path=OUT/f'yam_sharpa_{side}_nogripper.urdf';t.write(path);report[side]=dict(wrist_face_z=round(face,4),old_mount=was,link6_mass_removed=old,urdf=str(path),link5_radius=round(float(np.linalg.norm(in6[:,:2],axis=1).max()),3))
print(json.dumps(report,indent=0))
