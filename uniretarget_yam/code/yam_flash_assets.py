"""One bimanual URDF for flash_chord: both no-gripper YAM+Sharpa arms on a world root, placed where this episode's scene puts them."""
import argparse,json,xml.etree.ElementTree as ET
import mujoco
from pathlib import Path
from scipy.spatial.transform import Rotation as R
HERE=Path(__file__).parent/'yam_assets'
def build(episode,output):
    pl=json.loads((Path(episode)/'yam/yam_placement.json').read_text())['bases'];robot=ET.Element('robot',name='yam_sharpa_bimanual');ET.SubElement(robot,'link',name='world')
    for side,p in (('left','L'),('right','R')):
        r=ET.parse(HERE/f'yam_sharpa_{side}_nogripper.urdf').getroot();arm={l.get('name') for l in r.findall('link') if not l.get('name').startswith(side)};ren=lambda n:f'{p}_{n}' if n in arm else n;children={j.find('child').get('link') for j in r.findall('joint')};root=[l.get('name') for l in r.findall('link') if l.get('name') not in children][0]
        for l in r.findall('link'):l.set('name',ren(l.get('name')));robot.append(l)
        for j in r.findall('joint'):
            n=j.get('name')
            if n.startswith('joint'):j.set('name',f'{p}_arm_j{n[-1]}')
            elif j.find('child').get('link')==f'{side}_hand_C_MC':j.set('name',f'{p}_hand_mount')
            j.find('parent').set('link',ren(j.find('parent').get('link')));j.find('child').set('link',ren(j.find('child').get('link')));robot.append(j)
        q=pl[side]['quat'];rpy=R.from_quat([q[1],q[2],q[3],q[0]]).as_euler('xyz');fix=ET.SubElement(robot,'joint',name=f'{p}_base_fixed',type='fixed');ET.SubElement(fix,'origin',xyz=' '.join(f'{v:.8f}' for v in pl[side]['pos']),rpy=' '.join(f'{v:.8f}' for v in rpy));ET.SubElement(fix,'parent',link='world');ET.SubElement(fix,'child',link=ren(root))
    # The vendor table and back profile, exactly where the MuJoCo scene puts them.
    m=mujoco.MjModel.from_binary_path(str(Path(episode)/'yam/yam_model.mjb'))
    # Only the recording's round table (in the scene's support) stands in the workspace.
    for name in ():
        g=m.geom(name);w=g.quat;rpy=R.from_quat([w[1],w[2],w[3],w[0]]).as_euler('xyz');l=ET.SubElement(robot,'link',name=name);i=ET.SubElement(l,'inertial');ET.SubElement(i,'mass',value='1');ET.SubElement(i,'inertia',ixx='1',iyy='1',izz='1',ixy='0',ixz='0',iyz='0')
        for tag in ('visual','collision'):
            e=ET.SubElement(l,tag);ET.SubElement(ET.SubElement(e,'geometry'),'box',size=' '.join(f'{2*v:.6f}' for v in g.size))
        fix=ET.SubElement(robot,'joint',name=f'{name}_fixed',type='fixed');ET.SubElement(fix,'origin',xyz=' '.join(f'{v:.8f}' for v in g.pos),rpy=' '.join(f'{v:.8f}' for v in rpy));ET.SubElement(fix,'parent',link='world');ET.SubElement(fix,'child',link=name)
    output.parent.mkdir(parents=True,exist_ok=True);ET.ElementTree(robot).write(output);return output
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('episode');a=a.parse_args();print(build(a.episode,Path(a.episode)/'yam/flash/yam_sharpa_bimanual.urdf'))
