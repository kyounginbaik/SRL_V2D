TOOL = {
    "name":"construct_round_test_tube", "title":"Construct a round-bottom test tube",
    "type":"skill", "category":"Object Reconstruction", "status":"tested",
    "summary":"Build a free rigid open test tube with a closed round bottom and actual hollow contact geometry.",
    "description":"A parameterized tube with a closed spherical bottom and a polygonal annular wall. The spherical bottom extends into the cavity by one radius; the cavity is open above it. Twelve wall panels keep dexterous-hand collision costs bounded. The body origin is at the outer bottom. Parameters, positive inertial mass, cavity depth and body names are returned. This is a rigid grasp model, not a deformable or liquid model.",
    "inputs":[{"name":"out_dir","type":"path","description":"Asset output folder."},
              {"name":"name","type":"str","description":"Body and file prefix; default test_tube."},
              {"name":"radius","type":"float","description":"Outer radius in metres; default 0.006."},
              {"name":"length","type":"float","description":"Bottom-to-rim length in metres; default 0.085."},
              {"name":"wall","type":"float","description":"Wall thickness in metres; default 0.0006."},
              {"name":"mass","type":"float","description":"Total rigid mass in kilograms; default 0.006."},
              {"name":"sides","type":"int","description":"Annular contact panels, at least 8; default 12."},
              {"name":"rgba","type":"list[float]","description":"Tube colour and transparency; default pale glass."}],
    "outputs":[{"name":"built","type":"dict","description":"xml, root_body, bodies, cavity_depth_m, parameters and mass."}],
    "entrypoint":"construct_round_test_tube", "requirements":["mujoco>=3.10","numpy"],
    "tags":["tube","hollow","rigid","grasp","round bottom"]
}

from pathlib import Path
import json
import math
import re
import xml.etree.ElementTree as ET

def construct_round_test_tube(out_dir,name='test_tube',radius=0.006,length=0.085,wall=0.0006,mass=0.006,sides=12,rgba=None):
    radius,length,wall,mass=map(float,(radius,length,wall,mass))
    if not (0<wall<radius and length>4*radius and mass>0 and sides>=8 and int(sides)==sides):
        raise ValueError('Require 0 < wall < radius, length > 4 radii, positive mass and at least eight panels')
    if not re.fullmatch('[a-z][a-z0-9_]*',name):raise ValueError('name must be snake_case')
    out=Path(out_dir);out.mkdir(parents=True,exist_ok=True)
    rgba=rgba or [0.88,0.94,0.97,0.55]
    root=ET.Element('mujoco',model=name)
    ET.SubElement(root,'compiler',angle='radian')
    ET.SubElement(root,'option',timestep='0.002',integrator='implicitfast')
    world=ET.SubElement(root,'worldbody')
    body=ET.SubElement(world,'body',name=name)
    ET.SubElement(body,'freejoint',name=name+'_free')
    transverse=mass*(3*radius*radius+length*length)/12
    ET.SubElement(body,'inertial',pos=f'0 0 {length/2}',mass=str(mass),
                  diaginertia=f'{transverse} {transverse} {mass*radius*radius}')
    attrs=dict(rgba=' '.join(map(str,rgba)),friction='0.7 0.01 0.0002',condim='3')
    centre=radius-wall/2
    tangent=centre*math.tan(math.pi/int(sides))+0.00002
    for i in range(int(sides)):
        angle=2*math.pi*i/int(sides)
        ET.SubElement(body,'geom',name=f'{name}_wall_{i}',type='box',
            pos=f'{centre*math.cos(angle)} {centre*math.sin(angle)} {(length+radius)/2}',
            quat=f'{math.cos(angle/2)} 0 0 {math.sin(angle/2)}',
            size=f'{wall/2} {tangent} {(length-radius)/2}',**attrs)
    ET.SubElement(body,'geom',name=name+'_bottom',type='sphere',pos=f'0 0 {radius}',size=str(radius),**attrs)
    path=out/(name+'.xml');ET.ElementTree(root).write(path,encoding='unicode')
    params=dict(name=name,radius=radius,length=length,wall=wall,mass=mass,sides=int(sides),rgba=rgba)
    pp=out/(name+'_params.json');pp.write_text(json.dumps(params,indent=2))
    return dict(xml=str(path),params_json=str(pp),root_body=name,bodies=[name],
                mass_kg=mass,cavity_depth_m=length-2*radius,parameters=params)
