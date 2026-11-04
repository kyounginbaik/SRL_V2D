"""Stream Q phase 6: re-express board 1's BODY frame by the half turn about its normal (Ry180 = diag(-1,1,-1) in the body frame) in a
frozen scene WITHOUT changing the physical world: collision parts and visual mesh vertices transformed by Ry180, inertial pos/quat
transformed, keyframe orientation q -> q * Ry180 (origin unchanged). In frame B our slotted board-1 mesh has its slot on +z like
board 2 and like the compose board, so the kit registration onto the compose board is the near-identity (correct) one.
usage: make_scene_B.py IN_SCENE OUT_DIR VISUAL_SLOTTED_A.obj"""
import sys, re, glob, numpy as np, trimesh
from pathlib import Path
from scipy.spatial.transform import Rotation as R
scene, out, vis = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve(), Path(sys.argv[3]).resolve(); out.mkdir(parents=True, exist_ok=False)
Ry = np.diag([-1., 1., -1.]); qRy = R.from_matrix(Ry)
txt = scene.read_text()
coll_files = re.findall(r'<mesh name="wooden_piece_1_collision_\d+" file="([^"]+)"', txt); (out / 'collision').mkdir()
for f in coll_files:
    m = trimesh.load(f, force='mesh', process=False); m.vertices = m.vertices @ Ry.T; m.faces = m.faces  # det(Ry)=+1: orientation kept
    nf = out / 'collision' / Path(f).name; m.export(nf); txt = txt.replace(f'file="{f}"', f'file="{nf}"')
vf = re.search(r'<mesh name="wooden_piece_1_visual_0" file="([^"]+)"', txt).group(1)
mv = trimesh.load(vis, force='mesh', process=False); mv.vertices = mv.vertices @ Ry.T; nv = out / 'wooden_piece_1_visual_slotted_B.obj'; mv.export(nv); txt = txt.replace(f'file="{vf}"', f'file="{nv}"')
mi = re.search(r'(<body name="wooden_piece_1" pos="[^"]*" quat=")([^"]*)(">\s*<inertial pos=")([^"]*)(" quat=")([^"]*)(")', txt)
bq = np.array(mi.group(2).split(), float); ip = np.array(mi.group(4).split(), float); iq = np.array(mi.group(6).split(), float)
nbq = (R.from_quat(bq, scalar_first=True) * qRy).as_quat(scalar_first=True); nip = Ry @ ip; niq = (qRy * R.from_quat(iq, scalar_first=True)).as_quat(scalar_first=True)
fmt = lambda a: ' '.join(repr(float(x)) for x in a)
txt = txt[:mi.start(2)] + fmt(nbq) + mi.group(3) + fmt(nip) + mi.group(5) + fmt(niq) + txt[mi.end(6):]
import mujoco
m0 = mujoco.MjModel.from_xml_path(str(scene)); adr = int(m0.joint('wooden_piece_1_free').qposadr[0])
mk = re.search(r'(<key name="home" qpos=")([^"]*)(")', txt); vals = mk.group(2).split()
q = np.array(vals[adr + 3:adr + 7], float); nq = (R.from_quat(q, scalar_first=True) * qRy).as_quat(scalar_first=True)
for j in range(4): vals[adr + 3 + j] = repr(float(nq[j]))
txt = txt[:mk.start(2)] + ' '.join(vals) + txt[mk.end(2):]
(out / 'scene_B.xml').write_text(txt); print(out / 'scene_B.xml', 'visual B', nv)
