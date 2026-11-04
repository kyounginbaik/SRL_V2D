"""Versioned, image-supported audit/repair of initial pot/lid registration.

Never edits final_recon. Geometry, inertias, friction and robot caps are fixed.
"""
from pathlib import Path
import json
import cv2
import mujoco
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

B = Path(__file__).resolve().parents[1]
EP = B / 'episodes/episode_000001'
OUT = B / 'refinement_20261018/episode_000001/reconstruction_repair'
SCENE = B / 'runs/pilot_v2/episode_000001/seed_44/baseline/scene_physics.xml'


def load():
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    camera = dict(np.load(EP / 'camera_trajectory.npz'))
    meshes = {name: trimesh.load(B.parent / f'final_recon/episode_000001/assets/{name}_visual.obj',
                                force='mesh', process=False)
              for name in ('white_pot', 'white_pot_lid')}
    return model, data, camera, meshes


def project(world, camera, frame=0):
    T = camera['world_from_rgb'][frame]
    points = (world - T[:3, 3]) @ T[:3, :3]
    projected = points @ camera['K'].T
    return projected[:, :2] / projected[:, 2:3]


def mask(mesh, pose, camera, shape):
    points = mesh.vertices @ Rotation.from_quat(pose[3:], scalar_first=True).as_matrix().T + pose[:3]
    uv = project(points, camera)
    result = np.zeros(shape[:2], np.uint8)
    cv2.fillPoly(result, np.rint(uv[mesh.faces]).astype(np.int32), 255)
    return result


def outline(image, mesh, pose, camera, color):
    rendered = mask(mesh, pose, camera, image.shape)
    contours, _ = cv2.findContours(rendered, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, contours, -1, color, 2, cv2.LINE_AA)
    return image


def dominant_normal(mesh, bottom=False):
    normal = mesh.face_normals.copy()
    normal[normal[:, 0] < 0] *= -1
    selected = normal[:, 0] > .85
    if bottom:
        selected &= mesh.triangles_center[:, 0] < -.03
    result = np.average(normal[selected], weights=mesh.area_faces[selected], axis=0)
    return result / np.linalg.norm(result)


def align_axis(source, target):
    cross = np.cross(source, target)
    angle = np.arctan2(np.linalg.norm(cross), np.dot(source, target))
    return Rotation.from_rotvec(cross / max(np.linalg.norm(cross), 1e-12) * angle)


def seat_candidate():
    m, d, camera, meshes = load()
    poses = {name: np.r_[d.xpos[m.body(name).id], d.xquat[m.body(name).id]] for name in meshes}
    pot_normal = Rotation.from_quat(poses['white_pot'][3:], scalar_first=True).apply(dominant_normal(meshes['white_pot'], bottom=True))
    lid_normal = Rotation.from_quat(poses['white_pot_lid'][3:], scalar_first=True).apply(dominant_normal(meshes['white_pot_lid']))
    lid_column = int(m.joint('white_pot_lid_free').qposadr[0])
    candidate = poses['white_pot_lid'].copy()
    candidate[3:] = (align_axis(lid_normal, pot_normal) * Rotation.from_quat(candidate[3:], scalar_first=True)).as_quat(scalar_first=True)
    initial = candidate.copy()
    scan = []
    for distance in np.arange(-.01, .05001, .0005):
        candidate[:3] = initial[:3] + distance * pot_normal
        d.qpos[lid_column:lid_column + 7] = candidate
        mujoco.mj_forward(m, d)
        contacts = [c for c in d.contact if set((m.body(m.geom_bodyid[c.geom1]).name,
                                                m.body(m.geom_bodyid[c.geom2]).name)) == set(meshes)]
        penetration = max([max(0., -float(c.dist)) for c in contacts] + [0.])
        scan.append(dict(distance_m=float(distance), penetration_m=penetration))
        if penetration <= .0002:
            break
    source = cv2.imread(str(OUT / 'source_000.jpg'))
    rows = []
    for title, pose in [('Original tilted registration', poses['white_pot_lid']), ('Pot-normal-aligned seat candidate', candidate)]:
        panel = source.copy()
        outline(panel, meshes['white_pot'], poses['white_pot'], camera, (255, 150, 0))
        outline(panel, meshes['white_pot_lid'], pose, camera, (0, 0, 255))
        cv2.putText(panel, title, (650, 740), cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 255, 255), 2)
        rows.append(cv2.resize(panel[700:1350, 650:1450], (800, 650)))
    cv2.imwrite(str(OUT / 'seat_candidate_overlay.jpg'), np.hstack(rows))
    report = dict(provisional=True, source_unchanged=True, pot_normal=pot_normal.tolist(), lid_normal=lid_normal.tolist(),
                  original_lid_pose=poses['white_pot_lid'].tolist(), candidate_lid_pose=candidate.tolist(),
                  translation_change_m=float(np.linalg.norm(candidate[:3] - poses['white_pot_lid'][:3])),
                  rotation_change_deg=float(np.rad2deg((Rotation.from_quat(candidate[3:], scalar_first=True) *
                    Rotation.from_quat(poses['white_pot_lid'][3:], scalar_first=True).inv()).magnitude())), scan=scan)
    (OUT / 'seat_candidate.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='scan'}, indent=2), flush=True)


def pot_plane_candidate():
    m, d, camera, meshes = load()
    poses = {name: np.r_[d.xpos[m.body(name).id], d.xquat[m.body(name).id]] for name in meshes}
    support_path = B / 'refinement_20261018/expanded_batch/source_review/episode_000001/table_support.json'
    support = json.loads(support_path.read_text())
    normal = np.asarray(support['fitted_plane_normal_source'])
    manifest = json.loads((EP / 'input_manifest.json').read_text())
    shift = np.asarray(manifest['setup_from_source'])[:3, 3]
    plane_point = shift + np.array([0., 0., support['plane_z_coefficients'][2]])
    candidate = poses['white_pot'].copy()
    original_R = Rotation.from_quat(candidate[3:], scalar_first=True)
    old_normal = original_R.apply(dominant_normal(meshes['white_pot'], bottom=True))
    candidate[3:] = (align_axis(old_normal, normal) * original_R).as_quat(scalar_first=True)
    points = meshes['white_pot'].vertices @ Rotation.from_quat(candidate[3:], scalar_first=True).as_matrix().T + candidate[:3]
    correction = -np.min((points - plane_point) @ normal)
    candidate[:3] += correction * normal
    pot_column = int(m.joint('white_pot_free').qposadr[0])
    d.qpos[pot_column:pot_column + 7] = candidate
    mujoco.mj_forward(m, d)
    contacts = [c for c in d.contact if set((m.body(m.geom_bodyid[c.geom1]).name,
                                            m.body(m.geom_bodyid[c.geom2]).name)) == set(meshes)]
    penetration = max([max(0., -float(c.dist)) for c in contacts] + [0.])
    source = cv2.imread(str(OUT / 'source_000.jpg'))
    panels = []
    for title, pose in [('Original reconstruction', poses['white_pot']), ('Pot aligned with measured stereo table', candidate)]:
        panel = source.copy()
        outline(panel, meshes['white_pot'], pose, camera, (255, 150, 0))
        outline(panel, meshes['white_pot_lid'], poses['white_pot_lid'], camera, (0, 0, 255))
        cv2.putText(panel, title, (650, 740), cv2.FONT_HERSHEY_SIMPLEX, .75, (0, 255, 255), 2)
        panels.append(cv2.resize(panel[700:1350, 650:1450], (800, 650)))
    cv2.imwrite(str(OUT / 'pot_table_candidate_overlay.jpg'), np.hstack(panels))
    report = dict(provisional=True, original_pot_pose=poses['white_pot'].tolist(), candidate_pot_pose=candidate.tolist(),
                  unchanged_lid_pose=poses['white_pot_lid'].tolist(), table_normal=normal.tolist(), table_plane_point_setup=plane_point.tolist(),
                  translation_change_m=float(np.linalg.norm(candidate[:3] - poses['white_pot'][:3])),
                  rotation_change_deg=float(np.rad2deg((Rotation.from_quat(candidate[3:], scalar_first=True) * original_R.inv()).magnitude())),
                  max_pot_lid_penetration_m=penetration, support_source=str(support_path))
    (OUT / 'pot_table_candidate.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


def audit():
    OUT.mkdir(exist_ok=True, parents=True)
    m, d, camera, meshes = load()
    source = cv2.imread(str(OUT / 'source_000.jpg'))
    poses = {name: np.r_[d.xpos[m.body(name).id], d.xquat[m.body(name).id]] for name in meshes}
    rows = []
    stats = []
    lid_column = int(m.joint('white_pot_lid_free').qposadr[0])
    for dz in (0., .01, .02, .03, .04, .05):
        candidate = poses['white_pot_lid'].copy(); candidate[2] += dz
        d.qpos[lid_column:lid_column + 7] = candidate
        mujoco.mj_forward(m, d)
        contacts = [c for c in d.contact if set((m.body(m.geom_bodyid[c.geom1]).name,
                                                m.body(m.geom_bodyid[c.geom2]).name)) == set(meshes)]
        penetration = max([max(0., -float(c.dist)) for c in contacts] + [0.])
        panel = source.copy()
        outline(panel, meshes['white_pot'], poses['white_pot'], camera, (255, 150, 0))
        outline(panel, meshes['white_pot_lid'], candidate, camera, (0, 0, 255))
        cv2.putText(panel, f'Lid z +{dz*100:.0f} cm; deepest overlap {penetration*1000:.1f} mm',
                    (700, 710), cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 255, 255), 2)
        rows.append(cv2.resize(panel[700:1350, 650:1450], (640, 520)))
        stats.append(dict(lid_world_z_change_m=dz, max_pot_lid_penetration_m=penetration))
    cv2.imwrite(str(OUT / 'vertical_registration_overlay.jpg'), np.vstack([np.hstack(rows[:3]), np.hstack(rows[3:])]))
    (OUT / 'vertical_collision_scan.json').write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2), flush=True)


if __name__ == '__main__':
    audit()
    seat_candidate()
    pot_plane_candidate()
