#!/usr/bin/env python3
"""Collision-only cut of the plastic_dish_rack convex parts where they interpenetrate the mini_sweeper at its frame-0 reference
pose (both meshes are our reconstructions, posed by the reconstruction; their convex decompositions overlap by up to ~17 mm with
the six-prong rack, which wedges the brush into the rack so that lifting the brush drags/lifts the rack). For every (rack part A,
sweeper part B) pair with overlapping AABBs, a separating-axis test over the face normals of A and B finds the axis of least
overlap; if they overlap by more than TOL, A is sliced by the plane that bounds B on that axis (plus MARGIN) and replaced by the
convex hull of the kept side (a part that vanishes is dropped). Repeated until no pair overlaps by more than TOL. Visual mesh,
masses, inertias, body frames and object poses are untouched; only rack collision parts change. Writes a cut report.
usage: cut_rack_collision.py COLLISION_RECON_EPISODE_DIR [--tol 0.0005] [--margin 0.0005] [--frame 0]"""
import argparse, hashlib, json, shutil, xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np, trimesh
from scipy.spatial import ConvexHull


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def hull(v):
    h = ConvexHull(v); return v[h.vertices], h.equations[:, :3], float(h.volume)


def clip_convex(v, n, off):
    """Half-space clip of a convex polytope given by its vertices: kept vertices + hull-edge/plane intersections."""
    sd = v @ n - off; keep = v[sd >= 0]
    if len(keep) == 0: return None
    pts = [keep]; simp = ConvexHull(v).simplices; edges = set()
    for t in simp:
        for i in range(3): e = tuple(sorted((t[i], t[(i + 1) % 3]))); edges.add(e)
    for i, j in edges:
        if (sd[i] >= 0) != (sd[j] >= 0): u = sd[i] / (sd[i] - sd[j]); pts.append((v[i] + u * (v[j] - v[i]))[None])
    q = np.vstack(pts)
    try:
        if ConvexHull(q).volume < 1e-9: return None
    except Exception: return None
    return q


def sat(va, na, vb, nb):
    best = (np.inf, None, None)
    for n in np.vstack([na, nb]):
        a, b = va @ n, vb @ n
        o1, o2 = a.max() - b.min(), b.max() - a.min()
        if min(o1, o2) < best[0]: best = (min(o1, o2), n, ('keep_below', b.min()) if o1 <= o2 else ('keep_above', b.max()))
    return best


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('episode_dir', type=Path); ap.add_argument('--tol', type=float, default=.0005)
    ap.add_argument('--margin', type=float, default=.0005); ap.add_argument('--frame', type=int, nargs='+', default=[0]); ap.add_argument('--poses', type=Path, help='npz with world_from_object/names to use instead of the dir object_poses.npz'); a = ap.parse_args()
    d = a.episode_dir; A = d / 'assets'; z = np.load(a.poses or (d / 'object_poses.npz')); names = [str(x) for x in z['names']]
    rfs = [np.linalg.inv(z['world_from_object'][f][names.index('plastic_dish_rack')]) @ z['world_from_object'][f][names.index('mini_sweeper')] for f in a.frame]
    def parts_of(obj):
        root = ET.parse(A / f'{obj}.urdf').getroot(); return [A / c.find('geometry/mesh').get('filename') for c in root.find('link').findall('collision')]
    sw = []
    for p in parts_of('mini_sweeper'):
        v0 = np.asarray(trimesh.load(p, force='mesh', process=False).vertices)
        for rack_from_sweeper in rfs: v = v0 @ rack_from_sweeper[:3, :3].T + rack_from_sweeper[:3, 3]; sw.append(hull(v))
    rack_paths = parts_of('plastic_dish_rack'); rack = {}
    for p in rack_paths: rack[p] = hull(np.asarray(trimesh.load(p, force='mesh', process=False).vertices))
    before = {}; log = []
    for rnd in range(6):
        changed = False
        for p, (va, na, vol) in list(rack.items()):
            if va is None: continue
            for j, (vb, nb, _) in enumerate(sw):
                if np.any(va.min(0) > vb.max(0)) or np.any(vb.min(0) > va.max(0)): continue
                ov, n, (mode, h) = sat(va, na, vb, nb)
                if ov <= a.tol: continue
                before.setdefault(p.name, dict(sha256=sha(p), volume_cm3=vol * 1e6, max_overlap_mm=0.)); before[p.name]['max_overlap_mm'] = max(before[p.name]['max_overlap_mm'], ov * 1000)
                if mode == 'keep_below': n2, off = -n, -(h - a.margin)
                else: n2, off = n, h + a.margin
                kept = clip_convex(va, n2, off)   # keep x.n2 >= off
                if kept is None:
                    rack[p] = (None, None, 0.); log.append(dict(part=p.name, sweeper_part=j, round=rnd, overlap_mm=ov * 1000, action='dropped')); changed = True; break
                rack[p] = hull(kept); va, na, vol = rack[p]
                log.append(dict(part=p.name, sweeper_part=j, round=rnd, overlap_mm=ov * 1000, axis=n.tolist(), action='sliced', volume_after_cm3=vol * 1e6)); changed = True
        if not changed: break
    cut_dir = A / 'plastic_dish_rack_convex_repaired'; orig = A / 'plastic_dish_rack_convex_repaired_uncut'; shutil.copytree(cut_dir, orig)
    tree = ET.parse(A / 'plastic_dish_rack.urdf'); link = tree.getroot().find('link'); report_parts = {}
    for c in list(link.findall('collision')):
        p = A / c.find('geometry/mesh').get('filename')
        if p.name not in before: continue
        va = rack[p][0]
        if va is None: link.remove(c); p.unlink(); report_parts[p.name] = dict(before[p.name], after='dropped'); continue
        trimesh.Trimesh(va, ConvexHull(va).simplices, process=True).export(p)
        report_parts[p.name] = dict(before[p.name], after=dict(sha256=sha(p), volume_cm3=rack[p][2] * 1e6))
    ET.indent(tree); tree.write(A / 'plastic_dish_rack.urdf', encoding='utf-8', xml_declaration=True)
    # residual overlap check
    resid = 0.
    for p, (va, na, _) in rack.items():
        if va is None: continue
        for vb, nb, _ in sw:
            if np.any(va.min(0) > vb.max(0)) or np.any(vb.min(0) > va.max(0)): continue
            resid = max(resid, sat(va, na, vb, nb)[0])
    rep = dict(schema='stream_i_rack_collision_cut_v1', episode_dir=str(d), frames=a.frame, poses=str(a.poses or (d / 'object_poses.npz')), tol_m=a.tol, margin_m=a.margin, parts_total=len(rack_paths),
               parts_changed=len(report_parts), parts_dropped=sum(1 for v in report_parts.values() if v['after'] == 'dropped'), parts=report_parts, log=log,
               residual_max_overlap_mm=resid * 1000, uncut_copy=str(orig), visual_unchanged=True, masses_inertias_unchanged=True, object_poses_unchanged=True,
               script=str(Path(__file__).resolve()), script_sha256=sha(__file__))
    (d / 'rack_collision_cut.json').write_text(json.dumps(rep, indent=2)); print(json.dumps({k: rep[k] for k in ('parts_changed', 'parts_dropped', 'residual_max_overlap_mm')}))


if __name__ == '__main__': main()
