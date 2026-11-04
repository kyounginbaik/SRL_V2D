import trimesh, numpy as np, sys
sys.path.insert(0, 'compose_v2d')
from common import gt_mesh_path
TEAM = 'data/drive_results/V2D/extracted/public_meshes_20261008/selected_public'
for name, path in [('team blue_cup', f'{TEAM}/blue_cup/blue_cup.obj'), ('GT blue_cup', gt_mesh_path('blue_cup')),
                   ('team beige_cup', f'{TEAM}/beige_cup/beige_cup.obj'), ('GT beige_cup', gt_mesh_path('beige_cup'))]:
    m = trimesh.load(path, force='mesh'); hull = m.convex_hull
    ext = np.ptp(m.vertices, 0)
    vh = m.volume / hull.volume if m.is_watertight else float('nan')
    print(f'{name:16s} V={len(m.vertices):6d} F={len(m.faces):6d} watertight={m.is_watertight} vol/hull={vh:.2f} extents={np.round(ext / ext.max(), 2)}')
    c = m.vertices.mean(0); w, U = np.linalg.eigh(np.cov((m.vertices - c).T))
    gaps = [abs(w[i] - w[(i + 1) % 3]) + abs(w[i] - w[(i + 2) % 3]) - abs(w[(i + 1) % 3] - w[(i + 2) % 3]) for i in range(3)]
    ax = U[:, int(np.argmax(gaps))]
    L = np.abs((m.vertices - c) @ ax).max(); ts = np.linspace(-0.8 * L, 0.8 * L, 9); pts = c + np.outer(ts, ax)
    try:
        inside = m.contains(pts); print('   on-axis points inside the mesh (solid blob = all 1; hollow cup = 0 along the cavity):', inside.astype(int))
    except Exception as e:
        print('   contains failed:', e)
