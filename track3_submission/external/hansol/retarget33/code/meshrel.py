"""E31: relation of each v29 component mesh to the bundle's own mesh: vertex/face counts, and best rigid fit (Kabsch on equal vertex order, else ICP-free moment check)."""
import sys, json, hashlib, numpy as np, trimesh
from pathlib import Path
C = Path('/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028'); CA = Path('/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/common_v2_candidates'); X = Path(sys.argv[1])
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()[:8]
s29 = json.load(open(C / 'selection_v29.json')); out = {}
for e, t in s29.items():
    n = f'episode_{int(e):06d}'; mj = json.load(open(X / n / 'meshes.json'))
    for name, v in mj.items():
        b = X / n / v['file']; c = CA / t / n / 'meshes' / n / f'{name}.obj'
        if sha(b) == sha(c): continue
        mb = trimesh.load(b, force='mesh', process=False); mc = trimesh.load(c, force='mesh', process=False)
        r = dict(nv=(len(mb.vertices), len(mc.vertices)), nf=(len(mb.faces), len(mc.faces)))
        if len(mb.vertices) == len(mc.vertices):
            A = mb.vertices - mb.vertices.mean(0); B = mc.vertices - mc.vertices.mean(0); U, S, Vt = np.linalg.svd(A.T @ B); Rm = (U @ Vt).T
            res = np.linalg.norm(A @ Rm.T - B, axis=1)
            r.update(kabsch_rms_mm=float(np.sqrt((res ** 2).mean()) * 1000), rot_deg=float(np.degrees(np.arccos(np.clip((np.trace(Rm) - 1) / 2, -1, 1)))), t_mm=(mc.vertices.mean(0) - mb.vertices.mean(0)).round(4).tolist())
        r.update(extent_b=np.round(mb.extents, 4).tolist(), extent_c=np.round(mc.extents, 4).tolist(), area=(round(mb.area, 5), round(mc.area, 5)))
        out[f'{e}:{name}'] = r; print(e, name, json.dumps(r))
