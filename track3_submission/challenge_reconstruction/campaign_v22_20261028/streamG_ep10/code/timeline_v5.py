"""Per-frame aligned error vs a target (mirror of public_kit_score.aligned_errors, but per frame), plus declared (unaligned)
error.  usage: timeline_v5.py EP SUBMISSION_PARQUET RESULTS_PARQUET [TARGET_DIR] [BIN] [OUT_NPZ]"""
import sys, numpy as np, pandas as pd, trimesh
sys.path.insert(0, '/data1/SRL_V2D/challenge_submissions_SRL/analysis_20261020/scripts')
sys.path.insert(0, '/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/compose_target')
import public_kit_score as K
M = K.load_metric('MP-SR')
ep = int(sys.argv[1]); SUB = sys.argv[2]; RES = sys.argv[3]
T = sys.argv[4] if len(sys.argv) > 4 and sys.argv[4] else '/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/target_final_v5'
B = int(sys.argv[5]) if len(sys.argv) > 5 else 15; OUT = sys.argv[6] if len(sys.argv) > 6 else None
def ref_geometry(names, paths):
    g = dict(names=np.array(names))
    for i, (n, p) in enumerate(zip(names, paths)):
        m = trimesh.load(p, force='mesh'); m.merge_vertices()
        if len(m.faces) > 40000:
            try: m = m.simplify_quadric_decimation(face_count=40000)
            except Exception: pass
        m.remove_unreferenced_vertices(); v = np.asarray(m.vertices, np.float64); f = np.asarray(m.faces, np.int64)
        a, _ = M._surface_moments(v, f); ax, o = K.symmetry_axis(n, v)
        g.update({f'v{i}': v, f'f{i}': f, f'anchor{i}': a, f'points{i}': M._sample_mesh_surface(v, f, 10000, 0), f'axis{i}': ax, f'order{i}': np.int64(o)})
    return g
r = pd.read_parquet(RES); order = r.drop_duplicates('object_id').sort_values('object_id')[['object_id', 'object_body_name']].values.tolist()
names = [n for _, n in order]; d = f'{T}/episode_{ep:06d}'
tracks = [np.load(f'{d}/{n}_trajectory.npz')['T_world_obj'] for n in names]
sub = pd.read_parquet(SUB); sub = sub[sub.row_id.str.contains(f'/e{ep:06d}/')]
rows = []
for _, s in sub.iterrows():
    f = int(s.row_id.split('/f')[1][:6]); slot = int(s.row_id.split('/o')[1]); Tm = tracks[slot][min(f, len(tracks[slot]) - 1)]; c = K.pose_cols(Tm[None])
    rows.append(dict(row_id=s.row_id, episode_index=ep, world_index=0, frame_index=f, object_slot=slot, object_name=names[slot], **{'ref_' + k: float(v[0]) for k, v in c.items()}, reference_geometry_b64=''))
sol = pd.DataFrame(rows); geometry = ref_geometry(names, [f'{d}/{n}_mesh_metric.obj' for n in names])
merged = sol.merge(sub[['row_id', *M._REQUIRED_SUBMISSION_COLUMNS, 'object_mesh_b64']], on='row_id')
ach, ref, _, nm = M._episode_tensors(merged)
ach = M._normalise_quaternions(ach, 's', ValueError).copy(); ref = M._normalise_quaternions(ref, 's', ValueError).copy()
axes, orders = [], []
for slot, name in enumerate(nm):
    rr = merged[merged.object_slot == slot].sort_values(['world_index', 'frame_index']); enc = rr.object_mesh_b64.fillna('').astype(str)
    cand = M._decode_object_mesh(enc[enc.str.strip() != ''].iloc[0]); i = slot
    anchor = geometry[f'anchor{i}']; ca, rot = M._register_object_body(cand, (geometry[f'v{i}'], geometry[f'f{i}']), anchor, 1_000_003 * ep + 101 * slot)
    ach[:, :, slot] = M._convert_object_body(ach[:, :, slot], ca, rot)
    ref[:, :, slot, :3] += np.einsum('...ij,j->...i', M._rotation_matrix(ref[:, :, slot, 3:]), anchor)
    axes.append(geometry[f'axis{i}']); orders.append(int(geometry[f'order{i}']))
decl = ach.copy(); al = M._align_registered_scene(ach, ref, axes)
res = {}
for lab, A in (('aligned', al), ('declared', decl)):
    p = np.linalg.norm(A[..., :3] - ref[..., :3], axis=-1)[:, 0] * 100; o = np.degrees(M._symmetry_orientation_errors(A, ref, axes, orders))[:, 0]
    res[lab] = (p, o)
    print(f'== {lab}: ' + ' '.join(f'{n}:{p[:, i].mean():.2f}/{o[:, i].mean():.1f}' for i, n in enumerate(nm)))
p, o = res['aligned']; pd_, od = res['declared']
dv = (al[..., :3] - ref[..., :3])[:, 0] * 100
for s in range(0, len(p), B):
    e = min(len(p), s + B)
    print(f'f{s:3d}-{e-1:3d} ' + ' | '.join(f'{n[:9]} {p[s:e, i].mean():5.2f} {o[s:e, i].mean():5.1f} d({dv[s:e,i,0].mean():+5.1f},{dv[s:e,i,1].mean():+5.1f},{dv[s:e,i,2].mean():+5.1f}) z{ref[s:e,0,i,2].mean()*100:6.1f}' for i, n in enumerate(nm)))
if OUT: np.savez(OUT, names=np.array(nm), pos=p, ori=o, dpos=dv, ach=al, ref=ref, decl=decl)
