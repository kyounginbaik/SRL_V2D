"""Stream X8 (ep10): new world = N8 world_up (7.6 deg up-rotated target_final_v5 reference) with THINNER SIM-ONLY collision
proxies for the two cups (thinlib.py: blue inner wall R_in' = R_out - T_BLUE (rim lip removed), beige wall shifted inward by
D_BEIGE).  Visual meshes (= submitted meshes, final_recon 57bb269a.. / 2c041e69..) are untouched and still referenced from
stream L world_v1.  Writes only new paths:
  OUT/source_assets/{cup}_convex/part_*.obj, {cup}.urdf (mass 0.1 kg, COM/inertia from the new parts), collision_x8.json
  OUT/episode_target/episode_000010  = copy of world_up episode dir, input_manifest collision entries -> new parts
usage: build_world_x8.py OUT T_BLUE_MM D_BEIGE_MM"""
import sys, json, shutil, numpy as np
from pathlib import Path
import xml.etree.ElementTree as ET
sys.path.insert(0, str(Path(__file__).parent)); import thinlib as T
OUT = Path(sys.argv[1]).resolve(); TBm, DBm = float(sys.argv[2]), float(sys.argv[3])
WUP = Path('/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamN8_1_8_10_14_20_30/ep10/world_up')
LW = Path('/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamL_ep10/world_v1')
if OUT.exists(): raise SystemExit('refusing to overwrite ' + str(OUT))
SA = OUT / 'source_assets'; SA.mkdir(parents=True)
rep = dict(script=str(Path(__file__).resolve()), script_sha256=T.sha(__file__), thinlib_sha256=T.sha(Path(__file__).parent / 'thinlib.py'),
           t_blue_mm=TBm, d_beige_mm=DBm, base_collision=str(LW / 'source_assets'), objects={},
           rule='blue_cup: R_in = R_out - t_blue at every band edge (rim lip removed, R_out unchanged); beige_cup: R_in and R_out both - d_beige; '
                'same profile/sectors/push/base disk as stream L build_recon_world.py; mass 0.1 kg kept, COM and inertia recomputed from the parts; visual meshes unchanged',
           no_gt=True)
for n in ('blue_cup', 'beige_cup'):
    pr = T.profile(n); R_in, R_out = pr['R_in'].copy(), pr['R_out'].copy()
    if n == 'blue_cup': R_in2, R_out2 = R_out - TBm / 1000, R_out.copy()
    else: R_in2, R_out2 = R_in - DBm / 1000, R_out - DBm / 1000
    parts = T.parts_from(pr, R_in2, R_out2); cd = SA / f'{n}_convex'; cd.mkdir()
    for k, p in enumerate(parts): p.export(cd / f'part_{k:03d}.obj')
    shas = [T.sha(cd / f'part_{k:03d}.obj') for k in range(len(parts))]
    import hashlib; cat = hashlib.sha256(b''.join((cd / f'part_{k:03d}.obj').read_bytes() for k in range(len(parts)))).hexdigest()
    old_cat = hashlib.sha256(b''.join((LW / 'source_assets' / f'{n}_convex' / f'part_{k:03d}.obj').read_bytes() for k in range(len(parts)))).hexdigest()
    com, I = T.inertia(parts)
    tr = ET.parse(LW / 'source_assets' / f'{n}.urdf'); link = tr.getroot().find('link'); inert = link.find('inertial')
    assert inert.find('mass').get('value') == '0.1'
    inert.find('origin').set('xyz', ' '.join(repr(float(x)) for x in com))
    for kk, ax in zip(('ixx', 'ixy', 'ixz', 'iyy', 'iyz', 'izz'), ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))): inert.find('inertia').set(kk, repr(float(I[ax])))
    tr.write(SA / f'{n}.urdf', encoding='utf-8', xml_declaration=True)
    shutil.copy2(LW / 'source_assets' / f'{n}_visual.obj', SA / f'{n}_visual.obj'); assert T.sha(SA / f'{n}_visual.obj') == T.EXPECTED[n]
    zb = (pr['edges'] - pr['h0']) * 100
    rep['objects'][n] = dict(visual_sha256=T.EXPECTED[n], parts=len(parts), parts_concat_sha256=cat, stream_L_parts_concat_sha256=old_cat,
        part_sha256=shas, urdf=str(SA / f'{n}.urdf'), urdf_sha256=T.sha(SA / f'{n}.urdf'), mass_kg=0.1, com_body=com.tolist(), inertia_body=I.tolist(),
        band_edge_height_cm=np.round(zb, 2).tolist(),
        visual_raw_inner_radius_cm=np.round(pr['raw_in'] * 100, 3).tolist(), visual_outer_radius_cm=np.round(pr['R_out'] * 100, 3).tolist(),
        L_inner_radius_cm=np.round(R_in * 100, 3).tolist(), L_outer_radius_cm=np.round(R_out * 100, 3).tolist(),
        X8_inner_radius_cm=np.round(R_in2 * 100, 3).tolist(), X8_outer_radius_cm=np.round(R_out2 * 100, 3).tolist(),
        delta_inner_mm=np.round((R_in2 - R_in) * 1000, 2).tolist(), delta_outer_mm=np.round((R_out2 - R_out) * 1000, 2).tolist())
    print(n, 'concat', cat[:16], 'dIn mm', rep['objects'][n]['delta_inner_mm'], 'dOut mm', rep['objects'][n]['delta_outer_mm'], flush=True)
# episode dir copy with the manifest pointing at the new collision parts
E = OUT / 'episode_target' / 'episode_000010'; shutil.copytree(WUP / 'episode_target' / 'episode_000010', E, symlinks=True)
man = json.loads((E / 'input_manifest.json').read_text())
for o in man['objects']:
    nm = o['id']; vis = [x for x in o['meshes'] if x['kind'] == 'visual']; assert len(vis) == 1 and vis[0]['sha256'] == T.EXPECTED[nm]
    o['source_urdf'] = str(SA / f'{nm}.urdf'); o['source_urdf_sha256'] = T.sha(SA / f'{nm}.urdf')
    o['meshes'] = vis + [dict(source=str(p), sha256=T.sha(p), kind='collision') for p in sorted((SA / f'{nm}_convex').glob('part_*.obj'))]
man['x8_thinner_collision'] = dict(record=str(OUT / 'collision_x8.json'), note='X8: sim-only thinner cup collision proxies (blue lip removed / thinner wall, beige wall shifted inward); visual = submitted meshes unchanged; reference/targets unchanged (world_up)', base_episode_dir=str(WUP / 'episode_target' / 'episode_000010'))
(E / 'input_manifest.json').write_text(json.dumps(man, indent=2) + '\n')
rep['episode_dir'] = str(E); rep['base_episode_dir'] = str(WUP / 'episode_target' / 'episode_000010')
(OUT / 'collision_x8.json').write_text(json.dumps(rep, indent=1)); print('done', OUT)
