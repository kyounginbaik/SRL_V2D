"""E31: G18 mesh-only swap applied to a NEW E31 staged component (the G18v29 stage.py rule, in place on the E31 staging dir that the
unchanged stager just wrote; nothing outside it is touched). Replaces meshes/<episode>/wooden_spoon.obj with the G18 fix mesh (same body
frame) and records it; then asserts that every submitted mesh is byte-identical to the v29 build-v2 component's mesh.
usage: g18swap.py EP COMPONENT_DIR"""
import sys, json, shutil, hashlib
from pathlib import Path
ROOT = Path('/data1/SRL_V2D/challenge_reconstruction'); CAMP = ROOT / 'campaign_v22_20261028'; CAND = ROOT / 'campaign_v16_20261023/common_v2_candidates'
G18 = ROOT / 'reconstruction_recovery_20261026/streamG18_spoonmesh/mesh'
ep = int(sys.argv[1]); dst = Path(sys.argv[2]); name = f'episode_{ep:06d}'
assert 'teacher33/staging' in str(dst), dst
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest(); md5 = lambda p: hashlib.md5(Path(p).read_bytes()).hexdigest()
v29 = CAND / json.loads((CAMP / 'selection_v29.json').read_text())[str(ep)] / name / 'meshes' / name
mdir = dst / 'meshes' / name; rec = dict(stream='E31', episode=name, v29_meshes=str(v29))
if (mdir / 'wooden_spoon.obj').is_file() and not (dst / 'g18_meshswap.json').exists():
    fix = G18 / ('wooden_spoon_g18_sub15_16.obj' if ep in (15, 16) else 'wooden_spoon_g18.obj')
    out = mdir / 'wooden_spoon.obj'; old_sha, old_md5 = sha(out), md5(out)
    out.unlink(); shutil.copyfile(fix, out); new_sha = sha(out); assert new_sha == sha(fix)
    rec['swapped'] = dict(wooden_spoon=dict(old_sha256=old_sha, old_md5=old_md5, g18_source=str(fix), new_sha256=new_sha, new_md5=md5(out)))
    read = json.loads((dst / 'readiness.json').read_text())
    read['meshes']['wooden_spoon'].update(sha256=new_sha, g18_mesh_swap=dict(previous_sha256=old_sha, source=str(fix), transform='identity (G18 mesh built in this body frame)'))
    (dst / 'readiness.json').write_text(json.dumps(read, indent=2) + '\n')
    st = json.loads((dst / 'staged.json').read_text())
    if 'wooden_spoon' in st.get('meshes_sha256', {}):
        st['meshes_sha256']['wooden_spoon'] = new_sha; (dst / 'staged.json').write_text(json.dumps(st, indent=2) + '\n')
    (dst / 'g18_meshswap.json').write_text(json.dumps(rec, indent=1) + '\n')
mine = sorted(p.name for p in mdir.glob('*.obj')); ref = sorted(p.name for p in v29.glob('*.obj'))
assert mine == ref, (mine, ref)
bad = [n for n in mine if sha(mdir / n) != sha(v29 / n)]
assert not bad, f'meshes differ from v29: {bad}'
print('G18SWAP_OK', json.dumps(rec), 'all meshes == v29', mine)
