"""C30 copy of streamG18v29/code/stage.py: G18 MESH-ONLY swap of a freshly staged C30 component (spoon episodes), so the C30 runs
ship the same spoon mesh as v29. Copies SRC_STAGE/components/<ep> (never overwrites) and replaces ONLY meshes/<episode>/wooden_spoon.obj
with the G18 fix mesh (byte copy; same body frame). Poses, policy, actions, portable bundle untouched; the portable archive is copied.
usage: g18swap.py EP SRC_STAGE DST_STAGE"""
import sys, json, shutil, hashlib
from pathlib import Path
G18 = Path('/data1/SRL_V2D/challenge_reconstruction/reconstruction_recovery_20261026/streamG18_spoonmesh/mesh')
ep = int(sys.argv[1]); src_stage, stage = Path(sys.argv[2]), Path(sys.argv[3]); name = f'episode_{ep:06d}'
src = src_stage / 'components' / name; src_portable = src_stage / 'portable_archives' / f'{name}_portable.tar.gz'
assert src.is_dir() and src_portable.is_file(), (src, src_portable)
fix = G18 / ('wooden_spoon_g18_sub15_16.obj' if ep in (15, 16) else 'wooden_spoon_g18.obj')
dst = stage / 'components' / name
assert not stage.exists(), stage
shutil.copytree(src, dst, symlinks=True)
(stage / 'portable_archives').mkdir(); shutil.copy2(src_portable, stage / 'portable_archives' / src_portable.name)
(stage / 'portable_archives/SOURCE.txt').write_text(f'copied unchanged from {src_portable} (physics untouched; G18 mesh-only swap, C30 2026-11-01)\n')
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest(); md5 = lambda p: hashlib.md5(Path(p).read_bytes()).hexdigest()
out = dst / 'meshes' / name / 'wooden_spoon.obj'; old_sha, old_md5 = sha(out), md5(out)
out.unlink(); shutil.copyfile(fix, out); new_sha = sha(out); assert new_sha == sha(fix)
rec = dict(stream='C30 (G18v29 pattern)', episode=name, source_component=str(src), source_portable=str(src_portable),
           swapped=dict(wooden_spoon=dict(old_sha256=old_sha, old_md5=old_md5, g18_source=str(fix), new_sha256=new_sha, new_md5=md5(out))),
           unchanged={p.stem: sha(p) for p in sorted((dst / 'meshes' / name).glob('*.obj')) if p.stem != 'wooden_spoon'})
read = json.loads((dst / 'readiness.json').read_text())
read['meshes']['wooden_spoon'].update(sha256=new_sha, g18_mesh_swap=dict(previous_sha256=old_sha, source=str(fix), transform='identity (G18 mesh built in this body frame)'))
(dst / 'readiness.json').write_text(json.dumps(read, indent=2) + '\n')
st = json.loads((dst / 'staged.json').read_text())
if 'wooden_spoon' in st.get('meshes_sha256', {}):
    st['meshes_sha256']['wooden_spoon'] = new_sha; (dst / 'staged.json').write_text(json.dumps(st, indent=2) + '\n')
(dst / 'g18_meshswap.json').write_text(json.dumps(rec, indent=1) + '\n')
print(json.dumps(rec, indent=1))
