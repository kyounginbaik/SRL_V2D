"""G18 -> v29 MESH-ONLY swap (A9 streamA9_meshswap/code/stage.py pattern). Copies the v28 legacy component (never overwrites) and
replaces ONLY meshes/<episode>/wooden_spoon.obj with the G18 fix mesh (byte copy; G18 mesh is already in the component body frame).
Poses, policy, actions, the portable bundle (scene, collision parts, physics source meshes) are untouched; the portable archive is copied.
Usage: stage.py EP"""
import sys, json, shutil, hashlib
from pathlib import Path
ROOT = Path('/data1/SRL_V2D/challenge_reconstruction'); CAMP = ROOT / 'campaign_v22_20261028'; CAND = ROOT / 'campaign_v16_20261023/common_v2_candidates'
G18 = ROOT / 'reconstruction_recovery_20261026/streamG18_spoonmesh/mesh'
ep = int(sys.argv[1]); name = f'episode_{ep:06d}'
v28tag = json.loads((CAMP / 'selection_v28.json').read_text())[str(ep)]
done = json.loads((CAND / v28tag / name / 'completed.json').read_text())
assert done['exact_previous_physical_states_actions'] is True
src = Path(done['legacy_component'])
src_portable = src.parents[1] / 'portable_archives' / f'{name}_portable.tar.gz'; assert src_portable.is_file(), src_portable
fix = G18 / ('wooden_spoon_g18_sub15_16.obj' if ep in (15, 16) else 'wooden_spoon_g18.obj')
stage = CAMP / 'streamG18v29/staging' / f'ep{ep:02d}'; dst = stage / 'components' / name
assert not stage.exists(), stage
shutil.copytree(src, dst, symlinks=True)
(stage / 'portable_archives').mkdir(); shutil.copy2(src_portable, stage / 'portable_archives' / src_portable.name)
(stage / 'portable_archives/SOURCE.txt').write_text(f'copied unchanged from {src_portable} (physics untouched; G18 mesh-only swap, v29 2026-10-30)\n')
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest(); md5 = lambda p: hashlib.md5(Path(p).read_bytes()).hexdigest()
out = dst / 'meshes' / name / 'wooden_spoon.obj'; old_sha, old_md5 = sha(out), md5(out)
assert old_sha == done['meshes']['wooden_spoon']['sha256'], 'staged spoon != v28 build-v2 spoon'
out.unlink(); shutil.copyfile(fix, out); new_sha = sha(out); assert new_sha == sha(fix)
rec = dict(stream='G18v29', episode=name, v28_tag=v28tag, source_component=str(src), source_portable=str(src_portable),
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
