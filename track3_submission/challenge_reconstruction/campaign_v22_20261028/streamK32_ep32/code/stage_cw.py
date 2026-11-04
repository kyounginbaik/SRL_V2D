"""E31 cwmesh: MESH-ONLY copy of a selected build-v2 component (G18v29 stage.py pattern): copy the tag's legacy component + its portable
archive into streamK32_ep32/staging_cw/<CWTAG> (never overwrites), replace every meshes/<episode>/<object>.obj by the cwmesh_v5 file
(coworker bundle mesh, byte-exact or rigidly re-expressed in the component body frame, see cwmesh_v5/cwmesh_plan.json), update the
readiness / staged mesh sha256 records. Poses, policy, actions, scene and physics meshes are untouched.
usage: stage_cw.py EP TAG CWTAG"""
import sys, json, shutil, hashlib
from pathlib import Path
ROOT = Path('/data1/SRL_V2D/challenge_reconstruction'); CAND = ROOT / 'campaign_v16_20261023/common_v2_candidates'
E31 = ROOT / 'campaign_v22_20261028/streamK32_ep32'; CW = E31 / 'cwmesh_v5'
ep, tag, cwtag = int(sys.argv[1]), sys.argv[2], sys.argv[3]; name = f'episode_{ep:06d}'
done = json.loads((CAND / tag / name / 'completed.json').read_text()); assert done['exact_previous_physical_states_actions'] is True
src = Path(done['legacy_component']); src_portable = src.parents[1] / 'portable_archives' / f'{name}_portable.tar.gz'; assert src_portable.is_file(), src_portable
stage = E31 / 'staging_cw' / cwtag; dst = stage / 'components' / name; assert not stage.exists(), stage
shutil.copytree(src, dst, symlinks=True)
(stage / 'portable_archives').mkdir(); shutil.copy2(src_portable, stage / 'portable_archives' / src_portable.name)
(stage / 'portable_archives/SOURCE.txt').write_text(f'copied unchanged from {src_portable} (physics untouched; E31 cwmesh mesh-only swap 2026-11-01)\n')
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
plan = json.loads((CW / 'cwmesh_plan.json').read_text()); mdir = dst / 'meshes' / name; rec = dict(stream='E31 cwmesh', episode=name, source_tag=tag, source_component=str(src), swapped={})
names = sorted(p.stem for p in mdir.glob('*.obj')); assert names == sorted(k.split(':')[1] for k in plan if int(k.split(':')[0]) == ep), names
read = json.loads((dst / 'readiness.json').read_text()); st = json.loads((dst / 'staged.json').read_text())
for n in names:
    out = mdir / f'{n}.obj'; old = sha(out); new = CW / name / f'{n}.obj'; p = plan[f'{ep}:{n}']
    out.unlink(); shutil.copyfile(new, out); assert sha(out) == p['shipped_sha256']
    rec['swapped'][n] = dict(old_sha256=old, new_sha256=sha(out), mode=p['mode'], bundle_sha256=p['bundle_sha256'], rot_deg=p['rot_deg'], trans_mm=p['trans_mm'])
    if n in read.get('meshes', {}):
        read['meshes'][n].update(sha256=sha(out), e31_cwmesh_swap=dict(previous_sha256=old, mode=p['mode'], source=str(new)))
    if n in st.get('meshes_sha256', {}): st['meshes_sha256'][n] = sha(out)
(dst / 'readiness.json').write_text(json.dumps(read, indent=2) + '\n'); (dst / 'staged.json').write_text(json.dumps(st, indent=2) + '\n')
(dst / 'e31_cwmesh_swap.json').write_text(json.dumps(rec, indent=1) + '\n'); print(json.dumps(rec, indent=1))
