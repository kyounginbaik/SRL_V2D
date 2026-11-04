"""F9: apply stream A9's ep30 mesh swap to a NEW ep30 component staged from the same N8 / stream E world_T1 lineage (= D9 d9_meshswap25.py for ep30).
The swapped sweeper mesh is copied byte-identically from A9_ep30_sweeper (A9 stage.py transform, T_icp from A9 audit), after checking that the
new component's original submitted mesh is byte-identical to the one A9 swapped (same body frame). readiness.json mesh entries are updated
exactly as A9 stage.py does; the A9 record is copied with an F9 note. Portable bundle / scene / collision / poses untouched.
Finally every submitted mesh must equal A9_ep30_sweeper's file byte for byte.
usage: f9_meshswap30.py COMPONENT_DIR"""
import sys, json, shutil, hashlib, filecmp
from pathlib import Path
comp = Path(sys.argv[1]); A9 = Path('/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamA9_meshswap/staging/A9_ep30_sweeper/components/episode_000030')
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
rec = json.loads((A9 / 'a9_meshswap.json').read_text()); read = json.loads((comp / 'readiness.json').read_text()); a9read = json.loads((A9 / 'readiness.json').read_text())
out = dict(stream='F9', note='A9_ep30_sweeper mesh swap applied to this component (byte-identical copy of the A9 swapped mesh)', a9_record=str(A9 / 'a9_meshswap.json'), swapped={})
for obj, r in rec['swapped'].items():
    dst = comp / 'meshes/episode_000030' / f'{obj}.obj'
    assert sha(dst) == r['old_sha256'], (obj, sha(dst), r['old_sha256'])
    shutil.copy2(A9 / 'meshes/episode_000030' / f'{obj}.obj', dst); shutil.copy2(A9 / f'{obj}.a9_T_sub_from_v6.npy', comp / f'{obj}.a9_T_sub_from_v6.npy')
    assert sha(dst) == r['new_sha256']
    if obj in read.get('meshes', {}): read['meshes'][obj] = dict(read['meshes'][obj], **{k: a9read['meshes'][obj][k] for k in ('sha256', 'a9_mesh_swap')})
    out['swapped'][obj] = dict(old_sha256=r['old_sha256'], new_sha256=sha(dst))
names = sorted(p.name for p in (A9 / 'meshes/episode_000030').iterdir())
assert names == sorted(p.name for p in (comp / 'meshes/episode_000030').iterdir()), 'mesh set differs from A9'
for n in names: assert filecmp.cmp(A9 / 'meshes/episode_000030' / n, comp / 'meshes/episode_000030' / n, shallow=False), n
out['all_meshes_byte_identical_to_A9_ep30_sweeper'] = {n: sha(comp / 'meshes/episode_000030' / n) for n in names}
(comp / 'readiness.json').write_text(json.dumps(read, indent=2) + '\n')
shutil.copy2(A9 / 'a9_meshswap.json', comp / 'a9_meshswap.json'); (comp / 'f9_meshswap.json').write_text(json.dumps(out, indent=1) + '\n')
print(json.dumps(out))
