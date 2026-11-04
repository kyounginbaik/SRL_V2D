"""D9: apply stream A9's ep25 mesh swap to a NEW ep25 component staged from the same J8/E world_T1 lineage.
The swapped meshes are copied byte-identically from A9_ep25_boards (same files, same transforms: A9 stage.py, T_icp from A9 audit), after
checking that the new component's original submitted meshes are byte-identical to the ones A9 swapped (same body frames). readiness.json
mesh entries are updated exactly as A9 stage.py does; the A9 record is copied with a D9 note. Portable bundle / scene / poses untouched.
usage: d9_meshswap25.py COMPONENT_DIR"""
import sys, json, shutil, hashlib
from pathlib import Path
comp = Path(sys.argv[1]); A9 = Path('/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamA9_meshswap/staging/A9_ep25_boards/components/episode_000025')
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
rec = json.loads((A9 / 'a9_meshswap.json').read_text()); read = json.loads((comp / 'readiness.json').read_text()); a9read = json.loads((A9 / 'readiness.json').read_text())
out = dict(stream='D9', note='A9_ep25_boards mesh swap applied to this component (byte-identical copy of the A9 swapped meshes)', a9_record=str(A9 / 'a9_meshswap.json'), swapped={})
for obj, r in rec['swapped'].items():
    dst = comp / 'meshes/episode_000025' / f'{obj}.obj'
    assert sha(dst) == r['old_sha256'], (obj, sha(dst), r['old_sha256'])   # same original body frame as A9's source component
    shutil.copy2(A9 / 'meshes/episode_000025' / f'{obj}.obj', dst); shutil.copy2(A9 / f'{obj}.a9_T_sub_from_v6.npy', comp / f'{obj}.a9_T_sub_from_v6.npy')
    assert sha(dst) == r['new_sha256']
    if obj in read.get('meshes', {}): read['meshes'][obj] = dict(read['meshes'][obj], **{k: a9read['meshes'][obj][k] for k in ('sha256', 'a9_mesh_swap')})
    out['swapped'][obj] = dict(old_sha256=r['old_sha256'], new_sha256=sha(dst))
(comp / 'readiness.json').write_text(json.dumps(read, indent=2) + '\n')
shutil.copy2(A9 / 'a9_meshswap.json', comp / 'a9_meshswap.json'); (comp / 'd9_meshswap.json').write_text(json.dumps(out, indent=1) + '\n')
print(json.dumps(out))
