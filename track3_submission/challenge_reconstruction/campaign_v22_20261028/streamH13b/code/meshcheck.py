"""H13b: submitted meshes of a new component (staged + build-v2) vs the v27 component (staged + build-v2): sha256 must match; portable archive present.
usage: meshcheck.py EP NEW_STAGING_BASE NEW_TAG V27_STAGING_BASE V27_TAG"""
import sys, hashlib, json
from pathlib import Path
B = Path('/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/common_v2_candidates')
ep, nb, nt, ob, ot = int(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], Path(sys.argv[4]), sys.argv[5]; name = f'episode_{ep:06d}'
def shas(d): return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(d).rglob('*.obj'))}
new_s, old_s = shas(nb / 'components' / name / 'meshes'), shas(ob / 'components' / name / 'meshes')
new_b, old_b = shas(B / nt / name / 'meshes'), shas(B / ot / name / 'meshes')
arc = nb / 'portable_archives' / f'{name}_portable.tar.gz'
r = dict(ep=ep, staged_equal=new_s == old_s, build_equal=new_b == old_b, staged_equals_build=new_s == new_b, meshes={k: v[:8] for k, v in new_s.items()},
         archive=arc.exists() and hashlib.sha256(arc.read_bytes()).hexdigest()[:12],
         exact=json.loads((B / nt / name / 'completed.json').read_text())['exact_previous_physical_states_actions'])
print(json.dumps(r))
