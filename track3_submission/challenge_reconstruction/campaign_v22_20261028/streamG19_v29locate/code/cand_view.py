"""List candidate components per episode with outcomes vs each truth (K_ tables), build-v2 exact flag and portable archive."""
import json, os, sys
from pathlib import Path
S = '/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamG19_v29locate'
CD = Path('/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/common_v2_candidates'); OV = Path('/home/exaFLOPs/v22_overflow_20261028/common_v2_candidates')
def status(tag, ep):
    n = f'episode_{ep:06d}'; d = CD / tag / n if (CD / tag / n).is_dir() else OV / tag / n
    try: c = json.load(open(d / 'completed.json'))
    except Exception: return 'nocompl', None
    ok = c.get('exact_previous_physical_states_actions') is True
    leg = Path(c.get('legacy_component', '/x/y/z')); por = leg.parents[1] / 'portable_archives' / f'{n}_portable.tar.gz'
    return ('exact' if ok else 'NOTexact') + ('+port' if por.is_file() else '-noport'), str(d)
for ep in map(int, sys.argv[1:]):
    T = {l: json.load(open(f'{S}/out/tab/K_{l}/ep{ep}.json')) for l in ('scanv14', 'c21', 'c27', 'fro')}
    print(f'=== ep{ep}')
    for tag in sorted(T['scanv14']):
        if 'mp' not in T['scanv14'][tag]: continue
        st, _ = status(tag, ep)
        row = ' | '.join(f"{l}:{T[l][tag]['sp']}{T[l][tag]['mp']} " + ','.join(f'{a:.1f}/{b:.0f}' for a, b in T[l][tag]['obj']) if 'mp' in T[l].get(tag, {}) else f'{l}:NA' for l in T)
        print(f'{tag:28s} {st:14s} {row}')
