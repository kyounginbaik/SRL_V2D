"""usage: mkjobs.py K OUT.txt -> score lines for v29 components vs target_compose_vK_on_v14 (changed episodes only)"""
import json, sys
C = '/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028'; X = C + '/streamC30_compose'
k = sys.argv[1]; comp = json.load(open(X + '/out/v29_components.json')); r = json.load(open(f'{X}/out/changes_v{k}.json'))
L = [f"{e} {comp[e]} C30_v29ep{int(e):02d}_vs_c{k} {C}/target_compose_v{k}_on_v14" for e, er in r['episodes'].items() if er['changed']]
open(sys.argv[2], 'w').write('\n'.join(L) + '\n'); print(len(L))
