"""G19: kit-faithful per-episode outcomes (SP fail, MP fail) + per-object time-mean errors of every uploaded package (and optional
extra components) against one candidate truth. Uses the unchanged G13 kit wrapper (streamG13/code/kit.py -> public_kit_score).
usage: tab19.py LABEL SRC EP [PKG ...]
  SRC: target:<dir> | bundle:<dir> | fro
  PKG: names from tracks.PKG + v29, or name=<package dir with results/ and meshes/> (default: all 9 uploaded packages)
writes out/tab/<LABEL>/ep<EP>.json  {pkg: {sp, mp, obj: [[pos_cm, ori_deg], ...]}}"""
import sys, os, json, numpy as np
sys.path.insert(0, '/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamG13/code')
import tracks as T, kit
T.PKG['v29'] = '/data1/SRL_V2D/challenge_submissions_SRL/track3_v29_20261030/all_20_episodes'
S = '/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamG19_v29locate'
PK = ['first', 'pf6', 'v13', 'v21C', 'v22b', 'v24', 'v25', 'v26', 'v29']
lab, src, ep = sys.argv[1], sys.argv[2], int(sys.argv[3])
pk = sys.argv[4:] or PK


def truth(src, ep):
    if src == 'fro': return T.bundle(T.FRO, ep) if ep in (14, 15, 16, 20, 25, 28, 30) else T.bundle(f'{T.R}/final_recon', ep)
    if src.startswith('target:'): return T.target(None, ep, f'{src[7:]}/episode_{ep:06d}')
    if src.startswith('bundle:'): return T.bundle(src[7:], ep)
    raise SystemExit('bad src')


ref = truth(src, ep)
od = f'{S}/out/tab/{lab}'; os.makedirs(od, exist_ok=True); fn = f'{od}/ep{ep}.json'
out = json.load(open(fn)) if os.path.exists(fn) else {}
for p in pk:
    name, path = (p.split('=', 1) + [None])[:2] if '=' in p else (p, None)
    if name in out: continue
    try:
        ach = T.package(path or name, ep)
        pos, ori = kit.errors(ach, ref, ep); pm, om = pos.mean(0), ori.mean(0)
        sp = int(not ((pm.mean() <= 10) and (np.radians(om).mean() <= 0.5))); mp = int(not np.all((pm < 3) & (om < 30)))
        out[name] = dict(sp=sp, mp=mp, obj=[[round(float(a), 3), round(float(b), 2)] for a, b in zip(pm, om)], names=[a[0] for a in ach])
        print(lab, ep, name, sp, mp, out[name]['obj'], flush=True)
    except Exception as ex:
        print('ERR', lab, ep, name, repr(ex), flush=True); out[name] = dict(err=repr(ex))
    json.dump(out, open(fn, 'w'), indent=1)
