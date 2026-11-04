"""Make export TAG_OUT = TAG_IN poses + refined meshes from out/mf_<mode>/<ep>/<obj>.obj (GT-free meshfit)."""
import sys, json, shutil, kitx
from pathlib import Path
tin, tout, mode = sys.argv[1:4]; only = sys.argv[4].split(',') if len(sys.argv) > 4 else None
for e in kitx.TUNE + kitx.HELD:
    src = kitx.root(e) / 'exports' / tin; dst = kitx.root(e) / 'exports' / tout
    if dst.exists(): continue
    dst.mkdir()
    for f in src.iterdir():
        if f.is_file() and f.name != 'public_score.json': shutil.copy2(f, dst / f.name)
    mj = json.loads((src / 'meshes.json').read_text()) if (src / 'meshes.json').exists() else {}
    names = json.loads((kitx.root(e) / 'prepared.json').read_text())['record']['names']
    for n in names:
        mj.setdefault(n, str(kitx.root(e) / 'objects' / n / 'mesh.obj'))
        p = Path(f'/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamG15/out/mf_{mode}/{e}/{n}.obj')
        if p.exists() and (only is None or n in only): mj[n] = str(p)
    (dst / 'meshes.json').write_text(json.dumps(mj, indent=1))
    print(e, tout, mj)
