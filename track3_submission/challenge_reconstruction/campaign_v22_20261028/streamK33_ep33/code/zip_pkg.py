"""E31: zip a package folder (python zipfile, deflate, __pycache__ excluded; never overwrites). usage: zip_pkg.py SRL_ROOT PKG"""
import sys, zipfile, hashlib
from pathlib import Path
root, pkg = Path(sys.argv[1]), sys.argv[2]; out = root / f'{pkg}.zip'; assert not out.exists(), out
with zipfile.ZipFile(out, 'x', zipfile.ZIP_DEFLATED) as z:
    for p in sorted((root / pkg).rglob('*')):
        if p.is_file() and '__pycache__' not in p.parts: z.write(p, p.relative_to(root))
h = hashlib.sha256(out.read_bytes()).hexdigest(); print(out, out.stat().st_size, h)
with zipfile.ZipFile(out) as z: assert z.testzip() is None; print('entries', len(z.namelist()))
