"""zip PKG_DIR (e.g. challenge_submissions_SRL/track3_v30c8_20261101) to PKG_DIR.zip (paths relative to PKG_DIR's parent, no __pycache__/*.pyc); never overwrites.
usage: mkzip.py PKG_DIR"""
import sys, zipfile, hashlib
from pathlib import Path
d = Path(sys.argv[1]).resolve(); z = d.with_suffix('.zip'); assert d.is_dir() and not z.exists(), z
with zipfile.ZipFile(z, 'x', zipfile.ZIP_DEFLATED, allowZip64=True) as f:
    for p in sorted(d.rglob('*')):
        if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc':
            f.write(p, p.relative_to(d.parent))
h = hashlib.sha256(z.read_bytes()).hexdigest(); print(z, z.stat().st_size, h)
