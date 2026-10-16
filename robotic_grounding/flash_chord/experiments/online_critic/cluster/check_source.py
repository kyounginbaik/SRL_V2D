"""Refuse code/data drift between preparation and a queued launch."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
for name, base in (('source_manifest.json', root/'flash_chord'),
                   ('launcher_manifest.json', root), ('data_manifest.json', root)):
    records = json.loads((root/name).read_text())
    for relative, expected in records.items():
        path = base/relative
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise SystemExit(f'Missing or changed frozen file: {path}')
print('Frozen source, launcher and input checksums verified', flush=True)
