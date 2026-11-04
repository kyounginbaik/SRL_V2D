"""Fetch the public recording inventory and captions, without bulk media."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import hashlib
import json
import subprocess

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)
if not (DATA/'hub_info.json').exists():
    subprocess.run(['curl','-fLsS','--max-time','60',
                    'https://huggingface.co/api/datasets/ropedia-ai/egotools-data',
                    '-o',str(DATA/'hub_info.json')],check=True)
info = json.loads((DATA / 'hub_info.json').read_text())
revision = info['sha']
if not (DATA/'manifest.json').exists():
    subprocess.run(['curl','-fLsS','--max-time','60',
                    f'https://huggingface.co/datasets/ropedia-ai/egotools-data/resolve/{revision}/raw/manifest.json',
                    '-o',str(DATA/'manifest.json')],check=True)
manifest = json.loads((DATA / 'manifest.json').read_text())

def fetch(ep):
    entry = next(f for f in ep['files'] if f['path'].endswith('/caption.json'))
    dest = DATA / 'raw' / entry['path']
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = f'https://huggingface.co/datasets/ropedia-ai/egotools-data/resolve/{revision}/raw/{entry["path"]}'
    if not dest.exists():
        subprocess.run(['curl', '-fLsS', '--retry', '2', '--max-time', '120', url, '-o', str(dest)], check=True)
    digest = hashlib.sha256(dest.read_bytes()).hexdigest()
    if digest != entry['sha256']:
        raise ValueError(f'caption checksum mismatch: {ep["uuid"]}')
    caption = json.loads(dest.read_text())
    config = caption.get('config', {})
    return {'uuid': ep['uuid'], 'frames': ep['frame_count'], 'caption': str(dest.relative_to(ROOT)),
            'main_task': config.get('Main Task', ''), 'keys': list(caption), 'sha256': digest}

rows = []
with ThreadPoolExecutor(max_workers=10) as pool:
    for future in as_completed([pool.submit(fetch, ep) for ep in manifest['episodes']]):
        row = future.result()
        rows.append(row)
        print(json.dumps({'uuid':row['uuid'], 'main_task':row['main_task']}), flush=True)
rows.sort(key=lambda r:r['uuid'])
(DATA / 'caption_inventory.json').write_text(json.dumps({'revision': revision, 'episodes': rows}, indent=2))
print(f'Verified {len(rows)} caption files at {revision}', flush=True)
