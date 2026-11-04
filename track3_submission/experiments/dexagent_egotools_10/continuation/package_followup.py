"""Package clips, reviewable code and evidence; exclude models and temporary credentials."""
from pathlib import Path
import hashlib
import json
import zipfile
ROOT=Path(__file__).resolve().parents[1]
archive=ROOT/'dexagent_egotools_tested.zip'
with zipfile.ZipFile(ROOT/'dexagent_egotools_10.zip') as baseline:
    members={name:baseline.read(name) for name in baseline.namelist()}
allowed={'.py','.json','.jsonl','.yaml','.yml','.xml','.npz','.png','.jpg','.jpeg','.mp4','.log','.txt','.html','.patch'}
for path in (ROOT/'continuation').rglob('*'):
    if not path.is_file() or path.suffix not in allowed:continue
    relative=path.relative_to(ROOT)
    if any(part in {'transport','depth_needed','__pycache__','.search_prefixes'} for part in relative.parts):continue
    members[str(relative)]=path.read_bytes()
members['continuation/changes/README.txt']=(ROOT/'continuation/changes/README.txt').read_bytes()
members['README.txt']=b'Open continuation/report.html for the latest test results, or report.html for the original 10-task compatibility test.\nNo verified complete robot episode or robot success rate was obtained. The bundle includes source clips, code changes and failed physical diagnostics. Model checkpoints, environments, full raw depth recordings, HTTP headers and authentication are excluded.\n'
manifest={name:hashlib.sha256(data).hexdigest() for name,data in members.items()}
members['bundle_sha256.json']=json.dumps(manifest,indent=2).encode()
with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=3) as output:
    for name,data in sorted(members.items()):output.writestr(name,data)
with zipfile.ZipFile(archive) as output:
    assert output.testzip() is None
    clips=[name for name in output.namelist() if name.startswith('clips/') and name.endswith('.mp4')]
    assert len(clips)==10
print(json.dumps({'bundle':str(archive),'bytes':archive.stat().st_size,'members':len(members),'clips':len(clips)}))
