"""Resumable authenticated Hub downloads using checked HTTP byte ranges.

Credentials are read from the existing Hub login and passed through a private
temporary header file, never through logs or process arguments. Only official
Hub resolve URLs are used. Incomplete responses are never accepted as chunks.
"""
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import argparse
import hashlib
import json
import os
import re
import subprocess
import tempfile
import time

ROOT=Path(__file__).resolve().parent
p=argparse.ArgumentParser()
p.add_argument('kind',choices=['sam3','depth','grasp_hires'])
p.add_argument('--workers',type=int,default=8)
p.add_argument('--chunk-mb',type=int,default=4)
a=p.parse_args()
headers=[]
if a.kind=='sam3':
    from huggingface_hub import get_token,get_hf_file_metadata,hf_hub_url
    token=get_token()
    url=hf_hub_url('facebook/sam3','sam3.pt',revision='3c879f39826c281e95690f02c7821c4de09afae7')
    meta=get_hf_file_metadata(url,token=token)
    expected_hash=meta.etag.strip('"')
    assert re.fullmatch('[0-9a-f]{64}',expected_hash)
    size=meta.size
    cache=ROOT/'models/hf/hub/models--facebook--sam3'
    target=cache/'blobs'/expected_hash
    headers=['Authorization: Bearer '+token]
    revision=meta.commit_hash
else:
    uuid='02430554-623d-4a36-936e-b1ae200b43af'
    name='depth.hdf5' if a.kind=='depth' else 'stereo_left_for_rrd.mp4'
    asset=f'{uuid}/ep1/{name}'
    manifest=json.loads((ROOT/'data/manifest.json').read_text())
    ep=next(e for e in manifest['episodes'] if e['uuid']==uuid)
    entry=next(f for f in ep['files'] if f['path']==asset)
    size=entry['bytes'];expected_hash=entry.get('sha256')
    url=f'https://huggingface.co/datasets/ropedia-ai/egotools-data/resolve/c738802d9ac9bf1b79b7993592096a3dcd6391ec/raw/{asset}'
    target=ROOT/'data/raw'/asset
parts=ROOT/'downloads'/a.kind
parts.mkdir(parents=True,exist_ok=True)
chunk=a.chunk_mb*1024*1024
total=(size+chunk-1)//chunk

with tempfile.TemporaryDirectory(prefix='dexagent-hub-headers-') as private:
    auth=Path(private)/'headers'
    auth.write_text('\n'.join(headers)+'\n');auth.chmod(0o600)
    def fetch(i):
        start=i*chunk;end=min(size,(i+1)*chunk)-1
        dest=parts/f'{i:05d}.part'
        if dest.exists() and dest.stat().st_size==end-start+1:return i
        for attempt in range(12):
            temp=parts/f'{i:05d}.tmp'
            head=Path(private)/f'headers-{i}'
            cmd=['curl','--http1.1','-fLsS','--connect-timeout','6','--max-time','40',
                 '--max-filesize',str(end-start+1),'-H',f'Range: bytes={start}-{end}',
                 '-D',str(head),url+f'?download=true&chunk={i}&attempt={attempt}&ts={time.time_ns()}', '-o',str(temp)]
            if headers:cmd[1:1]=['-H','@'+str(auth)]
            result=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
            h=head.read_text(errors='replace') if head.exists() else ''
            ranges=re.findall(r'content-range:\s*bytes\s+(\d+)-(\d+)/(\d+)',h,re.I)
            if result.returncode==0 and ranges and tuple(map(int,ranges[-1]))==(start,end,size) and temp.stat().st_size==end-start+1:
                temp.replace(dest);return i
        raise RuntimeError(f'Chunk {i} failed after twelve attempts; curl exit {result.returncode}')
    done=0
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        for future in as_completed([pool.submit(fetch,i) for i in range(total)]):
            future.result();done+=1
            if done%10==0 or done==total:print(json.dumps({'kind':a.kind,'chunks':done,'total':total}),flush=True)

target.parent.mkdir(parents=True,exist_ok=True)
temporary=target.with_name(target.name+'.assembled')
digest=hashlib.sha256()
with temporary.open('wb') as out:
    for i in range(total):
        with (parts/f'{i:05d}.part').open('rb') as source:
            while data:=source.read(1024*1024):out.write(data);digest.update(data)
assert temporary.stat().st_size==size
actual=digest.hexdigest()
if expected_hash:assert actual==expected_hash,('checksum mismatch',actual,expected_hash)
temporary.replace(target)
if a.kind=='sam3':
    snapshot=cache/'snapshots'/revision
    snapshot.mkdir(parents=True,exist_ok=True)
    link=snapshot/'sam3.pt'
    if not link.exists():link.symlink_to(Path('../../blobs')/expected_hash)
    (cache/'refs').mkdir(exist_ok=True)
    (cache/'refs/main').write_text(revision)
(parts/'verified.json').write_text(json.dumps({'path':str(target),'bytes':size,'sha256':actual,'expected_sha256':expected_hash},indent=2))
print(json.dumps({'complete':a.kind,'bytes':size,'sha256':actual}),flush=True)
