"""Read selected public depth frames through verified, cached HTTP ranges."""
from pathlib import Path
import argparse
import hashlib
import io
import json
import re
import subprocess
import time
import h5py
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
REV='c738802d9ac9bf1b79b7993592096a3dcd6391ec'
class RecordedDepth(io.RawIOBase):
    chunk=1024*1024
    def __init__(self,uuid):
        super().__init__();self.position=0;self.receipts=[]
        self.asset=f'{uuid}/ep1/depth.hdf5'
        manifest=json.loads((ROOT/'data/manifest.json').read_text())
        episode=next(e for e in manifest['episodes'] if e['uuid']==uuid)
        self.size=next(f['bytes'] for f in episode['files'] if f['path']==self.asset)
        self.cache=ROOT/'downloads/selected_depth'/uuid;self.cache.mkdir(parents=True,exist_ok=True)
        self.url=f'https://huggingface.co/datasets/ropedia-ai/egotools-data/resolve/{REV}/raw/{self.asset}'
    def readable(self):return True
    def seekable(self):return True
    def tell(self):return self.position
    def seek(self,offset,whence=0):
        self.position=offset if whence==0 else self.position+offset if whence==1 else self.size+offset
        return self.position
    def readinto(self,buffer):
        if self.position>=self.size:return 0
        needed=min(len(buffer),self.size-self.position);done=0
        while done<needed:
            index=self.position//self.chunk
            start=index*self.chunk;end=min(self.size,start+self.chunk)-1
            part=self.cache/f'{index:05d}.part'
            if not part.exists():
                temporary=part.with_suffix('.tmp');headers=part.with_suffix('.headers')
                for attempt in range(6):
                    result=subprocess.run(['curl','--http1.1','-fLsS','--connect-timeout','15','--max-time','60',
                        '--max-filesize',str(end-start+1),'-H',f'Range: bytes={start}-{end}',
                        '-D',str(headers),self.url+f'?download=true&chunk={index}&ts={time.time_ns()}',
                        '-o',str(temporary)],capture_output=True)
                    ranges=re.findall(r'content-range:\s*bytes\s+(\d+)-(\d+)/(\d+)',headers.read_text(errors='replace') if headers.exists() else '',re.I)
                    if result.returncode==0 and ranges and tuple(map(int,ranges[-1]))==(start,end,self.size) and temporary.stat().st_size==end-start+1:
                        temporary.replace(part);break
                else:raise OSError(f'Cannot fetch source range {start}-{end}')
                print(json.dumps({'fetched_bytes':[start,end],'sha256':hashlib.sha256(part.read_bytes()).hexdigest()}),flush=True)
            count=min(needed-done,end+1-self.position)
            with part.open('rb') as stream:
                stream.seek(self.position-start);data=stream.read(count)
            if len(data)!=count:raise OSError('Incomplete cached source bytes')
            buffer[done:done+count]=data
            self.receipts.append({'start':self.position,'bytes':count,'part':str(part.relative_to(ROOT)),
                'sha256':hashlib.sha256(part.read_bytes()).hexdigest()})
            self.position+=count;done+=count
        return done

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('uuid');p.add_argument('frame',type=int);p.add_argument('out_dir')
    args=p.parse_args();out=Path(args.out_dir);out.mkdir(parents=True,exist_ok=True)
    source=RecordedDepth(args.uuid)
    with h5py.File(source,'r') as f:
        depth=f['depth'][args.frame];confidence=f['confidence'][args.frame]
        np.savez_compressed(out/'depth.npz',depths=depth[None],confidence=confidence[None])
        layout={'depth_shape':list(f['depth'].shape),'depth_dtype':str(f['depth'].dtype),'attributes':{k:np.asarray(v).tolist() for k,v in f.attrs.items()}}
    report=dict(source_url=source.url,revision=REV,source_frame_index=args.frame,layout=layout,reads=source.receipts)
    (out/'depth_source.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({'saved':str(out/'depth.npz'),'percentiles_m':np.nanpercentile(depth,[0,5,50,95,100]).tolist()}),flush=True)
