"""Read real downloaded HDF5 byte ranges without padding unavailable bytes.

The file length is known from the pinned release manifest. This is a range-backed
file object, not a completed-file claim: every read must be backed by actual
source bytes in the prefix, checked chunks, or the fetched tail, else it raises.
"""
from pathlib import Path
import io
import json
import re
import subprocess
import h5py
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'data/raw/02430554-623d-4a36-936e-b1ae200b43af/ep1/depth.hdf5'
FULLSIZE=487584800
CHUNK=4*1024*1024

class RangeSource(io.RawIOBase):
    def __init__(self):self.position=0;self.receipts=[]
    def readable(self):return True
    def seekable(self):return True
    def tell(self):return self.position
    def seek(self,offset,whence=0):
        self.position=offset if whence==0 else self.position+offset if whence==1 else FULLSIZE+offset
        return self.position
    def readinto(self,buffer):
        start=self.position;end=start+len(buffer)
        chunk=ROOT/'downloads/depth'/f'{start//CHUNK:05d}.part'
        tail=ROOT/'continuation/depth_tail.bin'
        if end<=SOURCE.stat().st_size:
            path=SOURCE;offset=start;kind='source_prefix'
        elif chunk.exists() and end<=min(FULLSIZE,(start//CHUNK+1)*CHUNK):
            path=chunk;offset=start%CHUNK;kind='checked_chunk'
        elif start>=486000000 and tail.exists() and tail.stat().st_size==1584800:
            path=tail;offset=start-486000000;kind='source_tail'
        else:
            index=start//CHUNK
            part_dir=ROOT/'continuation/depth_needed'
            part_dir.mkdir(exist_ok=True)
            path=part_dir/f'{index:05d}.part'
            chunk_start=index*CHUNK;chunk_end=min(FULLSIZE,(index+1)*CHUNK)-1
            if not path.exists():
                temp=path.with_suffix('.tmp');header=path.with_suffix('.headers')
                url='https://huggingface.co/datasets/ropedia-ai/egotools-data/resolve/c738802d9ac9bf1b79b7993592096a3dcd6391ec/raw/02430554-623d-4a36-936e-b1ae200b43af/ep1/depth.hdf5'
                subprocess.run(['curl','--http1.1','-fLsS','--retry','2','--max-time','90',
                    '--max-filesize',str(chunk_end-chunk_start+1),'-H',f'Range: bytes={chunk_start}-{chunk_end}',
                    '-D',str(header),url+f'?download=true&needed={index}','-o',str(temp)],check=True)
                ranges=re.findall(r'content-range:\s*bytes\s+(\d+)-(\d+)/(\d+)',header.read_text(),re.I)
                assert ranges and tuple(map(int,ranges[-1]))==(chunk_start,chunk_end,FULLSIZE)
                assert temp.stat().st_size==chunk_end-chunk_start+1
                temp.replace(path)
                print(f'Fetched actual depth source range {chunk_start}-{chunk_end}',flush=True)
            if end>chunk_end+1:
                # HDF5 reads may straddle independently checked range blocks.
                first=chunk_end+1-start
                temp_buffer=bytearray(len(buffer))
                self.readinto(memoryview(temp_buffer)[:first])
                self.readinto(memoryview(temp_buffer)[first:])
                buffer[:]=temp_buffer
                return len(buffer)
            offset=start-chunk_start;kind='on_demand_checked_chunk'
        with path.open('rb') as stream:stream.seek(offset);data=stream.read(len(buffer))
        if len(data)!=len(buffer):raise OSError(f'Incomplete actual byte range [{start},{end})')
        buffer[:]=data;self.position=end
        self.receipts.append({'start':start,'end':end,'kind':kind})
        return len(data)

if __name__=='__main__':
    source=RangeSource()
    with h5py.File(source,'r') as f:
        keys=list(f.keys())
        print('top level',len(keys),keys[:8],flush=True)
        for key in keys[:3]:
            value=f[key]
            print(key,type(value).__name__,getattr(value,'shape',None),getattr(value,'dtype',None),flush=True)
            if isinstance(value,h5py.Group):print('children',list(value.keys())[:5],flush=True)
        # Discover the first actual depth array without assuming its axis layout.
        datasets=[(k,f[k].shape,str(f[k].dtype)) for k in keys]
        (ROOT/'continuation/depth_layout.json').write_text(json.dumps(datasets,indent=2))
        print('datasets',datasets[:5],flush=True)
        depth=f['depth'][0]
        confidence=f['confidence'][0]
        np.savez_compressed(ROOT/'continuation/depth_frame0.npz',depths=depth[None],confidence=confidence[None])
        print('depth percentiles',np.nanpercentile(depth,[0,5,50,95,100]),'confidence',np.unique(confidence),flush=True)
        print('depth attrs',dict(f['depth'].attrs),'file attrs',dict(f.attrs),flush=True)
    (ROOT/'continuation/depth_read_receipts.json').write_text(json.dumps(source.receipts,indent=2))
