"""Reject non-finite saved policy tensors, including mutable normalization state."""
import json, struct
from pathlib import Path
import numpy as np

def check_checkpoint(path):
    path = Path(path)
    with path.open('rb') as f:
        size = struct.unpack('<Q', f.read(8))[0]
        header = json.loads(f.read(size))
        payload = f.read()
    bad = []
    for key, tensor in header.items():
        if key == '__metadata__':
            continue
        dtype = tensor['dtype']
        if dtype in ('F16', 'F32', 'F64'):
            start, end = tensor['data_offsets']
            values = np.frombuffer(payload[start:end], dtype={'F16':'<f2','F32':'<f4','F64':'<f8'}[dtype])
            if not np.isfinite(values).all():
                bad.append(key)
        elif dtype == 'BF16':
            start, end = tensor['data_offsets']
            values = np.frombuffer(payload[start:end], dtype='<u2')
            if ((values & 0x7f80) == 0x7f80).any():
                bad.append(key)
    if bad:
        raise ValueError(f'{path}: non-finite tensors: {bad}')
    return header['__metadata__']

if __name__ == '__main__':
    import sys
    for path in sys.argv[1:]:
        print(path, check_checkpoint(path))
