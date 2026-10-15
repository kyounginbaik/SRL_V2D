"""Resolve one scheduler allocation to its NVIDIA UUID without guessing indices.

Slurm installations may expose NVIDIA indices or device minor numbers. When
those identify different GPUs, require an explicit site convention or UUID.
Only the standard library is needed on the host.
"""
import csv
from dataclasses import dataclass
import io
import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET


@dataclass(frozen=True)
class GPU:
    index: int
    minor: int
    uuid: str


def parse_inventory(xml_text, index_csv):
    """Join explicit index and minor-number inventories by UUID."""
    minor_by_uuid = {}
    for device in ET.fromstring(xml_text).findall('gpu'):
        uuid = device.findtext('uuid', '').strip()
        minor = int(device.findtext('minor_number', '-1'))
        if not uuid.startswith('GPU-') or minor < 0 or uuid in minor_by_uuid:
            raise ValueError('Invalid or duplicate GPU UUID/minor in NVIDIA inventory')
        minor_by_uuid[uuid] = minor
    devices = []
    for row in csv.reader(io.StringIO(index_csv)):
        if not row:
            continue
        if len(row) != 2:
            raise ValueError('Expected NVIDIA index,uuid CSV inventory')
        index, uuid = int(row[0].strip()), row[1].strip()
        if index < 0 or uuid not in minor_by_uuid:
            raise ValueError('NVIDIA index and minor inventories do not agree')
        devices.append(GPU(index, minor_by_uuid[uuid], uuid))
    if (not devices or len(devices) != len(minor_by_uuid) or
            any(len({getattr(g, key) for g in devices}) != len(devices)
                for key in ('index', 'minor', 'uuid'))):
        raise ValueError('NVIDIA inventory must contain unique indices, minors and UUIDs')
    return tuple(devices)


def resolve_gpu(environ, devices):
    kind = environ.get('UNIRETARGET_GPU_ID_KIND', 'auto')
    if kind not in ('auto', 'minor', 'index'):
        raise ValueError('UNIRETARGET_GPU_ID_KIND must be auto, minor or index')
    by_uuid = {g.uuid: g for g in devices}
    explicit = environ.get('UNIRETARGET_GPU_UUID', '').strip()
    if explicit and explicit not in by_uuid:
        raise ValueError('UNIRETARGET_GPU_UUID must be one complete UUID from nvidia-smi')
    source = next((key for key in ('SLURM_STEP_GPUS', 'SLURM_JOB_GPUS', 'CUDA_VISIBLE_DEVICES')
                   if environ.get(key, '').strip()), None)
    if source is None:
        if explicit:
            return by_uuid[explicit]
        raise ValueError('No single GPU allocation found; provide UNIRETARGET_GPU_UUID or an allocated GPU environment')
    token = environ[source].strip()
    if ',' in token or not (token.isdecimal() or token in by_uuid):
        raise ValueError(f'{source} must identify exactly one GPU index, minor or complete UUID; got {token!r}')
    if token in by_uuid:
        candidates = {token}
    else:
        number = int(token)
        # CUDA_VISIBLE_DEVICES numeric entries are device indices, not Slurm
        # minor IDs. A site convention applies only to scheduler variables.
        effective_kind = 'index' if source == 'CUDA_VISIBLE_DEVICES' else kind
        candidates = {g.uuid for g in devices if
                      (effective_kind in ('auto', 'index') and g.index == number) or
                      (effective_kind in ('auto', 'minor') and g.minor == number)}
        if not candidates:
            raise ValueError(f'{source}={token} has no GPU match using {effective_kind} IDs')
    if explicit:
        if explicit not in candidates:
            raise ValueError('UNIRETARGET_GPU_UUID is outside the scheduler allocation; refusing another GPU')
        return by_uuid[explicit]
    if len(candidates) != 1:
        raise ValueError(f'{source}={token} is ambiguous: index and minor identify different GPUs. '
                         'Set UNIRETARGET_GPU_ID_KIND=minor or index for this site, '
                         'or UNIRETARGET_GPU_UUID to the allocated GPU UUID.')
    return by_uuid[next(iter(candidates))]


def main():
    try:
        xml_text = subprocess.check_output(['nvidia-smi', '-q', '-x'], text=True)
        index_csv = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid',
                                            '--format=csv,noheader,nounits'], text=True)
        device = resolve_gpu(os.environ, parse_inventory(xml_text, index_csv))
    except (ValueError, ET.ParseError, OSError, subprocess.CalledProcessError) as error:
        print(f'UniRetarget GPU allocation error: {error}', file=sys.stderr)
        return 2
    print(json.dumps(dict(index=device.index, minor=device.minor, uuid=device.uuid)), file=sys.stderr)
    print(device.uuid)
    return 0


if __name__ == '__main__':
    sys.exit(main())
