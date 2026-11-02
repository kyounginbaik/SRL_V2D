"""Resolve exactly one Slurm-assigned GPU; never select an unallocated device."""
import json
import os
import re
import subprocess
import sys
import xml.etree.ElementTree as ET


def allocated_device(environ):
    if not environ.get('SLURM_JOB_ID'):
        raise RuntimeError('a real Slurm allocation is required')
    # Direct srun launches can export STEP_GPUS without JOB_GPUS. Prefer the
    # step assignment so a step cannot escape its subset of a larger job.
    source = next((key for key in ('SLURM_STEP_GPUS', 'SLURM_JOB_GPUS') if environ.get(key)), None)
    if source is None:
        raise RuntimeError('Slurm supplied neither STEP_GPUS nor JOB_GPUS; refusing to guess')
    value = environ[source].strip()
    if not (value.isdecimal() or re.fullmatch(r'GPU-[0-9a-fA-F-]{36}', value)):
        raise RuntimeError('exactly one Slurm GPU device is required: ' + source)
    return source, value


def resolve_gpu(environ, xml):
    source, value = allocated_device(environ)
    root = ET.fromstring(xml)
    field = 'uuid' if value.startswith('GPU-') else 'minor_number'
    matches = [gpu for gpu in root.findall('gpu') if gpu.findtext(field) == value]
    if len(matches) != 1:
        raise RuntimeError('cannot uniquely map the Slurm GPU to a host NVIDIA device')
    gpu = matches[0]
    uuid = gpu.findtext('uuid', '')
    if not re.fullmatch(r'GPU-[0-9a-fA-F-]{36}', uuid):
        raise RuntimeError('invalid host NVIDIA UUID')
    return uuid, dict(allocation_source=source, allocated=value,
                      minor=gpu.findtext('minor_number'), uuid=uuid,
                      pci=gpu.findtext('pci/pci_bus_id'))


def main():
    # Check authorization before making a host GPU query.
    allocated_device(os.environ)
    uuid, evidence = resolve_gpu(os.environ, subprocess.check_output(['nvidia-smi', '-q', '-x']))
    print(json.dumps(evidence), file=sys.stderr)
    print(uuid)


if __name__ == '__main__':
    main()
