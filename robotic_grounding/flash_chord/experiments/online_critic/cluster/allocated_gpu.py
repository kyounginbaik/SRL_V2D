"""Map this cluster's Slurm /dev/nvidia minor allocation to a Docker GPU UUID."""
import os,subprocess,xml.etree.ElementTree as ET,json,sys
allocated=os.environ['SLURM_JOB_GPUS'].split(',')[0]
if allocated.startswith('GPU-'):
 print(allocated);sys.exit(0)
root=ET.fromstring(subprocess.check_output(['nvidia-smi','-q','-x']))
matches=[g for g in root.findall('gpu') if g.findtext('minor_number')==allocated]
if len(matches)!=1:raise RuntimeError(f'Cannot map allocated NVIDIA device minor {allocated}')
g=matches[0]
print(json.dumps({'slurm_job_gpus':os.environ['SLURM_JOB_GPUS'],'minor':g.findtext('minor_number'),'uuid':g.findtext('uuid'),'pci':g.findtext('pci/pci_bus_id')}),file=sys.stderr)
print(g.findtext('uuid'))
