"""Validate the actual GPU sampler before the integrated preflight."""
from pathlib import Path
import subprocess,sys
b=Path(__file__).resolve().parents[1]
if '--phase' in sys.argv and sys.argv[sys.argv.index('--phase')+1]=='smoke':
 for script in ['validate_eigen_extremes.py','validate_eigen.py']:
  subprocess.run([sys.executable,str(b/'code'/script)],check=True)
subprocess.run([sys.executable,str(b/'code/run_sequence.py'),*sys.argv[1:]],check=True)
