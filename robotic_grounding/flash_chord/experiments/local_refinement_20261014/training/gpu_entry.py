from pathlib import Path
import subprocess,sys,os
b=Path(__file__).resolve().parents[1]
if '--phase' in sys.argv and sys.argv[sys.argv.index('--phase')+1]=='smoke':
 subprocess.run([sys.executable,'-m','pytest','tests/test_local_correction.py','tests/runtime/test_wrist_limits.py','-q','-p','no:cacheprovider'],env={**os.environ,'JAX_PLATFORMS':'cpu'},check=True)
subprocess.run([sys.executable,str(b/'code/run_sequence.py'),*sys.argv[1:]],check=True)
