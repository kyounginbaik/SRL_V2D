"""Start the real trainer on CPU with few worlds, to catch config and layout errors before a GPU slot is spent."""
import runpy,sys
sys.argv=['train_flash_sac.py']+sys.argv[1:]
runpy.run_path('/workspace/scripts/train_flash_sac.py',run_name='__main__')
