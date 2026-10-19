from pathlib import Path
import sys,subprocess
b=Path(__file__).resolve().parents[1]
seed=sys.argv[1]
root=b/f"runs/pilot/episode_000001/chunk/seed_{seed}"
for ref in sorted((root/"training/references").glob("reference_*.npz")):
 version=int(ref.stem.split("_")[-1])
 subprocess.run([sys.executable,str(b/"code/refresh_overlay.py"),"pilot","episode_000001","chunk",seed,str(version)],check=True)
