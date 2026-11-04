"""score an assembled package vs a target with the kit scorer path (compose_target/score_vs_compose.py, per episode, 4 threads).
usage: pkgscore.py PKG_ALL20_DIR TARGET OUT.json"""
import sys, json, subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
P, T, O = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]); assert not O.exists(); O.parent.mkdir(parents=True, exist_ok=True)
S = '/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/compose_target/score_vs_compose.py'
tmp = O.with_suffix('.parts'); tmp.mkdir()
eps = [int(p.stem[-6:]) for p in sorted((P / 'results').glob('episode_*.parquet'))]
def one(e):
    f = tmp / f'{e}.json'; subprocess.run([sys.executable, '-B', S, str(P / 'track3_submission.parquet'), str(P / 'results'), str(T), str(f), str(e)], cwd='/tmp', check=True, capture_output=True)
    return json.loads(f.read_text())[str(e)]
with ThreadPoolExecutor(4) as ex: res = dict(zip(map(str, eps), ex.map(one, eps)))
mp = sum(v['scores']['MP-SR'] for v in res.values()); mppe = sum(v['scores']['MPPE'] for v in res.values()) / len(res); auc = sum(v['scores']['AUC'] for v in res.values()) / len(res)
out = dict(package=str(P), target=str(T), mp_pass=mp, n=len(res), mean_MPPE=mppe, mean_AUC=auc, failed=[e for e, v in res.items() if v['scores']['MP-SR'] != 1.0], episodes=res)
O.write_text(json.dumps(out, indent=1)); print(json.dumps({k: v for k, v in out.items() if k != 'episodes'}))
