"""Run all evidence for one episode: per-frame depth/mask evidence, spoon ball detection + marker fit, camera-path check."""
import sys, subprocess
ep = int(sys.argv[1]); D = '/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/h2h_spoon'; C = D + '/code'; py = sys.executable
run = lambda *a, env=None: subprocess.run([py, '-B', '-u', *a], check=False, env=env)
import os
run(f'{C}/detect_balls.py', str(ep), 'wooden_spoon', f'{D}/markers/balls_{ep}_wooden_spoon.json')
run(f'{C}/fit_markers.py', str(ep), 'wooden_spoon', f'{D}/markers/balls_{ep}_wooden_spoon.json', f'{D}/markers/fit_{ep}_wooden_spoon.json')
run(f'{C}/evidence_run.py', str(ep), f'{D}/evidence/ev_{ep}.json', '1')
run(f'{C}/campath.py', str(ep), f'{D}/campath/campath_{ep}.json', '10')
