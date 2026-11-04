"""Score list for Z9 public variants; folders whose world poses equal an already scored folder (r8_base / z9_inj6_none) reuse its score (deterministic scorer)."""
import json, shutil, sys
from pathlib import Path
import numpy as np
W = Path('/data1/SRL_V2D/challenge_reconstruction/reconstruction_recovery_20261026/canonical')
tags = sys.argv[1:]
lines = []
for split in ['public', 'public_heldout']:
    for root in sorted((W / split).glob('episode_*')):
        ref = {t: np.load(root / 'exports' / t / 'object_poses.npz')['world_from_object'] for t in ['r8_base', 'z9_inj6_none'] if (root / 'exports' / t / 'object_poses.npz').exists()}
        for tag in tags:
            f = root / 'exports' / tag
            if (f / 'public_score.json').exists(): continue
            w = np.load(f / 'object_poses.npz')['world_from_object']
            same = [t for t, r in ref.items() if t != tag and np.array_equal(r, w) and (root / 'exports' / t / 'public_score.json').exists()]
            if same:
                shutil.copy2(root / 'exports' / same[0] / 'public_score.json', f / 'public_score.json'); (f / 'score_reused_from.txt').write_text(same[0] + '\n')
            else:
                lines.append(str(f))
Path('/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamZ9_anchor_spoon/code/score_list.txt').write_text('\n'.join(lines) + '\n'); print(len(lines))
