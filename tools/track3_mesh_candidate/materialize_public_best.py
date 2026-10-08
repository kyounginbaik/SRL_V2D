"""Collect the selected video-derived public meshes in one reviewable folder."""
import json
import os
import shutil
from pathlib import Path

ROOT = Path(os.environ.get('V2D_MESH_BENCHMARK_ROOT',
                           '/data1/SRL_V2D/challenge_reconstruction/public_mesh_benchmark'))


def main():
    report = json.loads((ROOT / 'candidate_summary.json').read_text())
    out = ROOT / 'selected_video_only'
    out.mkdir(parents=True, exist_ok=True)
    for name, selected in report['objects'].items():
        source = Path(selected['mesh'])
        if not source.is_file():
            raise FileNotFoundError(source)
        target = out / f'{name}.glb'
        shutil.copy2(source, target)
        print(name, target, selected['mean_cd_o_cm'], flush=True)
    (out / 'selection_manifest.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
