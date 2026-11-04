#!/usr/bin/env python3
"""Prepare/evaluate an isolated scene variant without replacing prior trials."""
from pathlib import Path
import argparse
import json
import prepare_seven


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--episode', type=int, required=True)
    p.add_argument('--source-root', type=Path, required=True)
    p.add_argument('--workspace', type=Path, required=True)
    p.add_argument('--baseline', action='store_true')
    args = p.parse_args()
    prepare_seven.BASE = args.workspace.resolve()
    result = prepare_seven.prepare_episode(args.episode, args.source_root.resolve())
    if args.baseline:
        import run_seven_mpc
        run_seven_mpc.BASE = args.workspace.resolve()
        result = run_seven_mpc.run(args.episode, 'baseline_v1', samples=16,
            workers=4, max_minutes=10., seed=44, baseline_only=True)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
