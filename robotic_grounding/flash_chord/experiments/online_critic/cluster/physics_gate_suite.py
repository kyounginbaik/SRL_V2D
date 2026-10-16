"""Run the four GPU physics checks after CPU preflight, before training."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import uuid


CASES = ((1, 'no_hand_contacts'), (1, 'reference_control'),
         (2048, 'no_hand_contacts'), (2048, 'reference_control'))


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def case_result(worlds, mode, returncode, path):
    result = dict(worlds=worlds, mode=mode, returncode=returncode,
                  report=str(path), passed=False)
    if path.is_file():
        result['report_sha256'] = sha256(path)
        try:
            report = json.loads(path.read_text())
            result['passed'] = (returncode == 0 and report.get('passed') is True
                                and report.get('worlds') == worlds and report.get('mode') == mode)
        except (ValueError, AttributeError) as exc:
            result['error'] = str(exc)
    else:
        result['error'] = 'Physics process did not produce a report'
    return result


def suite_passed(cases):
    return ([(case['worlds'], case['mode']) for case in cases] == list(CASES)
            and all(case['passed'] is True for case in cases))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--episode', default='episode_000001')
    parser.add_argument('--seed', type=int, default=43)
    args = parser.parse_args()
    root = args.run_root.resolve()
    config = root/args.episode/'runs/smoke'/args.episode/'chunk'/f'seed_{args.seed}'/'launch_resolved.yaml'
    if not config.is_file():
        parser.error('Resolved smoke configuration is missing; run bash ROOT/run.sh preflight first')
    subprocess.run([sys.executable, str(root/'cluster/check_source.py')], check=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    output = root/'reports'/'physics_gates'/f'{args.episode}_seed_{args.seed}_{stamp}_{uuid.uuid4().hex[:8]}'
    output.mkdir(parents=True, exist_ok=False)
    cache = root/'warp_cache'/f'{args.episode}_seed_{args.seed}'
    cache.mkdir(parents=True, exist_ok=True)
    report = dict(passed=False, episode=args.episode, seed=args.seed, directory=str(output),
                  config_sha256=sha256(config),
                  manifest_sha256={name: sha256(root/name) for name in
                      ('source_manifest.json', 'launcher_manifest.json', 'data_manifest.json')},
                  cases=[])
    for worlds, mode in CASES:
        case_dir = output/f'{worlds}_{mode}'
        print(f'Physics gate: {worlds} worlds, {mode}; output {case_dir}', flush=True)
        process = subprocess.run([sys.executable, str(root/'cluster/physics_gate.py'),
            '--run-root', str(root), '--episode', args.episode, '--seed', str(args.seed),
            '--worlds', str(worlds), '--mode', mode, '--output', str(case_dir),
            '--cache-dir', str(cache)])
        report['cases'].append(case_result(worlds, mode, process.returncode, case_dir/'report.json'))
        (output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    report['passed'] = suite_passed(report['cases'])
    (output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(f'Physics gate suite passed={report["passed"]}; report {output/"report.json"}', flush=True)
    raise SystemExit(0 if report['passed'] else 1)


if __name__ == '__main__':
    main()
