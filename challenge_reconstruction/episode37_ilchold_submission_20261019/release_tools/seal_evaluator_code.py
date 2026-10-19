"""Commit only evaluator source and verify every copied byte against real Git."""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess

REPO = Path('/data1/SRL_V2D')
BASE = Path(__file__).resolve().parents[1]
WORKTREE = REPO / '.local/worktrees/track3-learned-submission-20261018'
SUFFIXES = {'.py', '.yaml', '.yml', '.json', '.toml', '.sh', '.sbatch', '.md', '.txt'}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    files = []
    for directory in ['code', 'package_code', 'release_tools', 'evaluator_sources',
                      'source/episode_000037/flash_chord', 'source/episode_000037/code']:
        for path in (BASE / directory).rglob('*'):
            if not path.is_file() or any(x in path.parts for x in ['__pycache__', '.git']):
                continue
            if path.suffix in SUFFIXES or path.name.startswith(('LICENSE', 'COPYING', 'Dockerfile')):
                files.append(path)
    files.extend([BASE / 'README.md', BASE / 'source_manifest.json'])
    files = sorted(set(files))
    paths = []
    for path in files:
        relative = path.relative_to(REPO)
        target = WORKTREE / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        paths.append(str(relative))
    for start in range(0, len(paths), 100):
        subprocess.run(['git', '-C', str(WORKTREE), 'add', '-f', '--', *paths[start:start + 100]], check=True)
    subprocess.run(['git', '-C', str(WORKTREE), 'commit', '-m',
                    'Prepare verified ilchold episode 37 replacement and common evaluator'], check=True)
    commit = subprocess.check_output(['git', '-C', str(WORKTREE), 'rev-parse', 'HEAD'], text=True).strip()
    verified = []
    for path, relative in zip(files, paths):
        data = subprocess.check_output(['git', '-C', str(WORKTREE), 'show', commit + ':' + relative])
        expected = sha(path.read_bytes())
        if sha(data) != expected:
            raise ValueError('Git bytes differ: ' + relative)
        verified.append(dict(path=relative, sha256=expected, git_bytes_match=True))
    record = dict(commit=commit, code_commit_url='https://github.com/kyounginbaik/SRL_V2D/commit/' + commit,
                  sources=verified, verified_files=len(verified), pushed=False)
    (BASE / 'git_provenance.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(dict(commit=commit, verified_files=len(verified))))


if __name__ == '__main__':
    main()
