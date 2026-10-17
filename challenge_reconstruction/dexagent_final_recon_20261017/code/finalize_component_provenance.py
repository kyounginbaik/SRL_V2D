"""Verify actual Git bytes before assigning producing-code provenance.

This records a reproducible local episode component, not a complete submission.
Every evaluator source must match both its recorded digest and the real commit.
Additional teacher/trainer files may be included as explicit lineage evidence.
"""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import tarfile

ROOT=Path(__file__).resolve().parents[3]


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())


def run(handoff,worktree,commit,source_code,archive=False):
    handoff,worktree=Path(handoff).resolve(),Path(worktree).resolve()
    readiness=read(handoff/'readiness.json')
    replay=read(handoff/'portable_validation/portable_validation.json')
    if not (readiness['episode_ready_for_packaging_review'] and replay['all_physical_states_actions_exact']
            and replay['source_object_targets_unchanged']):
        raise ValueError('Fresh exact learned-policy validation is required')
    actual=subprocess.check_output(['git','-C',str(worktree),'rev-parse',commit],text=True).strip()
    checkpoint=readiness['learned_policy_checkpoint']
    if sha(checkpoint['path'])!=checkpoint['sha256']:raise ValueError('Checkpoint changed')
    code={Path(x['path']).resolve():x['sha256'] for x in readiness['evaluation_code_manifest']}
    for file in source_code:
        path=Path(file).resolve();code[path]=sha(path)
    verified=[]
    for path,digest in code.items():
        relative=path.relative_to(ROOT)
        payload=subprocess.check_output(['git','-C',str(worktree),'show',actual+':'+relative.as_posix()])
        if sha(path)!=digest or hashlib.sha256(payload).hexdigest()!=digest:
            raise ValueError('Actual producing source differs: '+str(relative))
        verified.append(dict(path=str(relative),sha256=digest,git_bytes_match=True))
    provenance=dict(commit=actual,url='https://github.com/kyounginbaik/SRL_V2D/commit/'+actual,
        branch=subprocess.check_output(['git','-C',str(worktree),'branch','--show-current'],text=True).strip(),
        actual_sources_verified=verified,checkpoint_sha256=checkpoint['sha256'])
    output=handoff/'provenance/producing_code.json'
    output.parent.mkdir(exist_ok=True)
    if output.exists() and read(output)!=provenance:raise FileExistsError('Existing provenance differs')
    output.write_text(json.dumps(provenance,indent=2)+'\n')
    if archive:
        target=handoff/(readiness['episode']+'_portable.tar.gz')
        if target.exists():raise FileExistsError(target)
        with tarfile.open(target,'w:gz') as packed:
            for name in ['portable','readiness.json','results','meshes','validation','provenance','README.md']:
                if (handoff/name).exists():packed.add(handoff/name,arcname=handoff.name+'/'+name)
            packed.add(handoff/'portable_validation/portable_validation.json',
                       arcname=handoff.name+'/portable_validation/portable_validation.json')
        record=dict(path=str(target),bytes=target.stat().st_size,sha256=sha(target))
        (handoff/'archive.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(dict(handoff=str(handoff),commit=actual,verified_sources=len(verified),archive_created=archive)))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ['handoff','worktree','commit']:p.add_argument('--'+key,required=True)
    p.add_argument('--source-code',action='append',default=[])
    p.add_argument('--archive',action='store_true');run(**vars(p.parse_args()))
