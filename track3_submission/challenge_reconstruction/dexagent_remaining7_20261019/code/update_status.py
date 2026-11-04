#!/usr/bin/env python3
"""Summarize actual seven-episode files; never infer success from job completion."""
from pathlib import Path
import datetime
import json

BASE=Path(__file__).resolve().parents[1]

def main():
    result=dict(updated_at=datetime.datetime.now().astimezone().isoformat(),
        requested_episodes=[14,15,16,20,25,28,30],
        old_release='/data1/SRL_V2D/challenge_submissions/track3_mixed_ilchold_20261019/available_13_episodes',
        existing_13_preserved=True,hand_source='HaWoR',
        current_hand_root=str(BASE/'hawor_stereo_recon_v2'),
        active_workspace=str(BASE/'physics_collision_v2'),episodes=[],
        metrics_are_reference_proxies=True,full_submission_ready=False,uploaded=False)
    for number in result['requested_episodes']:
        name=f'episode_{number:06d}'
        item=dict(episode=name,
            corrected_hawor_complete=(BASE/'hawor_stereo_recon_v2'/name/'hawor_complete.json').exists(),
            collision_source_ready=(BASE/'collision_recon'/name/'collision_repair_provenance.json').exists(),
            scene_prepared=(BASE/'physics_collision_v2/episodes'/name/'preparation_complete.json').exists(),
            first_mpc_mp_sr=0.,submission_ready=False)
        p=BASE/'physics_collision_v2/runs/baseline_v1'/name/'baseline/result.json'
        if p.exists():
            r=json.loads(p.read_text());item['corrected_baseline']=dict(path=str(p),
                physics_verified=r['execution']['physics_verified'],
                reference_proxy_mp_sr=r['metrics']['reference_proxy_mp_sr'],
                objects=r['metrics']['official_style_reference_proxy']['objects'])
        for root in [BASE/'refinement',BASE/'physics_collision_v2/refinement']:
            p=root/name/'status.json'
            if p.exists():
                item['refinement_status']=dict(path=str(p),value=json.loads(p.read_text()))
        component=BASE/'common_components'/name/'completed.json'
        archive=BASE/'portable_archives'/f'{name}_portable.tar.gz'
        if component.exists():
            item['portable_component_report']=str(component)
            item['portable_component']=json.loads(component.read_text())
            item['portable_archive']=str(archive) if archive.exists() else None
            item['submission_ready']=archive.exists()
        result['episodes'].append(item)
    result['validated_new_components']=sum(x['submission_ready'] for x in result['episodes'])
    result['validated_total_components']=13+result['validated_new_components']
    # The final full package has its own official validation and release marker.
    temporary=BASE/'status.tmp.json';temporary.write_text(json.dumps(result,indent=2)+'\n')
    temporary.replace(BASE/'status.json')
    print(json.dumps([{k:x[k] for k in ['episode','corrected_hawor_complete','collision_source_ready','scene_prepared','submission_ready']} for x in result['episodes']],indent=2))

if __name__=='__main__':
    main()
