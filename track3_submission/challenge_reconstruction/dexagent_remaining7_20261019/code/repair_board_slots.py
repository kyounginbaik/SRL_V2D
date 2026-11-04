#!/usr/bin/env python3
"""Additional episode25 slot validation, preserving the prior repair sources."""
from pathlib import Path
import argparse
from types import SimpleNamespace
import numpy as np
import repair_collision_assets as repair

BASE=Path(__file__).resolve().parents[1]
repair.REPAIRS[25]=['wooden_piece_1','wooden_piece_2']
original_validate=repair.validate


def validate(directory,name,source_model,parts,meshes):
    report=original_validate(directory,name,source_model,parts,meshes)
    audit=repair.read(BASE/'review/episode25_narrow_slot_audit.json')
    points=[]
    for record in audit['records']:
        if record['recipient']==name:
            points.extend(record['false_probe_points_body_m'])
    points=np.asarray(points).reshape(-1,3)
    depth,_=repair.audit.penetration(points,parts)
    slot=dict(clear_reference_surface_probe_count=len(points),visual_clearance_threshold_mm=.5,
        maximum_allowed_penetration_mm=.25,max_false_penetration_mm=float(depth.max(initial=0)*1000),
        probes_blocked_over_0_25mm=int((depth>.00025).sum()),
        source_probe_report=str(BASE/'review/episode25_narrow_slot_audit.json'),
        source_probe_report_sha256=repair.sha(BASE/'review/episode25_narrow_slot_audit.json'),
        validation_adapter=str(Path(__file__).resolve()),validation_adapter_sha256=repair.sha(__file__))
    report['narrow_assembly_slot']=slot
    report['passed']=bool(report['passed'] and slot['probes_blocked_over_0_25mm']==0)
    return report


def main():
    p=argparse.ArgumentParser();p.add_argument('--object',choices=repair.REPAIRS[25]);p.add_argument('--merge',action='store_true')
    p.add_argument('--threshold',type=float,default=.01);p.add_argument('--max-hulls',type=int,default=64)
    args=p.parse_args()
    args=SimpleNamespace(episode=25,object=args.object,threshold=args.threshold,max_hulls=args.max_hulls,
        library=BASE/'collision_library_slots_v3',source_root=BASE/'hawor_stereo_recon_v2',
        output_root=BASE/'collision_recon_slots_v3',reuse_parts=None,merge=args.merge)
    repair.validate=validate
    if args.merge:repair.merge(args)
    elif args.object:repair.generate(args)
    else:p.error('--object or --merge is required')


if __name__=='__main__':main()
