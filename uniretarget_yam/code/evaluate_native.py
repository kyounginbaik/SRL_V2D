"""Entry point for canonical native-controller physical-feedback tests."""
import argparse,json
from pathlib import Path
import numpy as np
from flash_chord.retargeting.physical_reference_evaluator import evaluate_native_references
from flash_chord.training.checkpoint import read_checkpoint_metadata
from flash_chord.training.checkpoint_metadata import CONFIG_METADATA_KEY


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--steps',type=int)
    parser.add_argument('--replicas',type=int,default=2)
    parser.add_argument('--candidate',type=int,action='append')
    args=parser.parse_args()
    manifest=json.loads((args.plan/'manifest.json').read_text())
    configuration=json.loads(read_checkpoint_metadata(manifest['checkpoint'])[CONFIG_METADATA_KEY])
    with np.load(args.plan/'plan.npz') as archive:
        q=archive['joint_q']; target=archive['joint_target']
    selected=args.candidate or [0]
    result,_=evaluate_native_references(configuration,manifest,q[selected],target[selected],
         replicas=args.replicas,device=args.device,steps=args.steps,output=args.output)
    print(json.dumps(result),flush=True)

if __name__=='__main__': main()
