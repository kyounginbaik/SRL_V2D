"""Performance-gated shared VOC: every object must pass at the proposed lower scale."""
from dataclasses import dataclass, asdict
import json
import math
from pathlib import Path

@dataclass
class PerformanceVOCGate:
    object_count: int
    scales: tuple = (1., .75, .5, .25, .1, .05, .025, .01, 0.)
    required_passes: int = 3
    min_success: float = .9
    position_p95_m: float = .05
    orientation_p95_rad: float = math.radians(20)
    hand_rms_m: float = .06
    index: int = 0
    streak: int = 0
    reference_version: int = -1

    def __post_init__(self):
        if self.object_count < 1 or self.required_passes < 1:
            raise ValueError('positive object count and patience required')
        if not self.scales or self.scales[-1] != 0 or any(a<=b for a,b in zip(self.scales,self.scales[1:])):
            raise ValueError('VOC scales must decrease strictly to zero')

    @property
    def scale(self): return self.scales[self.index]
    @property
    def probe_scale(self): return self.scales[min(self.index+1,len(self.scales)-1)]

    def observe(self, metrics, reference_version):
        if self.reference_version != reference_version:
            self.streak=0;self.reference_version=reference_version
        checks={}
        for key,threshold,lower in [
            ('per_object_gate_success_fraction',self.min_success,False),
            ('per_object_position_p95_m',self.position_p95_m,True),
            ('per_object_orientation_p95_rad',self.orientation_p95_rad,True)]:
            values=metrics.get(key,[])
            checks[key]=len(values)==self.object_count and all(math.isfinite(x) and (x<=threshold if lower else x>=threshold) for x in values)
        hand=metrics.get('robot_position_rms_m',float('inf'))
        checks['hand_tracking']=math.isfinite(hand) and hand<=self.hand_rms_m
        checks['full_horizon']=metrics.get('full_horizon',False) is True
        checks['finite']=metrics.get('all_errors_finite',False) is True
        passed=all(checks.values());self.streak=self.streak+1 if passed else 0
        previous=self.scale;probe=self.probe_scale
        lowered=self.streak>=self.required_passes and self.index<len(self.scales)-1
        if lowered:self.index+=1;self.streak=0
        return dict(passed=passed,lowered=lowered,scale_before=previous,probe_scale=probe,
                    scale_after=self.scale,consecutive_passes=self.streak,checks=checks)

class PerformanceVOCController:
    def __init__(self, editor, options, output_dir):
        if editor.training.resume:
            raise ValueError('Adaptive VOC currently requires a fresh run; resuming without gate history is unsafe')
        self.editor=editor
        self.interval=int(options.get('interval',2097152))
        if self.interval<=0:raise ValueError('positive probe interval required')
        self.next_check=self.interval
        # Read names directly from the source roster instead of inferring object count from hands.
        import pyarrow.parquet as pq
        row=pq.read_table(editor.config['task']['parquet']).to_pylist()[0]
        self.object_names=list(row['object_body_names'])
        self.gate=PerformanceVOCGate(object_count=len(self.object_names),required_passes=int(options.get('required_passes',3)))
        self.output=Path(output_dir)
        (self.output/'voc_gate_manifest.json').write_text(json.dumps({'objects':self.object_names,'interval':self.interval,
          'gate':asdict(self.gate),'probe':'full video at next lower VOC, explicit frame-zero reset, no reset assistance',
          'shared_scale':True,'performance_thresholds_are_not_official_challenge_criteria':True},indent=2))
    @property
    def index(self):return self.gate.index
    @property
    def scale(self):return self.gate.scale
    def __call__(self, learner, step, logger):
        if step<self.next_check or self.gate.index==len(self.gate.scales)-1:return
        self.next_check=step+self.interval
        metrics,_,_=self.editor.rollout(self.editor.binding,learner,self.editor.training.seed+700000+step,
                                       voc_probe_scale=self.gate.probe_scale)
        decision=self.gate.observe(metrics,self.editor.version)
        record=dict(step=step,reference_version=self.editor.version,objects=self.object_names,metrics=metrics,**decision)
        with (self.output/'voc_gate.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
        (self.output/'voc_gate_state.json').write_text(json.dumps(asdict(self.gate),indent=2))
        if logger is not None:
            values={f'voc_gate/{k}':float(decision[k]) for k in ['passed','lowered','scale_before','probe_scale','scale_after','consecutive_passes']}
            for i,name in enumerate(self.object_names):
                for metric in ['per_object_gate_success_fraction','per_object_position_p95_m','per_object_orientation_p95_rad']:
                    values[f'voc_gate/{name}/{metric}']=metrics[metric][i]
            values['voc_gate/hand_rms_m']=metrics['robot_position_rms_m']
            logger.log(values,step=step)
        print('[performance VOC] '+json.dumps(record),flush=True)
