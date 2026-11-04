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


class StepVOCController:
    """Lower object assistance on a fixed ladder, reaching zero by 2/3 N."""

    def __init__(self, editor, options, output_dir):
        if editor.training.resume:
            raise ValueError('Step VOC requires a fresh run; partial bi-level resume is unsafe')
        self.total_steps = int(editor.training.total_environment_steps)
        if self.total_steps <= 0:
            raise ValueError('positive total_environment_steps required')
        self.scales = tuple(float(x) for x in options.get(
            'scales', (1., .75, .5, .25, .1, .05, .025, .01, 0.)))
        if len(self.scales) < 2 or self.scales[-1] != 0. or any(
            a <= b for a, b in zip(self.scales, self.scales[1:])):
            raise ValueError('Step VOC scales must strictly decrease to zero')
        self.editor = editor
        self.current_step = 0
        self.logged_index = 0
        self.output = Path(output_dir)
        last = len(self.scales) - 1
        milestones = [((2 * self.total_steps * i + 3 * last - 1) // (3 * last))
                      for i in range(1, last + 1)]
        (self.output / 'voc_schedule_manifest.json').write_text(json.dumps({
            'mode': 'step_schedule', 'total_environment_steps': self.total_steps,
            'zero_from_environment_step': milestones[-1],
            'scales': self.scales, 'milestones': milestones,
            'rule': 'staircase across first two thirds of N; zero for final third',
            'uses_physics_probes_for_gating': False,
        }, indent=2) + '\n')

    def index_at(self, step):
        return min(len(self.scales) - 1,
                   max(0, int(step)) * 3 * (len(self.scales) - 1) // (2 * self.total_steps))

    @property
    def index(self):
        return self.index_at(self.current_step)

    @property
    def scale(self):
        return self.scales[self.index]

    def __call__(self, learner, step, logger):
        previous = self.logged_index
        self.current_step = int(step)
        current = self.index
        if current == previous:
            return
        self.logged_index = current
        record = dict(step=self.current_step, mode='step_schedule',
                      reference_version=self.editor.version,
                      scale_before=self.scales[previous], scale_after=self.scales[current],
                      schedule_index=current, total_environment_steps=self.total_steps,
                      zero_from_environment_step=(2 * self.total_steps + 2) // 3)
        with (self.output / 'voc_gate.jsonl').open('a') as stream:
            stream.write(json.dumps(record) + '\n')
        (self.output / 'voc_schedule_state.json').write_text(json.dumps(record, indent=2) + '\n')
        if logger is not None:
            logger.log({'curriculum/voc_scale': self.scales[current],
                        'voc_schedule/scale': self.scales[current],
                        'voc_schedule/index': current}, step=self.current_step)
        print('[step VOC] ' + json.dumps(record), flush=True)


class YamProgressVOCController:
    """Lower object assistance when the policy passes the next lower level, with no time limit."""

    def __init__(self, editor, options, output_dir):
        if editor.training.resume:
            raise ValueError('Progress VOC requires a fresh run; resuming without gate history is unsafe')
        self.editor = editor
        self.scales = tuple(float(x) for x in options.get('scales', (1., .75, .5, .25, .1, .05, .025, .01, 0.)))
        if len(self.scales) < 2 or self.scales[-1] != 0. or any(a <= b for a, b in zip(self.scales, self.scales[1:])):
            raise ValueError('Progress VOC scales must strictly decrease to zero')
        self.interval = int(options.get('interval', 2500096))
        self.min_passed = float(options.get('min_passed', 0.75))
        self.required = int(options.get('required_passes', 2))
        self.index = 0
        self.streak = 0
        self.reference_version = -1
        self.next_check = self.interval
        self.output = Path(output_dir)
        (self.output / 'voc_gate_manifest.json').write_text(json.dumps(dict(
            mode='progress', scales=self.scales, interval=self.interval, min_passed=self.min_passed,
            required_passes=self.required, probe='next lower level, frozen policy, frame zero, worst object decides'), indent=2) + '\n')

    @property
    def scale(self):
        return self.scales[self.index]

    def __call__(self, learner, step, logger):
        if step < self.next_check or self.index == len(self.scales) - 1:
            return
        self.next_check = step + self.interval
        if self.reference_version != self.editor.version:
            self.reference_version = self.editor.version
            self.streak = 0
        probe = self.scales[self.index + 1]
        metrics, _ = self.editor.rollout(self.editor.binding, learner, self.editor.training.seed + 700000 + int(step), voc_probe_scale=probe)
        passed = bool(metrics['passed'] >= self.min_passed)
        self.streak = self.streak + 1 if passed else 0
        before = self.scale
        lowered = self.streak >= self.required
        if lowered:
            self.index += 1
            self.streak = 0
        record = dict(step=int(step), mode='progress', reference_version=self.editor.version, scale_before=before, probe_scale=probe,
                      scale_after=self.scale, lowered=lowered, streak=self.streak, passed=metrics['passed'],
                      frontier=metrics['frontier'], kinds=metrics['kinds'], cm=metrics['cm'], deg=metrics['deg'])
        with (self.output / 'voc_gate.jsonl').open('a') as stream:
            stream.write(json.dumps(record) + '\n')
        if logger is not None:
            logger.log({'curriculum/voc_scale': self.scale, 'voc_schedule/index': self.index}, step=int(step))
        print('[progress VOC] ' + json.dumps(record), flush=True)
