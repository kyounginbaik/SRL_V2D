from pathlib import Path
import json
from flash_chord.lifecycle.performance_voc import PerformanceVOCGate
ok=dict(per_object_gate_success_fraction=[.98,.98],per_object_position_p95_m=[.02,.02],per_object_orientation_p95_rad=[.1,.1],full_horizon=True,all_errors_finite=True)
g=PerformanceVOCGate(2)
assert g.scale==1
for _ in range(2): assert not g.observe(ok,0)['lowered']
assert g.observe(ok,0)['lowered'] and g.scale==.75
bad={**ok,'per_object_gate_success_fraction':[.99,.5]}
for _ in range(20):assert not g.observe(bad,0)['lowered'] and g.scale==.75
for _ in range(2):g.observe(ok,0)
assert not g.observe(ok,1)['lowered'] and g.streak==1
assert not g.observe({**ok,'full_horizon':False},1)['passed']
assert not g.observe({**ok,'per_object_position_p95_m':[float('nan'),.02]},1)['passed']
for _ in range(21):g.observe(ok,1)
assert g.scale==0
b=Path(__file__).resolve().parents[1];(b/'performance_gate_validation.json').write_text(json.dumps({'passed':True,'checks':['starts_at_one','three_pass_patience','both_objects_required','reference_change_resets_streak','incomplete_and_nan_rejected','reaches_zero_only_after_passes']}));print('PASS performance gate')
