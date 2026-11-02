"""Apply an explicitly fingerprinted task/physics protocol to a diagnostic."""
from copy import deepcopy
import hashlib
from pathlib import Path


def apply_task_protocol(configuration, manifest):
    result = deepcopy(configuration)
    protocol = manifest.get('evaluation_task_protocol')
    if protocol is None:
        return result
    path = Path(protocol['parquet'])
    if hashlib.sha256(path.read_bytes()).hexdigest() != protocol['parquet_sha256']:
        raise ValueError('evaluation task parquet changed')
    collision = protocol['collision']
    required = dict(robot_scope='all', hand_self_collision=True,
                    robot_object_collision=True, robot_support_collision=True, ground=True)
    if any(collision.get(key) != value for key, value in required.items()):
        raise ValueError('canonical evaluation requires the complete physical collision policy')
    if not result['scene']['support']:
        raise ValueError('canonical evaluation requires scene support')
    result['task']['parquet'] = str(path)
    result['collision'] = deepcopy(collision)
    return result
