"""C30: re-run an H13 window-beam pass with the SAME recipe (settings.json of the source pass) from a new start actor, scored vs
C30_TGT (env). usage: beam.py SRC_SETTINGS.json OUT_DIR key=value ...  (overrides; value 'None' drops the flag). Prints the command."""
import sys, json, shlex
src, out = sys.argv[1], sys.argv[2]; d = json.load(open(src)); a = dict(d['args']); a['out'] = out
for kv in sys.argv[3:]:
    k, v = kv.split('=', 1); a[k] = None if v == 'None' else v
script = d['script'].replace('streamH13b/code', 'streamC30_compose/h13b/code').replace('streamH13a/code', 'streamC30_compose/h13a/code')
cmd = [script]
for k, v in a.items():
    if v is None or v == '': continue
    cmd += ['--' + k.replace('_', '-'), str(v)]
print(' '.join(shlex.quote(c) for c in cmd))
