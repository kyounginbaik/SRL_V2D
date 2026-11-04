import sys, numpy as np
a, b, w = float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])
e = np.round(np.arange(a, b, w), 3); print(','.join(f'{x}:{min(round(x + w, 3), b)}' for x in e if b - x > 1e-6))
