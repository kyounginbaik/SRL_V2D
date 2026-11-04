import sys
a, b, s = float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]) if len(sys.argv) > 3 else 0.5
w = []; t = a
while t < b - 1e-6: w.append(f'{t:.2f}:{min(t + s, b):.2f}'); t += s
print(','.join(w))
