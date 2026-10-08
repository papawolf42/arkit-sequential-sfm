"""TEST ONLY: synthetic PLY verifies subprocess plumbing, NOT Gaussian training."""
import json
from pathlib import Path
import sys
import time

mode, capture, output = sys.argv[1:]
assert (Path(capture) / 'transforms.json').exists()
print(json.dumps({'progress': 0.5}), flush=True)
if mode == 'timeout':
    time.sleep(30)
if mode == 'failure':
    print('fixture failure', flush=True)
    sys.exit(7)
if mode == 'flood':
    print('x' * 100000, flush=True)
    time.sleep(30)
if mode == 'missing':
    sys.exit(0)
if mode == 'invalid':
    Path(output).write_text('not a ply')
else:
    props = ['x', 'y', 'z', 'f_dc_0', 'f_dc_1', 'f_dc_2', 'opacity',
             'scale_0', 'scale_1', 'scale_2', 'rot_0', 'rot_1', 'rot_2', 'rot_3']
    Path(output).write_text('ply\nformat ascii 1.0\nelement vertex 1\n' +
                            ''.join('property float ' + p + '\n' for p in props) +
                            'end_header\n' + ' '.join(['1'] * len(props)) + '\n')
