"""Inspect a capture manifest without modifying the source data."""
import argparse
import json
from pathlib import Path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('transforms', type=Path, help='Path to transforms.json')
    args = parser.parse_args()

    manifest = json.loads(args.transforms.read_text(encoding='utf-8'))
    frames = manifest['frames']
    print(json.dumps({
        'frame_count': len(frames),
        'first_frame': frames[0] if frames else None,
    }, indent=2))
