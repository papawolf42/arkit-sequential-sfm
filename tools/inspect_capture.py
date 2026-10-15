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
    first_image = args.transforms.parent / frames[0]['file_path'] if frames else None
    missing_images = []
    for frame in frames:
        image = args.transforms.parent / frame['file_path']
        if not image.is_file():
            missing_images.append(frame['file_path'])

    print(json.dumps({
        'frame_count': len(frames),
        'first_frame': frames[0] if frames else None,
        'first_image_exists': first_image.is_file() if first_image is not None else None,
        'missing_images': missing_images,
    }, indent=2))
