"""Inspect a capture manifest without modifying the source data."""
import argparse
import json
from pathlib import Path
import math

import numpy as np


def world_to_camera(camera_to_world):
    """유효한 강체 c2w 행렬을 w2c로 바꾼다(회전의 역은 전치)."""
    matrix = np.asarray(camera_to_world, dtype=float)
    rotation = matrix[:3, :3]
    position = matrix[:3, 3]
    inverse = np.eye(4)
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -rotation.T @ position
    return inverse


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

    camera_position = None
    camera_forward = None
    if frames:
        matrix = frames[0]['transform_matrix']
        camera_position = [matrix[0][3], matrix[1][3], matrix[2][3]]
        camera_forward = [-matrix[0][2], -matrix[1][2], -matrix[2][2]]

    # ARKit 추정 위치 사이의 직선거리이며 실제 이동 거리의 정답은 아니다.
    first_pair_distance = None
    first_pair_rotation = None
    if len(frames) >= 2:
        first_matrix = frames[0]['transform_matrix']
        next_matrix = frames[1]['transform_matrix']
        dx = next_matrix[0][3] - first_matrix[0][3]
        dy = next_matrix[1][3] - first_matrix[1][3]
        dz = next_matrix[2][3] - first_matrix[2][3]
        first_pair_distance = (dx ** 2 + dy ** 2 + dz ** 2) ** 0.5

        first_rotation = np.array(first_matrix)[:3, :3]
        next_rotation = np.array(next_matrix)[:3, :3]
        relative_rotation = first_rotation.T @ next_rotation
        cos_angle = (np.trace(relative_rotation) - 1) / 2
        # 부동소수점 반올림 오차로 acos의 입력 범위를 벗어나지 않게 한다.
        cos_angle = max(-1.0, min(1.0, float(cos_angle)))
        first_pair_rotation = math.degrees(math.acos(cos_angle))

    print(json.dumps({
        'frame_count': len(frames),
        'first_frame': frames[0] if frames else None,
        'first_image_exists': first_image.is_file() if first_image is not None else None,
        'missing_images': missing_images,
        'first_camera_position': camera_position,
        'first_pair_distance_m': first_pair_distance,
        'first_camera_forward': camera_forward,
        'first_pair_rotation_deg': first_pair_rotation,
    }, indent=2))
