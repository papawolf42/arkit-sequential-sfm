"""촬영 시각이 이웃한 이미지 쌍을 만든다. 원본은 수정하지 않는다."""
import argparse
import json
from pathlib import Path
import math

import numpy as np


def baseline_pairs(frames, min_offset=2, max_offset=8,
                   min_distance=0.3, max_distance=1.5, max_angle_deg=30):
    """근처 시간 구간에서 거리·시선 조건을 만족하는 후보를 고른다."""
    ordered = sorted(frames, key=lambda frame: frame['timestamp'])
    pairs = []
    for i, first in enumerate(ordered):
        first_matrix = np.asarray(first['transform_matrix'])
        for j in range(i + min_offset, min(i + max_offset + 1, len(ordered))):
            second = ordered[j]
            second_matrix = np.asarray(second['transform_matrix'])
            distance = np.linalg.norm(first_matrix[:3, 3] - second_matrix[:3, 3])
            forward_a = -first_matrix[:3, 2]
            forward_b = -second_matrix[:3, 2]
            direction_cos = np.dot(forward_a, forward_b) / (
                np.linalg.norm(forward_a) * np.linalg.norm(forward_b))
            if (min_distance <= distance <= max_distance
                    and direction_cos > math.cos(math.radians(max_angle_deg))):
                pairs.append((first['file_path'], second['file_path']))
    return pairs


def unique_pairs(pairs):
    """여러 선정 방식에서 나온 같은 이미지 쌍을 한 번만 비교한다."""
    result = []
    seen = set()
    for first, second in pairs:
        if first == second:
            continue
        pair = tuple(sorted((first, second)))
        if pair not in seen:
            seen.add(pair)
            result.append(pair)
    return result


def sequential_pairs(frames, overlap=1):
    """시간순 다음 overlap장과 연결한다. 실제 매칭 성공을 뜻하지 않는다."""
    if not isinstance(overlap, int) or overlap < 1:
        raise ValueError('overlap must be a positive integer')
    ordered = sorted(frames, key=lambda frame: frame['timestamp'])
    return [
        (first['file_path'], ordered[j]['file_path'])
        for i, first in enumerate(ordered)
        for j in range(i + 1, min(i + overlap + 1, len(ordered)))
    ]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('transforms', type=Path)
    parser.add_argument('--overlap', type=int, default=8,
                        help='Compare each frame with the next N frames (default: 8)')
    args = parser.parse_args()
    manifest = json.loads(args.transforms.read_text(encoding='utf-8'))
    frames = manifest['frames']
    pairs = sequential_pairs(frames, overlap=args.overlap)
    for first, second in unique_pairs(pairs):
        print(first, second)
