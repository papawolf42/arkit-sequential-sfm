"""가정한 Z-depth 평면에서 화면 격자의 재투영 포함 비율을 구한다."""
import math

import numpy as np

from tools.inspect_capture import project_world_point


def directional_overlap(source, target, depth, grid_size):
    """source의 가상 점 중 target 화면에 들어가는 비율. 가림은 검사하지 않는다."""
    camera_to_world = np.asarray(source['transform_matrix'], dtype=float)
    inside = 0
    # 픽셀 중심 좌표 관례: 화면 영역 [-0.5, w-0.5) × [-0.5, h-0.5).
    for row in range(grid_size):
        v = (row + 0.5) * source['h'] / grid_size - 0.5
        for column in range(grid_size):
            u = (column + 0.5) * source['w'] / grid_size - 0.5
            camera_point = [
                (u - source['cx']) * depth / source['fl_x'],
                (source['cy'] - v) * depth / source['fl_y'],
                -depth, 1,
            ]
            world_point = (camera_to_world @ camera_point)[:3]
            pixel = project_world_point(world_point, target)
            if pixel is not None:
                target_u, target_v = pixel
                if (-0.5 <= target_u < target['w'] - 0.5
                        and -0.5 <= target_v < target['h'] - 0.5):
                    inside += 1
    return inside / (grid_size * grid_size)


def overlap_by_depth(first, second, depths, grid_size=10):
    """각 카메라의 전방 축 깊이(m)별 양방향 추정 비율을 반환한다.

    광선 거리가 아닌 로컬 -Z 깊이이며, 각 방향에서 별도의 가상 평면을 만든다.
    실제 표면 겹침률·IoU·가림 판정이 아니다. 유효한 PINHOLE/OpenGL c2w 입력을 전제한다.
    """
    if not isinstance(grid_size, int) or grid_size < 1:
        raise ValueError('grid_size must be a positive integer')
    depths = list(depths)
    if any(not math.isfinite(depth) or depth <= 0 for depth in depths):
        raise ValueError('depths must be finite and positive')
    return [
        {
            'depth_m': depth,
            'a_to_b': directional_overlap(first, second, depth, grid_size),
            'b_to_a': directional_overlap(second, first, depth, grid_size),
        }
        for depth in depths
    ]
