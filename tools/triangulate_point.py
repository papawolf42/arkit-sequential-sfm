"""알려진 두 OpenGL 카메라와 픽셀 대응으로 점 하나를 선형 삼각측량한다."""
import numpy as np

from tools.inspect_capture import world_to_camera


def triangulate_point(pixel_a, pixel_b, first, second):
    """DLT/SVD 최소자승 추정. 퇴화·불안정한 자세에 대한 품질 판정은 별도다."""
    equations = []
    for pixel, frame in ((pixel_a, first), (pixel_b, second)):
        projection = world_to_camera(frame['transform_matrix'])[:3]
        u, v = pixel
        # OpenGL 전방 -Z, +Y 위쪽인 정규화 카메라 좌표.
        x = -(u - frame['cx']) / frame['fl_x']
        y = (v - frame['cy']) / frame['fl_y']
        equations.extend([x * projection[2] - projection[0],
                          y * projection[2] - projection[1]])
    _, _, vt = np.linalg.svd(equations)
    homogeneous = vt[-1]
    if abs(homogeneous[3]) < 1e-12:
        raise ValueError('Triangulated point is at infinity')
    return homogeneous[:3] / homogeneous[3]
