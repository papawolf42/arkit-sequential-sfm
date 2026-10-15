"""카메라 좌표와 월드 좌표의 왕복 변환을 확인한다."""
import numpy as np

from tools.inspect_capture import project_world_point, world_to_camera


def test_project_world_point_uses_opengl_camera_axes():
    frame = {
        'transform_matrix': [
            [0, -1, 0, 10], [1, 0, 0, 2], [0, 0, 1, 3], [0, 0, 0, 1],
        ],
        'fl_x': 100, 'fl_y': 200, 'cx': 320, 'cy': 240,
    }
    # 카메라 좌표 (1, 2, -4)에 해당하는 월드 점.
    assert project_world_point([8, 3, -1], frame) == (345, 140)
    assert project_world_point([10, 2, -1], frame) == (320, 240)
    assert project_world_point([10, 2, 3], frame) is None
    assert project_world_point([10, 2, 4], frame) is None


def test_world_to_camera_round_trip():
    camera_to_world = np.array([
        [0, -1, 0, 10],
        [1, 0, 0, 2],
        [0, 0, 1, 3],
        [0, 0, 0, 1],
    ], dtype=float)
    inverse = world_to_camera(camera_to_world)

    np.testing.assert_allclose(inverse @ camera_to_world, np.eye(4), atol=1e-12)
    np.testing.assert_allclose(inverse @ [10, 2, 3, 1], [0, 0, 0, 1], atol=1e-12)
    point = np.array([1, 2, -3, 1])
    np.testing.assert_allclose(inverse @ (camera_to_world @ point), point, atol=1e-12)
