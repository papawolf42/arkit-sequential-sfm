"""카메라 좌표와 월드 좌표의 왕복 변환을 확인한다."""
import numpy as np

from tools.inspect_capture import world_to_camera


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
