import numpy as np

from tools.inspect_capture import project_world_point
from tools.triangulate_point import triangulate_point


def test_triangulates_known_point_from_rotated_translated_cameras():
    first = {'transform_matrix': [[1, 0, 0, 0], [0, 1, 0, 0],
                                  [0, 0, 1, 0], [0, 0, 0, 1]],
             'fl_x': 400, 'fl_y': 450, 'cx': 320, 'cy': 240}
    second = dict(first, transform_matrix=[
        [0, -1, 0, 1], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    expected = [0.2, 0.3, -4]
    a = project_world_point(expected, first)
    b = project_world_point(expected, second)
    np.testing.assert_allclose(triangulate_point(a, b, first, second), expected, atol=1e-10)
