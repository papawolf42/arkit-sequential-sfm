import numpy as np

from tools.inspect_capture import project_world_point
from tools.triangulate_tracks import quality, triangulate_multiview


def test_three_view_point_and_quality():
    frames = []
    for x in [0, 1, 2]:
        frames.append({'transform_matrix': [[1, 0, 0, x], [0, 1, 0, 0],
                                            [0, 0, 1, 0], [0, 0, 0, 1]],
                       'fl_x': 400, 'fl_y': 450, 'cx': 320, 'cy': 240})
    expected = np.array([0.2, 0.3, -4])
    pixels = [project_world_point(expected, f) for f in frames]
    xyz = triangulate_multiview(pixels, frames)
    np.testing.assert_allclose(xyz, expected, atol=1e-10)
    depths, errors, angle = quality(xyz, pixels, frames)
    assert all(d > 0 for d in depths)
    np.testing.assert_allclose(errors, 0, atol=1e-10)
    assert 20 < angle < 30
