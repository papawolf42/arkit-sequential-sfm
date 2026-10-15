import numpy as np

from tools.pair_cloud import point_quality


def test_quality_for_known_front_point():
    frame = {'transform_matrix': np.eye(4).tolist(), 'fl_x': 100,
             'fl_y': 100, 'cx': 50, 'cy': 50}
    second = dict(frame, transform_matrix=[[1, 0, 0, 1], [0, 1, 0, 0],
                                           [0, 0, 1, 0], [0, 0, 0, 1]])
    quality = point_quality([0, 0, -5], [[50, 50], [30, 50]], [frame, second])
    np.testing.assert_allclose(quality['depths'], [5, 5])
    np.testing.assert_allclose(quality['reprojection_px'], [0, 0])
    assert quality['front_both']
    np.testing.assert_allclose(quality['angle_deg'], np.degrees(np.arctan(0.2)))
