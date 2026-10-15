import numpy as np

from tools.pose_residual import sampson_errors


def test_rectified_pair_has_zero_error_for_equal_vertical_coordinates():
    fundamental = np.array([[0, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=float)
    errors = sampson_errors([[10, 20], [10, 20]], [[15, 20], [15, 22]], fundamental)
    np.testing.assert_allclose(errors, [0, np.sqrt(2)])
