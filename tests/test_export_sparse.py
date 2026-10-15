from tools.export_sparse import keep_point


def test_strict_all_observation_filter_boundaries():
    point = dict(status='finite', front_all=True, max_angle_deg=2,
                 reprojection_px=[1, 3.99])
    assert keep_point(point)
    assert not keep_point(dict(point, reprojection_px=[1, 4]))
    assert not keep_point(dict(point, reprojection_px=[None, 1]))
    assert not keep_point(dict(point, front_all=False))
    assert not keep_point(dict(point, max_angle_deg=1.99))
