"""가정 깊이에서의 격자 기반 양방향 시야 겹침을 검증한다."""
import pytest

from tools.view_overlap import overlap_by_depth


def camera(x=0, opposite=False):
    sign = -1 if opposite else 1
    return {
        'w': 100, 'h': 100, 'fl_x': 100, 'fl_y': 100, 'cx': 50, 'cy': 50,
        'transform_matrix': [
            [sign, 0, 0, x], [0, 1, 0, 0],
            [0, 0, sign, 0], [0, 0, 0, 1],
        ],
    }


def test_identical_views_overlap_fully():
    assert overlap_by_depth(camera(), camera(), [1], grid_size=10) == [
        {'depth_m': 1, 'a_to_b': 1, 'b_to_a': 1},
    ]


def test_translation_overlap_changes_with_depth():
    rows = overlap_by_depth(camera(), camera(x=0.5), [1, 2], grid_size=10)
    assert rows[0]['a_to_b'] == pytest.approx(0.5)
    assert rows[0]['b_to_a'] == pytest.approx(0.5)
    assert rows[1]['a_to_b'] > rows[0]['a_to_b']
    assert rows[1]['b_to_a'] > rows[0]['b_to_a']


def test_opposite_views_do_not_overlap():
    row = overlap_by_depth(camera(), camera(opposite=True), [1])[0]
    assert row['a_to_b'] == row['b_to_a'] == 0


@pytest.mark.parametrize('depths, grid', [([0], 10), ([-1], 10), ([1], 0)])
def test_invalid_sampling_parameters(depths, grid):
    with pytest.raises(ValueError):
        overlap_by_depth(camera(), camera(), depths, grid_size=grid)
