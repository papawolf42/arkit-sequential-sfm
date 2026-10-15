"""시간순으로 이웃한 촬영 이미지 쌍을 만든다."""
from tools.capture_pairs import baseline_pairs, sequential_pairs, unique_pairs


def test_baseline_pairs_filters_distance_and_view_direction():
    def frame(name, x, forward_sign=1):
        return {'file_path': name, 'timestamp': x, 'transform_matrix': [
            [forward_sign, 0, 0, x], [0, 1, 0, 0],
            [0, 0, forward_sign, 0], [0, 0, 0, 1],
        ]}
    frames = [frame('a', 0), frame('b', 0.1), frame('c', 0.5),
              frame('d', 1), frame('e', 1.1, -1)]
    assert baseline_pairs(frames, min_offset=2, max_offset=3) == [('a', 'c'), ('a', 'd'), ('b', 'd')]
    # 기본 설정도 두 프레임 차이의 유효한 후보를 놓치지 않아야 한다.
    assert ('a', 'c') in baseline_pairs(frames)


def test_sequential_window_includes_all_offsets_without_pose_filters():
    frames = [{'file_path': str(i), 'timestamp': i} for i in range(10)]
    pairs = sequential_pairs(frames, overlap=8)
    assert len(pairs) == 44
    assert ('0', '8') in pairs
    assert ('0', '9') not in pairs


def test_unique_pairs_removes_reverse_and_self_pairs():
    pairs = [('b.jpg', 'a.jpg'), ('a.jpg', 'b.jpg'), ('a.jpg', 'a.jpg'),
             ('b.jpg', 'c.jpg'), ('b.jpg', 'c.jpg')]
    assert unique_pairs(pairs) == [('a.jpg', 'b.jpg'), ('b.jpg', 'c.jpg')]
    assert unique_pairs([]) == []


def test_sequential_pairs_sorts_by_timestamp():
    frames = [
        {'file_path': 'c.jpg', 'timestamp': 3},
        {'file_path': 'a.jpg', 'timestamp': 1},
        {'file_path': 'b.jpg', 'timestamp': 2},
    ]
    assert sequential_pairs(frames) == [('a.jpg', 'b.jpg'), ('b.jpg', 'c.jpg')]
    assert frames[0]['file_path'] == 'c.jpg'


def test_sequential_pairs_needs_two_frames():
    assert sequential_pairs([]) == []
    assert sequential_pairs([{'file_path': 'a.jpg', 'timestamp': 1}]) == []
