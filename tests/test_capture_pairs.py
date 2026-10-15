"""시간순으로 이웃한 촬영 이미지 쌍을 만든다."""
from tools.capture_pairs import sequential_pairs, unique_pairs


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
