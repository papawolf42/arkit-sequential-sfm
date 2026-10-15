"""첫 학습 단계: 실제 명령으로 캡처 JSON을 읽는다."""
import json
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize('image_exists', [True, False])
def test_inspect_capture_prints_count_first_frame_and_image_existence(tmp_path, image_exists):
    frame = {
        'file_path': 'images/example.jpg', 'timestamp': 12.5,
        'transform_matrix': [
            [0, -1, 0, 10],
            [1, 0, 0, 2],
            [0, 0, 1, 3],
            [0, 0, 0, 1],
        ],
    }
    capture = tmp_path / 'transforms.json'
    capture.write_text(json.dumps({'frames': [frame]}), encoding='utf-8')
    if image_exists:
        image = tmp_path / frame['file_path']
        image.parent.mkdir()
        image.write_bytes(b'file existence only; image decoding is a later step')
    script = Path(__file__).resolve().parents[1] / 'tools' / 'inspect_capture.py'

    result = subprocess.run(
        [sys.executable, str(script), str(capture)],
        capture_output=True, text=True, check=False,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {
        'frame_count': 1, 'first_frame': frame, 'first_image_exists': image_exists,
        'missing_images': [] if image_exists else [frame['file_path']],
        'first_camera_position': [10, 2, 3],
        'first_pair_distance_m': None,
        'first_camera_forward': [0, 0, -1],
    }


def test_inspect_capture_finds_missing_second_image(tmp_path):
    frames = [
        {'file_path': 'first.jpg', 'transform_matrix': [
            [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1],
        ]},
        {'file_path': 'second.jpg', 'transform_matrix': [
            [1, 0, 0, 3], [0, 1, 0, 4], [0, 0, 1, 12], [0, 0, 0, 1],
        ]},
    ]
    capture = tmp_path / 'transforms.json'
    capture.write_text(json.dumps({'frames': frames}), encoding='utf-8')
    (tmp_path / 'first.jpg').write_bytes(b'file existence only')
    script = Path(__file__).resolve().parents[1] / 'tools' / 'inspect_capture.py'

    result = subprocess.run(
        [sys.executable, str(script), str(capture)],
        capture_output=True, text=True, check=False,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)['missing_images'] == ['second.jpg']
    assert json.loads(result.stdout)['first_pair_distance_m'] == 13
