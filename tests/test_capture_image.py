"""저장 이미지와 manifest의 크기·방향 메타데이터를 비교한다."""
from PIL import Image

from tools.inspect_capture import inspect_image_metadata


def test_image_metadata_reports_size_mismatch_without_rotating(tmp_path):
    path = tmp_path / 'image.jpg'
    Image.new('RGB', (8, 6)).save(path)
    frame = {'w': 8, 'h': 6, 'image_orientation': 1}
    assert inspect_image_metadata(path, frame) == {
        'stored_size': [8, 6], 'size_matches': True,
        'manifest_orientation': 1, 'exif_orientation': None,
    }
    frame['w'] = 6
    assert inspect_image_metadata(path, frame)['size_matches'] is False
    assert Image.open(path).size == (8, 6)
