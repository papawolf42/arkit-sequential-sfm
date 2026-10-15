"""특징점의 픽셀 중심을 이미지에 표시한다."""
from PIL import Image

from tools.draw_features import draw_keypoints


def test_draw_keypoints_keeps_source_unchanged():
    source = Image.new('RGB', (20, 20), 'black')
    result = draw_keypoints(source, [[10.5, 10.5]])
    assert result.size == source.size
    assert result.getpixel((10, 10)) == (255, 0, 0)
    assert source.getpixel((10, 10)) == (0, 0, 0)
