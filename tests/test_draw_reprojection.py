from PIL import Image
from tools.draw_reprojection import draw_residuals


def test_marks_observation_and_projection_without_changing_source():
    source = Image.new('RGB', (40, 40))
    result = draw_residuals(source, [([10.5, 10.5], [20.5, 20.5])])
    assert result.getpixel((10, 7)) == (0, 255, 255)
    assert result.getpixel((20, 20)) == (255, 0, 255)
    assert source.getpixel((20, 20)) == (0, 0, 0)
