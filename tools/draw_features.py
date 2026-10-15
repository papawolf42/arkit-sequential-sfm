"""닫힌 COLMAP DB의 첫 이미지 특징점을 별도 PNG로 표시한다."""
import argparse
from pathlib import Path
import sqlite3

import numpy as np
from PIL import Image, ImageDraw


def draw_keypoints(image, keypoints):
    result = image.convert('RGB').copy()
    draw = ImageDraw.Draw(result)
    for point in keypoints:
        # COLMAP 픽셀 중심 (0.5, 0.5)을 Pillow 픽셀 인덱스 (0, 0)에 대응.
        x, y = point[0] - 0.5, point[1] - 0.5
        draw.ellipse((x - 1, y - 1, x + 1, y + 1), fill='red')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('database', type=Path)
    parser.add_argument('image_root', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a new output path')
    # 자신이 생성하고 추출이 종료된 DB만 사용한다. 활성 WAL DB에는 사용하지 않는다.
    uri = args.database.resolve().as_uri() + '?mode=ro&immutable=1'
    with sqlite3.connect(uri, uri=True) as connection:
        name, rows, cols, data = connection.execute(
            'SELECT images.name,keypoints.rows,keypoints.cols,keypoints.data '
            'FROM images JOIN keypoints USING(image_id) ORDER BY image_id LIMIT 1'
        ).fetchone()
    keypoints = np.frombuffer(data, dtype=np.float32).reshape(rows, cols)
    with Image.open(args.image_root / name) as image:
        result = draw_keypoints(image, keypoints)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.save(args.output)
    print(f'{rows} keypoints: {args.output}')
