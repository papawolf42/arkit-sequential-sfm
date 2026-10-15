"""별도 복사 DB에서 첫 두 이미지 매칭·기하 검증 후 대응선을 표시한다."""
import argparse
from pathlib import Path
import shutil
import sqlite3
import colorsys

import numpy as np
from PIL import Image, ImageDraw
import pycolmap


def match_and_draw(source_database, image_root, output_dir):
    source_database = Path(source_database)
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError('Use a new output directory')
    wal = Path(str(source_database) + '-wal')
    if wal.exists() and wal.stat().st_size:
        raise ValueError('Source database must be closed with no pending WAL')
    output_dir.mkdir(parents=True)
    database = output_dir / 'features.db'
    shutil.copy2(source_database, database)
    with sqlite3.connect(database) as connection:
        images = connection.execute('SELECT image_id,name FROM images ORDER BY image_id').fetchall()
    if len(images) != 2:
        raise ValueError('Exactly two extracted images are required')
    pairs_file = output_dir / 'pairs.txt'
    pairs_file.write_text(f'{images[0][1]} {images[1][1]}\n', encoding='utf-8')
    pairing = pycolmap.ImportedPairingOptions()
    pairing.match_list_path = str(pairs_file)
    matching = pycolmap.FeatureMatchingOptions()
    matching.num_threads = 2
    matching.use_gpu = False
    pycolmap.match_image_pairs(database, matching_options=matching,
                              pairing_options=pairing, device=pycolmap.Device.cpu)
    with sqlite3.connect(database) as connection:
        raw_count = connection.execute('SELECT rows FROM matches').fetchone()[0]
        count, data = connection.execute('SELECT rows,data FROM two_view_geometries').fetchone()
        matches = np.frombuffer(data, dtype=np.uint32).reshape(count, 2)
        points = []
        for image_id, _ in images:
            rows, cols, blob = connection.execute(
                'SELECT rows,cols,data FROM keypoints WHERE image_id=?', (image_id,)
            ).fetchone()
            points.append(np.frombuffer(blob, dtype=np.float32).reshape(rows, cols))
    with Image.open(Path(image_root) / images[0][1]) as a, Image.open(
            Path(image_root) / images[1][1]) as b:
        canvas = Image.new('RGB', (a.width + b.width, max(a.height, b.height)))
        canvas.paste(a, (0, 0))
        canvas.paste(b, (a.width, 0))
        draw = ImageDraw.Draw(canvas)
        # 전부 그리면 복잡하므로 검증된 대응에서 균등 간격으로 최대 200개 표시.
        selected = matches[np.linspace(0, count - 1, min(count, 200), dtype=int)] if count else []
        for index, (i, j) in enumerate(selected):
            x, y = points[0][i, :2] - 0.5
            u, v = points[1][j, :2] - 0.5
            rgb = colorsys.hsv_to_rgb((index * 0.61803398875) % 1, 0.85, 1)
            color = tuple(round(channel * 255) for channel in rgb)
            start = (float(x), float(y))
            end = (float(u + a.width), float(v))
            draw.line((*start, *end), fill=color, width=1)
            for px, py in (start, end):
                draw.ellipse((px - 3, py - 3, px + 3, py + 3), fill=color)
        canvas.save(output_dir / 'matches.png')
    print(f'Stored raw matches: {raw_count}; geometrically verified: {count}; displayed: {len(selected)}')
    return raw_count, count


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('database', type=Path)
    parser.add_argument('image_root', type=Path)
    parser.add_argument('output_dir', type=Path)
    args = parser.parse_args()
    match_and_draw(args.database, args.image_root, args.output_dir)
