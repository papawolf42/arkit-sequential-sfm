"""ARKit 자세에서 유도한 에피폴라 제약의 sqrt Sampson 근사 오차(px)."""
import argparse
import json
from pathlib import Path
import sqlite3

import numpy as np


def sampson_errors(points_a, points_b, fundamental):
    a = np.column_stack((points_a, np.ones(len(points_a))))
    b = np.column_stack((points_b, np.ones(len(points_b))))
    fa = a @ fundamental.T
    ftb = b @ fundamental
    numerator = np.sum(b * fa, axis=1) ** 2
    denominator = np.sum(fa[:, :2] ** 2 + ftb[:, :2] ** 2, axis=1)
    return np.sqrt(np.divide(numerator, denominator,
                            out=np.full_like(numerator, np.nan), where=denominator > 0))


def fundamental_from_frames(first, second):
    # OpenGL c2w → OpenCV c2w: 카메라 축만 변경하고 월드 축은 유지한다.
    axes = np.diag([1, -1, -1, 1])
    a = np.asarray(first['transform_matrix']) @ axes
    b = np.asarray(second['transform_matrix']) @ axes
    relative = np.linalg.inv(b) @ a
    rotation, translation = relative[:3, :3], relative[:3, 3]
    x, y, z = translation
    cross = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    def calibration(frame):
        return np.array([[frame['fl_x'], 0, frame['cx']],
                         [0, frame['fl_y'], frame['cy']], [0, 0, 1]])
    return np.linalg.inv(calibration(second)).T @ cross @ rotation @ np.linalg.inv(calibration(first))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('transforms', type=Path)
    parser.add_argument('database', type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.transforms.read_text(encoding='utf-8'))
    frames = {frame['file_path']: frame for frame in manifest['frames']}
    wal = Path(str(args.database) + '-wal')
    if wal.exists() and wal.stat().st_size:
        parser.error('Use a closed database without pending WAL')
    uri = args.database.resolve().as_uri() + '?mode=ro&immutable=1'
    with sqlite3.connect(uri, uri=True) as connection:
        pair_id, rows, blob = connection.execute(
            'SELECT pair_id,rows,data FROM two_view_geometries ORDER BY pair_id LIMIT 1'
        ).fetchone()
        image_ids = (pair_id // 2147483647, pair_id % 2147483647)
        matches = np.frombuffer(blob, dtype=np.uint32).reshape(rows, 2)
        points, selected_frames = [], []
        for image_id in image_ids:
            name = connection.execute('SELECT name FROM images WHERE image_id=?', (image_id,)).fetchone()[0]
            count, cols, data = connection.execute(
                'SELECT rows,cols,data FROM keypoints WHERE image_id=?', (image_id,)).fetchone()
            points.append(np.frombuffer(data, dtype=np.float32).reshape(count, cols)[:, :2])
            selected_frames.append(frames[name])
    # 픽셀 원점 관례 일치는 별도 검증 필요. 이번에는 DB 좌표를 그대로 사용한다.
    errors = sampson_errors(points[0][matches[:, 0]], points[1][matches[:, 1]],
                            fundamental_from_frames(*selected_frames))
    finite = errors[np.isfinite(errors)]
    print(json.dumps({'matches': rows, 'finite': len(finite),
                      'metric': 'sqrt_sampson_approx_px',
                      'median': float(np.median(finite)),
                      'p95': float(np.percentile(finite, 95)),
                      'max': float(np.max(finite))}, indent=2))
