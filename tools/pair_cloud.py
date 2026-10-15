"""첫 검증 쌍 전체를 삼각측량하고 필터 없는 진단·컬러 PLY를 저장한다."""
import argparse
import json
from pathlib import Path
import sqlite3

import numpy as np
from PIL import Image

from tools.inspect_capture import world_to_camera
from tools.triangulate_point import triangulate_point


def point_quality(xyz, pixels, frames):
    depths, errors, rays = [], [], []
    for pixel, frame in zip(pixels, frames):
        camera = world_to_camera(frame['transform_matrix']) @ np.append(xyz, 1)
        depth = -camera[2]
        depths.append(float(depth))
        if abs(depth) > 1e-12:
            projected = np.array([frame['fl_x'] * camera[0] / depth + frame['cx'],
                                  frame['cy'] - frame['fl_y'] * camera[1] / depth])
            errors.append(float(np.linalg.norm(projected - pixel)))
        else:
            errors.append(None)
        ray = np.asarray(xyz) - np.asarray(frame['transform_matrix'])[:3, 3]
        rays.append(ray / np.linalg.norm(ray))
    angle = np.degrees(np.arccos(np.clip(np.dot(*rays), -1, 1)))
    return {'depths': depths, 'front_both': all(d > 0 for d in depths),
            'reprojection_px': errors, 'angle_deg': float(angle)}


def export_pair(transforms, database, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError('Use a new output directory')
    database = Path(database)
    wal = Path(str(database) + '-wal')
    if wal.exists() and wal.stat().st_size:
        raise ValueError('Use a closed database without pending WAL')
    frames = {f['file_path']: f for f in json.loads(Path(transforms).read_text(encoding='utf-8'))['frames']}
    with sqlite3.connect(database.resolve().as_uri() + '?mode=ro&immutable=1', uri=True) as db:
        pair_id, count, data = db.execute(
            'SELECT pair_id,rows,data FROM two_view_geometries ORDER BY pair_id LIMIT 1').fetchone()
        matches = np.frombuffer(data, dtype=np.uint32).reshape(count, 2)
        ids = (pair_id // 2147483647, pair_id % 2147483647)
        points, selected, names = [], [], []
        for image_id in ids:
            name = db.execute('SELECT name FROM images WHERE image_id=?', (image_id,)).fetchone()[0]
            rows, cols, blob = db.execute('SELECT rows,cols,data FROM keypoints WHERE image_id=?',
                                         (image_id,)).fetchone()
            points.append(np.frombuffer(blob, dtype=np.float32).reshape(rows, cols)[:, :2])
            selected.append(frames[name])
            names.append(name)
    output.mkdir(parents=True)
    records, vertices = [], []
    with Image.open(Path(transforms).parent / names[0]) as image:
        image = image.convert('RGB')
        for i, j in matches:
            pixels = [points[0][i], points[1][j]]
            record = {'indices': [int(i), int(j)], 'pixels': [p.tolist() for p in pixels]}
            try:
                xyz = triangulate_point(*pixels, *selected)
                if not np.isfinite(xyz).all():
                    raise ValueError('nonfinite XYZ')
                record.update(xyz=xyz.tolist(), **point_quality(xyz, pixels, selected))
                x, y = np.floor(pixels[0]).astype(int)
                rgb = image.getpixel((int(np.clip(x, 0, image.width - 1)),
                                      int(np.clip(y, 0, image.height - 1))))
                vertices.append((*xyz.tolist(), *rgb))
                record['status'] = 'finite'
            except (ValueError, np.linalg.LinAlgError) as error:
                record.update(status='failed', reason=str(error))
            records.append(record)
    valid = [r for r in records if r['status'] == 'finite']
    def distribution(values):
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]
        return dict(zip(('min', 'p05', 'median', 'p95', 'max'),
                        np.percentile(values, [0, 5, 50, 95, 100]).tolist())) if len(values) else None
    summary = {'images': names, 'matches': count, 'finite': len(valid),
               'failed': count - len(valid), 'front_both': sum(r['front_both'] for r in valid),
               'angle_deg': distribution([r['angle_deg'] for r in valid]),
               'reprojection_px': [distribution([r['reprojection_px'][k] for r in valid
                                                if r['reprojection_px'][k] is not None]) for k in (0, 1)],
               'depths': [distribution([r['depths'][k] for r in valid]) for k in (0, 1)]}
    (output / 'points.json').write_text(json.dumps(records, indent=2, allow_nan=False), encoding='utf-8')
    (output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    header = f'ply\nformat ascii 1.0\nelement vertex {len(vertices)}\n'
    header += ''.join(f'property {kind} {name}\n' for kind, name in
                      [('double', 'x'), ('double', 'y'), ('double', 'z'),
                       ('uchar', 'red'), ('uchar', 'green'), ('uchar', 'blue')])
    (output / 'sparse.ply').write_text(header + 'end_header\n' + ''.join(
        ' '.join(map(str, vertex)) + '\n' for vertex in vertices), encoding='ascii')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('transforms', type=Path)
    parser.add_argument('database', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    export_pair(args.transforms, args.database, args.output)
