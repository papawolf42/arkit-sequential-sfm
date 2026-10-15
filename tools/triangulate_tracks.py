"""고정 ARKit 자세로 모든 트랙 DLT 추정. 품질 필터 없이 진단 저장."""
import json
from pathlib import Path
import sqlite3

import numpy as np

from tools.inspect_capture import world_to_camera


def triangulate_multiview(pixels, frames):
    equations = []
    for (u, v), frame in zip(pixels, frames):
        p = world_to_camera(frame['transform_matrix'])[:3]
        x = -(u - frame['cx']) / frame['fl_x']
        y = (v - frame['cy']) / frame['fl_y']
        equations.extend([x * p[2] - p[0], y * p[2] - p[1]])
    _, _, vt = np.linalg.svd(equations, full_matrices=False)
    if abs(vt[-1, 3]) < 1e-12:
        raise ValueError('point at infinity')
    return vt[-1, :3] / vt[-1, 3]


def quality(xyz, pixels, frames):
    depths, errors, rays = [], [], []
    for pixel, frame in zip(pixels, frames):
        point = world_to_camera(frame['transform_matrix']) @ np.append(xyz, 1)
        depth = -point[2]
        depths.append(float(depth))
        if abs(depth) < 1e-12:
            errors.append(None)
        else:
            projected = [frame['fl_x'] * point[0] / depth + frame['cx'],
                         frame['cy'] - frame['fl_y'] * point[1] / depth]
            errors.append(float(np.linalg.norm(np.asarray(projected) - pixel)))
        ray = xyz - np.asarray(frame['transform_matrix'])[:3, 3]
        norm = np.linalg.norm(ray)
        if norm == 0:
            raise ValueError('point at camera center')
        rays.append(ray / norm)
    cosine = np.clip(np.asarray(rays) @ np.asarray(rays).T, -1, 1)
    return depths, errors, float(np.degrees(np.arccos(cosine.min())))


def distribution(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return dict(zip(['min', 'median', 'p95', 'max'],
                    np.percentile(values, [0, 50, 95, 100]).tolist())) if len(values) else None


def run():
    output = Path('output/d6-triangulated')
    if output.exists():
        raise FileExistsError(output)
    tracks = json.loads(Path('output/d6-tracks/tracks.json').read_text())
    frames = {f['file_path']: f for f in json.loads(Path(
        'data/demo-day-latest/capture/transforms.json').read_text())['frames']}
    database = Path('output/d6-matching/features.db')
    wal = Path(str(database) + '-wal')
    if wal.exists() and wal.stat().st_size:
        raise ValueError('pending WAL')
    with sqlite3.connect(database.resolve().as_uri() + '?mode=ro&immutable=1', uri=True) as db:
        names = dict(db.execute('SELECT image_id,name FROM images'))
        keypoints = {i: np.frombuffer(blob, dtype=np.float32).reshape(rows, cols)[:, :2]
                     for i, rows, cols, blob in db.execute('SELECT image_id,rows,cols,data FROM keypoints')}
    records = []
    for track_id, track in enumerate(tracks):
        record = {'track_id': track_id, 'observations': track}
        pixels = [keypoints[i][j] for i, j in track]
        selected = [frames[names[i]] for i, _ in track]
        try:
            xyz = triangulate_multiview(pixels, selected)
            depths, errors, angle = quality(xyz, pixels, selected)
            if not np.isfinite(xyz).all() or not np.isfinite(depths).all() or not np.isfinite(angle):
                raise ValueError('nonfinite geometry')
            record.update(status='finite', xyz=xyz.tolist(), depths=depths,
                          front_all=all(d > 0 for d in depths),
                          reprojection_px=errors, max_angle_deg=angle)
        except (ValueError, np.linalg.LinAlgError) as error:
            record.update(status='failed', reason=str(error))
        records.append(record)
    summary = {'tracks': len(records), 'groups': {}}
    for label, predicate in [('two_views', lambda n: n == 2), ('three_plus', lambda n: n >= 3)]:
        group = [r for r in records if predicate(len(r['observations']))]
        valid = [r for r in group if r['status'] == 'finite']
        errors = [e for r in valid for e in r['reprojection_px'] if e is not None]
        summary['groups'][label] = {
            'tracks': len(group), 'finite': len(valid), 'failed': len(group) - len(valid),
            'front_all': sum(r['front_all'] for r in valid),
            'observations': sum(len(r['observations']) for r in group),
            'defined_residuals': len(errors),
            'reprojection_px_per_observation': distribution(errors),
            'track_max_reprojection_px': distribution([
                max(e for e in r['reprojection_px'] if e is not None)
                for r in valid if any(e is not None for e in r['reprojection_px'])]),
            'max_angle_deg_per_track': distribution([r['max_angle_deg'] for r in valid])}
    output.mkdir(parents=True)
    (output / 'points.json').write_text(json.dumps(records, allow_nan=False), encoding='utf-8')
    (output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    run()
