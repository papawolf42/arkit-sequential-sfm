"""고정 ARKit 포즈와 정제 트랙을 COLMAP legacy binary로 출력·native readback."""
import json
from pathlib import Path
import sqlite3
import struct

import numpy as np
import pycolmap


def export_model():
    output = Path('output/colmap-arkit-verified/sparse/0')
    if output.exists():
        raise FileExistsError(output)
    manifest = json.loads(Path('data/demo-day-latest/capture/transforms.json').read_text())
    frames = {f['file_path']: f for f in manifest['frames']}
    records = json.loads(Path('output/d6-triangulated/points.json').read_text())
    retained = json.loads(Path('output/d6-sparse/retained_track_ids.json').read_text())
    ply = Path('output/d6-sparse/sparse.ply').read_text().splitlines()
    vertices = np.array([[float(v) for v in line.split()] for line in ply[ply.index('end_header') + 1:]])
    assert len(vertices) == len(retained)
    database = Path('output/d6-matching/features.db')
    wal = Path(str(database) + '-wal')
    if wal.exists() and wal.stat().st_size:
        raise ValueError('pending WAL')
    with sqlite3.connect(database.resolve().as_uri() + '?mode=ro&immutable=1', uri=True) as db:
        names = dict(db.execute('SELECT image_id,name FROM images ORDER BY image_id'))
        keypoints = {i: np.frombuffer(blob, dtype=np.float32).reshape(rows, cols)[:, :2]
                     for i, rows, cols, blob in db.execute('SELECT image_id,rows,cols,data FROM keypoints')}
    backrefs = {i: {} for i in names}
    for point_id, track_id in enumerate(retained, 1):
        for image_id, index in records[track_id]['observations']:
            assert index not in backrefs[image_id]
            backrefs[image_id][index] = point_id
    output.mkdir(parents=True)
    poses = {}
    with (output / 'cameras.bin').open('wb') as file:
        file.write(struct.pack('<Q', len(names)))
        for image_id, name in names.items():
            f = frames[name]
            file.write(struct.pack('<iiQQ4d', image_id, 1, f['w'], f['h'],
                                   f['fl_x'], f['fl_y'], f['cx'], f['cy']))
    with (output / 'images.bin').open('wb') as file:
        file.write(struct.pack('<Q', len(names)))
        for image_id, name in names.items():
            c2w = np.asarray(frames[name]['transform_matrix']) @ np.diag([1, -1, -1, 1])
            # capture 회전은 float32 근사다. quaternion용 SO(3)에 투영한다.
            u, _, vt = np.linalg.svd(c2w[:3, :3])
            rotation_matrix = u @ np.diag([1, 1, np.linalg.det(u @ vt)]) @ vt
            pose = np.column_stack((rotation_matrix.T, -rotation_matrix.T @ c2w[:3, 3]))
            rotation = pycolmap.Rotation3d(pose[:3, :3])
            q = rotation.quat  # native xyzw -> binary wxyz
            poses[image_id] = np.linalg.inv(c2w)[:3]
            file.write(struct.pack('<i7di', image_id, q[3], *q[:3], *pose[:3, 3], image_id))
            file.write(name.encode('utf-8') + b'\0')
            file.write(struct.pack('<Q', len(keypoints[image_id])))
            for index, (x, y) in enumerate(keypoints[image_id]):
                file.write(struct.pack('<ddq', x, y, backrefs[image_id].get(index, -1)))
    with (output / 'points3D.bin').open('wb') as file:
        file.write(struct.pack('<Q', len(retained)))
        for point_id, (track_id, vertex) in enumerate(zip(retained, vertices), 1):
            r = records[track_id]
            np.testing.assert_array_equal(vertex[:3], r['xyz'])
            file.write(struct.pack('<Q3d3BdQ', point_id, *r['xyz'], *vertex[3:].astype(int),
                                   float(np.mean(r['reprojection_px'])), len(r['observations'])))
            for image_id, index in r['observations']:
                file.write(struct.pack('<ii', image_id, index))
    model = pycolmap.Reconstruction(output)
    assert len(model.images) == len(model.cameras) == len(names) == 240
    assert len(model.points3D) == len(retained) == 61025
    max_pose_delta, max_residual_delta, observations = 0., 0., 0
    for image_id, name in names.items():
        image = model.images[image_id]
        assert image.name == name
        np.testing.assert_allclose(image.points2D[0].xy, keypoints[image_id][0], atol=0, rtol=0)
        f = frames[name]
        np.testing.assert_array_equal(model.cameras[image.camera_id].params,
                                      [f['fl_x'], f['fl_y'], f['cx'], f['cy']])
        max_pose_delta = max(max_pose_delta, float(np.max(np.abs(
            image.cam_from_world().matrix() - poses[image_id]))))
    for point_id, track_id in enumerate(retained, 1):
        point, record = model.points3D[point_id], records[track_id]
        np.testing.assert_array_equal(point.xyz, record['xyz'])
        np.testing.assert_array_equal(point.color, vertices[point_id - 1, 3:])
        actual = [(e.image_id, e.point2D_idx) for e in point.track.elements]
        assert actual == [tuple(o) for o in record['observations']]
        for (image_id, index), old_error in zip(actual, record['reprojection_px']):
            image = model.images[image_id]
            observation = image.points2D[index]
            assert observation.point3D_id == point_id
            np.testing.assert_array_equal(observation.xy, keypoints[image_id][index])
            projected = model.cameras[image.camera_id].img_from_cam(image.cam_from_world() * point.xyz)
            error = float(np.linalg.norm(projected - observation.xy))
            max_residual_delta = max(max_residual_delta, abs(error - old_error))
            observations += 1
    print('pose/residual deltas:', max_pose_delta, max_residual_delta)
    # 비직교 float32 회전을 정규화한 차이: 잔차 허용 상한 0.01px.
    assert max_pose_delta < 1e-6 and max_residual_delta < 0.01
    unsupported = [names[i] for i in names if not backrefs[i]]
    assert len(unsupported) == 39
    report = {'images': len(model.images), 'cameras': len(model.cameras),
              'points': len(model.points3D), 'observations': observations,
              'unsupported_images': unsupported, 'max_pose_matrix_delta': max_pose_delta,
              'max_residual_delta_px': max_residual_delta,
              'note': 'No BA. Legacy binary read by installed pycolmap; original JPEGs not copied.'}
    (output.parent.parent / 'verification.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    export_model()
