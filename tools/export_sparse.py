"""전방·최대 시차각 >=2도·모든 관측 잔차 <4px로 컬러 희소점 출력."""
import json
from pathlib import Path
import sqlite3

import numpy as np
from PIL import Image


def keep_point(point):
    errors = point.get('reprojection_px', [])
    return (point['status'] == 'finite' and point['front_all']
            and point['max_angle_deg'] >= 2 and bool(errors)
            and all(e is not None and np.isfinite(e) and e < 4 for e in errors))


def run():
    root = Path('output/d6-sparse')
    if root.exists():
        raise FileExistsError(root)
    records = json.loads(Path('output/d6-triangulated/points.json').read_text())
    valid = [p for p in records if p['status'] == 'finite']
    front = [p for p in valid if p['front_all']]
    angle = [p for p in front if p['max_angle_deg'] >= 2]
    kept = [p for p in records if keep_point(p)]
    database = Path('output/d6-matching/features.db')
    wal = Path(str(database) + '-wal')
    if wal.exists() and wal.stat().st_size:
        raise ValueError('pending WAL')
    with sqlite3.connect(database.resolve().as_uri() + '?mode=ro&immutable=1', uri=True) as db:
        names = dict(db.execute('SELECT image_id,name FROM images'))
        keypoints = {i: np.frombuffer(blob, dtype=np.float32).reshape(rows, cols)[:, :2]
                     for i, rows, cols, blob in db.execute('SELECT image_id,rows,cols,data FROM keypoints')}
    # 첫 관측의 RGB를 사용한다. 충돌 색 평균이나 점 병합은 하지 않는다.
    colors = {}
    for image_id in sorted({p['observations'][0][0] for p in kept}):
        with Image.open(Path('data/demo-day-latest/capture') / names[image_id]) as source:
            image = source.convert('RGB')
            for p in kept:
                i, j = p['observations'][0]
                if i != image_id:
                    continue
                x, y = np.floor(keypoints[i][j]).astype(int)
                colors[p['track_id']] = image.getpixel((int(np.clip(x, 0, image.width - 1)),
                                                        int(np.clip(y, 0, image.height - 1))))
    def stats(points):
        errors = [e for p in points for e in p['reprojection_px'] if e is not None and np.isfinite(e)]
        return {'points': len(points), 'observations': sum(len(p['observations']) for p in points),
                'supported_images': len({i for p in points for i, _ in p['observations']}),
                'residual_median_p95_max_px': np.percentile(errors, [50, 95, 100]).tolist() if errors else None}
    supported = {i for p in kept for i, _ in p['observations']}
    graph = {i: set() for i in names}
    for p in kept:
        ids = [i for i, _ in p['observations']]
        for i in ids:
            graph[i].update(set(ids) - {i})
    unseen, sizes = set(names), []
    while unseen:
        pending, component = [min(unseen)], set()
        while pending:
            i = pending.pop()
            if i not in component:
                component.add(i)
                pending.extend(graph[i] - component)
        unseen -= component
        sizes.append(len(component))
    report = {'policy': 'front_all AND max_angle_deg >= 2 AND every residual < 4px; no mean alternative',
              'before': stats(valid), 'after': stats(kept),
              'sequential_counts': {'finite': len(valid), 'front_all': len(front),
                                    'angle_at_least_2': len(angle), 'residual_below_4': len(kept)},
              'groups_after': {label: stats([p for p in kept if (len(p['observations']) == 2) == two])
                               for label, two in [('two_views', True), ('three_plus', False)]},
              'camera_count': len(names), 'coobserved_component_sizes': sorted(sizes, reverse=True),
              'unsupported_images': [name for i, name in names.items() if i not in supported]}
    root.mkdir(parents=True)
    header = f'ply\nformat ascii 1.0\nelement vertex {len(kept)}\n'
    header += ''.join(f'property {kind} {name}\n' for kind, name in
                      [('double', 'x'), ('double', 'y'), ('double', 'z'),
                       ('uchar', 'red'), ('uchar', 'green'), ('uchar', 'blue')])
    (root / 'sparse.ply').write_text(header + 'end_header\n' + ''.join(
        ' '.join(map(str, [*p['xyz'], *colors[p['track_id']]])) + '\n' for p in kept), encoding='ascii')
    (root / 'retained_track_ids.json').write_text(json.dumps([p['track_id'] for p in kept]))
    (root / 'summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    run()
