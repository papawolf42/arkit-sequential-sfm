"""검증 대응의 연결 성분을 트랙으로 묶고 이미지 중복 성분은 보류한다."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sqlite3

import numpy as np


def build_tracks(edges):
    parent = {}
    def find(node):
        parent.setdefault(node, node)
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node
    for a, b in edges:
        root_a, root_b = find(a), find(b)
        if root_a != root_b:
            parent[root_b] = root_a
    components = {}
    for node in parent:
        components.setdefault(find(node), []).append(node)
    tracks, conflicts = [], []
    for component in sorted(sorted(c) for c in components.values()):
        image_ids = [image for image, _ in component]
        target = tracks if len(set(image_ids)) == len(image_ids) else conflicts
        target.append(component)
    return tracks, conflicts


def export_tracks(database, output):
    database, output = Path(database), Path(output)
    if output.exists():
        raise FileExistsError('Use a new output directory')
    wal = Path(str(database) + '-wal')
    if wal.exists() and wal.stat().st_size:
        raise ValueError('Use a closed database without pending WAL')
    with sqlite3.connect(database.resolve().as_uri() + '?mode=ro&immutable=1', uri=True) as db:
        images = dict(db.execute('SELECT image_id,name FROM images'))
        def edges():
            for pair_id, count, blob in db.execute(
                    'SELECT pair_id,rows,data FROM two_view_geometries WHERE rows>0 ORDER BY pair_id'):
                a, b = divmod(pair_id, 2147483647)
                for i, j in np.frombuffer(blob, dtype=np.uint32).reshape(count, 2):
                    yield (a, int(i)), (b, int(j))
        tracks, conflicts = build_tracks(edges())
    supported = {image for track in tracks for image, _ in track}
    summary = {'tracks': len(tracks), 'observations': sum(map(len, tracks)),
               'length_histogram': dict(sorted(Counter(map(len, tracks)).items())),
               'conflicting_components': len(conflicts),
               'conflicting_observations': sum(map(len, conflicts)),
               'supported_images': len(supported),
               'unsupported_images': [name for key, name in images.items() if key not in supported],
               'policy': 'quarantine entire conflicting component; no greedy splitting or edge deletion'}
    output.mkdir(parents=True)
    for filename, value in [('tracks.json', tracks), ('conflicts.json', conflicts),
                            ('images.json', images), ('summary.json', summary)]:
        (output / filename).write_text(json.dumps(value, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('database', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    export_tracks(args.database, args.output)
