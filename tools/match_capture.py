"""전체 시간 인접 후보를 별도 DB에서 매칭한다."""
import json
import os
from pathlib import Path
import shutil
import sqlite3
import time

import pycolmap

from tools.capture_pairs import sequential_pairs, unique_pairs


def run():
    root = Path('output/d6-matching')
    root.mkdir(parents=True, exist_ok=False)
    source = Path('output/d6-all240/features.db')
    wal = Path(str(source) + '-wal')
    if wal.exists() and wal.stat().st_size:
        raise ValueError('Source DB must be closed without pending WAL')
    database = root / 'features.db'
    shutil.copy2(source, database)
    frames = json.loads(Path('data/demo-day-latest/capture/transforms.json').read_text())['frames']
    pairs = unique_pairs(sequential_pairs(frames, overlap=8))
    assert len(frames) == 240 and len(pairs) == 1884
    pair_file = root / 'pairs.txt'
    pair_file.write_text(''.join(f'{a} {b}\n' for a, b in pairs), encoding='utf-8')
    cuda = bool(pycolmap.has_cuda)
    threads = min(16, os.cpu_count() or 1)
    matching = pycolmap.FeatureMatchingOptions()
    matching.use_gpu = cuda
    matching.num_threads = threads
    pairing = pycolmap.ImportedPairingOptions()
    pairing.match_list_path = str(pair_file)
    pairing.block_size = 64
    verification = pycolmap.TwoViewGeometryOptions()
    start = time.perf_counter()
    print(f'START pairs={len(pairs)} cuda={cuda} threads={threads}', flush=True)
    pycolmap.match_image_pairs(database, matching_options=matching,
                              pairing_options=pairing, verification_options=verification,
                              device=pycolmap.Device.cuda if cuda else pycolmap.Device.cpu)
    elapsed = time.perf_counter() - start
    with sqlite3.connect(database) as db:
        images = dict(db.execute('SELECT image_id,name FROM images'))
        raw = dict(db.execute('SELECT pair_id,rows FROM matches'))
        geometry = dict(db.execute('SELECT pair_id,rows FROM two_view_geometries'))
    name_ids = {name: key for key, name in images.items()}
    expected = {min(name_ids[a], name_ids[b]) * 2147483647 + max(name_ids[a], name_ids[b])
                for a, b in pairs}
    assert set(raw) == set(geometry) == expected
    graph = {key: set() for key in images}
    for pair_id, rows in geometry.items():
        if rows:
            a, b = divmod(pair_id, 2147483647)
            graph[a].add(b)
            graph[b].add(a)
    components, unseen = [], set(images)
    while unseen:
        pending, component = [min(unseen)], set()
        while pending:
            node = pending.pop()
            if node in component:
                continue
            component.add(node)
            pending.extend(graph[node] - component)
        unseen -= component
        components.append(sorted(images[node] for node in component))
    report = {'candidates': len(pairs), 'stored_raw_pairs': len(raw),
              'stored_geometry_pairs': len(geometry),
              'verified_nonempty_pairs': sum(n > 0 for n in geometry.values()),
              'stored_raw_matches': sum(raw.values()), 'verified_matches': sum(geometry.values()),
              'cuda': cuda, 'configured_threads': threads, 'matching_seconds': elapsed,
              'component_sizes': sorted([len(c) for c in components], reverse=True),
              'components': components,
              'isolated_images': [images[key] for key in graph if not graph[key]]}
    (root / 'summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    run()
