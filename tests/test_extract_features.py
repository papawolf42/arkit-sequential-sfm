"""추출 범위와 기존 DB 보호를 확인한다."""
import pytest

from tools.extract_features import extract_first_frame


def test_extract_two_frames_with_individual_intrinsics(tmp_path):
    import json
    import sqlite3
    from PIL import Image
    frames = []
    for i in range(2):
        name = f'{i}.png'
        Image.new('RGB', (32, 32), 'white').save(tmp_path / name)
        frames.append({'file_path': name, 'fl_x': 40 + i, 'fl_y': 40,
                       'cx': 16, 'cy': 16})
    manifest = tmp_path / 'transforms.json'
    manifest.write_text(json.dumps({'frames': frames}), encoding='utf-8')
    database = tmp_path / 'features.db'
    extract_first_frame(manifest, database, count=2)
    with sqlite3.connect(database) as connection:
        assert connection.execute('SELECT COUNT(*) FROM images').fetchone()[0] == 2
        assert connection.execute('SELECT COUNT(*) FROM cameras').fetchone()[0] == 2


def test_existing_database_is_not_overwritten(tmp_path):
    database = tmp_path / 'features.db'
    database.write_bytes(b'keep')
    with pytest.raises(FileExistsError):
        extract_first_frame(tmp_path / 'missing.json', database)
    assert database.read_bytes() == b'keep'
