import io
import json
import zipfile
import stat
import sys
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image


@pytest.mark.parametrize('name', ['../escape', '/absolute', 'C:/evil', 'images\\evil',
                                   'images/../evil', 'images/CON', 'images/file:ads',
                                   'images/trailing.'])
def test_unsafe_zip_paths(tmp_path, name):
    with client(tmp_path) as c:
        assert c.post('/jobs', content=capture({name: b'x'}), headers=AUTH).status_code == 422
    assert not list(tmp_path.glob('*/capture'))


def test_symlink_rejected(tmp_path):
    body = io.BytesIO(capture())
    with zipfile.ZipFile(body, 'a') as z:
        info = zipfile.ZipInfo('images/link')
        info.create_system = 3
        info.external_attr = (stat.S_IFLNK | 0o777) << 16
        z.writestr(info, '../../outside')
    with client(tmp_path) as c:
        assert c.post('/jobs', content=body.getvalue(), headers=AUTH).status_code == 422


@pytest.mark.parametrize('options,extra,code', [
    ({'max_upload': 20}, {}, 413),
    ({'max_files': 1}, {}, 422),
    ({'max_extracted': 10}, {}, 422),
    ({'max_ratio': 2}, {'bomb.txt': b'a' * 100000}, 422),
    ({}, {'transforms.json': b'{}'}, 422),
    ({}, {'transforms.json': b'{"frames":[{"file_path":"../escape"}]}'}, 422),
    ({}, {'images/one.png': b'not image'}, 422),
])
def test_bounded_invalid_uploads(tmp_path, options, extra, code):
    with client(tmp_path, **options) as c:
        assert c.post('/jobs', content=capture(extra), headers=AUTH).status_code == code


def test_job_capacity_and_content_type(tmp_path):
    with client(tmp_path, max_jobs=1) as c:
        assert c.post('/jobs', content=capture(), headers={'Authorization': 'Bearer test-token'}).status_code == 415
        assert c.post('/jobs', content=capture(), headers=AUTH).status_code == 202
        assert c.post('/jobs', content=capture(), headers=AUTH).status_code == 429

def capture(entries=None):
    image = io.BytesIO()
    Image.new('RGB', (4, 4)).save(image, 'PNG')
    manifest = {'camera_model': 'OPENCV', 'fl_x': 10, 'fl_y': 10,
                'cx': 2, 'cy': 2, 'w': 4, 'h': 4,
                'frames': [{'file_path': 'images/one.png',
                            'transform_matrix': [[1, 0, 0, 0], [0, 1, 0, 0],
                                                 [0, 0, 1, 0], [0, 0, 0, 1]]}]}
    data = {'transforms.json': json.dumps(manifest).encode(), 'images/one.png': image.getvalue()}
    data.update(entries or {})
    out = io.BytesIO()
    with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED) as z:
        for name, value in data.items():
            info = zipfile.ZipInfo()
            info.filename = name  # Preserve malicious backslashes on Windows for safety tests.
            info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info, value)
    return out.getvalue()


def client(tmp_path, **kw):
    from server import Settings, create_app
    return TestClient(create_app(Settings(data_dir=tmp_path, api_token='test-token', **kw)))



@pytest.mark.parametrize('mode,expected', [('success', 'succeeded'), ('failure', 'failed'),
                                          ('missing', 'failed'), ('invalid', 'failed'),
                                          ('timeout', 'failed'), ('flood', 'failed')])
def test_fixture_subprocess_not_real_training(tmp_path, mode, expected):
    command = [sys.executable, str(Path(__file__).with_name('fixture_backend.py').resolve()),
               mode, '{capture}', '{artifact}']
    with client(tmp_path, command=command, timeout=1, max_log=4096) as c:
        job_id = c.post('/jobs', content=capture(), headers=AUTH).json()['id']
        deadline = time.monotonic() + 8
        while time.monotonic() < deadline:
            job = c.get('/jobs/' + job_id, headers=AUTH).json()
            if job['status'] in ('succeeded', 'failed'):
                break
            time.sleep(0.05)
        assert job['status'] == expected
        log = c.get(f'/jobs/{job_id}/logs', headers=AUTH).json()['log']
        assert len(log.encode()) <= 4096
        artifact = c.get(f'/jobs/{job_id}/artifact', headers=AUTH)
        assert artifact.status_code == (200 if expected == 'succeeded' else 409)
        if expected == 'succeeded':
            assert artifact.content.startswith(b'ply')
            assert job['progress'] == 1.0
            (tmp_path / job_id / 'output' / 'splat.ply').write_text('corrupt')
            assert c.get(f'/jobs/{job_id}/artifact', headers=AUTH).status_code == 409
        else:
            assert job['error']


AUTH = {'Authorization': 'Bearer test-token', 'Content-Type': 'application/zip'}


def test_duplicate_zip_and_restart_recovery(tmp_path):
    body = io.BytesIO(capture())
    with zipfile.ZipFile(body, 'a') as z:
        z.writestr('IMAGES/ONE.PNG', b'collision')
    with client(tmp_path) as c:
        assert c.post('/jobs', content=body.getvalue(), headers=AUTH).status_code == 422
        job_id = c.post('/jobs', content=capture(), headers=AUTH).json()['id']
    import sqlite3
    with sqlite3.connect(tmp_path / 'jobs.sqlite3') as db:
        db.execute("UPDATE jobs SET status='running' WHERE id=?", (job_id,))
    with client(tmp_path) as c:
        job = c.get('/jobs/' + job_id, headers=AUTH).json()
        assert job['status'] == 'failed'
        assert 'restarted' in job['error']


def test_missing_executable_fails_without_killing_worker(tmp_path):
    with client(tmp_path, command=[str(tmp_path / 'nonexistent-trainer')]) as c:
        for _ in range(2):
            job_id = c.post('/jobs', content=capture(), headers=AUTH).json()['id']
            deadline = time.monotonic() + 5
            job = {}
            while time.monotonic() < deadline:
                job = c.get('/jobs/' + job_id, headers=AUTH).json()
                if job['status'] == 'failed':
                    break
                time.sleep(0.05)
            assert job['status'] == 'failed'
            assert job['error']


def test_ui_and_configuration(tmp_path, monkeypatch):
    from server import Settings, create_app
    with client(tmp_path) as c:
        assert 'type="file"' in c.get('/').text
        for suffix in ('', '/logs', '/artifact'):
            assert c.get('/jobs/nope' + suffix).status_code == 401
        assert c.post('/jobs', content=b'bad zip', headers=AUTH).status_code == 422
    with pytest.raises(RuntimeError):
        create_app(Settings(data_dir=tmp_path, api_token=''))
    with pytest.raises(ValueError):
        create_app(Settings(data_dir=tmp_path, api_token='x', command='bad'))
    monkeypatch.setenv('TRAIN_MAX_STEPS', '500')
    monkeypatch.setenv('TRAIN_MAX_GAUSSIANS', '10000')
    assert Settings().max_steps == 500
    assert Settings().max_gaussians == 10000


def test_upload_auth_and_persistent_blocked_job(tmp_path):
    with client(tmp_path) as c:
        assert c.get('/health').json()['status'] == 'ok'
        assert c.post('/jobs', content=capture()).status_code == 401
        response = c.post('/jobs', content=capture(), headers=AUTH)
        assert response.status_code == 202
        job_id = response.json()['id']
        job = c.get(f'/jobs/{job_id}', headers=AUTH).json()
        assert job['status'] == 'blocked'
        assert 'backend' in job['error']
        assert c.get(f'/jobs/{job_id}/artifact', headers=AUTH).status_code == 409
    with client(tmp_path) as c:
        assert c.get(f'/jobs/{job_id}', headers=AUTH).json()['status'] == 'blocked'
        assert c.get('/jobs/not-an-id', headers=AUTH).status_code == 404
