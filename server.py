"""Single-process capture ingestion server. Training is never simulated."""
import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
import hmac
import json
import os
from pathlib import Path
import sqlite3
import threading
import time
import uuid

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse


@dataclass
class Settings:
    data_dir: Path = field(default_factory=lambda: Path(os.getenv('CAPTURE_DATA_DIR', 'data')))
    api_token: str = field(default_factory=lambda: os.getenv('CAPTURE_API_TOKEN', ''))
    command: list[str] = field(default_factory=lambda: json.loads(os.getenv('TRAIN_COMMAND_JSON', '[]')))
    max_upload: int = 256 * 1024 * 1024
    max_extracted: int = 512 * 1024 * 1024
    max_files: int = 2000
    max_ratio: int = 100
    max_jobs: int = 20
    timeout: float = 3600
    max_log: int = 2 * 1024 * 1024
    max_artifact: int = 1024 * 1024 * 1024
    max_steps: int = field(default_factory=lambda: int(os.getenv('TRAIN_MAX_STEPS', '1000')))
    max_gaussians: int = field(default_factory=lambda: int(os.getenv('TRAIN_MAX_GAUSSIANS', '100000')))
    image_downscale: int = field(default_factory=lambda: int(os.getenv('TRAIN_IMAGE_DOWNSCALE', '4')))

    def __post_init__(self):
        if not isinstance(self.command, list) or any(not isinstance(x, str) or not x for x in self.command):
            raise ValueError('TRAIN_COMMAND_JSON must be an argv JSON array of nonempty strings')
        for value in (self.max_upload, self.max_extracted, self.max_files, self.max_ratio,
                      self.max_jobs, self.timeout, self.max_log, self.max_artifact,
                      self.max_steps, self.max_gaussians, self.image_downscale):
            if value <= 0:
                raise ValueError('Resource limits must be positive')


class Store:
    def __init__(self, root):
        self.path = root / 'jobs.sqlite3'
        with self.connect() as db:
            db.execute('CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, status TEXT, '
                       'error TEXT, progress REAL, created REAL)')
            db.execute("UPDATE jobs SET status='failed', error='Server restarted during training' "
                       "WHERE status='running'")

    def connect(self):
        return sqlite3.connect(self.path, timeout=30)

    def get(self, job_id):
        with self.connect() as db:
            db.row_factory = sqlite3.Row
            row = db.execute('SELECT * FROM jobs WHERE id=?', (job_id,)).fetchone()
        if row is None:
            raise HTTPException(404, 'Job not found')
        return dict(row)

    def update(self, job_id, status, error=None, progress=None):
        with self.connect() as db:
            db.execute('UPDATE jobs SET status=?, error=?, progress=? WHERE id=?',
                       (status, error, progress, job_id))


def create_app(settings=None):
    s = settings or Settings()
    if not s.api_token:
        raise RuntimeError('CAPTURE_API_TOKEN must be set; refusing unauthenticated startup')
    s.data_dir = Path(s.data_dir).resolve()
    s.data_dir.mkdir(parents=True, exist_ok=True)
    store = Store(s.data_dir)
    stop = threading.Event()
    admission = asyncio.Lock()

    @asynccontextmanager
    async def lifespan(app):
        worker = threading.Thread(target=worker_loop, args=(s, store, stop), daemon=True)
        worker.start()
        yield
        stop.set()
        await asyncio.to_thread(worker.join, 10)

    app = FastAPI(title='Capture Training Prototype', lifespan=lifespan)

    def auth(request):
        supplied = request.headers.get('authorization', '')
        if not hmac.compare_digest(supplied.encode(), ('Bearer ' + s.api_token).encode()):
            raise HTTPException(401, 'Bearer token required')

    @app.get('/health')
    def health():
        return {'status': 'ok', 'backend_configured': bool(s.command)}

    @app.get('/')
    def index():
        return HTMLResponse(Path(__file__).with_name('index.html').read_text(encoding='utf-8'))

    @app.post('/jobs', status_code=202)
    async def upload(request: Request):
        auth(request)
        if request.headers.get('content-type', '').split(';')[0] != 'application/zip':
            raise HTTPException(415, 'Use application/zip raw request body')
        async with admission:
            with store.connect() as db:
                count = db.execute('SELECT COUNT(*) FROM jobs').fetchone()[0]
            if count >= s.max_jobs:
                raise HTTPException(429, 'Job storage limit reached; operator cleanup required')
            job_id = uuid.uuid4().hex
            folder = s.data_dir / job_id
            folder.mkdir()
            try:
                archive = folder / 'capture.zip'
                size = 0
                with archive.open('wb') as f:
                    async for chunk in request.stream():
                        size += len(chunk)
                        if size > s.max_upload:
                            raise HTTPException(413, 'Upload exceeds limit')
                        f.write(chunk)
                await asyncio.to_thread(extract_capture, archive, folder / 'capture', s)
                archive.unlink()
                status = 'queued' if s.command else 'blocked'
                error = None if s.command else 'No training backend configured (TRAIN_COMMAND_JSON)'
                with store.connect() as db:
                    db.execute('INSERT INTO jobs VALUES (?,?,?,?,?)',
                               (job_id, status, error, None, time.time()))
            except BaseException:
                import shutil
                shutil.rmtree(folder, ignore_errors=True)
                raise
        return store.get(job_id)

    @app.get('/jobs/{job_id}')
    def status(job_id: str, request: Request):
        auth(request)
        return store.get(job_id)

    @app.get('/jobs/{job_id}/logs')
    def logs(job_id: str, request: Request):
        auth(request)
        store.get(job_id)
        path = s.data_dir / job_id / 'train.log'
        return {'log': path.read_text(errors='replace') if path.exists() else ''}

    @app.get('/jobs/{job_id}/artifact')
    def artifact(job_id: str, request: Request):
        auth(request)
        job = store.get(job_id)
        path = s.data_dir / job_id / 'output' / 'splat.ply'
        if job['status'] != 'succeeded' or not valid_ply(path, s.max_artifact):
            raise HTTPException(409, 'No validated splat artifact available')
        return FileResponse(path, media_type='application/octet-stream', filename='splat.ply')

    return app


def safe_name(name):
    import re
    if not isinstance(name, str) or not name or len(name) > 240:
        raise ValueError('Invalid path')
    parts = name.split('/')
    for part in parts:
        if (part in ('', '.', '..') or part[-1:] in ('.', ' ')
                or re.search(r'[\\\\:\x00-\x1f<>"|?*]', part)
                or part.split('.')[0].upper() in {'CON', 'PRN', 'AUX', 'NUL',
                    *(f'COM{i}' for i in range(10)), *(f'LPT{i}' for i in range(10))}):
            raise ValueError('Unsafe path')
    return name


def extract_capture(archive, target, settings):
    import stat
    import warnings
    import zipfile
    from PIL import Image
    import numpy as np
    try:
        target.mkdir()
        total = 0
        seen = set()
        with zipfile.ZipFile(archive) as z:
            infos = z.infolist()
            if len(infos) > settings.max_files:
                raise ValueError('Too many ZIP entries')
            for info in infos:
                if '\\' in info.orig_filename or '\x00' in info.orig_filename:
                    raise ValueError('Unsafe original ZIP filename')
                name = safe_name(info.filename.rstrip('/') if info.is_dir() else info.filename)
                key = name.casefold()
                mode = stat.S_IFMT(info.external_attr >> 16)
                if (key in seen or mode not in (0, stat.S_IFREG, stat.S_IFDIR)
                        or info.flag_bits & 1
                        or info.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED)
                        or info.file_size > settings.max_extracted
                        or info.file_size > max(1, info.compress_size) * settings.max_ratio):
                    raise ValueError('Unsafe ZIP entry, duplicate, or compression ratio')
                seen.add(key)
                total += info.file_size
                if total > settings.max_extracted:
                    raise ValueError('Expanded ZIP exceeds limit')
            actual = 0
            for info in infos:
                dest = target / info.filename
                if info.is_dir():
                    dest.mkdir(parents=True, exist_ok=True)
                    continue
                dest.parent.mkdir(parents=True, exist_ok=True)
                with z.open(info) as src, dest.open('xb') as dst:
                    while chunk := src.read(65536):
                        actual += len(chunk)
                        if actual > settings.max_extracted:
                            raise ValueError('Expanded ZIP exceeds limit')
                        dst.write(chunk)
        manifest_path = target / 'transforms.json'
        if manifest_path.stat().st_size > 8 * 1024 * 1024:
            raise ValueError('Manifest too large')
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        frames = manifest.get('frames')
        if not isinstance(frames, list) or not 1 <= len(frames) <= settings.max_files:
            raise ValueError('Nonempty frames required')
        for frame in frames:
            name = frame['file_path']
            if name.startswith('./'):
                name = name[2:]
            image = target / safe_name(name)
            if not name.startswith('images/') or image.suffix.lower() not in ('.png', '.jpg', '.jpeg'):
                raise ValueError('Frame must reference an image under images/')
            matrix = np.asarray(frame['transform_matrix'], dtype=float)
            if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
                raise ValueError('Finite 4x4 camera transform required')
            if not np.allclose(matrix[3], [0, 0, 0, 1]):
                raise ValueError('Invalid homogeneous transform')
            for key in ('fl_x', 'fl_y', 'cx', 'cy', 'w', 'h'):
                value = frame.get(key, manifest.get(key))
                if not isinstance(value, (int, float)) or not np.isfinite(value):
                    raise ValueError('Camera intrinsics required')
                if key in ('fl_x', 'fl_y', 'w', 'h') and value <= 0:
                    raise ValueError('Positive focal length and dimensions required')
            with warnings.catch_warnings():
                warnings.simplefilter('error', Image.DecompressionBombWarning)
                with Image.open(image) as im:
                    if im.width * im.height > 40_000_000:
                        raise ValueError('Image pixel limit exceeded')
                    if (im.width != frame.get('w', manifest.get('w'))
                            or im.height != frame.get('h', manifest.get('h'))):
                        raise ValueError('Image dimensions disagree with intrinsics')
                    im.verify()
    except (ValueError, KeyError, TypeError, AttributeError, OSError, RuntimeError, EOFError,
            zipfile.BadZipFile, Image.DecompressionBombWarning, Image.DecompressionBombError) as exc:
        raise HTTPException(422, 'Invalid capture: ' + str(exc)[:200]) from exc


def worker_loop(settings, store, stop):
    while not stop.wait(0.1):
        with store.connect() as db:
            row = db.execute("SELECT id FROM jobs WHERE status='queued' ORDER BY created LIMIT 1").fetchone()
        if not row:
            continue
        job_id = row[0]
        store.update(job_id, 'running')
        try:
            run_training(settings, store, stop, job_id)
        except Exception as exc:
            store.update(job_id, 'failed', str(exc)[:1000])


def kill_process_tree(process):
    import signal
    import subprocess
    if os.name == 'nt':
        subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    else:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    if process.poll() is None:
        process.kill()
    process.wait(timeout=5)


def run_training(s, store, stop, job_id):
    import queue
    import subprocess
    root = s.data_dir / job_id
    output = root / 'output'
    output.mkdir(exist_ok=True)
    artifact = output / 'splat.ply'
    values = {'capture': str(root / 'capture'), 'output': str(output),
              'artifact': str(artifact), 'job_id': job_id, 'max_steps': str(s.max_steps),
              'max_gaussians': str(s.max_gaussians), 'image_downscale': str(s.image_downscale)}
    argv = [arg.format_map(values) for arg in s.command]
    env = os.environ.copy()
    env.pop('CAPTURE_API_TOKEN', None)
    env.pop('TRAIN_COMMAND_JSON', None)
    process = subprocess.Popen(argv, shell=False, cwd=root, env=env,
                               stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, start_new_session=os.name != 'nt')
    chunks = queue.Queue(maxsize=32)
    reader_stop = threading.Event()

    def read_output():
        try:
            while not reader_stop.is_set():
                data = process.stdout.read1(4096)
                if not data:
                    break
                while not reader_stop.is_set():
                    try:
                        chunks.put(data, timeout=0.1)
                        break
                    except queue.Full:
                        pass
        finally:
            process.stdout.close()

    reader = threading.Thread(target=read_output, daemon=True)
    reader.start()
    start = time.monotonic()
    written = 0
    pending = b''
    error = None
    try:
        with (root / 'train.log').open('wb') as log:
            while process.poll() is None or reader.is_alive() or not chunks.empty():
                if stop.is_set():
                    error = 'Server shutting down'
                elif time.monotonic() - start > s.timeout:
                    error = 'Training timeout exceeded'
                elif artifact.exists() and artifact.stat().st_size > s.max_artifact:
                    error = 'Artifact size limit exceeded'
                if error:
                    break
                try:
                    chunk = chunks.get(timeout=0.1)
                except queue.Empty:
                    continue
                remaining = s.max_log - written
                log.write(chunk[:remaining])
                log.flush()
                written += min(len(chunk), remaining)
                if len(chunk) > remaining:
                    error = 'Training log limit exceeded'
                    break
                pending += chunk
                while b'\n' in pending:
                    line, pending = pending.split(b'\n', 1)
                    try:
                        value = json.loads(line).get('progress')
                        if isinstance(value, (int, float)) and 0 <= value <= 1:
                            store.update(job_id, 'running', progress=float(value))
                    except (ValueError, AttributeError):
                        pass
                pending = pending[-4096:]
    finally:
        # Kill the process group even after a wrapper exits, preventing orphaned trainers.
        kill_process_tree(process)
        reader_stop.set()
        reader.join(timeout=5)
    if error:
        raise RuntimeError(error)
    if process.returncode != 0:
        raise RuntimeError(f'Training command exited with code {process.returncode}')
    if not valid_ply(artifact, s.max_artifact):
        raise RuntimeError('Training produced no valid Gaussian splat PLY at output/splat.ply')
    store.update(job_id, 'succeeded', progress=1.0)


def valid_ply(path, limit):
    import numpy as np
    from plyfile import PlyData
    try:
        if path.is_symlink() or not path.is_file() or not 0 < path.stat().st_size <= limit:
            return False
        # Reject hostile vertex counts / list properties before the parser allocates arrays.
        with path.open('rb') as f:
            header = bytearray()
            while len(header) < 65536:
                line = f.readline(1024)
                header.extend(line)
                if line.strip() == b'end_header':
                    break
                if not line:
                    return False
            else:
                return False
        text = header.decode('ascii').replace('\r\n', '\n')
        import re
        vertices = re.findall(r'^element vertex (\d+)$', text, re.M)
        if len(vertices) != 1 or not 0 < int(vertices[0]) <= 10_000_000:
            return False
        if 'property list' in text or len(re.findall(r'^element ', text, re.M)) != 1:
            return False
        # Fixed scalar properties require at least one byte each, even for ASCII.
        if int(vertices[0]) * len(re.findall(r'^property ', text, re.M)) > path.stat().st_size:
            return False
        ply = PlyData.read(str(path), mmap='r')
        data = ply['vertex'].data
        required = ['x', 'y', 'z', 'f_dc_0', 'f_dc_1', 'f_dc_2', 'opacity',
                    'scale_0', 'scale_1', 'scale_2', 'rot_0', 'rot_1', 'rot_2', 'rot_3']
        if not set(required).issubset(data.dtype.names or ()):
            return False
        for name in data.dtype.names:
            if data[name].dtype.kind != 'f' or not np.isfinite(data[name]).all():
                return False
        rotations = np.column_stack([data[f'rot_{i}'] for i in range(4)])
        return bool((np.linalg.norm(rotations, axis=1) > 0).all())
    except (OSError, ValueError, KeyError, UnicodeError, OverflowError):
        return False


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(create_app(), host='127.0.0.1', port=8000, workers=1)
