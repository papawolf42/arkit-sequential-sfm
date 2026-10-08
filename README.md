# Capture training server prototype

Working ingestion/job orchestration; **no bundled trainer, no GPU training validation**. Missing `TRAIN_COMMAND_JSON` produces persistent `blocked` jobs, never fake success. Tested on Windows with Python 3.12. Portable to Linux/WSL; trainer executable/environment is operator-provided. No SSH or global CUDA installation is needed by this API.

## Run

```bash
uv sync --python 3.12
# Set a long random token in your shell/environment; do not commit it.
export CAPTURE_API_TOKEN='<your-secret-token>'
uv run python server.py
```

Default: `http://127.0.0.1:8000/`, one process, one worker. Startup refuses an empty token. `CAPTURE_DATA_DIR` defaults to `./data`; use a private local disk directory. The web UI accepts ZIPs and offers status/log refresh and authenticated PLY download. Token remains in the page memory only, not localStorage or URL. UI is minimal and has no automatic polling.

For LAN/mobile access, terminate TLS on a reverse proxy and explicitly bind the app if needed:

```bash
uv run uvicorn server:create_app --factory --host 0.0.0.0 --port 8000 --workers 1
```

**Do not use multiple workers, multiple instances sharing data, or reload during training.** This is a single-process prototype. Reverse proxy must impose body size (256 MiB or less), header/body read deadlines, connection limits, and request rate limits. Apply firewall rules. Do not expose the bearer token over plain HTTP or public internet. Health/UI are public; all job routes require auth. Disable/restrict `/docs` and `/openapi.json` at the proxy if unwanted. API token grants access to all jobs; there are no tenant accounts.

## API contract

All job routes: `Authorization: Bearer <CAPTURE_API_TOKEN>`.

| Method | Path | Contract |
|---|---|---|
| GET | `/health` | Public `{status: "ok", backend_configured: bool}`; liveness, not GPU readiness |
| POST | `/jobs` | Raw ZIP body, `Content-Type: application/zip` (not multipart); 202 job |
| GET | `/jobs/{id}` | Job `{id,status,error,progress,created}`; progress null unless backend reports it; created Unix seconds |
| GET | `/jobs/{id}/logs` | `{log: string}` bounded combined stdout/stderr |
| GET | `/jobs/{id}/artifact` | Binary `splat.ply` only after successful exit + validated output; otherwise 409 |

Errors: 401 auth, 404 unknown job, 409 unavailable artifact, 413 compressed upload limit, 415 content type, 422 invalid capture, 429 retained-job capacity. IDs are generated UUID hex. Statuses: `blocked`, `queued`, `running`, `failed`, `succeeded`. Interrupted running jobs become failed on next startup; queued jobs resume. Blocked jobs do not auto-retry when a backend is later configured: submit again. No delete/cancel/retry API yet.

```bash
curl -H "Authorization: Bearer $CAPTURE_API_TOKEN" \
  -H 'Content-Type: application/zip' --data-binary @capture.zip http://127.0.0.1:8000/jobs
curl -H "Authorization: Bearer $CAPTURE_API_TOKEN" http://127.0.0.1:8000/jobs/JOB_ID
curl -H "Authorization: Bearer $CAPTURE_API_TOKEN" \
  http://127.0.0.1:8000/jobs/JOB_ID/artifact --output splat.ply
```

ZIP layout must have root `transforms.json` and `images/*.png|jpg|jpeg`. Nerfstudio convention: `frames` with `file_path` (optional leading `./`) and finite 4x4 `transform_matrix` camera-to-world matrices; last row `[0,0,0,1]`. Require `fl_x,fl_y,cx,cy,w,h` globally or per frame; image dimensions must match. Preserve Nerfstudio/OpenGL camera axes; the chosen trainer must interpret these and distortion parameters correctly. This checks structure, not pose accuracy, overlap, calibration quality, or whether a scene is reconstructable. One-frame input is accepted structurally but cannot deliver useful training; capture many overlapping sharp views.

Rejects traversal, absolute/drive/UNC paths, symlinks/special files, Windows device/ADS aliases, duplicate/case-colliding names, encrypted/unsupported ZIP methods, corrupt images, oversized manifests/images and ZIP expansion. Defaults: 256 MiB compressed, 512 MiB expanded, 2,000 entries, 100:1 ratio, 8 MiB manifest, 40M pixels/image, 20 retained jobs. Counts include blocked/failed/completed jobs, bounding capture retention. Operator cleanup requires stopping server, backing up SQLite, then removing selected job directories and their corresponding database rows. Never manipulate live jobs. Protect data directory from other local users and symlinks.

## Real training integration

Set `TRAIN_COMMAND_JSON` to a JSON **argv array**, not shell text. Executable and script paths should be absolute. Server starts one subprocess at a time with `shell=False`, stdin closed, cwd the private job directory, combined logs, and API token stripped from its environment. Trainer is trusted operator code, not a sandbox. Tokens/commands cannot be supplied by uploads.

Supported substitutions in each argv item:

- `{capture}`: extracted capture directory
- `{output}`: job output directory
- `{artifact}`: exact destination `output/splat.ply`
- `{job_id}`
- `{max_steps}`, `{max_gaussians}`, `{image_downscale}`: operator resource hints

Example **contract only**, not an included or validated adapter:

```bash
export TRAIN_COMMAND_JSON='["/absolute/trainer-env/bin/python","/absolute/real_adapter.py","--capture","{capture}","--artifact","{artifact}","--max-steps","{max_steps}","--max-gaussians","{max_gaussians}","--image-downscale","{image_downscale}"]'
export TRAIN_MAX_STEPS=1000
export TRAIN_MAX_GAUSSIANS=100000
export TRAIN_IMAGE_DOWNSCALE=4
```

The adapter must actually train from the supplied poses/intrinsics, export real learned Gaussians, and copy/atomically rename the final PLY to `{artifact}`. An official Graphdeco trainer is **not directly compatible** with Nerfstudio transforms: it needs a verified dataset converter or a compatible training frontend. No implicit COLMAP/reconstruction conversion is claimed here. Adapter may use another Python environment, local GPU or RTX3090 Linux environment, without changing server dependencies.

For the first RTX4070 Laptop 8GB experiment, keep existing workloads untouched; begin with small capture sets, downscale 4, capped steps and Gaussians. These placeholders are **hints only until the real adapter enforces them**; they do not reserve/cap VRAM and do not establish that training fits currently free VRAM. No CUDA, nvcc, conda, WSL or GPU operation has been performed by this prototype.

Optional stdout lines `{"progress": 0.5}` set progress (0..1). Other output is plain logs. Null means unknown, not invented percentage. Success forces 1.0. Default timeout 3,600 seconds, logs capped at 2 MiB, PLY at 1 GiB/10M vertices. Exceeding timeout/log limit fails and terminates the spawned process tree; GPU processes unrelated to that subprocess are never touched. Shutdown also terminates this server's active trainer. On Windows, trainers that detach grandchildren require stronger OS job-object isolation for production.

PLY is parsed, nonempty, finite floating Gaussian fields required (`x/y/z`, `f_dc_0..2`, `opacity`, `scale_0..2`, `rot_0..3`) with nonzero quaternion. Supports ASCII/binary scalar vertex PLY; no list properties. File is revalidated on download. This verifies Gaussian file structure, **not learning provenance or rendered quality**. Uploaded PLYs are never promoted to output.

Other `Settings` limits can be set in a Python factory. Disk quota/container limits are required for untrusted trainers: polling the final artifact is not a filesystem quota and does not bound arbitrary trainer scratch/checkpoint outputs. GPU hard quotas, sandboxing, robust Windows descendant containment, cancellation, retention API, multi-user auth, distributed scheduling and real trainer validation remain deployment work.

## Verification

```bash
uv run pytest -q --basetemp=.test-tmp
uv run ruff check server.py tests
```

Tests use private temporary directories, authenticate uploads, verify persistence/status, ZIP paths/symlinks/expansion bounds and missing/corrupt output. `tests/fixture_backend.py` is explicitly **synthetic test-only subprocess plumbing**, never a real trainer: success/failure/missing/invalid/timeout/log-flood paths are exercised. No GPU result is asserted.
