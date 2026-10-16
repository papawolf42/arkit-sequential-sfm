"""COLMAP 이미지 이름(images/...)을 유지하는 독립 학습 데이터셋 생성."""
import hashlib
import json
from pathlib import Path
import shutil

import pycolmap


def run():
    source = Path('data/demo-day-latest/capture')
    model = Path('output/colmap-arkit-verified/sparse/0')
    target = Path('output/arkit-baseline-dataset')
    if target.exists():
        raise FileExistsError(target)
    reconstruction = pycolmap.Reconstruction(model)
    # COLMAP에 images/000000.jpg로 기록됐으므로 학습기 images 루트 아래에도 유지.
    image_root = target / 'images'
    hashes = {}
    for image in reconstruction.images.values():
        src, dst = source / image.name, image_root / image.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        original = hashlib.sha256(src.read_bytes()).hexdigest()
        assert hashlib.sha256(dst.read_bytes()).hexdigest() == original
        hashes[image.name] = original
    sparse = target / 'sparse/0'
    sparse.mkdir(parents=True)
    for name in ['cameras.bin', 'images.bin', 'points3D.bin']:
        shutil.copy2(model / name, sparse / name)
        assert (model / name).read_bytes() == (sparse / name).read_bytes()
    reread = pycolmap.Reconstruction(sparse)
    assert len(reread.images) == len(hashes) == 240
    assert len(reread.points3D) == 61025
    (target / 'image_hashes.json').write_text(json.dumps(hashes, indent=2), encoding='utf-8')
    print(f'{target.resolve()}: 240 JPEG hashes matched; 61025 points; image root={image_root}')


if __name__ == '__main__':
    run()
