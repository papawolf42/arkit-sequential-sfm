"""COLMAP 모델을 기존 Minmi trainer의 고정 카메라 NPZ 계약으로 연결."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
import pycolmap


def run():
    root = Path('output/arkit-baseline-dataset')
    target = root / 'platform-input.npz'
    if target.exists():
        raise FileExistsError(target)
    model = pycolmap.Reconstruction(root / 'sparse/0')
    ordered = sorted(model.images.values(), key=lambda i: i.name)
    poses, ks, images = [], [], []
    for image in ordered:
        pose = np.eye(4)
        pose[:3] = image.cam_from_world().inverse().matrix()
        poses.append(pose)
        fx, fy, cx, cy = model.cameras[image.camera_id].params
        ks.append([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        with Image.open(root / 'images' / image.name) as rgb:
            images.append(np.asarray(rgb.convert('RGB')))
    points = np.array([p.xyz for p in model.points3D.values()], dtype=np.float32)
    colors = np.array([p.color for p in model.points3D.values()], dtype=np.float32) / 255
    centers = np.asarray(poses)[:, :3, 3]
    scale = float(np.linalg.norm(centers - centers.mean(0), axis=1).max())
    np.savez(target, points=points, colors=colors, c2w=np.array(poses, dtype=np.float32),
             Ks=np.array(ks, dtype=np.float32), images=np.array(images, dtype=np.uint8),
             scene_scale=np.float32(scale), train_indices=np.arange(240, dtype=np.int64))
    config = dict(target='local', subset='all-240', steps=5000, model='3dgs',
                  needle_prune=False, max_gaussians=61025, growth=False, pose_opt=False,
                  scale_flat=False, opacity_entropy=False, pearson_depth=False,
                  decoupled_prune=False, effective_rank=False, surfel_init=False,
                  checkpoint_every=1000, train_indices=list(range(240)))
    (root / 'baseline-config.json').write_text(json.dumps(config, indent=2))
    print('Prepared all240 native1280x960, seed61025, fixed poses, 5000steps; growth disabled')


if __name__ == '__main__':
    run()
