"""기존 gsplat 환경에서 학습 시점 RGB 비교 렌더링."""
import json
import argparse
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from gsplat import rasterization

root = Path(__file__).resolve().parents[1] / 'output'
data = np.load(root / 'arkit-baseline-dataset/platform-input.npz')
parser = argparse.ArgumentParser()
parser.add_argument('--run', default='arkit-baseline-training')
out = root / parser.parse_args().run
checkpoint = torch.load(out / 'checkpoint.pt', map_location='cuda', weights_only=True)
p = checkpoint['params']
results = []
with torch.no_grad():
    for index in [0, 60, 120, 180, 239]:
        c2w = torch.tensor(data['c2w'][index], device='cuda')
        k = torch.tensor(data['Ks'][index], device='cuda')
        target = data['images'][index]
        h, w = target.shape[:2]
        rgb, _, _ = rasterization(means=p['means'], quats=p['quats'],
            scales=p['scales'].exp(), opacities=p['opacities'].sigmoid(),
            colors=torch.cat([p['sh0'], p['shN']], dim=1),
            viewmats=torch.linalg.inv(c2w)[None], Ks=k[None], width=w, height=h,
            sh_degree=3, packed=False)
        rgb = rgb[0].clamp(0, 1).cpu().numpy()
        mse = float(np.mean((rgb - target / 255.) ** 2))
        results.append({'capture_index': index, 'training_view_psnr': -10 * np.log10(mse)})
        comparison = np.concatenate([target, (rgb * 255).astype(np.uint8)], axis=1)
        Image.fromarray(comparison).save(out / f'comparison-{index:06d}.png')
(out / 'render_metrics.json').write_text(json.dumps(results, indent=2))
print(json.dumps(results, indent=2))
