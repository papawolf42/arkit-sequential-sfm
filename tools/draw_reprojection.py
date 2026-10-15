"""관측점(청록 원)과 XYZ 재투영점(자홍 십자)을 원본 크기로 비교한다."""
import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw

from tools.inspect_capture import project_world_point


def draw_residuals(image, pairs):
    result = image.convert('RGB').copy()
    draw = ImageDraw.Draw(result)
    for observed, projected in pairs:
        # 양쪽 좌표에 같은 표시용 -0.5 오프셋. 기하 계산값은 바꾸지 않는다.
        x, y = (float(value) - 0.5 for value in observed)
        u, v = (float(value) - 0.5 for value in projected)
        draw.line((x, y, u, v), fill='yellow', width=1)
        draw.ellipse((x - 3, y - 3, x + 3, y + 3), outline='cyan')
        draw.line((u - 2, v, u + 2, v), fill='magenta')
        draw.line((u, v - 2, u, v + 2), fill='magenta')
    return result


def export_overlays(transforms, cloud, output):
    transforms, cloud, output = Path(transforms), Path(cloud), Path(output)
    if output.exists():
        raise FileExistsError('Use a new output directory')
    frames = {f['file_path']: f for f in json.loads(transforms.read_text(encoding='utf-8'))['frames']}
    names = json.loads((cloud / 'summary.json').read_text(encoding='utf-8'))['images']
    records = json.loads((cloud / 'points.json').read_text(encoding='utf-8'))
    output.mkdir(parents=True)
    for index, name in enumerate(names):
        pairs = []
        for record in records:
            if record['status'] != 'finite':
                continue
            projected = project_world_point(record['xyz'], frames[name])
            if projected is not None:
                pairs.append((record['pixels'][index], projected))
        with Image.open(transforms.parent / name) as image:
            overlay = draw_residuals(image, pairs)
            overlay.save(output / f'reprojection_{index}.png')
            # 최대 오차 위치 주변 96px 영역을 표시와 함께 8배 확대한다.
            worst = max(pairs, key=lambda p: sum((a - b) ** 2 for a, b in zip(*p)))
            x, y = worst[0]
            left = max(0, min(image.width - 96, int(x) - 48))
            top = max(0, min(image.height - 96, int(y) - 48))
            overlay.crop((left, top, left + 96, top + 96)).resize(
                (768, 768), Image.Resampling.NEAREST).save(output / f'worst_zoom_{index}.png')
        print(f'{name}: {len(pairs)} overlays; worst crop origin=({left},{top})')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('transforms', type=Path)
    parser.add_argument('cloud', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    export_overlays(args.transforms, args.cloud, args.output)
