"""촬영 시각이 이웃한 이미지 쌍을 만든다. 원본은 수정하지 않는다."""
import argparse
import json
from pathlib import Path


def sequential_pairs(frames):
    """시간순 바로 다음 프레임과만 연결한다. 실제 매칭 성공을 뜻하지 않는다."""
    ordered = sorted(frames, key=lambda frame: frame['timestamp'])
    return [
        (first['file_path'], second['file_path'])
        for first, second in zip(ordered, ordered[1:])
    ]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('transforms', type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.transforms.read_text(encoding='utf-8'))
    for first, second in sequential_pairs(manifest['frames']):
        print(first, second)
