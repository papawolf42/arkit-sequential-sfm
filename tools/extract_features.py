"""첫 이미지의 SIFT 특징점을 별도 COLMAP DB에 추출한다. 카메라 자세는 아직 가져오지 않는다."""
import argparse
import json
from pathlib import Path

import pycolmap


def extract_first_frame(transforms, database):
    transforms = Path(transforms)
    database = Path(database)
    if database.exists():
        raise FileExistsError(f'Use a new output database: {database}')
    manifest = json.loads(transforms.read_text(encoding='utf-8'))
    frame = manifest['frames'][0]
    database.parent.mkdir(parents=True, exist_ok=True)
    reader = pycolmap.ImageReaderOptions()
    reader.camera_model = 'PINHOLE'
    reader.camera_params = ','.join(str(frame[key]) for key in ('fl_x', 'fl_y', 'cx', 'cy'))
    options = pycolmap.FeatureExtractionOptions()
    options.num_threads = 2
    options.use_gpu = False
    pycolmap.extract_features(
        database_path=database, image_path=transforms.parent,
        image_names=[frame['file_path']], reader_options=reader,
        extraction_options=options, device=pycolmap.Device.cpu,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('transforms', type=Path)
    parser.add_argument('database', type=Path)
    args = parser.parse_args()
    extract_first_frame(args.transforms, args.database)
