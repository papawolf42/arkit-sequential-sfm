"""첫 학습 단계: 실제 명령으로 캡처 JSON을 읽는다."""
import json
from pathlib import Path
import subprocess
import sys


def test_inspect_capture_prints_count_and_first_frame(tmp_path):
    frame = {'file_path': 'images/example.jpg', 'timestamp': 12.5}
    capture = tmp_path / 'transforms.json'
    capture.write_text(json.dumps({'frames': [frame]}), encoding='utf-8')
    script = Path(__file__).resolve().parents[1] / 'tools' / 'inspect_capture.py'

    result = subprocess.run(
        [sys.executable, str(script), str(capture)],
        capture_output=True, text=True, check=False,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {'frame_count': 1, 'first_frame': frame}
