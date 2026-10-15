# 0003: 2단계 — 순차 삼각측량 및 Local Bundle Adjustment 명세서

본 문서는 **1단계에서 생성된 최적 매칭 그래프(`matching_pairs.txt`)**와 **ARKit 초기 자세**를 바탕으로, PyCOLMAP을 연동하여 10개 단절 문제를 해결하는 순차 삼각측량 및 Local BA 모듈의 상세 규격을 정의한다.

---

## 1. 개요 및 파이프라인 흐름

전통적인 COLMAP Incremental Mapper는 사진 간 매칭만으로 초기 2장을 골라 원점(Gauge)을 잡고 점진적으로 카메라를 붙여나가다가, 텍스처가 부족한 바닥 카펫에서 확장이 끊겨 10개의 독립 모델로 쪼개졌습니다.

2단계에서는 **모든 카메라의 초기 6DoF 자세가 이미 ARKit 월드 좌표계 위에 단일하게 정렬되어 있다는 사실**을 활용합니다:

```
[ 1단계 matching_pairs.txt (1,200쌍) ]
                  │
                  ▼
[ PyCOLMAP 특징점 추출 및 지정 쌍 매칭 (match_image_pairs) ]
                  │
                  ▼
[ ARKit 고정 자세 기반 3D 점 삼각측량 (point_triangulator) ]
                  │
                  ▼
[ 슬라이딩 윈도우 Local BA (Ceres / BetweenFactor 상대 자세 제약) ]
                  │
                  ▼
[ 루프 클로저 발생 시 Global BA (누적 VIO 드리프트 일괄 분산) ]
```

---

## 2. 핵심 구현 모듈 규격 (`tools/run_sequential_sfm.py`)

### ① CLI 인터페이스
```bash
uv run python tools/run_sequential_sfm.py \
    --database output/database.db \
    --image-dir data/demo-day-latest/capture/images \
    --pairs output/matching_pairs.txt \
    --transforms data/demo-day-latest/capture/transforms.json \
    --output-dir output/sparse_model
```

### ② 단계별 실행 절차
1. **특징점 추출 및 매칭 (Feature Extraction & Pair Matching):**
   - PyCOLMAP `extract_features`: SIFT 특징점 추출 (각 이미지당 최대 2,048개).
   - PyCOLMAP `match_image_pairs`: `matching_pairs.txt`에 지정된 ~1,200쌍에 대해서만 특징점 2D 대응점 매칭 수행.
2. **ARKit 초기 카메라 주입 (Model Injection):**
   - COLMAP의 `Reconstruction` 객체를 생성하고, `transforms.json`의 240개 카메라 파라미터($K_i$)와 4×4 자세($T_i$)를 직접 등록.
   - 모든 240개 카메라가 처음부터 "등록된(Registered)" 상태로 시작.
3. **삼각측량 (Triangulation via `point_triangulator`):**
   - `pycolmap.triangulate_points`: 고정된 카메라 자세를 기반으로 매칭된 2D 트랙들을 공간상에 삼각측량하여 3D 점 생성.
   - 단절 없이 240개 전 뷰에 걸친 단일 통합 3D 포인트 클라우드 확보.
4. **Local Bundle Adjustment (Sliding Window):**
   - $k=10$ 프레임 크기의 윈도우를 한 칸씩 전진시키며, 최근 관측된 3D 점과 카메라 자세만 부분 최적화.
   - 인접 프레임 간 ARKit 상대 변환 $\|T_i^{-1} T_{i+1} - \Delta A_i\|$을 비용 함수에 주입하여 VIO 상대 이동성을 보존.

---

## 3. 합격 판정 기준 (Acceptance Criteria)

1. **컴포넌트 수:** 기존 10개로 분열되던 모델이 **단 1개의 단일 연결 모델**로 통합되어야 함.
2. **등록 카메라 수:** 240장 중 최소 235장 이상(98%+)이 단일 모델에 등록 완료되어야 함.
3. **평균 재투영 오차:** 3D 희소점들의 평균 재투영 오차가 $1.5\text{px}$ 이하로 수렴해야 함.
4. **산출물:** `output/sparse_model/cameras.bin`, `images.bin`, `points3D.bin` (COLMAP 표준 바이너리 포맷) 정상 생성.
