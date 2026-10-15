# ARKit-Guided SfM + AI Depth Fusion — 구현 설계서 v0.1

작성 기준: 2026-10-15. **설계 제안**이며 아직 해당 저장소에서 실행하거나 성능을 측정한 결과가 아니다.

## 0. 목표와 현재 데이터

**목표:** LiDAR 없는 iPhone XR에서 수집한 RGB·ARKit 자세를 이용하여, 텍스처 부족 구간까지 일관된 깊이를 추정하고, 그 결과로 이중벽·Floater가 적은 3DGS 장면을 만든다. 모든 기술을 한 번에 결합하지 않고, 모듈·실험별로 추가 효과를 검증한다.

사용자가 제공한 데이터·코드 분석 요약:

| 항목 | 입력 조건 |
|---|---|
| 데이터 경로 | `data/demo-day-latest/capture/transforms.json` |
| 이미지 | 240장, 1280×960 JPEG, CoreImage 압축 품질 0.82 |
| 획득 | `ARFrame.capturedImage`와 동일 `ARFrame`에서 pose·K·timestamp 저장 |
| 기간 | 123.44초; 프레임 간 약 0.517초 |
| 인접 이동 | 최소 3.5cm, 평균 16.6cm, 최대 41.5cm (ARKit 추정) |
| 추적 | 저장 프레임의 `tracking_state == normal` |
| 세션 | 제공된 분석에 따르면 단일 연속 ARKit 세션 |
| 기존 SfM | 과거 실험에서 10개 컴포넌트, 일부 미등록 이미지 및 이중벽 문제 보고 |

**주의:** 위 내용은 사용자가 제공한 코드/통계의 요약이다. 실제 `transforms.json` 원본은 이 설계 단계에서 직접 읽거나 검증하지 않았다. `normal`은 정확도 보증이 아니고, 연속 자세도 전역 드리프트를 배제하지 못한다.

## 1. 아키텍처: 세 종류의 관측, 하나의 신뢰도 체계

```text
RGB 240장 + ARKit T_c2w + K + timestamp
   |
   +--> A. 순차/선택적 이미지 매칭 + SfM + Local/Global BA
   |          |-- 검증된 카메라와 희소 3D 점 / 재투영 오차
   |          '-- 희소 깊이 앵커 (sparse anchor)
   |
   +--> B. Pose-conditioned AI Depth (DA3; 창별 추론)
              '-- 조밀한 깊이 후보 + 모델 confidence
                         |
       [깊이 척도 보정 + 시점 간 검증 + 가림 판단]
                         |
       [검증된 dense anchor + 인접 프레임 depth warp]
                         |
       [출처·신뢰도 보존형 depth fusion / 미확인 마스크]
                         |
       +--> 3DGS: RGB-only 기준선과 depth-aware 학습 비교
       '---> (추후) depth factor + ARKit 상대자세 factor로 BA 보완
```

**비협상 원칙**

1. 특징점 기반 삼각측량은 **희소** 깊이를 만든다. 희소점을 워핑만 해서는 흰 벽 전체가 조밀하게 채워지지 않는다. 조밀한 깊이는 AI 또는 별도의 표면 가정이 제공해야 한다.
2. **AI 깊이 = 후보**, SfM = 기하학적 근거, ARKit = 자세 초기값·인접 이동 제약. 어느 하나도 항상 정답은 아니다.
3. 모델이 생성한 깊이를 여러 번 워핑해도 새로운 독립 증거가 생기지 않는다. 원본 출처·최초 프레임·전파 횟수를 추적해 **자기 증폭**을 막는다.
4. 가림·새로 드러난 표면·거울·반사·움직이는 물체·모델 간 충돌은 `UNKNOWN`으로 남기는 것이 강제로 채우는 것보다 낫다.
5. **첫 버전에서는 카메라 자세를 고정하고 Depth의 효과만 검증한다.** Depth가 유용하다는 증거가 생기면 그다음에 카메라 자세의 공동 최적화를 시도한다.

## 2. 저장 형식과 모듈 경계

원본 `capture/`는 절대 수정하지 않는다. 모든 산출물은 버전과 설정 파일 해시가 있는 별도 디렉터리에 기록한다.

```text
data/demo-day-latest/capture/           # 원본, read-only
work/hybrid_v01/
  config.yaml                           # 매칭·깊이·신뢰도·학습 파라미터
  00_normalized/
    frames.jsonl                         # frame_id, filename, timestamp, W/H, 출처
    cameras.npz                         # K_original,K_saved,T_arkit_c2w,T_cv_w2c
    validation.json                     # 크기·축변환·시간·회전 검증
  01_matches/
    pairs_local.txt                     # 국소 이미지 후보
    pairs_loop.txt                      # 재방문 후보(기하 검증 전)
    database.db                         # COLMAP 특징점·매칭
    match_report.csv                    # inliers, baseline, angle, overlap
  02_sfm/
    colmap/                              # cameras/images/points3D, 버전별 형식
    poses_refined.npz                   # 보정된 T_c2w, K
    tracks.csv                          # track 길이·오차·삼각측량 각도
    sparse_anchors.npz                  # frame_id,u,v,z,quality,point3d_id
  03_ai_depth/
    windows.json                        # 입력 이미지 창과 모델/체크포인트
    raw/000001.npz                      # depth, conf_raw, K, pose_version
    aligned/000001.npz                  # depth_z, confidence, valid, scale
    alignment_report.csv                # scale/offset/앵커 수/잔차
  04_fusion/
    warped/000001.npz                   # 후보별 z, source, age, occlusion
    fused/000001.npz                    # depth_z, weight, source_code, valid
    fusion_report.csv                   # coverage, conflicts, unsupported_ratio
  05_3dgs/
    rgb_only/                            # 동일 카메라·학습 조건 기준선
    ai_depth_only/
    warped_depth_only/
    fused_depth/
  reports/                               # 정량표·시각화·실험 구성 해시
```

**픽셀 단위 depth 파일 계약 (`*.npz`)**

- `depth_z: float32[H,W]` — **카메라 +Z 축 깊이**. 광선 길이(ray distance)가 아님; NaN은 미확인.
- `weight: float32[H,W]` — [0,1] 범위의 **실험적 신뢰 점수**. 통계적으로 교정된 확률은 아님.
- `valid: bool[H,W]` — 학습 Loss와 평가에 사용할 수 있는 픽셀.
- `source_code: uint8[H,W]` — 0 unknown / 1 sparse_geom / 2 propagated_geom / 3 validated_ai / 4 ai_only.
- `support_count: uint8[H,W]` — 유효한 **서로 다른 원천 시점**의 지지 수. 같은 예측의 반복 워핑은 중복 산정하지 않음.
- `origin_frame_id`, `propagation_hops`, `pose_version`, `model_id`, `image_transform_id`는 메타데이터로 저장.
- `depth_confidence`와 `camera_confidence`를 별도 기록. 하나의 숫자로 합치기 전에 원인 분석이 가능해야 함.

## 3. 구현 단계별 명세

### P0 — 데이터·좌표 검증 (가장 먼저)

1. `transforms.json`을 읽는 **프로젝트별 어댑터** 작성. 실제 필드명을 보기 전에는 스키마를 가정하지 않는다.
2. 동일 frame ID로 JPEG·timestamp·`K`·`T_c2w` 결합. 길이 240, 파일 존재 여부, unique ID, finite matrix, 회전행렬 정규직교성 검사.
3. ARKit의 camera-to-world 및 축 방향을 OpenCV/COLMAP의 world-to-camera로 변환. 일반적인 카메라 축 변경 후보는 `diag(1,-1,-1)`이지만 **저장된 사진 방향**과 맞는지 시각 검증 후 확정한다.
4. 1280×960 리사이즈에 맞춰 `K`도 스케일·크롭·회전 변환. `K_saved = H_pixel @ K_original`이 성립하도록 픽셀 변환 행렬 `H_pixel`을 명시한다.
5. 적어도 3장에 대해 알려진 특징점 3D 좌표를 재투영하고, 중심·위아래·좌우가 기대한 위치인지 시각 검증한다.
6. 0.517초, 3.5~41.5cm 결과를 원본 값으로 재계산. 거리/회전 큰 구간·이미지 선명도·ARKit 변환의 불연속을 로그로 남긴다.

**통과 기준:** `T_c2w @ T_w2c ≈ I`, `K`가 리사이즈된 사진과 일치, reprojection 좌우 반전/90도 회전 문제 없음. 여기서 실패하면 이후 단계 금지.

### P1 — ARKit-guided 이미지 매칭 그래프

각 이미지마다 기본적으로 시간상 ±1~5장과 매칭한다. 삼각측량 후보는 **이동 거리와 예상 겹침**을 이용해 별도로 선정한다. 추가로 이미지 검색으로 과거 장면 재방문 후보를 제안한다.

- **국소 후보:** 시간 차가 작고 시야가 겹치는 이미지.
- **충분한 시차 후보:** ARKit 카메라 중심 간 baseline이 적당하고 회전으로 인해 겹침이 사라지지 않은 이미지.
- **전역 후보:** 이미지 검색(BoW/특징 검색)에서 발견된 원거리 시점; ARKit 좌표상 가까운 것만 고르지 않는다(드리프트 때문).
- **최종 승인:** 실제 특징점 매칭의 RANSAC/기하 검증을 통과한 쌍만 SfM 그래프의 엣지로 확정한다.

이 단계는 COLMAP `sequential_matcher` + loop detection으로 먼저 기준선을 만들고, 별도 pair list를 구성한 개선안을 비교한다. 프레임 이웃이라는 이유만으로 실측 동일 구조를 관측했다고 취급하지 않는다.

**산출:** `pairs_local.txt`, `pairs_loop.txt`, 유효 매칭 그래프, 약한 연결 구간 목록.

### P2 — 카메라 자세·희소 깊이 앵커

1. **A-기준선:** 기존 COLMAP 결과 보존.
2. **B-초기화:** ARKit 자세를 COLMAP 알려진 자세로 내보내고 `point_triangulator`로 검증된 트랙을 삼각측량.
3. **C-보정:** ARKit을 초기값으로 Local BA 수행. 상대 자세 factor는 별도 실험에서 도입해 효과를 분리.
4. **D-전역:** 검증된 루프/멀리 떨어진 이미지 연결을 얻으면 Global BA 및 재삼각측량.
5. 프레임별 희소 깊이 앵커 생성. 각 3D 점을 보정된 카메라로 투영해 `z`와 신뢰 정보를 저장.

깊이 앵커 품질은 `track length`, `reprojection error`, `triangulation angle`, `pose uncertainty`로 판단한다. 예시 초깃값: 관측 3뷰 이상, 재투영 오차 1~2px 이하, 삼각측량 각도 1.5~2° 이상. **이는 튜닝 전 가설 값이지 검증된 권장 임곗값이 아니다.**

프레임이 희소점 없이 등록되더라도 **pose-only**로 계속 보존할 수 있다. 단 `pose-known`과 `geometry-validated`는 별도 상태로 구분한다.

### P3 — AI Depth 생성: DA3를 우선 실험

[Depth Anything 3 공식 API](https://github.com/ByteDance-Seed/Depth-Anything-3/blob/main/docs/API.md)는 `image`, `extrinsics`(world-to-camera), `intrinsics`를 입력받고 `prediction.depth`, 선택적으로 `prediction.conf`를 제공한다.

1. 먼저 가벼운 `DA3-BASE`로 작은 구간을 테스트한다(공식 모델 카드상 Apache 2.0). 대형 체크포인트는 모델별 비상업 조건 확인.
2. **8~16프레임 단위로 시작**, 인접 창 사이 3~5프레임 중첩. 수치는 GPU 메모리와 실제 모델 출력 안정성에 따라 조정한다.
3. `extrinsics`는 **OpenCV 스타일 W2C**, `intrinsics`는 **해당 입력 영상 크기에 맞춘 K**. 모델 내부 resize 후의 깊이 배열 크기도 기록한다.
4. `align_to_input_ext_scale=True`를 사용해 입력 카메라 척도에 정렬하되, 이것만으로 깊이가 정답 미터값이라고 단정하지 않는다.
5. 비교용으로 같은 창에서 `pose-conditioned`와 `pose-free` 추정을 모두 기록한다. ARKit pose가 틀리면 conditioning이 독이 될 수도 있다.
6. 창 경계에서 깊이 척도와 중첩 영역의 일관성을 계산한다. 입력 모델 confidence는 **정답 확률이 아니라 추정 신호**로 취급한다.

실제 함수 호출 형식(로딩 및 패키지 버전은 공식 API 확인):

```python
# 실행 코드의 핵심 API 사용 형태
from depth_anything_3.api import DepthAnything3
model = DepthAnything3.from_pretrained("depth-anything/DA3-BASE").to("cuda")
pred = model.inference(
    image=image_paths,             # 길이 N
    extrinsics=w2c_array,         # shape [N, 4, 4], OpenCV convention
    intrinsics=K_array,           # shape [N, 3, 3], image dimensions
    align_to_input_ext_scale=True,
    process_res=504,
    infer_gs=False,
)
depth = pred.depth
conf = getattr(pred, "conf", None)
```

### P4 — AI 깊이 정렬(Alignment)과 dense anchor 생성

- 희소 SfM 3D 점을 같은 프레임에 투영하여 **기하 깊이와 AI 예측 깊이의 대응 샘플**을 얻는다.
- DA3처럼 직접 깊이 `z`를 출력하면 우선 robust scale 보정(`z_geom ≈ s z_ai`)부터 검토한다. 필요하면 offset까지 비교하지만 지나치게 유연한 보정은 모델 오류를 감출 수 있다.
- 상대 **역깊이**를 출력하는 다른 모델을 쓰면 `1/z` 영역의 scale-shift 정렬을 우선 검토한다. `z`와 `1/z` 보정식을 혼용하면 안 된다.
- 유효 앵커가 적거나 특정 작은 영역에 몰려 있으면 보정 신뢰도 `low`로 지정하고, 주변 창의 보정을 참고하되 불확실성을 높인다.
- dense anchor는 **AI 깊이를 정렬했으며, 다른 관측 뷰에서의 투영/가림/깊이 일관성 검사까지 통과한 픽셀**만 포함한다. 희소 SfM 점도 고신뢰 별도 채널로 남긴다.

### P5 — Depth Warping (중요한 기하 단계)

원본 프레임 s의 픽셀 `p=(u,v)`와 해당 Z-depth `z_s`에서:

```text
X_s = z_s * inverse(K_s) * [u, v, 1]
X_w = T_s_c2w * [X_s, 1]
X_t = T_t_w2c * X_w
if X_t.z <= 0: reject
p_t = project(K_t * X_t)       # 픽셀 좌표
z_t = X_t.z                   # target 카메라 Z-depth
```

단순 nearest-neighbor 옮기기가 아니라, target depth map에서 **z-buffer(splatting)**로 겹친 표면의 앞쪽을 처리하고, 필요하면 sub-pixel 투영과 hole filling을 적용한다. RGB 차이는 무늬 없는 벽에서는 판별력이 낮으므로 **기하·가림 검사를 우선**한다.

필수 검사:

- 소스·대상 카메라의 깊이·이미지 영역 유효성
- 같은 target 픽셀을 여러 표면이 차지할 때 앞/뒤 가림(z-buffer)
- target 자체 깊이가 존재하면 `|z_warp - z_target| / max(z_warp,z_target)` 등 정규화 차이
- 다른 관측 뷰로 역투영했을 때의 round-trip 위치·깊이 일관성
- 원래 `source_code`, `origin_frame_id`, 전파 횟수 보존; **최대 1-hop에서 시작**
- ARKit 또는 최적화된 pose가 바뀌면 워핑 결과 캐시 무효화

**시작 실험의 깊이 일치 임계값은 예컨대 상대 5~10%**를 탐색하되, 실제 모델의 잔차 분포를 보고 결정한다. 엄격한 기준은 깊이 절단을, 느슨한 기준은 허공 표면 유입을 만들 수 있다.

### P6 — 출처·신뢰도 기반 Fusion

**단순 평균하지 않는다.** 두 후보가 서로 다른 표면(전경/배경)에 속하면 평균값은 공중에 가짜 표면을 만든다. 먼저 가림과 다중 시점 일치로 *같은 표면 후보끼리* 그룹화한 뒤, 해당 그룹에서 robust median/Huber 등의 추정을 수행한다.

픽셀별 상태와 정책:

| 상태 | 출처/상태 | 초기 정책 |
|---|---|---|
| `GEOMETRY_VERIFIED` | 여러 뷰 삼각측량, 낮은 오차 | 강한 anchor. 단 희소 픽셀에 한정 |
| `PROPAGATED_GEOM` | 신뢰 가능한 깊이를 1-hop 전파 | 가림·pose 검사 통과 시 중상 신뢰 |
| `MULTIVIEW_AI` | 정렬된 AI 깊이, 다른 관측으로 일관성 확인 | 중간 신뢰; 기하 앵커 없는 영역 표시 |
| `AI_ONLY` | AI만 존재하거나 검증 근거 약함 | 시각화·후보로 보존; 기본 3DGS depth Loss에는 제외 또는 매우 약하게 |
| `UNKNOWN` | 서로 충돌·가림·검증 실패 | 0 weight, 깊이 강제 주입 금지 |

confidence는 `source reliability × pose reliability × visibility × cross-view agreement × scale-alignment quality` 등의 요소로 **실험적으로** 구성하되, 관측값과 중간 점수를 모두 저장한다. AI confidence와 다중 시점 일관성은 공통 모델의 오류 때문에 서로 독립적이지 않을 수 있다. 점수의 정확도는 사후 보정한다.

### P7 — 3DGS에 연결

**최초 실험에서는 카메라·RGB 이미지·희소 초기 Gaussian·학습 스텝·Seed를 모두 동일하게 고정**하고 깊이 감독만 달리한다. 그 후 dense anchor로 초기 점을 보강하는 실험은 별도로 수행한다.

- 선택 1: 기존 3DGS 공식 Depth regularization. 원본은 *inverse depth* 비교를 사용하므로 모델의 깊이 정의와 변환을 맞춰야 한다.
- 선택 2: gsplat의 `render_mode="RGB+ED"` 등으로 projection Z expected depth를 얻어, 픽셀의 신뢰 마스크를 이용한 추가 손실을 구성한다. opacity가 너무 낮은 픽셀은 깊이 감독에서 제외한다.

개념적 깊이 손실:

```text
L = L_RGB + lambda_D * sum(valid * weight * robust(depth_rendered - depth_fused))
                      / max(sum(valid * weight), eps)
```

여기서 `depth_rendered`와 `depth_fused`의 단위·좌표축·표현(`z`, inverse-z, ray distance)이 동일해야 한다. Gaussian 렌더링의 depth는 항상 단단한 표면의 깊이와 같은 것은 아니므로 투명·불확실한 픽셀을 주의한다.

## 4. 실험 매트릭스 — 무엇 때문인지 알 수 있도록

| ID | 변경점 | 카메라 고정? | 핵심 질문 |
|---|---|---|---|
| E0 | 현재 RGB-only 3DGS | 기존 | 원래 품질·시간·메모리 기준 |
| E1 | ARKit-guided pair selection + SfM | 변경 | 이미지 연결·카메라 정합 개선? |
| E2 | E1의 카메라를 고정하고 AI Depth Loss만 추가 | 예 | Depth만으로 Floaters 감소? |
| E3 | E1 + 기하 anchor 워핑 Depth Loss | 예 | 순수 기하 전파는 얼마나 도움? |
| E4 | E1 + 정렬된 AI Depth + 검증된 warp fusion | 예 | 신뢰도 융합이 AI-only보다 나은가? |
| E5 | E4 기반 상대 자세 Factor + BA | 아니오 | Depth와 ARKit이 pose까지 개선하는가? |
| E6 | E4와 동일조건 + dense point 초기화 | 예 | 깊이로 만든 초기 점이 이득인가? |

**E2, E3, E4는 동일한 카메라·초기 Gaussian·RGB·스텝·Seed를 사용**해야 깊이 정책의 인과관계를 볼 수 있다. E1/E0는 입력 pose 변경으로 별도의 차이이다. E5/E6은 하나씩 변인을 늘린다.

### 평가 데이터 누수 방지

- 학습 평가용 사진 일부(예: 10~15% 내외)를 **처음부터 홀드아웃**한다. 홀드아웃은 SfM 입력, DA3 다중 시점 입력, depth fusion, 3DGS 학습 모두에서 제외한다.
- 홀드아웃의 카메라 자세는 평가를 위해 별도로 신뢰할 수 있게 구해야 한다. 단순히 ARKit 자세만 사용하는 경우 렌더 지표는 pose 오차에 영향을 받는다.
- 3DGS 평가는 PSNR/SSIM/LPIPS 외에도 **새로운 카메라 이동 경로 영상의 Floater 개수·이중벽·깊이 불연속**을 동일 기준으로 기록한다.
- 실측 가능한 길이 2~3개(문 너비, 테이블 폭 등)를 줄자/도면과 비교하면 *실제 치수의 정확도* 검증에 유용하다. AI 깊이의 자체 일관성과 별개인 외부 검증이다.
- 최소한 GPU 시간, CPU 시간, 최고 VRAM, 사용된 프레임 수, 유효한 3D track 수, depth valid 비율을 기록한다.

## 5. 테스트와 실패 처리

1. **좌표·투영:** `T_c2w @ T_w2c = I`, 예제 3D 점의 픽셀 roundtrip 오차가 수치 오차 이내.
2. **같은 프레임 워프:** `warp(i→i)`는 유효 픽셀에서 원래 depth를 보존(양자화/리샘플링 오차 허용).
3. **앞뒤 가림:** 한 픽셀에 2m 벽과 5m 벽을 투영하면 2m가 전경으로 채택, 3.5m를 평균으로 생성하지 않음.
4. **무늬 없는 벽:** 영상 색상이 균일해 photometric test가 무력해도 기하 검사는 동작; 증거 없으면 `AI_ONLY`/`UNKNOWN` 유지.
5. **추적 점프:** 급격한 상대 pose 변화가 발견되면 edge를 약화하고 구간을 분리해 진단.
6. **모델 규모/척도:** DA3의 좌표·depth 유형이 기대값과 다르면 명시적 오류 및 로그 후 중단.
7. **캐시 무효화:** BA가 pose를 바꾸면 해당 pose 버전으로 생성한 warp/consistency/fusion 재계산.
8. **빈 값:** 앵커가 없는 프레임을 임의 깊이로 채우지 않고, 의존성·신뢰도 낮음을 표시.

## 6. 개발 순서, 파일 단위 작업, 완료 조건

| 작업 | 구현 파일(제안) | 완료 조건 |
|---|---|---|
| T0. 데이터 검증 | `hybrid/normalize_capture.py`, `test_geometry.py` | 모든 사진/K/pose 정합, 축변환 검증 |
| T1. 후보 쌍 그래프 | `hybrid/select_pairs.py` | 후보 쌍·유효 매칭률·카메라 연결 그래프 출력 |
| T2. 기본 SfM | `hybrid/run_colmap.py` | 원본 대비 사용 프레임·track·기하 오차 비교 |
| T3. DA3 깊이 추론 | `hybrid/infer_depth.py` | 모든 테스트 창의 depth/conf/K/pose-version 저장 |
| T4. 깊이 앵커·정렬 | `hybrid/align_depth.py` | 앵커 매칭, 척도 잔차, 검증 실패 구분 |
| T5. 재투영 워프 | `hybrid/warp_depth.py`, `test_warp.py` | 항등 워프·z-buffer·가림 테스트 통과 |
| T6. Fusion/신뢰도 | `hybrid/fuse_depth.py` | depth/source/valid/quality 및 진단 화면 출력 |
| T7. 3DGS 연결 | `hybrid/export_depth_for_3dgs.py` | RGB-only vs Depth 실험 결과 재현 |
| T8. 하이브리드 BA | `hybrid/pose_factors.*` (마지막 단계) | 상대자세+영상 재투영 함께 최적화·기준선 비교 |

**추천 MVP 절단선:** T0~T5를 **30~50장 구간**에서 먼저 수행하고, T6~T7에 들어가기 전에 워핑된 벽·바닥의 깊이를 시각화한다. 후속 전체 240장 처리·BA 확장은 첫 효과를 본 뒤 한다.

## 7. 설정 초안 (확정값이 아닌 비교 실험 시작값)

```yaml
source:
  transforms: data/demo-day-latest/capture/transforms.json
  readonly: true
matching:
  temporal_radius: 5
  loop_retrieval_topk: 5
  max_candidate_pairs_per_frame: 18  # 초기 계산 예산, 미검증
sfm:
  min_track_views: 3
  max_reprojection_error_px: 2.0  # 임시값
  min_triangulation_angle_deg: 1.5  # 임시값
ai_depth:
  model: depth-anything/DA3-BASE
  pose_conditioned: true
  process_res: 504
  window_frames: 12
  overlap_frames: 4
  align_to_input_ext_scale: true
warp:
  max_hops: 1
  relative_depth_consistency: 0.07  # 임시값
fusion:
  min_independent_view_support: 2
  include_ai_only_in_loss: false
training:
  fixed_seed: 42
  compare_same_camera_pose: true
  train_val_split_manifest: work/hybrid_v01/eval_split.json
```

모든 숫자는 첫 비교를 시작하기 위한 임시값이고, 결과가 좋게 나온 뒤 평가 데이터에 맞춰 뒤늦게 변경하면 안 된다. 유효한 개발용 구간에서 탐색 후 홀드아웃으로 확인한다.

## 8. 아직 확인되지 않은 정보

- 실제 `transforms.json`의 필드 구조, 행렬이 행 우선/열 우선인지, saved JPEG 회전·크롭 방식.
- `K`가 원본 ARFrame 해상도 기준인지, JPEG 해상도로 이미 변환됐는지.
- 현재 COLMAP·3DGS 구현체의 정확한 버전/커밋, 학습용 카메라 convention.
- 기존 10개 컴포넌트 실패가 **특징점 매칭 부족**인지, **잘못된 대응**인지, **자세/내부 파라미터 오류**인지.
- 홀드아웃 장면에 대해 외부 정확도를 검증할 실제 길이·Depth 기준 데이터의 확보 가능성.

위 사항은 구현 초기에 확인하면 된다. 현재 단계에서 결과 품질 개선이나 속도 향상을 확정적으로 약속할 근거는 없다.

## 9. 핵심 참고 구현과 문서

- [COLMAP Tutorial — Sequential Matching/Loop Detection](https://colmap.github.io/tutorial.html)
- [COLMAP — Known Camera Poses & Point Triangulator](https://colmap.github.io/faq.html#reconstruct-sparse-dense-model-from-known-camera-poses)
- [COLMAP — Output Format](https://colmap.github.io/format.html)
- [Depth Anything 3 — 공식 Python API](https://github.com/ByteDance-Seed/Depth-Anything-3/blob/main/docs/API.md)
- [Depth Anything 3 — 공식 모델·라이선스](https://github.com/ByteDance-Seed/Depth-Anything-3)
- [gsplat — Rasterization Depth Modes](https://docs.gsplat.studio/main/apis/rasterization.html)
- [원본 3D Gaussian Splatting — Depth Regularization 구현](https://github.com/graphdeco-inria/gaussian-splatting/blob/main/train.py)
- [DN-Splatter — Depth/Normal Supervision 참고](https://github.com/maturk/dn-splatter)

**최종 판단:** ARKit은 움직임을 운반하고, SfM은 관측된 기하를 강하게 고정하며, AI Depth는 부족한 표면을 *제안*한다. 검증을 통과한 깊이만 3DGS에 강하게 반영한다. 이 설계의 성공은 더 많은 Gaussian을 만드는 것이 아니라, **거짓 표면을 만들지 않으면서 유효한 기하 정보를 늘리는 것**으로 평가한다.
