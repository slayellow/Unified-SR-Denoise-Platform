# CameraModel 모듈 문서

| 문서번호 | 버전 | 작성일 | 작성자 | 시스템 | 상위 문서 |
|---|---|---|---|---|---|
| USRD-MOD-CAM-001 | 0.3 | 2026-10-08 | JSHONG | unified-sr-denoise-platform | USRD-RESEARCH-003 §1·§4 (EO Denoise v3) |

개정 이력
| 버전 | 일자 | 변경 내용 | 작성자 |
|---|---|---|---|
| 0.1 | 2026-10-08 | 최초 작성. MC-G105 Aperture 9 / 3D NR 5 실측으로 값 맞춤 | JSHONG |
| 0.2 | 2026-10-08 | 정답 모드 `target`(camera / optics / source) 추가(MREQ-CAM-07). 소유자 목표 정리: Denoise 출력 = 고품질 원본 형태(C안), 비교로 B안. 희소 결함(고정 핫픽셀, 움직임 모드 점 노이즈) 추가(MREQ-CAM-08, 소유자 승인 2026-10-08) | JSHONG |
| 0.3 | 2026-10-12 | MREQ-CAM-06의 clean 쌍 설명을 코드 동작에 맞춤(정답 = `target`이 고른 영상, 채택한 EXP-08-1은 `source`라 (원본, 원본)). 코드 변경 없음 | JSHONG |

이 저장소에는 SRS가 없다. 상위 요구는 연구 계획서 `docs/20261008-RESEARCH-EODenoiseV3.md`이며, 이 모듈이 그 연구의 학습 입력(LR)과 정답(target)을 만든다. 모듈 약어 `CAM`.

## 1 요구
| ID | 유형 | 요구사항 | 수용 기준 | 상위 | 검증 |
|---|---|---|---|---|---|
| MREQ-CAM-01 | 기능 | 모듈은 깨끗한 BGR 패치 하나에서 같은 크기의 (LR, target) 쌍을 만들어야 한다. LR = 광학 흐림 → 샤프닝 전 노이즈 → 샤프닝 → 4:2:0 → 8-bit. target은 MREQ-CAM-07의 모드를 따르며 기본(`camera`)은 같은 경로에서 노이즈만 뺀 영상이다 | uint8, 입력과 같은 shape. `camera` 모드에서 노이즈 배율 0이면 LR과 target이 비트 단위로 같다 | RESEARCH §1 | T |
| MREQ-CAM-02 | 기능 | 모듈의 샤프닝은 실측 Aperture 9 응답을 재현하여야 한다: 9×9 커널 K(방향성 있음, D4 대칭)의 응답 1 + b(Y)·K̂(f)로 키우고, 세기 b는 국소 밝기에 따라 달라진다 | 밝은 평탄부(Y 160) 저진폭 정현파 0.375 cycles/px 증폭이 커널 응답 ±10 %. 같은 정현파의 증폭이 Y 45에서 Y 160보다 작음. σ1.5로 흐린 120 DN 계단(인쇄 회색 패치 엣지 수준)의 넘침 2~10 % (실측 4.8 %) | RESEARCH §1 | T |
| MREQ-CAM-03 | 기능 | 모듈의 샤프닝 전 노이즈는 모드별 밝기-노이즈 표(NLF)와 방사형 스펙트럼을 따라야 한다. 색 노이즈는 4:2:0 반해상도 평면에 더한다 | 평탄 패치에서 휘도 노이즈 표준편차가 표 × 배율의 ±10 %, 색 노이즈 표준편차가 설정값 × 배율의 ±15 % | RESEARCH §1 | T |
| MREQ-CAM-04 | 차단 | 모듈은 실측 값이 빠졌거나 모양이 틀린 설정을 조용히 기본값으로 바꾸지 않고 거부하여야 한다 | 키 이름을 담은 `ValueError`. 통과 0건 | RESEARCH §2 | T |
| MREQ-CAM-05 | 기능 | 같은 난수 생성기 상태에서는 출력이 같아야 한다(검증 세트 재생성용) | 같은 시드 2회 비트 동일, 다른 시드는 다름 | RESEARCH §2 | T |
| MREQ-CAM-06 | 기능 | `DenoiseDataset`은 `camera_model.enabled`일 때 확률 `prob`로 카메라 쌍을, 나머지는 기존 열화(replay, 정답 = HR)를 써야 한다. `clean_prob`는 두 경로 각각에 적용된다: 카메라 경로의 clean 쌍은 (정답, 정답)이고 정답은 `target`이 고른 영상이다(기본 `camera`면 카메라 정답, `source`면 원본), replay 경로의 clean 쌍은 (HR, HR). 꺼져 있으면 기존 동작과 같다 | 켜면 `camera`·`optics`는 target ≠ HR, `source`는 target = HR이고 LR ≠ target(TC-CAM-06-8). clean 쌍은 LR = target(TC-CAM-06-3). 끄면 target = HR. 시작 로그에 상태 한 줄 | RESEARCH §1 | T |
| MREQ-CAM-07 | 기능 | 모듈은 `camera_model.target`으로 정답을 고를 수 있어야 한다: `camera`(기본, 카메라가 노이즈 없이 냈을 영상, A안), `optics`(광학 흐림만 거친 영상, ISP 전, B안), `source`(입력 원본, C안). LR은 모드와 무관하다. 그 밖의 값은 거부한다 | 같은 난수 상태에서 모드만 바꾸면 LR이 비트 동일하고 target만 바뀐다. `source`의 target = 입력, `optics`의 target = 같은 σ의 Gaussian 흐림을 반올림한 영상. 노이즈 배율 0이면 LR = 카메라 정답. 다른 값은 `ValueError(target)` | RESEARCH §1 (소유자 목표, 2026-10-08) | T |
| MREQ-CAM-08 | 기능 | 모듈은 설정이 있을 때 LR에만 희소 결함을 넣어야 한다: `hot_pixels`(고정 핫픽셀: 밀도 개/Mpx, 세기 DN, 모양 커널)는 샤프닝 **전** 휘도에, 모드별 `impulses`(점 노이즈: 밀도, 세기, 밝은 점 비율)는 샤프닝 **후** 휘도에. 설정이 없으면 출력은 이전과 비트 동일하다 | 실측 a0 핫픽셀(13 DN, 십자 모양)을 넣으면 샤프닝 뒤 중심 52.5~57 DN의 ±25 %, ±2화소 어두운 테. 점 노이즈 개수가 포아송 5σ 이내, 밝은 점 비율 ±0.07, 다른 모드에는 없음. target은 바뀌지 않음 | RESEARCH §1 (소유자: 노이즈 전부 정리) | T |

## 2 설계
### 2.1 구성 요소
| 항목 | 내용 |
|---|---|
| 책임 | MC-G105(Aperture 9, 3D NR 5) 출력을 흉내 낸 학습 쌍 생성 |
| 파일 | `src/data/degradations.py`(카메라 모델 함수), `src/data/datasets.py`(`DenoiseDataset` 연결, 시작 로그) |
| 개발 상태 | 신규 |
| 자원 | CPU, DataLoader 작업자 안. 256×256 패치에서 FFT 3회 + 필터 몇 개 |
| 의존 | NumPy, OpenCV |

### 2.2 실행 개념
`DenoiseDataset.__getitem__` → 크롭 → (clean / 카메라 / replay) 분기 → `apply_camera_model(patch, cfg, rng)`. 난수는 인자로 받은 `numpy.random.Generator`를 쓰고, 없으면 전역 NumPy 상태에서 시드를 뽑는다. DataLoader 작업자별 시드 규칙을 그대로 따르기 위해서다.

### 2.3 인터페이스
| 함수 | 입력 → 출력 | MREQ |
|---|---|---|
| `validate_camera_model_cfg(cfg)` | 설정 dict → 없음. 틀리면 `ValueError(키 이름)` | 04 |
| `bgr_to_ycbcr601(img)` / `ycbcr601_to_bgr(ycc)` | float BGR [0,255] ↔ BT.601 제한 범위 YCbCr | 01 |
| `camera_sharpen(y, sharpen_cfg, gain_scale=1.0)` | Y 평면(제한 범위 DN) → 샤프닝된 Y | 02 |
| `shaped_noise(h, w, bands, nps, rng)` | 방사형 스펙트럼 → 분산 1 노이즈 | 03 |
| `camera_luma_noise(y, mode, scale, rng)` | Y 평면 → 샤프닝 전 휘도 노이즈(NLF × 배율) | 03 |
| `camera_chroma_noise(hh, hw, mode, scale, rng)` | 반해상도 크기 → (Cb, Cr) 노이즈 | 03 |
| `sparse_impulses(h, w, density, amplitude, rng, bright_share=1.0)` | 크기 → 점 노이즈 평면(포아송 개수, 균등 세기, 부호) | 08 |
| `apply_camera_model(img_bgr, cfg, rng=None)` | uint8 BGR → (LR, target) uint8 BGR | 01, 03, 05, 07, 08 |

## 3 상세 설계
### 3.1 `apply_camera_model`
1. 표본마다 뽑는다: 광학 σ ~ U(`optics_sigma`), 노이즈 모드(`noise_modes[i].prob`), 노이즈 배율 ~ log-U(`noise_scale`), 샤프닝 배율 ~ U(1 ± `gain_jitter`).
2. BGR float에 Gaussian(σ) → BT.601 제한 범위 YCbCr. Cb·Cr는 `INTER_AREA`로 반으로 줄인다(카메라 NV12).
3. 카메라 정답: Y_t = S(Y). LR: Y_n = Y + n_Y, n_Y = `shaped_noise`(휘도 스펙트럼) × NLF(국소 밝기) × 배율, Y_l = S(Y_n). 반해상도 Cb·Cr에 색 노이즈(색 스펙트럼, 표준편차 × 배율)를 LR에만 더한다.
4. 둘 다 Y·Cb·Cr를 8-bit로 반올림(NV12 양자화) → Cb·Cr를 `INTER_LINEAR`로 원래 크기 → BGR → [0,255] 반올림.
5. target 모드(MREQ-07): `camera` = 3·4단계의 카메라 정답, `optics` = 2단계의 흐린 BGR float를 반올림(4:2:0·샤프닝 없음), `source` = 입력 그대로. 난수는 모드와 무관하게 같은 순서로 뽑으므로 LR은 모드에 따라 바뀌지 않는다.
6. 희소 결함(MREQ-08, LR만): `hot_pixels`가 있으면 포아송 개수(밀도 × 면적)의 위치에 세기 U(`amplitude`)를 찍고 `footprint`로 넓혀 샤프닝 전 Y_n에 더한다. 그다음 샤프닝. 고른 모드에 `impulses`가 있으면 샤프닝된 Y에 1화소 점(부호는 `bright_share` 확률로 +)을 더한다. 난수는 기존 뽑기 뒤에, 키가 있을 때만 뽑는다(키가 없으면 출력 비트 동일, 기존 val 재생성으로 확인).
- 노이즈 배율 0이면 3단계의 노이즈가 0이 되어 LR = 카메라 정답(`camera` 모드에서는 LR = target, MREQ-01). 이때 희소 결함도 넣지 않는다.

### 3.2 `camera_sharpen` (실측 맞춤, 2026-10-08)
- d = K * Y (9×9 커널, 합 0, 상하·좌우 반전과 전치에 대칭(D4)이지만 등방은 아님: 같은 주파수에서 대각 방향 증폭이 가로·세로의 약 2배).
- b = 국소 밝기(G_σ * Y, σ = `brightness_sigma`)에서 `brightness_y`→`brightness_gain` 선형 보간(양 끝 고정).
- Y' = Y + `gain_scale` · b · (`gain_pos`·max(d,0) − `gain_neg`·max(−d,0)).
- 값 맞춤 순서와 근거: `docs/CONFIG_RATIONALE.md` 14절. 1) K: 밝은 종이 2-D 노이즈 스펙트럼 비 √(NPS_a9/NPS_a0)에 D4 대칭 커널을 최소제곱으로 맞춤, 2) b(Y): 밝기 구간별 노이즈 표준편차 비, 3) ± 비대칭: Aperture 0 평균 영상을 Aperture 9 평균 영상에 어파인 정렬해 엣지 근처 화소로 맞춤(평균 증폭은 1로 고정해 1)을 건드리지 않음).
- 엣지 세기에 따른 억제 항은 맞춰 보았으나 뺐다: 맞춘 기준점(σ1.5 흐린 영상에서 81 DN/px)에 8-bit 영상의 어떤 엣지도 닿지 않아(차트 최대 33 DN/px) 작용하지 않는 항이었다. 실제 카메라도 차트의 가장 강한 엣지(검정 사각형)에서 모델보다 넘침이 크거나 같았다(실측 약 8 % vs 모델 약 3 %).

### 3.3 `shaped_noise`
흰 노이즈 FFT × √(방사형 스펙트럼 보간, 양 끝 고정) → 역FFT 실수부 → 표준편차 1로 정규화. 원형 경계(주기적)는 노이즈라 문제없다.

## 4 단위 시험 케이스
| TC | MREQ | 구분 | 테스트 이름 | 입력 | 기대 결과 |
|---|---|---|---|---|---|
| TC-CAM-01-1 | 01 | 정상 | `test_TC_CAM_01_1_PairShapeAndType` | 256×256 BGR | LR·target uint8, 입력과 같은 shape |
| TC-CAM-01-2 | 01 | 경계 | `test_TC_CAM_01_2_ZeroNoiseGivesIdenticalPair` | `noise_scale` [0, 0] | LR == target |
| TC-CAM-01-3 | 01 | 경계 | `test_TC_CAM_01_3_OddSizePatch` | 255×257 | shape 유지 |
| TC-CAM-02-1 | 02 | 정상 | `test_TC_CAM_02_1_SmallSignalGainMatchesKernel` | Y 160 위 진폭 1 DN 정현파 0.375 c/px | 커널 응답 ±10 % |
| TC-CAM-02-2 | 02 | 정상 | `test_TC_CAM_02_2_DarkAreasAreSharpenedLess` | 같은 정현파, Y 45 | 증폭 < Y 160 증폭 |
| TC-CAM-02-3 | 02 | 정상 | `test_TC_CAM_02_3_SoftEdgeOvershootInMeasuredRange` | σ1.5 흐린 120 DN 계단 | 넘침 2~10 % |
| TC-CAM-03-1 | 03 | 정상 | `test_TC_CAM_03_1_LumaNoiseFollowsNLF` | Y 120 평탄 평면, 배율 1.3 | 표준편차 = NLF(120) × 1.3 ±10 % |
| TC-CAM-03-2 | 03 | 정상 | `test_TC_CAM_03_2_ChromaNoiseOnHalfResPlanes` | 128×128 반해상도, 배율 0.8 | Cb·Cr 표준편차 = 설정 × 0.8 ±15 % |
| TC-CAM-03-3 | 03 | 정상 | `test_TC_CAM_03_3_ShapedNoiseHasUnitVarianceAndShape` | 정지 모드 휘도 스펙트럼 | 분산 1, 이웃 상관 > 0.3 |
| TC-CAM-03-4 | 03 | 정상 | `test_TC_CAM_03_4_PairNoiseLevelOnFlatGray` | 평탄 회색(Y 120), 샤프닝 끔, 쌍 생성 경로 전체 | LR − target 휘도 표준편차 = NLF(120) ±10 %, Cb·Cr 표준편차 = 설정의 0.75~1.0배(반해상도 → 이중선형 확대) |
| TC-CAM-03-5 | 03 | 경계 | `test_TC_CAM_03_5_TinyPlaneGivesFiniteNoise` | 1×1 평면, 2×2 패치 | 노이즈 0, LR − target < 40 DN |
| TC-CAM-04-1 | 04 | 오류 | `test_TC_CAM_04_1_InvalidConfigRejected` (parametrize 14) | 커널 짝수 크기, NLF 정렬 안 됨, 모드 확률 합 ≠ 1, σ 범위 역순, `sharpen` 없음, 밝기 표 중복, 스펙트럼 0, `gain_jitter` ≥ 1, `noise_scale` [0, 1.4](로그 균등이라 하한 0 불가), 음수 모드 확률, `target: raw`, 짝수 크기 `hot_pixels.footprint`, `impulses.bright_share` 1.5, `hot_pixels.density` 역순 | `ValueError`, 키 이름 포함 |
| TC-CAM-04-2 | 04 | 정상 | `test_TC_CAM_04_2_ShippedConfigIsValid` | `configs/data/bicubicpp/denoise.yaml` | 통과 |
| TC-CAM-05-1 | 05 | 정상 | `test_TC_CAM_05_1_SameSeedSameOutput` | 같은 시드 2회 / 다른 시드 | 동일 / 다름 |
| TC-CAM-05-2 | 05 | 정상 | `test_TC_CAM_05_2_ValSetRegeneratesBitIdentically` | `make_val_camera.py` 2회 | 파일 4개가 모두 있고 HR·LR 비트 동일 |
| TC-CAM-04-3 | 04 | 오류 | `test_TC_CAM_04_3_ValSetRejectsDisabledCameraModel` | `enabled: false` | `ValueError` |
| TC-CAM-06-1 | 06 | 정상 | `test_TC_CAM_06_1_DatasetUsesCameraTarget` | prob 1, clean 0 | target ≠ HR |
| TC-CAM-06-2 | 06 | 정상 | `test_TC_CAM_06_2_DatasetLegacyPathUnchanged` | `enabled: false` | target == HR |
| TC-CAM-06-3 | 06 | 정상 | `test_TC_CAM_06_3_CleanPairUsesCameraTarget` | clean_prob 1 | LR == target ≠ HR |
| TC-CAM-06-4 | 06 | 정상 | `test_TC_CAM_06_4_StartupLogReportsCameraModel` | 켜짐 / 결함 키 추가 / 빈 설정 | `camera_model on`·`target=camera`·`hot_pixels=off` / `hot_pixels=[20, 60]`·`impulses` / `camera_model off` |
| TC-CAM-06-5 | 06 | 오류 | `test_TC_CAM_06_5_SRDatasetRejectsCameraModel` | SRDataset에 켜진 블록 | `ValueError` (조용히 무시하지 않음) |
| TC-CAM-04-4 | 04 | 오류 | `test_TC_CAM_04_4_MalformedBlockStopsDatasetInit` | `noise_modes` 없음 | DenoiseDataset 생성 시 `ValueError` |
| TC-CAM-06-7 | 06 | 정상 | `test_TC_CAM_06_7_ProbSplitsCameraAndReplay` | `prob` 0.5, 200회 | 카메라 정답 비율 0.38~0.62 |
| TC-CAM-06-8 | 06, 07 | 정상 | `test_TC_CAM_06_8_DatasetSourceTargetIsHR` | `target: source`, prob 1 | target == HR, LR ≠ target |
| TC-CAM-07-1 | 07 | 정상 | `test_TC_CAM_07_1_TargetModeChangesOnlyTarget` (parametrize 2: optics, source) | 같은 시드, 모드만 바꿈 | LR 비트 동일, target = 기대 영상(입력 / 같은 σ 흐림), 카메라 정답과 다름 |
| TC-CAM-07-2 | 07 | 경계 | `test_TC_CAM_07_2_ZeroNoiseSourceTargetKeepsCameraInput` | `noise_scale` [0, 0], `target: source` | LR = 카메라 정답, target = 입력 |
| TC-CAM-08-1 | 08 | 정상 | `test_TC_CAM_08_1_HotPixelLooksLikeMeasuredDefect` | Y 160 평탄, 노이즈 0, 핫픽셀 60개/Mpx × 13 DN | 중심 39~71 DN, ±2화소 < −5 DN, target은 핫픽셀 없는 결과와 같음 |
| TC-CAM-08-2 | 08 | 정상 | `test_TC_CAM_08_2_ImpulseCountAmplitudeAndSign` | 1000×1000, 400개/Mpx, 밝은 점 0.31 / 밀도 0 | 개수 포아송 5σ, 세기 범위, 비율 ±0.07 / 0개 |
| TC-CAM-08-3 | 08 | 정상 | `test_TC_CAM_08_3_ImpulsesOnlyInTheirMode` | 움직임 모드에만 점 노이즈 | 움직임 모드 LR에 점 > 100개, 정지 모드 LR − target ≤ 2 DN |

## 5 단위 시험 결과
| 일자 | 환경 | 명령 | 결과 |
|---|---|---|---|
| 2026-10-08 | Ubuntu 22.04, Python 3.10.12, NumPy 2.2.6, OpenCV 4.12.0, PyTorch 2.9.0 | `PYTHONPATH=. python3 -m pytest tests/test_camera_model.py -q` | P 28 / F 0 |
| 2026-10-08 (검토 반영 후) | 동일 | 같은 명령 | P 32 / F 0 |
| 2026-10-08 (v0.2: 정답 모드, 희소 결함) | 동일 | `PYTHONPATH=. python3 -m pytest tests/test_camera_model.py -q` | P 43 / F 0. 기존 설정으로 val 재생성 → 기존 `val_denoise_cam_a9nr5` HR·LR 비트 동일 |
| 2026-10-08 (v0.2) | 동일 | `PYTHONPATH=. python3 -m pytest tests -q` | P 151 / F 3. 실패 3건은 아래 행과 같은 기존 환경 문제(`test_config_guards.py`) |
| 2026-10-08 | 동일 | `PYTHONPATH=. python3 -m pytest tests -q` | P 126 / F 3. 실패 3건은 이 변경 전부터 있던 환경 문제: `test_config_guards.py`의 두 건은 테스트 설정이 가리키는 `/home/jshong/Code/datasets/SR/...`가 없음, `test_eo_phase2_starts_from_the_phase1_output`는 Phase-1 체크포인트 경로가 없음 |

실측 대조 (2026-10-08, 10배 3 m 차트, `docs/CONFIG_RATIONALE.md` 14.4): 샤프닝을 실제 Aperture 0 노이즈에 적용한 결과가 실제 Aperture 9와 노이즈 스펙트럼 비 0.95~1.03(0.025~0.675 cycles/px 14대역), 밝기 구간별 표준편차 ±4 %, 2화소 상관 −0.44 vs −0.44. 합성 노이즈까지 모델로 만든 경우 스펙트럼 비 0.86~1.03, 표준편차 −4~+12 %, 2화소 상관 −0.42.

## 6 미해결 · 결정
- 움직임 모드(`motion_3dnr_inactive`)의 값은 Aperture 9 / 3D NR 0 클립 1개에서 유도했다. 같은 Aperture에서 3D NR 0이 3D NR 5보다 밝기 노이즈가 낮게 나온 점(6.2 vs 12.8 DN)은 설명되지 않았다. 확률을 낮게(0.3) 두고, 실외 이동 장면을 찍게 되면 다시 맞춘다.
- 샤프닝은 회의실 차트(10배 3 m)로만 맞췄다. 1배 클립에 그대로 적용하면 노이즈는 맞지만(스펙트럼 비 0.90~0.98) 엣지 근처 고주파가 실제보다 크다(24.3 vs 20.2). 운용이 망원 위주라 그대로 두고, `gain_jitter` ±10 %로 일부 덮는다. `docs/CONFIG_RATIONALE.md` 14.4.
- 아주 강한 엣지(정반사 등)에서 카메라가 증폭을 줄이는지는 관측하지 못했다(차트 최대 기울기에서는 줄이지 않았다).
- 카메라 톤 곡선(검정 뭉개짐), 디모자이크 가짜 색은 모델에 없다. 가짜 색은 RESEARCH HYP-03.
- 희소 결함 값(2026-10-08 원본 클립 실측, `results/denoise_release_v1/analysis/noise_census/`): 고정 핫픽셀은 1배·10배 Aperture 0 클립에서 같은 위치 49개(평탄 1.49 Mpx, 약 33개/Mpx, 10 DN 이상만 셈), 세기 11~18 DN(10~90 %), 십자 모양(이웃 0.35), 무채색. Aperture 9에서 같은 결함이 52.5~57 DN + 테 −0.22. 모델은 13 DN 입력에서 62 DN + 테 −0.23. 점 노이즈는 3D NR 0 클립에서만 많다(약 380개/Mpx, ±38~46 DN, 밝은 점 31 %). 3D NR 5 정지 장면은 1~1.5개/Mpx라 정지 모드에는 넣지 않았다. 3D NR 0 클립에서는 핫픽셀이 샤프닝을 거의 받지 않아(11 DN, 테 없음) 그 클립의 샤프닝이 운용 설정과 다르다는 기존 의문(6절 첫 항)을 뒷받침한다. 그래서 점 노이즈는 샤프닝 후 실측값 그대로 넣는다.
- 가로·세로 줄무늬(FPN, 프레임별 띠)는 측정 한계(0.1~0.2 DN) 수준이라 넣지 않았다. 프레임 전체 밝기 흔들림(최대 1.2 DN)은 조명·AE 몫이라 Denoise 대상이 아니다.
