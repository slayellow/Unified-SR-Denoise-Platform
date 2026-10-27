# ChartScore 모듈 문서

| 문서번호 | 버전 | 작성일 | 작성자 | 시스템 | 상위 문서 |
|---|---|---|---|---|---|
| USRD-MOD-CHART-001 | 0.3 | 2026-10-08 | JSHONG | unified-sr-denoise-platform | USRD-RESEARCH-003 §2 (평가 프로토콜 P1) |

개정 이력
| 버전 | 일자 | 변경 내용 | 작성자 |
|---|---|---|---|
| 0.1 | 2026-10-08 | 최초 작성. 회의실 차트 분석 스크립트(`results/denoise_release_v1/analysis/scripts/`)를 저장소로 옮김(소유자 승인 2026-10-08) | JSHONG |
| 0.2 | 2026-10-08 | 프로토콜 P1.1(소유자 승인 2026-10-08): P1은 엣지 위치를 노이즈 낀 입력의 행·열마다 따로 맞춰, 노이즈만 정확히 뺀 완벽한 Denoise도 넘침 +5.6 %p, MTF50 비 1.73으로 읽었다(검토 `results/denoise_release_v1/analysis/finetune_review/`). 엣지 직선을 입력에서 한 번 맞춰 출력에 재사용하고, 넘침은 프레임 평균 프로파일에서 잰다. 보조 지표(점 보존, 질감 비, 실측 핫픽셀 잔차, 사각형 엣지 더한 넘침) 추가 | JSHONG |
| 0.3 | 2026-10-15 | 출하 Denoise v1 기준값 재현(MREQ-CHART-06, TC-CHART-06-1) 추가 | JSHONG |

이 저장소에는 SRS가 없다. 상위 요구는 연구 계획서 `docs/20261008-RESEARCH-EODenoiseV3.md` §2의 평가 프로토콜 P1이며, 이 모듈이 그 실기 지표를 계산한다. 이 모듈의 동작을 바꾸면 프로토콜 버전도 올려야 한다. 모듈 약어 `CHART`.

## 1 요구
| ID | 유형 | 요구사항 | 수용 기준 | 상위 | 검증 |
|---|---|---|---|---|---|
| MREQ-CHART-01 | 기능 | 모듈은 차트 프레임에서 5° 경사 검정 사각형을 찾아 화면 배율(mm/px)과 차트 좌표 → 화소 좌표 변환을 정하여야 한다 | 합성 프레임에서 중심 1 px, 배율 2 % 이내. 사각형이 없으면 `ValueError` | RESEARCH §2 | T |
| MREQ-CHART-02 | 기능 | 모듈은 같은 프레임의 카메라 입력과 Denoise 출력에서 다음을 같은 방법으로 재야 한다: 회색 패치 P4 평탄부 노이즈(휘도 고주파 σ), P2·P3 윗변 넘침/꺼짐(%), 사각형 왼변·윗변 MTF50과 넘침, P3 색 고주파 σ, 가장 작은 두 글자 줄의 색 고주파 σ(가짜 색). 엣지 직선은 카메라 입력에서 한 번 맞춰 출력에도 쓴다(사각형: 문턱 윤곽의 직선 맞춤, 패치: P0~P3 공통 윗변을 열 묶음 위치로 맞춤) | 합성 입력에서: 노이즈 σ 2배 → 지표 2배 ±5 %, 넘침 10 % 계단 → 10 ±2 %, σ1 흐린 경사 엣지 MTF50 = 0.187 ±10 % | RESEARCH §2 | T |
| MREQ-CHART-03 | 기능 | 모듈은 프레임별 값의 중앙값과, 출력 − 입력 넘침(더해진 오버슈트, 패치 P2·P3와 사각형)·출력/입력 MTF50(유지율)·출력/입력 노이즈 비를 클립마다 보고하여야 한다. 넘침·꺼짐은 프레임 평균 엣지 프로파일에서 잰다(프레임마다 잰 최댓값은 노이즈로 부풀려진다) | 항등 모델에서 더해진 오버슈트 0, 유지율 1, 노이즈 비 1. **노이즈 낀 입력(10 DN, 10프레임)에 완벽한 Denoise(출력 = 노이즈 없는 원본)**: 더한 넘침 P2 ±2, P3 ±3.5, 사각형 ±2 %p, MTF50 비 1 ±0.06 (측정 정밀도: 시드 묶음 6개에서 표준편차 P2 0.7, P3 1.2, 사각형 0.4~0.7 %p, MTF 비 0.02) | RESEARCH §1 | T |
| MREQ-CHART-05 | 기능 | 모듈은 차트에서 볼 수 없는 위험을 보조 지표로 보고하여야 한다: (1) 점 보존: 회색 평탄면에 1·2·3화소, +20·60·150 DN 점을 넣어 카메라 모델(같은 노이즈 뽑기, 점 유무)과 Denoise를 거친 출력 최댓값 / 원본 최댓값, (2) 질감 비: 깨끗한 영상 중앙 512 영역의 라플라시안 분산 출력/입력, (3) 실측 센서 핫픽셀 위치의 잔차(화소 − 5×5 중앙값) 출력/입력 | 항등 모델에서 점 보존 = 카메라 입력 값, 질감 비 1, 핫픽셀 비 1. σ2 흐림 모델은 점 보존·질감 비가 항등의 절반 미만, 3×3 중앙값 모델은 핫픽셀 비 < 0.2. 영상 폴더가 비면 `FileNotFoundError` | RESEARCH §2 | T |
| MREQ-CHART-04 | 차단 | 모듈은 잘못된 입력을 조용히 채점하지 않아야 한다: 프레임 수 부족, 클립 sha256 목록 불일치, 클립 크기가 NV12 1920×1080의 배수가 아님 | 체크포인트를 열기 전 `ValueError` | RESEARCH §2 | T |
| MREQ-CHART-06 | 회귀 | 출하 BicubicPPDenoise[v1](`results/denoise_release_v1/checkpoint/best.pth`, EXP-09-1)을 P1.1 기본 클립·핫픽셀·질감 폴더와 릴리스의 카메라 config로 채점하면, `analysis/p1_scores/P1.1/EXP-09-1.json` 기록값과 같아야 한다 | 숫자 값 161개가 비트 단위로 같음(NaN은 NaN끼리 같음으로 본다). 체크포인트·기록·클립·질감 폴더가 없으면 skip | RESEARCH §2 | T |

## 2 설계
### 2.1 구성 요소
| 항목 | 내용 |
|---|---|
| 책임 | 체크포인트 하나를 회의실 차트 원본 NV12 클립에 FP32로 돌려 P1 실기 지표를 낸다 |
| 파일 | `examples/evaluation/score_eo_chart.py` (CLI + 라이브러리 함수) |
| 개발 상태 | 신규 (`results/denoise_release_v1/analysis/scripts/`의 `cap_metrics.py`, `patch_halo.py`, `cap_metrics_geom.py`를 정리해 옮김) |
| 자원 | CPU(기본, IEEE FP32). 클립당 10프레임 × 1280×720 Denoise 1회 |
| 의존 | `examples/evaluation/score_grid_artifact.load_model`, OpenCV, NumPy, SciPy, PyTorch |

### 2.2 실행 개념
1. 클립(`이름=경로`)마다 크기를 확인하고 프레임 `start::step`을 고른다(기본 `20::10`, 앞 10장). 장수가 `--min_frames`보다 적거나 `--manifest`의 sha256과 다르면 거부한다. 모두 체크포인트를 열기 전이다.
2. 프레임마다: NV12 → BGR, 사각형 검출로 배율·좌표 변환, 차트 중심을 담는 1280×720 원화소 창을 잘라 Denoise(배포와 같은 입력 크기), 창 안에서 카메라 입력과 출력의 지표를 잰다.
3. 클립마다 프레임 중앙값과 출력 − 입력 / 출력 ÷ 입력 파생값을 `--json`에 쓴다(클립 sha256, 체크포인트 경로·md5·epoch, 프레임 번호 포함).

### 2.3 인터페이스
| 함수 | 입력 → 출력 | MREQ |
|---|---|---|
| `find_square(y, side_mm)` | 휘도 평면 → (중심, mm/px, 윤곽). 없으면 `ValueError` | 01 |
| `mm_to_px(pt, c_px, c_mm, scale)` / `box_px(...)` | 차트 mm → 화소 | 01 |
| `edge_mtf(y, contour, axis)` | 사각형 왼변(1)·윗변(0) → `mtf50`, `over_pct`, `under_pct`, `esf`(정규화 엣지 프로파일) | 02 |
| `top_edge_line(y, boxes)` / `patch_lines(...)` | 패치 윗변들 → 공통 직선(기울기, 행 오프셋) | 02 |
| `top_edge_profile(y, box, line)` / `edge_overshoot(prof)` / `top_edge_overshoot(...)` | 패치 윗변 → 정렬 프로파일 → (넘침 %, 꺼짐 %) | 02, 03 |
| `esf_overshoot(e)` | 정규화 엣지 프로파일 → (넘침 %, 꺼짐 %) | 03 |
| `chart_window(bgr, chart)` | 전체 프레임 → 창 원점, 사각형 중심·배율·윤곽 | 01 |
| `hot_residual(bgr, xy)` | 위치 목록 → 잔차 중앙값 | 05 |
| `dot_retention(run, camera_cfg)` / `texture_ratio(run, root)` | Denoise 함수 → 보조 지표 | 05 |
| `flat_noise(y, box)` / `chroma_noise(bgr, box)` | 영역 → 고주파 σ | 02 |
| `score_frame(bgr, chart, model_fn, hot_xy=None)` | 프레임 한 장 → 입력·출력 지표(+ 핫픽셀 잔차) | 02, 05 |
| `summarize(frames)` | 프레임 지표 목록 → 중앙값 + 프레임 평균 프로파일의 넘침 + 파생값 | 03 |
| `main(args)` | CLI | 04 |

## 3 상세 설계
- 차트 기하(mm, 페이지 왼쪽 위 기준): `z10` = `eo_chart_a4_tele.pdf`(사각형 16 mm, 중심 (95.17, 97.67), 패치 5개 12×10 mm), `z01` = `eo_chart_a4.pdf`(사각형 36 mm, 중심 (41.5, 66.5), 패치 28×23 mm). 차트 생성 스크립트 `results/denoise_release_v1/analysis/scripts/make_chart*.py`와 같은 값.
- 사각형: Y < 80 연결 요소 중 최소 외접 사각형의 가로세로비 0.85~1.15, 채움 > 0.9, 기울기 5±3°인 가장 큰 것.
- MTF50: 경사 엣지 4배 과표본 ESF → LSF(Hann) → FFT, 0.5를 처음 지나는 주파수. 엣지 직선은 입력에서 찾은 문턱(Y < 80) 윤곽의 행별 왼쪽 끝을 직선 맞춤한 것이고, 출력도 같은 윤곽·직선을 쓴다(P1은 행마다 노이즈 낀 기울기 중심을 맞춰 입력 ESF만 흐려졌다).
- 패치 넘침: P0~P3 윗변은 인쇄상 한 직선이다. 패치마다 열을 6묶음으로 나눠 묶음 평균 프로파일의 엣지 위치(가장 큰 기울기 ±3화소의 기울기 중심)를 구하고, 24점으로 직선 하나를 맞춘다. 열마다 그 직선으로 이동해 평균한 프로파일을 프레임별로 모으고, 프레임 평균 프로파일에서 종이 쪽 넘침과 패치 쪽 꺼짐을 단차 대비 %로 잰다. 사각형 넘침도 프레임 평균 ESF에서 잰다.
- 핫픽셀 잔차: 실측 위치(`results/denoise_release_v1/analysis/noise_census/hot_pixels_a0.json`, Aperture 0 두 클립에서 같은 위치 10 DN 초과 49개, 창 안 약 19개)에서 화소 − 5×5 중앙값의 중앙값.
- 점 보존: Y 120 회색 96×96, 점 중심 (48, 48), 카메라 모델 `configs/data/bicubicpp/denoise.yaml`, 시드 0~4의 중앙값. 질감 비: `val_denoise/HR` 앞 10장.
- 평탄부 노이즈: 영역 가장자리 20 %를 뺀 안쪽에서 Y − G(σ1.5)*Y의 표준편차. 색은 YCrCb의 Cr·Cb에서 같은 방식으로 재 두 값의 평균.
- Denoise 입력: BGR → RGB [0,1], 출력 RGB → BGR 8-bit 반올림(배포 경로와 같은 8-bit).

## 4 단위 시험 케이스
| TC | MREQ | 구분 | 테스트 이름 | 입력 | 기대 결과 |
|---|---|---|---|---|---|
| TC-CHART-01-1 | 01 | 정상 | `test_TC_CHART_01_1_SquareFoundWithScale` | 합성 차트 프레임(0.166 mm/px) | 중심 1 px, 배율 2 % 이내 |
| TC-CHART-01-2 | 01 | 오류 | `test_TC_CHART_01_2_NoSquareRejected` | 흰 프레임 | `ValueError` |
| TC-CHART-02-1 | 02 | 정상 | `test_TC_CHART_02_1_FlatNoiseScalesWithSigma` | σ 3 / 6 노이즈 | 비 2.0 ±5 %, 평탄 영상 0 |
| TC-CHART-02-2 | 02 | 정상 | `test_TC_CHART_02_2_OvershootOfKnownStep` | 종이 쪽 10 % 넘침 계단 | 10 ±2 % |
| TC-CHART-02-3 | 02 | 정상 | `test_TC_CHART_02_3_MTF50OfGaussianEdge` | σ1 흐린 경사 사각형 | 0.187 ±10 % |
| TC-CHART-03-1 | 03 | 정상 | `test_TC_CHART_03_1_IdentityModelIsNeutral` | 합성 차트 + 노이즈, 항등 모델 | 더해진 오버슈트 0, 유지율 1, 노이즈 비 1 |
| TC-CHART-03-3 | 03 | 정상 | `test_TC_CHART_03_3_NoisyPerfectDenoiserIsNeutral` | 샤프닝(넘침 약 8 %) 합성 차트 + 노이즈 10 DN, 10프레임, 출력 = 노이즈 없는 원본 | 입력 넘침 P2 > 5 %, 더한 넘침 P2 ±2, P3 ±3.5, 사각형 ±2 %p, MTF50 비 1 ±0.06. P1 채점기에서는 실패(+5.6 %p, 1.73 / 1.38) |
| TC-CHART-03-2 | 01 | 정상 | `test_TC_CHART_03_2_GeometryRoundTrip` | 사각형 기준 16.6 mm 오른쪽 점 | 100 px 오른쪽, 패치 상자 크기 양수 |
| TC-CHART-04-1 | 04 | 오류 | `test_TC_CHART_04_1_TooFewFramesRejected` | 5프레임 클립, `--min_frames 10` | `ValueError`, 체크포인트를 열기 전 |
| TC-CHART-04-2 | 04 | 오류 | `test_TC_CHART_04_2_ManifestMismatchRejected` | 다른 sha256 | `ValueError` |
| TC-CHART-04-3 | 04 | 오류 | `test_TC_CHART_04_3_BadClipSizeRejected` | NV12 프레임 배수가 아닌 파일 | `ValueError` |
| TC-CHART-05-1 | 05 | 정상 | `test_TC_CHART_05_1_DotRetentionIdentityAndBlur` | 항등 / σ2 흐림, 1·2화소 +60 | 항등 = 카메라 값, 흐림 < 절반, 2화소 > 1화소 > 0.3 |
| TC-CHART-05-2 | 05 | 정상·오류 | `test_TC_CHART_05_2_TextureRatio` | 무작위 질감 3장 / 빈 폴더 | 항등 1, 흐림 < 0.5 / `FileNotFoundError` |
| TC-CHART-05-3 | 05 | 정상 | `test_TC_CHART_05_3_HotPixelResidual` | 합성 차트, 창 안 8곳에 +40 DN | 입력 잔차 40 ±2, 항등 비 1, 3×3 중앙값 비 < 0.2 |
| TC-CHART-06-1 | 06 | 정상 | `test_TC_CHART_06_1_ReleaseDenoiseReproducesP11` | 출하 Denoise v1, P1.1 클립 3개 | 숫자 값 161개가 EXP-09-1 기록과 같음 |

## 5 단위 시험 결과
| 일자 | 환경 | 명령 | 결과 |
|---|---|---|---|
| 2026-10-08 | Ubuntu 22.04, Python 3.10.12, NumPy 2.2.6, OpenCV 4.12.0, PyTorch 2.9.0 | `CUDA_VISIBLE_DEVICES= PYTHONPATH=. python3 -m pytest tests/test_score_eo_chart.py -q` | P 10 / F 0 |
| 2026-10-08 (v0.2, P1.1) | 동일 | 같은 명령 | 수정 전 TC-CHART-03-3 실패(재현), 수정 후 P 14 / F 0 |
| 2026-10-09 (v0.2) | 동일 | `PYTHONPATH=. python3 -m pytest tests -q` | P 155 / F 3. 실패 3건은 기존 환경 문제(`test_config_guards.py`, 데이터·체크포인트 경로 없음) |
| 2026-10-15 (v0.3) | Ubuntu 22.04, Python 3.10.12, PyTorch 2.9.0, CPU | `python3 -m pytest -q -rs tests/test_score_eo_chart.py` | P 15 / F 0 / skip 0. 변형 시험: 디노이저 출력의 8-bit 반올림을 빼면 TC-CHART-06-1 실패(평탄부 노이즈·넘침·MTF50 값이 달라짐) |

실측 교차 확인 (2026-10-08, P1 — 넘침·MTF 비는 노이즈 편향으로 무효, 평탄부 노이즈·색만 유효. P1.1 값은 `results/denoise_release_v1/analysis/p1_scores/P1.1/`): 10배 a9 평탄부 노이즈 17.59 → 3.69, P2 더한 넘침 +3.9 %p, MTF50 비 1.64 / 1.34. 1배 a9 +11.9 %p. 회의실 분석(평균·단일 프레임 기준 17.2 → 3.6, +1.5 %p(단일 프레임) / +13.3 %p)과 같은 방향·크기. 결과 `results/denoise_release_v1/analysis/p1_scores/BL1_v2.json`.

## 6 미해결 · 결정
- 차트 두 장(10배용, 1배용)의 기하만 안다. 다른 차트를 쓰면 `CHARTS`에 추가하고 프로토콜 버전을 올린다.
- 가짜 색 지표는 보고만 한다(RESEARCH HYP-03).
