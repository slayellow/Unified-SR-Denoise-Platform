# GridScore 모듈 문서

| 문서번호 | 버전 | 작성일 | 작성자 | 시스템 | 상위 문서 |
|---|---|---|---|---|---|
| USRD-MOD-GRID-001 | 0.4 | 2026-09-28 | JSHONG | unified-sr-denoise-platform | USRD-RESEARCH-001 §2 (평가 프로토콜 P2) |

개정 이력
| 버전 | 일자 | 변경 내용 | 작성자 |
|---|---|---|---|
| 0.1 | 2026-09-28 | 최초 작성 | JSHONG |
| 0.2 | 2026-09-29 | 프로토콜 P2: 기본 프레임 집합·인덱스 교체, `--min_frames`·`--manifest`로 장수 부족·목록 불일치 거부(MREQ-GRID-03), 기본 장치 cpu, json에 frames·index·manifest·device 기록, TC-GRID-03-2~03-6 추가, `grid_score` 반환 표기 정정 | JSHONG |
| 0.3 | 2026-10-15 | 출하 IR x4 기준값 재현(MREQ-GRID-04, TC-GRID-04-1) 추가. §5의 P2 기준값을 출하본 ep89로 정정(ep84는 10-06에 교체된 모델) | JSHONG |
| 0.4 | 2026-10-17 | `load_model`이 옆의 `train_config.yaml` 대신 체크포인트에 저장된 model config를 읽음(없으면 `ValueError`). MREQ-GRID-04에 그 요구를 넣고 TC-GRID-04-2·04-3 추가 | JSHONG |

이 저장소에는 SRS가 없다. 상위 요구는 연구 계획서 `docs/20260928-RESEARCH-X4IRGrid.md` §2의 평가 프로토콜(현재 P2)이며, 이 모듈이 그 주 지표를 계산한다. 따라서 이 모듈의 동작을 바꾸면 프로토콜 버전도 올려야 한다.

## 1 요구
| ID | 유형 | 요구사항 | 수용 기준 | 상위 | 검증 |
|---|---|---|---|---|---|
| MREQ-GRID-01 | 기능 | 모듈은 SR 출력에서 업샘플러 주기(`period`)의 격자 성분을 k/period cycles/px 스펙트럼 피크 대 주변 중앙값 비로 가로(h)·세로(v)·대각(d)별로 산출하여야 한다 | 격자 없는 장면 < 3, 0.5 GV 격자 > 20, 가로 전용 격자는 v < 3, 비정수 bin(주기 3)도 검출 | RESEARCH §2 | T |
| MREQ-GRID-02 | 기능 | 모듈은 같은 입력의 bicubic 확대 대비 SR 출력의 선명도비(평균 Sobel 크기 비)를 산출하여야 한다 | 자기 자신 = 1.0, 블러 σ2 < 0.9 | RESEARCH §1 제약 | T |
| MREQ-GRID-03 | 차단 | 모듈은 잘못된 입력을 조용히 채점하지 않고 거부하여야 한다: period < 2, 선택된 프레임이 0장이거나 `--min_frames`(기본 50, 프로토콜 장수)보다 적음, 선택된 프레임의 이름·sha256이 `--manifest` 목록과 순서까지 같지 않음 | `ValueError` + 사유, 체크포인트를 열기 전. 정확히 `--min_frames`장이고 목록과 같은 선택은 통과. 통과 0건 | RESEARCH §2 | T |
| MREQ-GRID-04 | 회귀 | 출하 IR x4(ep89, `checkpoints/bicubicpp_deploy/finetune_bicubicplusplus_ir_sr_x4_dim32_v3`)를 저장 config로 다시 만들어 P2로 채점하면, 그 폴더의 `grid_best.json` 기록값과 같아야 한다 | 숫자 키 9개(점수 7 + epoch, n_frames)가 비트 단위로 같음. 체크포인트·기록·프레임이 없으면 skip. 모델은 체크포인트에 저장된 config로만 만들고, 저장된 config가 없으면 옆의 yaml로 대신하지 않고 `ValueError` | RESEARCH §2 | T |

## 2 설계
### 2.1 구성 요소
| 항목 | 내용 |
|---|---|
| 책임 | 체크포인트 하나를 실기 프레임에 FP32로 돌려 격자 점수와 선명도비를 낸다 |
| 파일 | `examples/evaluation/score_grid_artifact.py` (CLI + 라이브러리 함수) |
| 개발 상태 | 신규. 2026-09-22 보드 검증 스크립트의 전역 위상평균 지표를 대체(국소 부호가 상쇄돼 격자를 놓쳤음) |
| 자원 | CPU(기본, IEEE FP32). GPU도 되지만 Ampere에서 conv가 TF32로 돌 수 있어 프로토콜 채점에는 쓰지 않는다. 50프레임 × 1280×720 FFT, CPU로 체크포인트당 약 10초(2026-09-29 실측) |
| 의존 | `src.models.build_model`, `src.engine.checkpoint.normalize_state_dict`, OpenCV, NumPy, PyTorch |

### 2.2 실행 개념
1. 정렬된 PNG 목록에서 `--index start:stop:step`으로 프레임 선택. 0장이거나 `--min_frames`보다 적으면 거부. `--manifest`가 있으면 선택된 파일의 이름·sha256을 목록과 순서까지 대조하고 다르면 거부. 둘 다 체크포인트를 열기 전이다.
2. 체크포인트에 저장된 model config(`config.model`)로 모델을 만들고 `model_state_dict`(학습이 EMA를 저장했으면 EMA)를 로드. 저장된 config가 없으면 `ValueError`(2026-10-17부터. 전에는 옆의 `train_config.yaml`을 읽어, 릴리스 폴더처럼 yaml이 없는 곳의 체크포인트를 채점하지 못했다).
3. 프레임마다 보드 디지털줌과 같은 LR 크롭 → 모델 → 휘도(BT.601) → 8-bit 반올림. 파워스펙트럼(Hann 창)을 누적하고, 같은 크롭의 bicubic 대비 선명도비를 기록.
4. 누적 스펙트럼에서 k/period(≤ 1/2)마다 h·v·d 피크비를 계산. 선명도비는 프레임 중앙값.

프레임마다 격자의 부호가 내용에 따라 바뀌어도 파워(제곱)는 상쇄되지 않는다. 이것이 전역 위상평균과의 차이다. 스펙트럼을 여러 프레임에 걸쳐 합하는 것은 잡음 bin의 분산을 줄이기 위해서다(단일 프레임에서는 우연만으로 피크비가 약 4까지 나온다).

### 2.3 인터페이스
| 함수 | 입력 → 출력 | MREQ |
|---|---|---|
| `power_spectrum(img)` | 2-D float → Hann 창 파워스펙트럼 | 01 |
| `peak_ratio(spec, fy, fx)` | (fy, fx) ±1 bin 최대 / 17×17 주변(중앙 5×5 제외) 중앙값 | 01 |
| `grid_score(spec, period)` | k/period(≤ 1/2)마다 h·v·d. period 4면 `1/4h, 1/4v, 1/4d, 1/2h, 1/2v, 1/2d`. period < 2 → `ValueError` | 01, 03 |
| `sharpness_ratio(img, ref)` | 평균 Sobel 크기 비 | 02 |
| `load_model(ckpt, device)` | (model, scale, epoch) | 04 |
| `upscale(model, lr_rgb, device)` | uint8 RGB → 8-bit 반올림 휘도 | 04 |
| `score(model, scale, frames, crop, device)` | 격자 점수 + `sharpness` | 01, 02 |
| `main(args)` | CLI. 선택 0장·`--min_frames` 미만·목록 불일치 → `ValueError`. `--json`으로 점수와 ckpt·epoch·n_frames·frames·index·manifest·device 저장 | 03 |

## 3 상세 설계
- 피크 bin: `round(f × N)`. 1280/3처럼 정수가 아니어도 ±1 bin 최대로 잡는다.
- 주변: 피크 주위 17×17 bin에서 중앙 5×5를 뺀 중앙값. 격자 자신의 누설을 배경에 넣지 않기 위함이다.
- 이름 규칙: k/period가 1/2이면 `"1/2"`, 아니면 약분하지 않은 `"k/period"`.
- 기본값 (P2): 프레임 `results/release_v3/eval/ir92_raw_600`(저장소 루트 기준 상대 경로), 크롭 `160,166,320,180`(x4 IR 디지털줌), 인덱스 `0:600:12`(50장), `--min_frames 50`, `--manifest results/release_v3/eval/ir92_p2_frames.sha256`(`''`이면 대조 생략), `--device cpu`. 모두 인자로 바꿀 수 있다. 다른 프레임 폴더를 쓸 때는 `--manifest`도 바꾸거나 `''`로 꺼야 한다. P1 기본값은 프레임 `260922_TEST/RAW/92/20260922_112645/raw`, 인덱스 `100:2400:46`이었고 폴더가 없어졌다.

## 4 단위 시험 케이스
| TC | MREQ | 구분 | 테스트 이름 | 입력 | 기대 결과 |
|---|---|---|---|---|---|
| TC-GRID-01-1 | 01 | 정상 | `test_TC_GRID_01_1_SmoothSceneScoresNearOne` | σ1 블러 잡음 8장 | 1/4 h·v·d < 3 |
| TC-GRID-01-2 | 01 | 정상 | `test_TC_GRID_01_2_SubGVGridStandsOut` | + 0.5 GV 주기-4 격자 | 1/4h, 1/4d > 20 |
| TC-GRID-01-3 | 01 | 경계 | `test_TC_GRID_01_3_HorizontalOnlyGridLeavesVerticalFlat` | 가로 전용 격자 | h > 20, v < 3 |
| TC-GRID-01-4 | 01 | 경계 | `test_TC_GRID_01_4_OddPeriodBinIsFound` | 주기 3 (비정수 bin) | 1/3h > 20 |
| TC-GRID-01-5 | 03 | 오류 | `test_TC_GRID_01_5_PeriodBelowTwoRejected` | period = 1 | `ValueError` |
| TC-GRID-02-1 | 02 | 정상 | `test_TC_GRID_02_1_SharpnessRatioOfSelfIsOne` | 같은 영상 | 1.0 |
| TC-GRID-02-2 | 02 | 정상 | `test_TC_GRID_02_2_BlurLowersSharpness` | σ2 블러 | < 0.9 |
| TC-GRID-03-1 | 03 | 오류 | `test_TC_GRID_03_1_EmptyFrameSelectionRejected` | 인덱스 범위 밖 (0장) | `ValueError`, 체크포인트를 열기 전 |
| TC-GRID-03-2 | 03 | 오류 | `test_TC_GRID_03_2_TooFewFramesRejected` | 11장 선택, 기본 `--min_frames 50` | `ValueError`, 체크포인트를 열기 전 |
| TC-GRID-03-3 | 03 | 경계 | `test_TC_GRID_03_3_MinFramesIsInclusive` | 11장 선택, `--min_frames 11`, `--manifest ''` | 프레임 검사 통과(없는 체크포인트에서 `FileNotFoundError`) |
| TC-GRID-03-4 | 03 | 오류 | `test_TC_GRID_03_4_MinFramesBelowOneStillRejectsEmpty` | 0장 선택, `--min_frames 0` / `-1` | `ValueError`, 체크포인트를 열기 전 |
| TC-GRID-03-5 | 03 | 오류 | `test_TC_GRID_03_5_FramesNotMatchingManifestRejected` | 목록과 바이트가 다른 프레임 / 한 칸 밀린 선택 | `ValueError`, 체크포인트를 열기 전 |
| TC-GRID-03-6 | 03 | 정상 | `test_TC_GRID_03_6_FramesMatchingManifestAccepted` | 목록과 같은 3장 | 대조 통과(`FileNotFoundError`까지 진행) |
| TC-GRID-04-1 | 04 | 정상 | `test_TC_GRID_04_1_ShippedIRx4ReproducesRecordedP2` | 출하 IR x4 체크포인트, P2 프레임 50장 | 숫자 키 9개가 `grid_best.json`과 같음 |
| TC-GRID-04-2 | 04 | 정상 | `test_TC_GRID_04_2_LoadsFromSavedConfigWithoutYamlBeside` | 옆에 yaml이 없는 작은 x4 체크포인트 | 저장된 config로 로드, scale 4, epoch, 가중치 일치 |
| TC-GRID-04-3 | 04 | 오류 | `test_TC_GRID_04_3_NoSavedConfigRejected` | config 없는 체크포인트 + 옆의 yaml | `ValueError`(옆 yaml로 대체하지 않음) |

## 5 단위 시험 결과
| 일자 | 환경 | 명령 | 결과 |
|---|---|---|---|
| 2026-09-28 | Ubuntu 22.04, Python 3.10, NumPy/OpenCV/PyTorch 2.9.0 | `python3 -m pytest tests/test_score_grid_artifact.py -q` | P 8 / F 0 |
| 2026-09-29 | 동일 | `PYTHONPATH=. python3 -m pytest tests/test_score_grid_artifact.py -q` | P 15 / F 0 |
| 2026-10-15 | Ubuntu 22.04, Python 3.10.12, PyTorch 2.9.0, CPU | `python3 -m pytest -q -rs tests/test_score_grid_artifact.py` | P 16 / F 0 / skip 0. 변형 시험: `upscale`의 8-bit 반올림을 빼면 TC-GRID-04-1 실패 |
| 2026-10-17 | 같음 | `python3 -m pytest -q -rs tests/test_score_grid_artifact.py` | P 18 / F 0 / skip 0. `load_model`이 저장된 config를 읽게 바꾼 뒤 TC-GRID-04-1(출하 IR x4 P2 재현) 그대로, 새 TC-GRID-04-2·04-3은 바꾸기 전 코드에서 실패 |

실측 교차 확인 (2026-09-28, CLI, P1): 출하 IR x4 v3(ep84) `1/4h 77.34 · 1/4v 5.50 · 1/4d 63.84 · 선명도 1.53`, Phase-1 v3(ep113) `5.41 · 2.65 · 26.54 · 1.07`. 09-23 스크래치 측정(채널 0 기준)의 출하 모델 값 77.34 / 63.84와 일치. Phase-1 값의 차이(6.3 / 19.1)는 휘도 채점과 Phase-1이 무채색 출력 모델이 아니라서 생긴다.

P2 기준값 (CLI, CPU): 출하 IR x4 ep89(2026-10-06 수정판) `1/4h 5.41 · 1/4v 5.48 · 1/4d 14.30 · 선명도 1.20`(`grid_best.json`, TC-GRID-04-1이 재현). 교체 전 ep84는 `63.07 · 3.62 · 38.90 · 1.52`, Phase-1 ep113은 `6.66 · 2.44 · 23.04 · 1.08`(2026-09-29). 에폭별 값은 RESEARCH §3.

## 6 미해결 · 결정
- 정지 장면(예: 09-14 EO 책등 캡처)에서는 장면 자체의 성분이 피크비에 섞인다. 그 경우 bicubic 행을 대조로 두고, 장면 기여가 작은 방향(v, d)으로 판정한다.
- 피크비는 배경(장면)에 대한 상대값이라 장면이나 주파수가 다르면 직접 비교할 수 없다. 같은 프로토콜 집합 안에서만 비교한다(P1 값과 P2 값은 비교하지 않는다).
- EO 입력(보드 `.rgba` 텐서)과 이미 계산된 출력만 채점하는 모드는 없다. EO 원인 실험과 보드 검증 전에 추가하고, 그때 프로토콜 버전을 올린다.
- TC-GRID-04-1은 x4(PixelShuffle 2단) 경로만 덮는다. x3(ConvTranspose) 경로까지 고정하려면 출하 IR x3의 P2 기록을 새로 남겨야 한다.
