# ReleaseGolden 모듈 문서

| 문서번호 | 버전 | 작성일 | 작성자 | 시스템 | 상위 문서 |
|---|---|---|---|---|---|
| USRD-MOD-REL-001 | 0.3 | 2026-10-15 | JSHONG | unified-sr-denoise-platform | 리팩토링 검토 2026-10-15 (실행 계획 1단계 안전망) |

개정 이력
| 버전 | 일자 | 변경 내용 | 작성자 |
|---|---|---|---|
| 0.1 | 2026-10-15 | 최초 작성. 출하 Bicubic++ SR v3와 BicubicPPDenoise[v1]의 재현을 지키는 golden 테스트 | JSHONG |
| 0.2 | 2026-10-15 | 출하 데이터 레시피 digest(MREQ-REL-04, TC-REL-04-1) 추가. 게이트 기준값은 각 게이트 모듈(MREQ-GRID-04, MREQ-CHART-06)에 둠 | JSHONG |
| 0.3 | 2026-10-16 | 가우시안 노이즈 반올림 수정으로 TC-REL-04-1 기록값 14개를 다시 잼(EO·IR Phase-2, Denoise). 수정 전 기록은 `de502f6` | JSHONG |

이 저장소에는 SRS가 없다. 상위 요구는 리팩토링 검토(2026-10-15)의 하드 제약이다. 리팩토링 뒤에도 출하 체크포인트가 저장된 config로 그대로 로드되고, 같은 배포 출력과 같은 W16A16 export 그래프를 내야 한다. 검토 당시 출하 가중치를 로드하는 테스트는 0개였다. 모듈 약어 `REL`.

## 1 요구
| ID | 유형 | 요구사항 | 수용 기준 | 상위 | 검증 |
|---|---|---|---|---|---|
| MREQ-REL-01 | 회귀 | 출하·학습 기준 체크포인트 11개(SR v3 6, DN v1, Phase-1 x2/x3/x4, EO 소프트닝 넷)는 파일이 바뀌지 않고, 내장 model config가 현재 config와 같고, strict 로드되고, 배포 변환 뒤 출력이 같아야 한다 | 체크포인트 md5, 내장 model == config의 model 블록, `load_state_dict` strict 통과, 배포 state_dict와 고정 입력 출력의 sha256이 기록값과 같음 | 하드 제약 | T |
| MREQ-REL-02 | 회귀 | 출하 export 경로(`prepare_deploy_model` → W16A16 QuantSim → 빈 MMP → export)가 출하 그래프를 그대로 내야 한다 | IR 6개와 DN 1개는 ONNX md5와 param encodings가 출하 파일과 같음. EO 3개는 ONNX md5가 기록된 CPU 기준선과 같음. 10개 모두 activation encodings 이름 목록이 출하 파일과 같음 | 하드 제약 | T |
| MREQ-REL-03 | 회귀 | 출하 구조 4종의 시드 고정 초기값이 바뀌지 않아야 한다(Phase-1 재학습 재현) | `torch.manual_seed(0)` 뒤 `build_model` state_dict sha256이 기록값과 같음 | 하드 제약 | T |
| MREQ-REL-04 | 회귀 | 출하 데이터 config 4개(`configs/data/bicubicpp/`)가 같은 학습 쌍(LR/HR)을 만들어야 한다. 드물게 뽑히는 연산도 덮도록, 켜진 연산을 모두 확률 1로 강제한 경우도 본다 | 합성 영상 1장, 샘플마다 `random`·`np.random` 시드 고정, 24쌍의 sha256이 config × 스케일 × (출하 그대로 / 강제) 20개 모두 기록값과 같음. OpenCV 4.12.0·NumPy 2.2.6이 아니면 skip | 하드 제약 | T |

## 2 설계
| 항목 | 내용 |
|---|---|
| 파일 | `tests/test_release_golden.py`, `tests/test_data_recipe_golden.py` (새 파일) |
| 개발 상태 | 신규. 제품 코드는 바꾸지 않는다 |
| 기준(oracle) | git이 추적하지 않는 `results/release_v3/`, `results/denoise_release_v1/`, `checkpoints/`의 출하 파일. 파일이 없으면 그 케이스는 skip이고, 학습 서버에서는 skip이 0이어야 한다(`pytest -rs`로 확인) |
| 의존성 | 추가 없음. AIMET은 `pytest.importorskip("aimet_torch")`로 가져온다 |

## 3 상세 설계
- 배포 변환은 출하 흐름과 같은 순서로 한다: strict 로드 → `eval()` → `switch_to_deploy()`(2026-10-17부터 IR 무채색 접기 포함) → `6_apply_qat._check_achromatic_fold`(importlib). 소프트닝 넷은 `Trainer._load_denoise_net`처럼 `normalize_state_dict`로 로드하고 achromatic 접기는 하지 않는다.
- 출력 해시는 `torch.set_num_threads(1)`에서 `torch.rand(1, 3, 48, 64)`(seed 0)로 잰다.
- export는 데이터 없이 한다. 보정 배치는 무작위 2개라서 activation encodings 값은 비교하지 않는다(출하 흐름도 보정에 시드가 없다). ONNX와 param encodings는 보정과 무관하다.
- EO는 2026-09-10에 GPU로 export돼서(릴리스 폴더에는 09-14에 복사) CPU export와 W16 scale이 1 ulp 다르다. 그래서 EO의 기준은 이 테스트를 작성한 커밋(`d064868` 기준 코드)의 CPU export md5다.
- AIMET이 작업 디렉터리에 파일을 쓰므로 export 테스트는 `tmp_path`로 chdir한다.
- 의도한 변경으로 기록값을 바꿀 때는 같은 커밋에서 상수를 다시 재고, 이유를 커밋과 이 문서 5절에 적는다.

## 4 단위 시험 케이스
| TC | MREQ | 구분 | 테스트 이름 | 입력 | 기대 결과 |
|---|---|---|---|---|---|
| TC-REL-01-1 | 01 | 정상 | `test_TC_REL_01_1_CheckpointMatchesConfigAndLoadsStrict` (parametrize 11) | 체크포인트 11개 | md5 일치, 내장 model == config, strict 로드 |
| TC-REL-01-2 | 01 | 정상 | `test_TC_REL_01_2_DeployOutputUnchanged` (parametrize 11) | 배포 변환 뒤 고정 입력 | state_dict + 출력 sha256 일치 |
| TC-REL-02-1 | 02 | 정상 | `test_TC_REL_02_1_AimetExportMatchesShipped` (parametrize 10) | 보드 프로파일 10개 | ONNX md5, param encodings, activation 이름 목록 일치 |
| TC-REL-03-1 | 03 | 정상 | `test_TC_REL_03_1_SeededInitUnchanged` (parametrize 4) | x2 auto, x3 deconv, x4 split, DN 절반 4층 | state_dict sha256 일치 |
| TC-REL-04-1 | 04 | 정상 | `test_TC_REL_04_1_DataRecipeUnchanged` (parametrize 20) | 데이터 config 4개 × 스케일 × 2모드 | 24쌍 sha256 일치 |

## 5 단위 시험 결과
| 일자 | 환경 | 명령 | 결과 |
|---|---|---|---|
| 2026-10-15 | Ubuntu 22.04, Python 3.10.12, PyTorch 2.9.0, AIMET 2.29.0, CPU | `CUDA_VISIBLE_DEVICES= NUMPY_MADVISE_HUGEPAGE=0 python3 -m pytest -q -rs tests/test_release_golden.py` | P 36 / F 0 / skip 0, 17.5 s. IR 6개와 DN의 ONNX md5가 출하 파일과 같음(DN `81ac40c7…`, IR x4 `5de6458e…`) |
| 2026-10-15 | 같음 | 메모리에서만 코드를 바꾼 변형 시험(`-p` 플러그인) | 본체 skip의 Add 입력 순서만 바꿈(출력 수치 동일) → TC-REL-02-1 10개 실패. 배포 뒤 `conv1` 가중치 × (1 + 1e-6) → TC-REL-01-2 10개와 TC-REL-02-1 10개 실패. 소프트닝 넷(SVFocusSRNet)은 영향 없음 |
| 2026-10-15 | 같음 | 전체 `pytest -q -rs` | P 210 / F 0, 61.6 s. 실행 뒤 `checkpoints/`·`results/`에 새 파일 없음 |
| 2026-10-15 (v0.2) | 같음, OpenCV 4.12.0, NumPy 2.2.6 | `python3 -m pytest -q -rs tests/test_data_recipe_golden.py` | P 20 / F 0, 12 s. 기록값은 다른 프로세스에서 잰 값과 같음(결정적). 변형 시험: 가우시안 잡음을 버림 대신 반올림하면 그 잡음을 쓰는 14개 실패, bicubic만 쓰는 Phase-1 6개는 통과 |
| 2026-10-16 (v0.3) | 같음 | `python3 -m pytest -q tests/test_data_recipe_golden.py` | 가우시안 노이즈 반올림 수정 뒤 EO·IR Phase-2와 Denoise 14개가 의도대로 바뀌어 기록값을 다시 쟀다. Phase-1 6개는 그대로. 다시 잰 뒤 P 20 / F 0 |
| 2026-10-16 | 같음 | `CUDA_VISIBLE_DEVICES= NUMPY_MADVISE_HUGEPAGE=0 python3 -m pytest -q -rs tests/test_release_golden.py tests/test_data_recipe_golden.py tests/test_make_val_lr.py` | 실행 계획 7단계(구세대 모델 8종·KD·config 42개 삭제) 뒤 P 59 / F 0 / skip 0. 출하 체크포인트, 소프트닝 넷, export, 데이터 레시피 모두 기록값과 같음. 전체 `pytest -q -rs` P 246 / F 0 |

## 6 미해결 · 결정
- 기록값은 이 서버의 PyTorch 2.9.0 / AIMET 2.29.0 기준이다. 버전을 올리면 기록값이 바뀔 수 있으므로 requirements 버전은 고정해 둔다.
- 게이트 기준값(P2 grid, P1.1 chart)은 게이트 모듈 문서에 있다(MREQ-GRID-04, MREQ-CHART-06).
- 학습 분포를 바꾸는 수정(실행 계획 4단계)은 MREQ-REL-04의 기록값을 같은 커밋에서 다시 잰다. 수정 전 커밋이 출하 레시피의 재현 기준이 된다.
