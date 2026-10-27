# BicubicPPDenoise 모듈 문서

| 문서번호 | 버전 | 작성일 | 작성자 | 시스템 | 상위 문서 |
|---|---|---|---|---|---|
| USRD-MOD-BPPD-001 | 0.4 | 2026-10-13 | JSHONG | unified-sr-denoise-platform | USRD-RESEARCH-003 §4 (HYP-07) |

개정 이력
| 버전 | 일자 | 변경 내용 | 작성자 |
|---|---|---|---|
| 0.1 | 2026-10-12 | 최초 작성. Bicubic++에 scale 1(EO Denoise) 경로 추가(소유자 승인 2026-10-12: W16A16에서 30 ms 예산 안에 드는 Denoise 구조 3종 학습 — 절반 해상도 본체 4층·2층, 원 해상도 ReLU) | JSHONG |
| 0.2 | 2026-10-13 | 절반 4층(EXP-09-1, ep287)을 BicubicPPDenoise[v1]로 배포 결정(소유자 2026-10-13). 학습 설정을 `configs/train/Denoise/`로 옮기고 실험 설정 삭제. 릴리스 `results/denoise_release_v1/` | JSHONG |
| 0.3 | 2026-10-16 | 조용히 다른 모델이 되던 조합을 거부(MREQ-BPPD-03 확장): SR `legacy`+`use_rep`(ECB가 꺼졌음), ECB+`bias`(경계에서 접기가 틀림), 정수가 아닌 배율, `build_model`의 모르는 키(TypeError). TC-BPPD-03-1 parametrize 8, TC-BPPD-03-3 추가 | JSHONG |
| 0.4 | 2026-10-17 | IR 무채색 접기를 `6_apply_qat.py`에서 `switch_to_deploy`로 옮김(OD-8, MREQ-BPPD-02 확장, TC-BPPD-02-3). 출하 ONNX md5·가중치 encodings 그대로, tools의 IR FP32 출력은 반올림 경계에서만 1 DN 차. `clamp_output` 기본값 True(OD-14, TC-BPPD-02-4): 체크포인트 131개·config 36개가 모두 True를 명시해 바뀌는 모델 없음 | JSHONG |

이 저장소에는 SRS가 없다. 상위 요구는 연구 계획서 `docs/20261008-RESEARCH-EODenoiseV3.md`의 HYP-07(W16A16 Denoise 구조)이다. 근거: 2026-10-12 보드 98 실측(burst)에서 같은 SVFocusDenoiseNet 그래프(v3 C)가 W8A8 7.7 ms, W16A16 31.0 ms(4.0배)였다(v2 W8A8 7.1 ms 대비 4.4배). W16 구조 벤치 7종 중 v2 W8 지연(7.1 ms)보다 빠른 것은 절반 해상도 Bicubic++ 본체 2층(5.1 ms)뿐이고, 4층(7.9 ms)은 1.1배, 원 해상도 ReLU(11.9 ms)는 1.7배다. 모듈 약어 `BPPD`.

## 1 요구
| ID | 유형 | 요구사항 | 수용 기준 | 상위 | 검증 |
|---|---|---|---|---|---|
| MREQ-BPPD-01 | 기능 | `BicubicPlusPlus(scaling_factor=1)`은 입력과 같은 크기의 영상을 내야 한다. `legacy`는 stride-2 stem → 절반 해상도 본체 → PixelShuffle(2), `auto`는 원 해상도 본체에 shuffle 없음. 출력 = clamp(입력 + 잔차), 잔차는 0에 가깝게 시작한다. `n_body`(stem 뒤 3x3 conv 수, 2 이상)로 깊이를 고른다 | 출력 shape = 입력, 초기 출력과 입력의 평균 차 < 0.02. `n_body` 2의 키는 SR 모델과 같은 `conv0/1/2, conv_out`, 늘린 층은 `body_extra.*`. SR(scale ≥ 2) 모델의 키는 바뀌지 않는다 | HYP-07 | T |
| MREQ-BPPD-02 | 기능 | ECB 학습 가지는 stride-1 conv 모두에 쓰고, `switch_to_deploy` 후 3x3 하나로 정확히 접혀야 한다. 내보낸 그래프는 HTP 변환 템플릿(`final_act` 출력)과 맞아야 한다. IR(`achromatic_output`) 모델은 `switch_to_deploy`가 채널 평균을 마지막 층 가중치로 접어야 한다 | 접기 전후 출력 차 ≤ 1e-5, ECB 가지 키가 남지 않음. IR: 접은 뒤 `achromatic_output` False, 세 출력 그룹 가중치 같음, ONNX에 ReduceMean·Expand 없음, 두 번 불러도 같음. ONNX: `legacy` DepthToSpace 1개, `auto` 0개(DepthToSpace(1) 아님), 마지막 노드 Clip, Add 2개(본체 skip + 입력 skip). AIMET QuantSim W16A16 1280x720 생성·보정 통과 | HYP-07 | T |
| MREQ-BPPD-03 | 차단 | 지원하지 않는 조합은 만들 때 거부하고, 절반 해상도의 홀수 크기 입력은 조용히 넘어가지 않아야 한다. 조용히 다른 모델이 되는 조합(SR `legacy`+`use_rep`, ECB+`bias`, 정수가 아닌 배율)과 `build_model`의 모르는 키도 거부한다 | `deconv`+scale 1, scale 0, `achromatic_output`+scale 1, SR의 `n_body` ≠ 2, `n_body` < 2 → `ValueError`. 홀수 크기 → 입력 덧셈에서 `RuntimeError`(forward에 크기 검사를 두지 않아 torch.fx/AIMET 추적에 데이터 의존 분기가 없음). SR `legacy`+`use_rep`, `use_rep`+`bias`, 배율 1.5 → `ValueError`. `build_model`에 모르는 키(예: `use_reps`) → `TypeError` | HYP-07 | T |

## 2 설계
| 항목 | 내용 |
|---|---|
| 파일 | `src/models/bicubicplusplus.py`(scale 1 분기, `n_body`, 입력 skip, ECB 허용 범위), `src/models/__init__.py`(`n_body` 전달) |
| 개발 상태 | 기존 모듈 수정. v0.1~0.3: SR 경로(scale ≥ 2)의 그래프·키·기본값은 그대로. v0.4: `clamp_output` 기본값 True(모든 config·체크포인트가 이미 True 명시), IR SR의 무채색 접기가 `switch_to_deploy`로 이동(출하 ONNX md5 그대로) |
| 학습 설정 | `configs/train/Denoise/bicubicppdenoise.yaml`(배포 v1, 절반 4층 = EXP-09-1). 절반 2층(EXP-10-1)·원 해상도 ReLU(EXP-11-1) 설정은 배포 결정(2026-10-13) 뒤 지웠다. 셋 다 EXP-08-1(C 처음부터)과 모델 외 전부 같았다 |

## 3 상세 설계
- 생성자: `self.denoise = scaling_factor == 1`. `legacy`는 factor 2(stem stride 2 × scale 1), `auto`는 factor 1 → `nn.Identity()`로 shuffle 생략.
- ECB: SR `legacy`에는 ECB 형태가 없어 `use_rep`을 주면 거부한다(2026-10-16 전에는 조용히 껐다). Denoise는 새 경로라 stride-1 conv(conv1, conv2, body_extra, conv_out, `auto`의 conv0)에 ECB를 쓰고, `conv3()`가 stride-2 stem은 평범한 conv로 둔다.
- 잔차 시작값: `conv_out`의 파라미터에 0.01을 곱한다(SVFocusDenoiseNet의 tail ×0.01과 같은 방식).
- forward: 본체 `act(conv1) → act(conv2) → act(body_extra…) + x0` → `conv_out` → (shuffle) → `x + y` → `final_act`.
- `switch_to_deploy`: 직계 ECB와 `body_extra` 안의 ECB를 접고, 이어서 `achromatic_output`이면 `_fold_achromatic`이 채널 평균(PixelShuffle 경로는 출력 3그룹 평균, deconv는 dim 1 평균)을 마지막 층에 접고 `achromatic_output`을 False로 둔다. 멱등.

## 4 단위 시험 케이스
| TC | MREQ | 구분 | 테스트 이름 | 입력 | 기대 결과 |
|---|---|---|---|---|---|
| TC-BPPD-01-1 | 01 | 정상 | `test_TC_BPPD_01_1_ShapeAndNearIdentityAtInit` (parametrize 4: legacy/auto × n_body 2/4) | 2×3×48×64 | shape 같음, 평균 차 < 0.02 |
| TC-BPPD-01-2 | 01 | 정상 | `test_TC_BPPD_01_2_BodyDepthAndSRKeysUnchanged` | n_body 2/4, SR auto x2 | 키 집합 일치 |
| TC-BPPD-02-1 | 02 | 정상 | `test_TC_BPPD_02_1_DeployFoldIsExact` (parametrize 3) | 가지 가중치를 흔든 모델 | 접기 전후 ≤ 1e-5, ECB 키 없음 |
| TC-BPPD-02-2 | 02 | 정상 | `test_TC_BPPD_02_2_OnnxGraph` (parametrize 2) | 접은 모델 ONNX | DepthToSpace 1 / 0, 마지막 Clip, Add 2 |
| TC-BPPD-02-3 | 02 | 정상 | `test_TC_BPPD_02_3_AchromaticFoldInSwitchToDeploy` (parametrize 6: x2 auto, x3 deconv, x4 split × ECB 유무) | 가중치를 흔든 IR 모델 | `achromatic_output` False, 그룹 가중치 같음, 출력 차 < 1e-5, 두 번째 호출 동일, ONNX에 ReduceMean·Expand 없음 |
| TC-BPPD-02-4 | 02 | 정상 | `test_TC_BPPD_02_4_ClampIsTheDefault` | `clamp_output`을 안 준 config / `False` | `final_act`(Hardtanh) 있음 / 없음 |
| TC-BPPD-03-1 | 03 | 오류 | `test_TC_BPPD_03_1_InvalidConfigRejected` (parametrize 8) | 지원 안 하는 조합(SR `legacy`+`use_rep`, ECB+`bias`, 배율 1.5 포함) | `ValueError` |
| TC-BPPD-03-3 | 03 | 오류 | `test_TC_BPPD_03_3_UnknownModelKeyRejected` | `build_model`에 `use_reps` | `TypeError` |
| TC-BPPD-03-2 | 03 | 오류 | `test_TC_BPPD_03_2_HalfResFailsLoudlyOnOddSize` | 47×64 입력 | `RuntimeError` |

## 5 단위 시험 결과
| 일자 | 환경 | 명령 | 결과 |
|---|---|---|---|
| 2026-10-12 | Ubuntu 22.04, Python 3.10.12, PyTorch 2.9.0, AIMET 2.29.0 | `CUDA_VISIBLE_DEVICES= PYTHONPATH=. python3 -m pytest tests/test_bicubicpp_denoise.py tests/test_build_model_contract.py tests/test_reparam_equivalence.py -q` | 구현 전 TC-BPPD-01-1 실패(scale 1 거부), 구현 후 P 65 / F 0. AIMET QuantSim W16A16 1280x720: legacy 4층·2층, auto ReLU 모두 생성·보정 통과 |
| 2026-10-16 (v0.3) | 같음, CPU | `python3 -m pytest -q tests/test_bicubicpp_denoise.py tests/test_build_model_contract.py tests/test_reparam_equivalence.py tests/test_release_golden.py` | 추가한 4개는 수정 전 실패(재현), 수정 후 P 105 / F 0. 디스크의 Bicubic++ config 36개와 체크포인트 131개가 모두 빌드·strict 로드됨 |
| 2026-10-17 (v0.4) | 같음, CPU | `python3 -m pytest -q tests/test_bicubicpp_denoise.py tests/test_release_golden.py` | TC-BPPD-02-3 6개는 접기를 옮기기 전 실패, 옮긴 뒤 통과. TC-BPPD-02-4는 기본값을 바꾸기 전 실패, 바꾼 뒤 통과. golden 전부 통과. 실제 `6_apply_qat.py` export(IR x3·EO x4·DN)의 ONNX md5·가중치 encodings 그대로 |

## 6 미해결 · 결정
- 절반 해상도의 화질 위험(1화소 디테일, 화면 이동 시 위상 의존, PixelShuffle 격자, 윤곽 제거 약화)은 학습 후 P1.1 + 보조 지표 + GridScore로 판정한다(RESEARCH HYP-07).
- 학습 손실은 EXP-08-1과 같게 두었다(edge 0.05 gray 크기형). 격자가 보이면 다음 축으로 signed/none을 시험한다.
- **결정 (2026-10-13)**: 절반 4층을 BicubicPPDenoise[v1]로 배포한다. 보드 98 W16A16 실측과 근거는 `results/denoise_release_v1/MANIFEST.md`.
