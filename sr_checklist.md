# SR MTKD 준비 체크리스트

## 개요

| 항목 | 내용 |
|---|---|
| 작성일 | 2026-06-02 |
| 목적 | Super Resolution MTKD 적용 전, 현재 x4 baseline/deploy GPU 모델의 문제 양상 정의 |
| 핵심 판단 | Teacher 모델 학습은 baseline failure mode와 real sensor/domain mismatch를 분리한 뒤 진행 |

---

## 1. 현재 확인된 기준

- [x] GitHub repo 최신 상태 기준으로 확인
- [x] 실제 checkpoint config 경로 확인: `checkpoints/csuav_deploy`
- [x] `checkpoints/csuav_deploy`의 SR train/finetune `data_config.yaml`과 현재 `configs/data` 설정 비교
- [x] 파일 바이트 기준으로는 동일하지 않음
- [x] 실제 degradation 동작 기준으로는 동일함

### Degradation Baseline 판정

| 대상 | 현재 config | 판정 |
|---|---|---|
| SR train x2/x4 | `configs/data/sr_train.yaml` | 동작 동일 |
| SR finetune EO x2/x4 | `configs/data/sr_finetune_eo.yaml` | 동작 동일 |
| SR finetune IR x2/x4 | `configs/data/sr_finetune_ir.yaml` | 동작 동일 |

### 주의 사항

| 항목 | 내용 |
|---|---|
| `data_config.yaml` | 현재 `configs/data` 설정을 degradation baseline으로 사용 가능 |
| `train_config.yaml` | 현재 `configs/train`, `configs/finetune` 설정과 전체 동일하지 않음 |
| 기존 `csuav_deploy` checkpoint | `n_resblocks: 8` 계열 |
| 현재 block4 config | `configs/train/SVFocusSRNet/*dim32_block4*.yaml`은 기존 deploy 재현 기준과 분리 필요 |

---

## 2. MTKD Teacher 후보

- [ ] Mamba 계열
- [ ] EvaSR
- [ ] MaIR

### 현재 판단

- [x] Teacher 학습/MTKD 설계 전에 현재 deploy 계열 x4 GPU baseline의 실패 양상 분석이 우선
- [x] x2보다 x4가 정보 손실이 크므로 baseline 문제 분석은 x4부터 진행
- [x] 먼저 PyTorch GPU FP32 결과를 확인해야 모델 한계와 ONNX/NPU/quantization 이슈를 분리할 수 있음

---

## 3. LR/HR 기준 정리

### 실제 임무장비 입력

- 실제 드론 임무장비 Raw/Image 입력에는 대응되는 HR/GT가 없음
- 따라서 real input만으로 PSNR/SSIM 기반 정답 비교는 불가능
- real input 분석은 no-reference 지표와 시각적 failure 분류 중심으로 진행

### 필요한 검증 트랙

| 트랙 | 데이터 | 목적 |
|---|---|---|
| Synthetic GT benchmark | HR 원본에서 degradation pipeline으로 LRx4 생성 | 모델 자체 성능과 failure mode 정량 분석 |
| Real sensor validation | 실제 임무장비 Raw/Image 입력 | 실사용 분포에서 domain mismatch와 artifact 증폭 확인 |

---

## 4. 회사에서 준비할 데이터/파일

### 필수 준비 항목

- [ ] 현재 deploy 기준 x4 PyTorch checkpoint 경로 확인
  - EO/IR 구분
  - x4 구분
  - `.pth` weight 실제 위치
- [ ] checkpoint 생성 당시 config 확보
  - `train_config.yaml`
  - `data_config.yaml`
- [ ] Synthetic benchmark용 HR 이미지 폴더 선정
  - 최소 20장
  - 가능하면 50장 이상
  - 건물, 간판, 글자, 도로선, 전선, 격자, 나뭇잎, 평탄 영역 포함
- [ ] 실제 임무장비 Raw/Image 입력 샘플 폴더 선정
  - HR 없음
  - 주/야간, 고주파 장면, 평탄 장면, 원거리 구조물 포함

### 권장 결과 폴더 구조

```text
results/sr_x4_baseline_analysis/
  synthetic/
    HR/
    LRx4/
    bicubic/
    pred_deploy_gpu/
    metrics/
    comparisons/
  real_sensor/
    input/
    bicubic/
    pred_deploy_gpu/
    metrics_no_ref/
    comparisons/
  configs/
    train_config.yaml
    data_config.yaml
  notes/
    failure_summary.md
```

---

## 5. Synthetic GT Benchmark 작업

- [ ] 현재 baseline degradation config로 HR -> LRx4 생성
- [ ] 동일 LRx4 입력에 대해 bicubic 결과 생성
- [ ] 동일 LRx4 입력에 대해 deploy x4 GPU 모델 추론
- [ ] `LRx4 / Bicubic / Deploy GPU / HR` 비교 이미지 생성
- [ ] 정량 지표 계산
  - PSNR
  - SSIM
  - LPIPS
  - Edge/gradient similarity
  - High-frequency energy
  - Error map
- [ ] 실패 유형 분류
  - edge collapse
  - texture loss
  - oversmoothing
  - ringing
  - aliasing
  - color/tone shift
  - degradation mismatch 의심

---

## 6. Real Sensor Validation 작업

- [x] 실제 Raw/Image 입력에 대해 center crop `320x180` 생성
- [x] 실제 Raw/Image 입력에 대해 bicubic x4 결과 생성
- [x] 실제 Raw/Image 입력에 대해 deploy x4 GPU 모델 추론
- [x] `Input / Bicubic / Deploy GPU / Diff heatmap` 비교 이미지 생성
- [x] no-reference 지표 계산
  - 계산 완료: NIQE, BRISQUE, PIQE (`pyiqa`), sharpness, edge density, high-frequency energy, ringing score, local contrast, color/tone shift proxy
- [x] 시각적 failure 분류
  - noise amplification
  - sensor artifact amplification
  - false texture
  - edge oversharpening
  - edge smoothing
  - color/tone instability

### 2026-06-05 Real Sensor x4 검증 결과

| 항목 | 내용 |
|---|---|
| 입력 | `results/260602_mc_g105_probe_42/raw` |
| 샘플 수 | 42장 |
| 처리 기준 | 원본 `1920x1080` 중앙 `320x180` crop 후 x4 SR `1280x720` 생성 |
| Deploy x4 config | `checkpoints/csuav_deploy/finetune_svfocussrnet_eo_sr_x4_dim32_epoch100_bs_16_ga_2_lr_5e-5/train_config.yaml` |
| Deploy x4 checkpoint | `checkpoints/csuav_deploy/finetune_svfocussrnet_eo_sr_x4_dim32_epoch100_bs_16_ga_2_lr_5e-5/best.pth` |
| 결과 폴더 | `results/sr_x4_baseline_analysis/real_sensor` |
| 보고서 | `results/sr_x4_baseline_analysis/real_sensor/report.html`, `report.md` |
| 비교 이미지 | `results/sr_x4_baseline_analysis/real_sensor/comparisons/overview_top_risk.jpg` |
| 지표 표 | `metrics/per_image_metrics.csv`, `metrics/summary_by_scene.csv`, `metrics/summary_by_zoom.csv` |

요약:

- Deploy GPU x4는 Bicubic 대비 평균 sharpness ratio `2.33`, edge density ratio `1.95`로 edge/detail을 적극적으로 살리는 경향이 있음.
- 평균 low-frequency luma MAE `0.45`, chroma MAE `0.68`로 큰 tone/color drift 신호는 낮음.
- No-reference IQA는 낮을수록 좋은 방향이며, Deploy가 Bicubic 대비 평균 NIQE `8.86 -> 8.18`, BRISQUE `67.22 -> 48.72`, PIQE `79.46 -> 60.93`으로 개선됨.
- failure label count: `edge_oversharpening_or_ringing` 35장, `noise_or_false_texture_amplification` 11장, `large_local_deviation_from_bicubic` 10장, `no_strong_failure_signal` 7장.
- NIQE/BRISQUE/PIQE는 `sr` container의 `pyiqa` 구현으로 계산했으며, HR/GT 없는 no-reference 지표이므로 top-risk frame 정성 확인과 함께 해석해야 함.

---

## 7. 판단 기준

### Synthetic에서도 성능이 나쁠 때

| 판단 | 후속 방향 |
|---|---|
| 모델 capacity 문제 가능성 높음 | Teacher/MTKD 적용 우선순위 높음 |
| loss 설계 문제 가능성 있음 | loss 구성과 weight 재검토 |

후보 개선 항목:

- output KD
- feature KD
- frequency/edge KD
- perceptual/texture 관련 loss 재조정

### Synthetic은 괜찮고 Real에서 나쁠 때

| 판단 | 후속 방향 |
|---|---|
| degradation pipeline과 실제 센서 입력 사이 domain mismatch 가능성 높음 | Teacher 학습보다 degradation pipeline 보정 우선 |

후보 개선 항목:

- real sensor blur/noise/sharpening 특성 반영
- compression/ISP artifact 반영
- scene/zoom 조건별 degradation profile 분리

### GPU FP32는 괜찮고 Deploy/NPU에서 나쁠 때

| 판단 | 후속 방향 |
|---|---|
| ONNX 변환, quantization, NPU runtime 이슈 가능성 높음 | MTKD보다 export/quantization 검증 우선 |

---

## 8. 다음 액션

- [x] 회사에서 x4 deploy PyTorch checkpoint 실제 경로 확인
- [ ] Synthetic benchmark용 HR 샘플 폴더 선정
- [x] Real sensor 입력 샘플 폴더 선정: `results/260602_mc_g105_probe_42/raw`
- [x] x4 GPU inference 결과 생성
- [x] 비교 이미지와 지표를 repo `results/sr_x4_baseline_analysis/`에 저장
- [x] 분석 결과를 바탕으로 SR finetune 방향 결정
  - [x] Deploy SR은 no-reference IQA는 개선하지만 edge boost/ringing/false texture risk가 큼
  - [x] Denoise 후단 SR을 가정하고 optical blur, aliasing/resampling, mild compression/ISP artifact, denoise residual artifact 중심의 2-stage degradation으로 정리
  - [x] 학습 방식은 Deploy checkpoint에서 바로 1-phase finetune으로 진행

---

## 9. 2026-06-05 SR Finetune 실행

- [x] x4 2-stage degradation config 추가: `configs/data/sr_finetune_eo_denoised_input_x4_2stage.yaml`
- [x] x2 2-stage degradation config 추가: `configs/data/sr_finetune_eo_denoised_input_x2_2stage.yaml`
- [x] x4 1-phase finetune config 추가: `configs/finetune/SVFocusSRNet/svfocussrnet_4x_eo_denoised_input_1phase.yaml`
- [x] x2 1-phase finetune config 추가: `configs/finetune/SVFocusSRNet/svfocussrnet_2x_eo_denoised_input_1phase.yaml`
- [x] YAML numeric parsing 이슈 수정
  - [x] `src/losses/losses.py`: `eps`, threshold, kernel size, loss weight를 numeric type으로 캐스팅
  - [x] `eps: 1e-3` 문자열 파싱으로 인한 `CharbonnierLoss` TypeError 해결
- [/] x2 SR finetune 진행 중
  - [x] epoch 1 validation 완료: loss `0.086243`, PSNR `25.7627`, SSIM `0.7144`, LPIPS `0.3871`, NIQE `5.6560`
  - [/] 2026-06-05 18:12 KST 기준 epoch 2 약 `85/653`, GPU 4 약 `15.5GB`
- [/] x4 SR finetune 진행 중
  - [/] 2026-06-05 18:12 KST 기준 epoch 1 약 `462/1306`, GPU 1 약 `26.9GB`
- [ ] 월요일 오전 x2/x4 finetune validation 추세 확인
- [ ] fixed 42 real sensor probe로 Deploy vs finetuned SR 재비교

## Quick Priority

| 우선순위 | 작업 | 상태 |
|---:|---|---|
| 1 | x4 deploy GPU baseline checkpoint 확보 | 완료 |
| 2 | Synthetic GT benchmark 구성 | 대기 |
| 3 | Real sensor no-reference validation 구성 | 완료 |
| 4 | x4 baseline failure mode 정리 | 완료 |
| 5 | x2/x4 SR finetune 진행 및 Monday validation 확인 | 진행 중 |
