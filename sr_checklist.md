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

- [ ] 실제 Raw/Image 입력에 대해 bicubic 결과 생성
- [ ] 실제 Raw/Image 입력에 대해 deploy x4 GPU 모델 추론
- [ ] `Input / Bicubic / Deploy GPU` 비교 이미지 생성
- [ ] no-reference 지표 계산
  - NIQE
  - BRISQUE 또는 PIQE 가능 시
  - sharpness
  - edge density
  - high-frequency energy
  - ringing score
  - local contrast
  - color/tone shift
- [ ] 시각적 failure 분류
  - noise amplification
  - sensor artifact amplification
  - false texture
  - edge oversharpening
  - edge smoothing
  - color/tone instability

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

- [ ] 회사에서 x4 deploy PyTorch checkpoint 실제 경로 확인
- [ ] Synthetic benchmark용 HR 샘플 폴더 선정
- [ ] Real sensor 입력 샘플 폴더 선정
- [ ] x4 GPU inference 결과 생성
- [ ] 비교 이미지와 지표를 repo `results/sr_x4_baseline_analysis/`에 저장
- [ ] 분석 결과를 바탕으로 MTKD 설계 여부 결정

---

## Quick Priority

| 우선순위 | 작업 | 상태 |
|---:|---|---|
| 1 | x4 deploy GPU baseline checkpoint 확보 | 진행 예정 |
| 2 | Synthetic GT benchmark 구성 | 대기 |
| 3 | Real sensor no-reference validation 구성 | 대기 |
| 4 | x4 baseline failure mode 정리 | 대기 |
| 5 | Teacher/MTKD loss 설계 확정 | 대기 |
