# EO Denoise 재설계 및 Scratch Multi-Teacher KD 검토 메모

작성일: 2026-05-11

목표: EO 센서 파라메터 최적화 이후 확보한 이미지와 현재 GitLab repo의 denoise 파이프라인을 기준으로, Denoise 모델을 1-stage/2-stage 학습 curriculum으로 재설계할지 검토한다. NAFNet, Restormer, Deploy-family 모델은 pretrained output을 그대로 쓰는 것이 아니라, 동일한 Denoise Degradation Pipeline으로 scratch 학습한 Teacher 후보로 두고 Student 배포 모델에 지식을 주입하는 방안을 평가한다.

검토 관점:
- 센서/열화 파이프라인
- Multi-Teacher Knowledge Distillation
- 배포 Student 및 QNN/AIMET 제약
- 실험 설계 및 평가 프로토콜

---

## 1. 결론 요약

제안 방향은 충분히 시도할 가치가 있다. 다만 바로 Multi-Teacher KD로 들어가면 실험 비용이 커지고, 현재 degradation profile 또는 validation set이 잘못된 상태에서는 여러 Teacher가 같은 오류를 더 강하게 Student에 주입할 위험이 있다.

현재 기준 추천 접근은 다음과 같다.

1. 센서 분석 결과를 먼저 machine-usable sensor profile로 고정한다.
2. Teacher 모델을 scratch부터 같은 data config로 학습한다.
3. 1-stage Teacher와 2-stage Teacher를 먼저 비교한다.
4. 2-stage가 실제 EO strata에서 이길 때만 KD 단계로 간다.
5. Student는 supervised pretrain 후 cached MTKD finetune을 수행한다.
6. 배포 전에는 fused graph 기준 ONNX/AIMET/QNN 검증을 반드시 통과해야 한다.

중요한 판단:
- 2-stage는 inference 구조가 아니라 training curriculum으로 유지한다.
- NAFNet/Restormer는 offline quality Teacher로만 사용한다.
- 현재 Deploy-family 모델은 quality oracle이 아니라 deployment-prior/legacy-stability Teacher로 낮은 weight에서만 검토한다.
- KD loss는 GT supervision을 대체하지 않는다. `L_gt`는 반드시 anchor로 유지한다.
- Teacher output 단순 평균은 위험하다. disagreement가 큰 영역에서는 KD weight를 낮춰야 한다.

---

## 2. 용어 정리

현재 repo에는 두 종류의 "stage" 개념이 섞일 수 있으므로 문서와 config에서 명확히 분리한다.

### Degradation Pipeline Stage

`src/data/datasets.py`의 `apply_configured_degradation()`은 `degradation.stage1`과 `degradation.stage2`를 한 샘플 안에서 순차 적용한다.

- `degradation.stage1`: unprocess noise, blur, resize, gaussian/poisson noise, jpeg, sinc 등 일반/물리 기반 열화
- `degradation.stage2`: target resize, detail attenuation, signal instability, color cast, common/chroma noise, hot pixels, final jpeg 등 센서/조건 특화 열화

이것은 학습 curriculum의 1-stage/2-stage와 다르다.

### 1-Stage Training

Teacher 또는 Student를 scratch부터 한 번에 학습한다.

- common noise
- sensor-specific degradation
- day/night, zoom conditional profile
- hot pixel, chroma, tint, detail attenuation 등

위 요소를 하나의 training run에서 모두 섞어 학습한다.

장점:
- 실험 단순
- end-to-end로 최종 분포에 바로 적응
- checkpoint/운영 관리가 쉬움

리스크:
- 모델이 common noise와 sensor artifact를 분리해서 배우지 못할 수 있음
- 강한 센서 특화 열화가 early training을 불안정하게 만들 수 있음
- small Student는 모든 변동성을 한 번에 흡수하다가 oversmoothing으로 갈 수 있음

### 2-Stage Training

학습을 두 phase로 나눈다.

- Phase 1: common/generic denoise degradation으로 scratch pretraining
- Phase 2: MC-G105 등 센서 특화 degradation으로 finetuning

장점:
- 먼저 안정적인 denoise prior를 만든 뒤 센서 특화 artifact에 적응 가능
- sensor profile이 바뀌어도 Phase 2만 재실험할 수 있음
- Teacher 후보의 학습 안정성이 좋아질 가능성

리스크:
- Phase 1 분포가 너무 generic하면 Phase 2에서 catastrophic forgetting 또는 domain gap이 생김
- Phase 2가 paired validation PSNR을 낮출 수 있음
- finetune 결과가 real EO에서는 좋아도 기존 validation에서는 나빠질 수 있으므로 평가 set 설계가 중요

### KD Stage

Teacher 학습 이후 Student를 학습하는 별도 단계다.

- Teacher 학습: scratch 기반 1-stage/2-stage 비교
- Student 학습: supervised pretrain + cached teacher output 기반 MTKD finetune

KD stage는 Teacher training stage와 분리해서 관리한다.

---

## 3. 현재 Repo 상태에서 확인된 사실

### 이미 구현된 것

- `DenoiseDataset`은 SRDataset 상속에서 분리되어 same-resolution clean/noisy pair를 만든다.
- `apply_configured_degradation()`은 data config 기반으로 공통 열화와 센서 특화 열화를 함께 적용한다.
- `configs/data/denoise_generic_baseline.yaml`은 common/generic denoise profile에 가깝다.
- `configs/data/denoise_mc_g105_v1.yaml`, `denoise_mc_g105_v2.yaml`은 MC-G105 day/night, zoom conditional profile을 포함한다.
- `examples/analysis/analyze_mc_g105_sensor_capture.py`는 temporal noise, hot-pixel 후보, tint bias, dark-region noise, edge density, high-frequency energy를 계산한다.
- `UnifiedLoss`는 Charbonnier, edge, SSIM, color consistency, hot-pixel masked loss 등을 지원한다.
- `SVFocusSRNet(scale=1)`은 input skip + residual 구조로 denoise Student에 적합한 형태다.
- `tools/export.py`는 `switch_to_deploy()`를 호출하여 RepBlock을 export 전에 fuse한다.

### 아직 없는 것

- NAFNet/Restormer 모델 registry 및 training config
- Teacher training 전용 config set
- deterministic degradation sample/cache 생성 도구
- Teacher output cache dataset
- KD loss: output KD, residual KD, feature/frequency KD, teacher disagreement weighting
- KD trainer branch
- Teacher checkpoint/data_config hash 기반 cache manifest
- real EO strata 기반 평가 프로토콜
- fused graph 기준 AIMET/QNN 검증 흐름

### 주의할 점

현재 `tools/train.py`는 denoise task에서 기본 data config를 `configs/data/denoise.yaml`로 잡는다. 따라서 Teacher와 Student가 "동일한 degradation design"으로 학습하려면 반드시 `--data_config configs/data/denoise_generic_baseline.yaml` 또는 `--data_config configs/data/denoise_mc_g105_v2.yaml`처럼 명시해야 한다.

또한 현재 MC-G105 finetune config는 `pretrained_path`를 사용하므로, user가 말한 scratch Teacher recipe와는 다르다. 새 Teacher config는 `pretrained_path: null`을 명확히 둬야 한다.

---

## 4. 센서 분석 접근

현재 분석 스크립트는 triage 용도로 좋다. 다만 production sensor profile을 만들기에는 아직 부족하다.

### 이미 보는 항목

- day/night, zoom label
- RGB/YCrCb 평균 및 표준편차
- tint bias: R-G, B-G
- edge density
- high-frequency energy
- dark-region ratio/noise std
- hot-pixel 후보 수
- burst 기반 temporal noise

### 추가로 확보해야 할 capture

- dark frame: lens cap 또는 완전 암부 조건
- flat field frame: 균일 조명/균일 패턴
- static burst: 고정 장면 5장 이상, 가능하면 16장 이상
- exposure/gain/zoom/focus metadata
- day/night x low/mid/high zoom strata
- flat/edge/texture/small-target ROI

### 현재 분석의 한계

- burst grouping이 filename 마지막 digit 기반 heuristic이다.
- temporal noise는 alignment/motion masking 없이 grayscale stack std로 계산된다.
- hot pixel은 per-image local residual 기반이라 작은 밝은 표적을 hot pixel로 오탐할 수 있다.
- row/column FPN, PRNU/DSNU, gain-dependent variance curve는 아직 직접 산출하지 않는다.

### 산출물 목표

최종적으로는 문서가 아니라 training config가 읽을 수 있는 profile이 필요하다.

후보:
- `configs/noise/mc_g105_sensor_profile.yaml`
- hot pixel candidate map: `.npy` 또는 `.png`
- zoom/time별 degradation probability/range
- tint/color cast range
- temporal/common noise range
- detail attenuation range

---

## 5. 1-Stage vs 2-Stage 학습 판단

객관적으로는 2-stage가 더 그럴듯하지만, 항상 이긴다고 보면 안 된다. 특히 current paired validation에서 sensor-specific finetune이 generic baseline보다 낮게 나올 수 있으므로 real EO 평가 strata가 필요하다.

### 비교해야 할 실험

| ID | 목적 | Init | Data/degradation | 판단 |
| --- | --- | --- | --- | --- |
| B1 | generic denoise baseline | scratch | generic/common profile | common denoise 기준선 |
| C1 | 1-stage sensor-profiled | scratch | common + MC-G105 sensor profile 한 번에 학습 | sensor profile 단독 효과 |
| C2 | 2-stage no KD | scratch -> finetune | Phase 1 common, Phase 2 MC-G105 profile | curriculum 효과 |

### 2-stage 채택 조건

2-stage는 다음 조건을 만족할 때 채택한다.

- C2가 C1보다 real EO strata에서 우수
- flat-region noise 감소
- chroma/tint error 감소
- hot-pixel 후보 감소
- edge/small-target 보존이 악화되지 않음
- paired validation PSNR/SSIM이 크게 무너지지 않음

### 중단 조건

다음 중 하나면 KD로 넘어가지 않고 sensor profile/degradation부터 다시 손본다.

- C1/C2 모두 current deploy baseline보다 real EO 육안 품질이 나쁨
- C2가 C1보다 명확히 좋지 않음
- sensor-specific profile이 small target을 hot pixel처럼 지우는 방향으로 학습됨
- chroma/tint 보정은 좋아졌지만 detail attenuation이 과도함

---

## 6. Scratch Teacher Training 설계

Teacher는 pretrained generic restoration 모델의 output을 그대로 가져오는 것이 아니라, 동일한 denoise degradation design으로 scratch부터 학습한다.

### Teacher 후보

| Teacher | 사용 목적 | 객관적 리스크 |
| --- | --- | --- |
| NAFNet | CNN 계열 고성능/효율 복원 teacher, local detail 보존 | repo에 구현 없음, synthetic bias를 같이 배울 수 있음 |
| Restormer | global context, structured artifact, chroma 안정성 teacher | 무겁고 over-smoothing/chroma over-correction 가능 |
| Deploy-family Teacher | 현재 배포 안정성, quantization/deploy prior | 기존 artifact를 Student에 복제할 수 있음 |

NAFNet/Restormer는 runtime 후보가 아니라 offline teacher다. Student가 흡수하기 어려운 고주파/전역 보정을 강하게 강요하면 오히려 small Student가 불안정해질 수 있다.

### Teacher 학습 원칙

- 모든 Teacher는 동일한 명시적 data config로 학습한다.
- train config만으로 degradation design을 암묵적으로 기대하지 않는다.
- `pretrained_path: null`로 scratch 학습 recipe를 따로 둔다.
- Teacher별 best checkpoint를 real EO strata에서 검증한다.
- 나쁜 Teacher는 ensemble에 넣지 않거나 region별 weight를 낮춘다.

### 추천 순서

1. NAFNet 1-stage scratch
2. NAFNet 2-stage scratch
3. Restormer 1-stage scratch
4. Restormer 2-stage scratch
5. Deploy-family 1-stage/2-stage scratch 또는 current deploy baseline comparison
6. Teacher별 real EO strata 평가
7. Teacher cache 생성

---

## 7. Multi-Teacher KD 설계

MTKD는 GT supervision의 대체물이 아니다. Teacher output은 Student의 regularizer이며, clean/pseudo-clean target에 대한 supervised loss가 중심이어야 한다.

### 권장 loss

```text
L = L_gt
  + w_kd     * L_teacher_output
  + w_res    * L_residual
  + w_edge   * L_edge
  + w_chroma * L_chroma
  + w_freq   * L_frequency
  + w_cons   * L_aug_consistency
```

권장 방향:
- `L_gt`: Charbonnier 중심, 가장 큰 weight
- `L_teacher_output`: Teacher output과 Student output의 weighted L1/Charbonnier
- `L_residual`: `(input - output)` residual/noise map distillation
- `L_edge`: Sobel/Laplacian 기반, 작은 표적 보존 확인 필요
- `L_chroma`: tint/chroma 안정성 보조
- `L_freq`: low/high frequency residual consistency
- `L_aug_consistency`: augmentation 전후 일관성

VGG perceptual loss는 EO denoise 기본값으로 추천하지 않는다. 자연 이미지 prior가 EO small target/detail 보존과 충돌할 수 있다.

### Teacher weighting

단순 평균은 기본안으로 두지 않는다.

권장:
- Teacher disagreement가 낮은 영역: KD weight 증가
- Teacher disagreement가 높은 영역: KD weight 감소
- edge/small-target 후보 영역: Teacher output보다 GT/edge 보존 loss 우선
- chroma/tint 영역: chroma 안정성이 검증된 Teacher weight 증가
- Deploy-family Teacher: 낮은 weight의 deployment-prior regularizer로만 사용

주의:
- Deploy Teacher가 D3 단독 실험에서 artifact를 줄이지 못하면 all-teacher ensemble에서 제외한다.
- Student와 너무 비슷한 deploy-family Teacher는 self-copying 효과가 커질 수 있다.

---

## 8. Teacher Cache 요구사항

현재 `DenoiseDataset`은 random crop과 random degradation을 on-the-fly로 만든다. 따라서 Teacher cache를 image path만으로 만들면 재현성이 깨진다.

### Cache key

cache entry는 최소한 다음을 포함해야 한다.

- source image path
- crop 좌표
- random seed
- degradation config hash
- degradation parameter sample
- teacher model name
- teacher checkpoint hash
- color order
- tensor range
- patch size

### Cache payload

후보:
- noisy input `lr`
- clean target `hr`
- teacher output per teacher
- teacher residual: `lr - teacher_output`
- teacher variance/disagreement map
- optional confidence map

저장 형식 후보:
- LMDB
- Zarr
- memmap
- sharded `.pt` 또는 `.npz`

fp16 teacher output 저장을 기본으로 검토하되, metric 계산용 subset은 fp32도 보관할 수 있다.

---

## 9. Student 및 배포 제약

Student는 배포 제약이 먼저다. Teacher 성능이 좋아도 Student가 QNN/AIMET에서 느리거나 quantization 손실이 크면 실패다.

### 주 Student 후보

- `SVFocusSRNet(scale=1, basic RepBlock, n_resblocks=2)`
- dim24 vs dim32는 실제 QNN latency로 결정
- `QuickDenoiseOpt`는 Quick path를 쓸 경우 `QuickDenoiseNet`보다 scale=1 구조가 깨끗하다.

### 주의할 점

- `use_advanced_rep`는 fuse 후 수치 동등성 및 quantized graph 검증 전까지 보수적으로 사용한다.
- Restormer/NAFNet Teacher의 복잡한 복원 함수를 small Student가 그대로 흡수하지 못할 수 있다.
- dim32/block2도 1080p 실시간 영상에서는 가벼운 모델이 아닐 수 있다.

### 배포 검증 요구

현재 `tools/export.py`는 `switch_to_deploy()` 후 ONNX export를 수행한다. 그러나 AIMET 경로에서도 같은 fused graph를 기준으로 calibration/QAT/export해야 한다.

필수 확인:
- fused ONNX와 AIMET/QNN graph가 동일한 deploy 구조인지
- static target resolution
- PTQ/QAT 전후 PSNR/SSIM/LPIPS/NIQE 변화
- 실제 device latency
- 메모리 사용량
- small-target/edge 보존 육안 평가

---

## 10. 실험 프로토콜

### 평가 strata

모든 모델은 global average만 보지 말고 다음 strata로 나눠 평가한다.

- day / night
- low / mid / high zoom
- flat / edge / texture / small-target ROI
- static burst / video sequence
- raw capture / deploy output / candidate output

### 실험 matrix

| ID | 목적 | Init | Data/degradation | Teacher | 판단 |
| --- | --- | --- | --- | --- | --- |
| A0 | Sensor report | none | optimized EO captures | none | sensor profile 생성 |
| B0 | Current deploy baseline | existing | current validation + real EO | none | 배포 기준선 고정 |
| B1 | Current generic baseline | scratch | generic/common denoise | none | synthetic 기준선 |
| C1 | 1-stage sensor-profiled scratch | scratch | common + MC-G105 profile | none | profile 단독 효과 |
| C2 | 2-stage no KD | scratch -> finetune | Phase 1 common, Phase 2 MC-G105 | none | curriculum 효과 |
| T1 | NAFNet Teacher | scratch | best of C1/C2 recipe | none | quality teacher 후보 |
| T2 | Restormer Teacher | scratch | best of C1/C2 recipe | none | global/chroma teacher 후보 |
| T3 | Deploy-family Teacher | scratch or existing baseline | best of C1/C2 recipe | none | deployment prior 후보 |
| D1 | Student KD NAFNet | supervised pretrain | cached inputs | NAFNet | local/detail KD 효과 |
| D2 | Student KD Restormer | supervised pretrain | cached inputs | Restormer | structured/chroma KD 효과 |
| D3 | Student KD Deploy | supervised pretrain | cached inputs | Deploy-family | stability prior 효과 |
| D4 | Student KD NAFNet+Restormer | supervised pretrain | cached inputs | adaptive 2-teacher | quality ensemble 효과 |
| D5 | Student KD all | supervised pretrain | cached inputs | adaptive 3-teacher | Deploy Teacher 추가 가치 |
| E0 | Quant/deploy check | best C/D | same eval set | none | latency/QAT 통과 |

### Decision criteria

- C2가 C1을 real EO strata에서 이길 때만 2-stage를 채택한다.
- D 계열이 C2보다 나을 때만 MTKD를 채택한다.
- D3가 의미 있게 좋지 않으면 Deploy Teacher는 D5에서 제외한다.
- adaptive KD가 simple average보다 낫지 않으면 복잡한 weighting을 미룬다.
- FP32에서 좋아도 QAT/PTQ 후 망가지면 Student architecture 또는 loss weight를 다시 조정한다.

---

## 11. 구현 작업 목록

필요한 코드 작업:

1. NAFNet/Restormer model registry 추가 또는 외부 training wrapper 정리
2. Teacher train config 생성
3. deterministic degradation sample generator 추가
4. teacher output cache 생성 tool 추가
5. cache manifest schema 정의
6. cached KD dataset 추가
7. KD loss 추가
8. KD trainer branch 추가
9. teacher disagreement/confidence map 계산 추가
10. fused graph 기준 AIMET/QNN 검증 경로 정리

처음부터 모두 구현하지 말고, C1/C2가 baseline을 이기는지 먼저 확인한 뒤 KD 관련 구현을 시작한다.

---

## 12. 최종 추천

현재 가장 안전한 접근은 다음 순서다.

1. MC-G105 sensor analysis를 보강해 `sensor_profile.yaml` 수준으로 만든다.
2. generic/common profile과 MC-G105 profile을 분리한 data config를 명확히 만든다.
3. Student/Teacher 공통으로 1-stage scratch와 2-stage scratch를 비교한다.
4. real EO strata에서 2-stage가 이길 때만 2-stage를 기본 curriculum으로 채택한다.
5. NAFNet/Restormer는 scratch-trained offline Teacher로 학습한다.
6. Teacher별 성능을 먼저 검증하고, 나쁜 Teacher는 KD ensemble에서 제외한다.
7. Student는 supervised pretrain 후 cached MTKD finetune한다.
8. Deploy-family Teacher는 낮은 weight의 stability prior로만 테스트한다.
9. 최종 후보는 fused ONNX/AIMET/QNN 기준으로 latency와 quantization 손실까지 확인한다.

요약하면, user의 2-stage 및 scratch Teacher MTKD 방향은 합리적이다. 그러나 성공 조건은 “Teacher를 많이 쓰는 것”이 아니라, 센서 profile과 평가 strata를 먼저 고정하고, 1-stage/2-stage와 Teacher 조합을 냉정하게 ablation하는 것이다. KD는 마지막 성능 압축 단계이지, sensor degradation 설계 실패를 보정하는 만능 장치가 아니다.

---

## References

- NAFNet: Simple Baselines for Image Restoration, arXiv:2204.04676, https://arxiv.org/abs/2204.04676
- Restormer: Efficient Transformer for High-Resolution Image Restoration, arXiv:2111.09881 / CVPR 2022, https://arxiv.org/abs/2111.09881
- Distilling the Knowledge in a Neural Network, arXiv:1503.02531, https://arxiv.org/abs/1503.02531
- Practical Deep Raw Image Denoising on Mobile Devices, arXiv:2010.06935, https://arxiv.org/abs/2010.06935
- Feature-Align Network with Knowledge Distillation for Efficient Denoising, arXiv:2103.01524, https://arxiv.org/abs/2103.01524
