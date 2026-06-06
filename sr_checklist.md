# SR Finetune 평가 체크리스트

## 개요

| 항목 | 내용 |
|---|---|
| 작성일 | 2026-06-06 |
| 다음 확인일 | 2026-06-08 월요일 |
| 현재 결론 | SR x2/x4 finetune은 진행 중이며, 월요일에는 real sensor 입력에서 Deploy SR 대비 개선 여부를 판단한다. |
| 핵심 비교 | 기존 Deploy Denoise 출력을 고정 입력으로 두고 Deploy SR과 Finetune SR을 비교한다. |

---

## 1. 현재 결정

- [x] x4 real sensor baseline 분석 완료
- [x] Deploy SR은 Bicubic 대비 no-reference IQA는 개선하지만 edge boost/ringing/false texture risk가 큼
- [x] SR 개선은 raw noise 제거보다 Denoise 후단 입력을 가정하고 진행
- [x] 현재 finetune 방향은 conservative 1-phase finetune
- [x] 2-stage degradation은 유지
- [x] loss는 hallucination/ringing을 줄이는 방향으로 구성
  - Charbonnier
  - SSIM
  - TV
  - color consistency
  - 약한 perceptual/edge loss

---

## 2. 월요일 핵심 평가

월요일 평가는 SR 모델 차이를 분리하기 위해 입력을 고정한다.

| 비교축 | 입력 | SR 모델 | 목적 |
|---|---|---|---|
| Baseline | 기존 Deploy Denoise 출력 | Deploy SR | 현재 생산 기준 |
| Candidate | 기존 Deploy Denoise 출력 | Finetune SR | finetune 개선 여부 판단 |

### 기준 입력

| 항목 | 경로/기준 |
|---|---|
| fixed probe raw | `results/260602_mc_g105_probe_42/raw` |
| 기존 Deploy Denoise 출력 | `results/260602_mc_g105_probe_42/deploy` |
| SR real sensor 분석 결과 | `results/sr_x4_baseline_analysis/real_sensor` |
| 기존 분석 스크립트 | `tools/analyze_sr_real_sensor_x4.py` |

---

## 3. 평가 방식

실제 real sensor 입력에는 HR/GT가 없으므로 PSNR/SSIM 중심으로 판단하지 않는다.

| 평가 항목 | 판단 기준 |
|---|---|
| NIQE/BRISQUE/PIQE | Deploy SR 대비 Finetune SR이 악화되지 않아야 함 |
| Sharpness / edge density | 과도한 edge boost가 줄어드는지 확인 |
| High-frequency energy | false texture나 noise amplification이 줄어드는지 확인 |
| Ringing score | edge 주변 ringing이 감소하는지 확인 |
| Local contrast | 과도한 contrast 변화가 없는지 확인 |
| Tone/color proxy | Denoise 출력 대비 luma/chroma drift가 과도하지 않은지 확인 |
| Top-risk visual review | edge/ringing/false texture/frame별 artifact를 직접 확인 |

---

## 4. 참고 비교

주 평가는 Deploy Denoise 출력을 고정 입력으로 사용한다. 단, 참고용으로 아래 비교도 같이 보면 좋다.

| 참고 비교 | 목적 |
|---|---|
| Raw/Input -> Deploy SR | Denoise 없이 SR을 걸었을 때 artifact amplification 확인 |
| Raw/Input -> Finetune SR | Finetune SR의 raw 입력 robustness 확인 |
| Bicubic x4 | SR 모델이 단순 upsampling 대비 어떤 artifact를 만드는지 확인 |

참고 비교는 최종 판단의 보조 자료이며, 주 판단축은 `Deploy Denoise -> Deploy SR` vs `Deploy Denoise -> Finetune SR`이다.

---

## 5. 월요일 판단 분기

### Finetune SR이 Deploy SR보다 개선될 때

- [ ] fixed 42 real sensor probe에서 top-risk frame 개선 확인
- [ ] edge oversharpening/ringing 감소 확인
- [ ] false texture/noise amplification 감소 확인
- [ ] no-reference IQA가 유지 또는 개선되는지 확인
- [ ] Deploy 후보로 유지하고 추가 epoch 진행 여부 판단

### Finetune SR이 Deploy SR보다 악화될 때

- [ ] 해당 checkpoint는 보류
- [ ] loss weight 재조정
- [ ] edge/perceptual loss를 더 낮추거나 TV/color consistency 비중 재검토
- [ ] degradation이 실제 Denoise output 분포와 맞는지 재확인
- [ ] 필요 시 이전 epoch checkpoint와 비교

---

## 6. SR MTKD로 넘어가기 전 조건

- [ ] Deploy SR baseline failure mode가 명확히 정리되어야 함
- [ ] Finetune SR이 Deploy SR 대비 어떤 실패를 줄였는지 확인되어야 함
- [ ] Synthetic GT benchmark가 별도로 구성되어야 함
- [ ] Real sensor no-reference 개선과 synthetic GT 정량 개선을 분리해서 판단해야 함
- [ ] Teacher 모델 선택은 위 조건 확인 후 진행

---

## 7. 월요일 작업 순서

1. SR x2/x4 finetune 학습 상태 확인
2. epoch 3-5 validation trend 확인
3. Deploy Denoise output을 입력으로 Deploy SR / Finetune SR 추론
4. fixed 42 real sensor probe 비교 리포트 생성
5. no-reference metric과 top-risk frame 확인
6. x2/x4 finetune 계속 진행 여부 결정
7. 결과가 충분히 안정적이면 SR MTKD 설계로 넘어갈 준비

---

## Quick Priority

| 우선순위 | 작업 | 상태 |
|---:|---|---|
| 1 | SR x2/x4 finetune validation trend 확인 | 월요일 진행 |
| 2 | Deploy Denoise -> Deploy SR 결과 확보 | 기존 출력 활용 |
| 3 | Deploy Denoise -> Finetune SR 결과 생성 | 월요일 진행 |
| 4 | fixed 42 probe 비교 리포트 생성 | 월요일 진행 |
| 5 | SR MTKD 필요성 및 방향 판단 | finetune 평가 이후 |
