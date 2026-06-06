# Denoise Teacher Gate 및 KD 재설계 체크리스트

## 개요

| 항목 | 내용 |
|---|---|
| 작성일 | 2026-06-06 |
| 다음 확인일 | 2026-06-08 월요일 |
| 현재 결론 | Denoise KD는 일시 중단. Tone 완화 degradation을 적용한 Teacher 모델을 먼저 재평가한다. |
| 핵심 판단 | Teacher가 Deploy 대비 실사용 개선을 보일 때만 KD 재설계를 시작한다. |

---

## 1. 현재 결정

- [x] 기존 `dim32 block2` Student 기반 MTKD/STKD는 deployable 개선으로 보기 어렵다고 판단
- [x] Validation 지표보다 fixed 42 real sensor probe 결과를 우선 판단 기준으로 설정
- [x] 기존 KD 문제는 단일 원인으로 확정하지 않음
  - Student capacity 부족
  - full-output KD target 과다
  - Teacher tone/detail trade-off
  - loss weight 설계 문제
- [x] 현재 방향은 KD 즉시 재시작이 아니라 Teacher gate 재검증 후 KD 재설계

---

## 2. 월요일 평가 대상

| 대상 | 역할 | 확인 목적 |
|---|---|---|
| Deploy Denoise | 생산 baseline | 실제 교체 가치 판단 기준 |
| Restormer tone-safe Teacher | tone/color safety 후보 | RAW-like 상태에서 벗어나 denoise 효과가 생겼는지 확인 |
| NAFNet tone-safe Teacher | detail/edge Teacher 후보 | 기존 NAFNet의 tone-down 문제가 완화됐는지 확인 |

### 기준 데이터

| 항목 | 경로/기준 |
|---|---|
| fixed probe raw | `results/260602_mc_g105_probe_42/raw` |
| 기존 Deploy 출력 | `results/260602_mc_g105_probe_42/deploy` |
| 비교 단위 | 동일 42장 real sensor probe |
| 기존 분석 스크립트 | `tools/analyze_denoise_real_sensor_42.py` |
| 기존 KD 분석 리포트 | `results/denoise_real_sensor_42_kd_analysis/report.md` |

---

## 3. Teacher Acceptance Gate

Teacher는 validation PSNR/SSIM만으로 채택하지 않는다. 아래 조건을 fixed 42 real sensor probe에서 확인한다.

| 평가 항목 | 판단 기준 |
|---|---|
| Noise suppression | Deploy 대비 평탄 영역 noise-like high frequency가 줄어드는지 확인 |
| Tone stability | `lowfreq_luma_mae`, luma mean shift가 Deploy 대비 과도하게 커지면 보류 |
| Color stability | `chroma_mae`가 Deploy 대비 커지면 보류 |
| Edge/detail preservation | `strong_edge_grad_ratio`가 과도하게 낮아지면 smoothing으로 판단 |
| No-reference IQA | NIQE/BRISQUE/PIQE가 Deploy와 동급 이상인지 확인 |
| Visual review | top-risk frame에서 tone-down, oversmoothing, residual noise, color shift를 직접 확인 |

### 모델별 Gate

| Teacher | 통과 조건 | 보류 조건 |
|---|---|---|
| NAFNet tone-safe | Detail/edge가 Deploy보다 좋고 tone-down이 제한적 | 이전처럼 큰 음수 Y shift 또는 low-frequency tone drift 발생 |
| Restormer tone-safe | Tone/color가 안전하면서 denoise 효과가 명확 | RAW-like 출력에 가까워 noise 제거 효과가 부족 |

---

## 4. 월요일 판단 분기

### Teacher가 Deploy 대비 개선을 보일 때

- [ ] Teacher 후보를 KD source로 채택
- [ ] KD target 범위를 재설계
- [ ] 44K Student를 유지할 경우 full-output KD는 피하고 narrow target부터 시작
- [ ] 후보 loss:
  - residual/noise KD
  - local artifact KD
  - tone-gated weak output KD
  - edge/frequency KD
- [ ] 필요 시 수백 K급 Student 후보를 별도 검토

### Teacher가 Deploy 대비 개선을 못 보일 때

- [ ] KD 재시작 금지
- [ ] Teacher/degradation 재설계 우선
- [ ] NAFNet tone-safe degradation 범위 재검토
- [ ] Restormer가 RAW-like이면 추가 epoch 또는 loss/degradation 재설계
- [ ] Student KD 실험은 Teacher gate 통과 전까지 보류

---

## 5. 월요일 작업 순서

1. Restormer tone-safe와 NAFNet tone-safe 학습 상태 확인
2. 최신 `last.pth`/`best.pth` timestamp와 validation log 확인
3. fixed 42 real sensor probe에 대해 최신 Teacher 추론 수행
4. Deploy / Restormer / NAFNet 비교 리포트 생성
5. metric table과 top-risk comparison image 확인
6. Teacher 채택 여부 결정
7. 채택된 Teacher가 있을 때만 KD 재설계안 작성

---

## Quick Priority

| 우선순위 | 작업 | 상태 |
|---:|---|---|
| 1 | Tone-safe Teacher 최신 checkpoint 확인 | 월요일 진행 |
| 2 | fixed 42 probe 동일 평가 실행 | 월요일 진행 |
| 3 | Deploy 대비 Teacher 가치 판단 | 월요일 진행 |
| 4 | KD target 범위 재설계 | Teacher gate 이후 |
| 5 | Student capacity 재검토 | Teacher gate 이후 |
