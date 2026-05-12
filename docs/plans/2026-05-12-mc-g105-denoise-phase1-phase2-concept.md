# MC-G105 Denoise Phase 1/2 학습 컨셉

작성일: 2026-05-12

목적:
- 공개 HR 데이터셋(DIV2K, Flickr2K 등)을 기반으로 MC-G105 EO 센서용 Denoise 모델을 학습하기 위한 Phase 1/Phase 2 컨셉을 정리한다.
- 아직 추가 촬영 데이터 3개 케이스가 모두 확보되지 않았으므로, 본 문서는 YAML config 확정 전 설계 메모로 둔다.
- 현재 `denoise_mc_g105_v1.yaml`, `denoise_mc_g105_v2.yaml`는 추후 삭제 가능하지만, `v3` 설계가 확정되기 전까지는 보존한다.

---

## 1. 현재 센서 분석에서 얻은 사실

현재까지 분석한 데이터:
- 실내 흰 벽 `flat`
- 렌즈 가림 `dark`
- 오전 흐림 야외 `morning/cloudy`
  - `building`
  - `mixed`
  - `road_shaded`

핵심 관찰:
- 실내 흰 벽에서는 green bias가 강하지 않았다.
- 야외 오전 흐림에서는 장면에 따라 green cast가 뚜렷하게 나타났다.
- 특히 `road_shaded` 1x/3x/5x에서 green cast와 chroma mottle이 강했다.
- 7x에서는 green cast를 무조건 강하게 보기보다 detail attenuation, shading/vignetting, 초점/광학계 영향이 더 중요해 보인다.
- 렌즈 가림 기준 hot pixel 후보는 존재하지만, 여러 센서에 일반화해야 하므로 특정 위치 mask는 학습에 사용하지 않는다.
- hot pixel은 위치가 아니라 발생 밀도만 랜덤 주입하는 방향이 맞다.
- 현재 프레임 간 변화 지표는 순수 temporal sensor noise가 아니라 흔들림, AE/AWB/AF, 압축, 장면 질감 변화가 섞인 proxy로 봐야 한다.

---

## 2. 전체 학습 방향

공개 HR 이미지는 MC-G105의 실제 clean 원본이 아니다. 따라서 공개 HR 데이터셋을 "센서 영상 그 자체"로 보지 않고, clean scene prior로 사용한다.

학습 입력은 다음처럼 만든다.

```text
Public HR clean image
-> synthetic degradation
-> noisy/degraded input
```

Denoise task에서는 해상도를 낮추는 Super-Resolution LR이 아니라, 같은 해상도의 noisy input을 만든다.

```text
HR clean target: 256x256
Noisy input:     256x256
scale:           1
target_resize:   false
```

따라서 현재 설계의 중심은 `HR -> LR`이라기보다 `clean HR -> noisy same-resolution input`이다.

---

## 3. Phase 1: Generic Denoise Prior

### 목표

Phase 1은 MC-G105를 정밀하게 흉내내는 단계가 아니다.

목표는 공개 HR 데이터셋에서 모델이 다음 능력을 먼저 갖도록 만드는 것이다.

- 약한~중간 노이즈 제거
- clean detail 보존
- edge와 작은 구조물 유지
- 색을 과도하게 중립화하지 않는 안정성
- denoise residual prior 형성

즉, Phase 1은 "모델 체력 만들기"에 가깝다.

### 권장 열화 범위

| 열화 항목 | Phase 1 권장 |
| --- | --- |
| clean/near-clean | `clean_prob 0.10 ~ 0.20` |
| unprocess/read/shot noise | 약~중간 |
| Gaussian noise | `sigma 1 ~ 3` |
| Poisson noise | 약~중간 |
| common noise | `sigma 1 ~ 4` |
| JPEG | 약하게, `quality 95 ~ 99` |
| blur | 아주 약하게 |
| green cast | 끄거나 매우 약하게 |
| chroma mottle | 끄거나 매우 약하게 |
| hot pixel | 끄거나 아주 약하게 |
| detail attenuation | 강하게 넣지 않음 |
| resize/target resize | 사용하지 않음 |
| IR noise | 사용하지 않음 |

### 주의점

Phase 1에서 너무 많은 센서 특화 열화를 넣으면 다음 문제가 생길 수 있다.

- 모델이 normal texture를 noise처럼 지울 수 있다.
- green cast 제거 편향이 생겨 실제 녹색/회색 구조를 왜곡할 수 있다.
- chroma texture가 뭉개질 수 있다.
- 7x detail attenuation을 너무 일찍 학습해 전체적으로 oversmoothing될 수 있다.

따라서 Phase 1은 약하고 넓게 가는 것이 좋다.

### 학습 길이

Student 기준 초기 권장:

| 항목 | 권장 |
| --- | --- |
| Epoch | 250 ~ 350 |
| 기준 | generic validation plateau 이후 30~50 epoch 추가 |
| 선택 기준 | PSNR뿐 아니라 near-clean identity, edge retention, flat noise reduction 확인 |

현재 300 epoch 수준의 generic baseline은 큰 방향에서 적절하다.

---

## 4. Phase 2: MC-G105 Sensor Adaptation

### 목표

Phase 2는 MC-G105에서 실제로 관측된 센서/ISP 특성을 조건부로 주입하는 단계다.

중요한 점:
- "MC-G105 노이즈를 무조건 세게 넣기"가 아니다.
- 장면, 줌, 밝기/그늘 조건에 따라 다르게 넣어야 한다.
- 7x라고 해서 green cast를 무조건 강하게 넣으면 안 된다.

### 주요 열화 축

| 열화 축 | 반영 이유 | 주의점 |
| --- | --- | --- |
| Scene-conditional green cast | 야외 특정 장면에서 green bias 확인 | 전역 고정값으로 넣으면 위험 |
| Chroma mottle | `road_shaded`, `mixed`에서 색 얼룩 증가 | 고주파 RGB noise보다 저주파 색 얼룩 형태가 적합 |
| Shading/vignetting | flat 1x/7x에서 밝기 불균일 확인 | 1x와 7x 패턴을 분리하는 것이 좋음 |
| Detail attenuation | 7x에서 디테일 저하 경향 | 너무 강하면 denoise가 아니라 blur 학습이 됨 |
| Hot pixel density | dark에서 반복 후보 확인 | 위치 mask 금지, 랜덤 density만 사용 |
| Mild raw/ISP noise | 기본 sensor noise proxy | raw 물리값으로 과해석하지 않음 |
| Signal/auto instability proxy | 일부 burst에서 프레임 간 변화 큼 | 순수 temporal noise로 해석하지 않음 |

### 조건부 profile 초안

| 조건 | Green Cast | Chroma Mottle | Shading | Detail Attenuation | Instability |
| --- | --- | --- | --- | --- | --- |
| flat/bright | 거의 없음 | 약함 | 1x/7x 강함 | 약함 | 약함 |
| building 1x~5x | 약~중간 | 약~중간 | 약~중간 | 줌에 따라 증가 | 약함 |
| building 7x | 중간 | 약~중간 | 중간 | 강함 | 약~중간 |
| mixed 1x/3x | 중간~강함 | 중간 | 약~중간 | 약~중간 | 약함 |
| mixed 5x/7x | 약함 | 중간 | 중간 | 중간 | 7x 약~중간 |
| road_shaded 1x/3x/5x | 강함 | 강함 | 중간 | 중간 | 강함 |
| road_shaded 7x | 약함 | 중간~강함 | 강함 | 강함 | 강함 |

### Phase 2 샘플링 비율 초안

초기에는 너무 강하게 몰아가지 않는다.

| Bucket | 권장 비율 | 의미 |
| --- | ---: | --- |
| Generic/mild replay | 40~50% | Phase 1에서 배운 clean/detail 보존 능력 유지 |
| MC-G105 scene-specific | 35~45% | 현재 분석 기반 조건부 열화 |
| Stress case | 10~20% | road_shaded, high zoom, strong shading 등 hard case |
| clean/near-clean | 5~10% | 과도한 denoise/oversmoothing 방지 |

Student에서는 `clean_prob`를 `0.10 ~ 0.15` 정도로 유지하는 것이 안전해 보인다.

### Phase 2 학습 길이

| 항목 | 권장 |
| --- | --- |
| Epoch | 80 ~ 120부터 우선 확인 |
| Snapshot | 30/60/90/120/180 모두 평가 |
| 선택 기준 | 마지막 epoch가 아니라 noise-detail Pareto 기준 |

Phase 2를 길게 돌리면 noise는 줄어도 작은 표적, 선 구조, texture가 함께 사라질 수 있다. 따라서 checkpoint 선택은 PSNR뿐 아니라 real probe 분석과 육안 비교를 함께 봐야 한다.

---

## 5. Hot Pixel 설계 원칙

현재 결론:
- 특정 장비의 hot pixel 위치 mask는 학습에 사용하지 않는다.
- 여러 센서에 일반화해야 하므로 위치는 매 샘플 랜덤으로 생성한다.
- 렌즈 가림 분석에서 얻은 density 범위만 참고한다.

초기 범위:

| 항목 | 값 |
| --- | --- |
| density | `0.0001 ~ 0.0006` |
| 위치 | random coordinate |
| 크기 | 대부분 1px |
| blob | 사용하더라도 매우 낮은 확률, 1~2px 제한 |

주의:
- hot pixel을 Phase 2의 주인공으로 만들면 실제 야외 문제보다 과대학습될 수 있다.
- 작은 밝은 표적을 hot pixel처럼 지우는 실패를 반드시 검사해야 한다.

---

## 6. Validation 설계

실제 MC-G105 캡처 데이터는 반드시 활용한다. 다만 GT가 없으므로 checkpoint 선택용 정답 validation이 아니라 real-domain probe로 둔다.

### Validation 구성

| 검증 세트 | 역할 |
| --- | --- |
| fixed generic synthetic val | 일반 denoise 정량 평가 |
| fixed MC-G105 synthetic val | GT가 있는 센서 특화 정량 평가 |
| real MC-G105 flat/dark/outdoor probe | 실제 센서 영상에서 부작용 확인 |
| 추가 3개 케이스 | 새 조건 generalization 확인 |

### Synthetic fixed validation

반드시 고정해야 한다.

```text
Public HR holdout
-> fixed seed degradation
-> noisy input + clean GT 저장
```

이렇게 해야 모델 변화와 validation sample 변화가 섞이지 않는다.

평가 지표 후보:
- PSNR
- SSIM / MS-SSIM
- LPIPS 또는 DISTS
- edge retention
- color error
- hot pixel removal rate

### Real MC-G105 probe

현재 `flat/dark/morning_cloudy`는 다음 용도로 사용한다.

- synthetic degradation range 보정
- 모델 출력의 실제 센서 영상 변화 확인
- green cast, chroma mottle, hot pixel, detail loss 부작용 검사

하지만 다음 용도로는 쓰지 않는다.

- 단독 최종 성능 주장
- checkpoint 선택의 유일한 기준
- GT 기반 PSNR/SSIM 평가

Real probe에서는 input/output을 같은 분석 스크립트로 돌려 변화량을 본다.

확인할 항목:

| 항목 | 좋아지는 방향 | 실패 신호 |
| --- | --- | --- |
| green_excess | 과한 green cast 감소 | 정상 색까지 과도하게 중립화 |
| chroma_mottle | 색 얼룩 감소 | 색 디테일 소실 |
| hot_pixel_ratio | 후보 감소 | 작은 밝은 표적 삭제 |
| temporal_std | 과한 프레임 변화 감소 | 전체 디테일 저하와 혼동 가능 |
| high_freq_energy | 적절히 유지 | 큰 폭 감소 시 oversmoothing 의심 |
| edge_density | 유지 또는 소폭 감소 | 선/경계 구조 소실 |

---

## 7. 비교 실험 설계

최소 비교 실험:

| ID | 목적 | 학습 방식 |
| --- | --- | --- |
| B1 | generic baseline | Phase 1 only |
| C1 | 1-stage MC-G105 | scratch부터 MC-G105 mixed profile |
| C2 | 2-stage MC-G105 | Phase 1 generic -> Phase 2 MC-G105 |

공정 비교 조건:
- C1과 C2의 총 update 수를 맞춘다.
- C2가 300+120 epoch라면 C1도 그에 준하는 update 수로 비교한다.
- C1은 full mixture와 ramp-up mixture 두 버전을 검토할 수 있다.

2-stage 채택 기준:
- C2가 C1 대비 synthetic MC-G105 val에서 개선된다.
- real probe에서 green/chroma/hot-pixel 문제가 줄어든다.
- edge/high-frequency retention이 크게 무너지지 않는다.
- 작은 밝은 표적이나 선 구조가 지워지지 않는다.
- paired synthetic PSNR/SSIM이 크게 하락하지 않는다.

초기 기준:
- C2가 C1 대비 PSNR `-0.1 ~ -0.2 dB` 이상 크게 떨어지면 보류한다.
- noise 감소와 detail 보존의 균형이 더 중요하다.

---

## 8. Multi-Teacher KD로 넘어가기 전 조건

KD는 Phase 1/2 supervised 실험 이후에 진행한다.

KD가 sensor profile 실패나 oversmoothing을 해결해주지는 못한다. 따라서 먼저 supervised C1/C2 비교가 끝나야 한다.

KD 진입 조건:
- sensor profile과 validation strata가 고정됨
- C1/C2 중 하나가 current deploy baseline보다 real probe에서 안정적임
- 선택된 supervised Student가 oversmoothing 없이 통과함
- NAFNet/Restormer Teacher가 scratch 학습 기준으로 Student보다 명확히 우수함
- deterministic degradation/cache가 준비됨
- 단일 Teacher KD가 먼저 이득을 보임

권장 순서:

```text
NAFNet KD
-> Restormer KD
-> NAFNet + Restormer adaptive KD
-> Deploy-family Teacher 추가 여부 판단
```

Student KD에서는 GT loss를 anchor로 둔다.

초기 loss weight 감각:

| Loss | 권장 |
| --- | --- |
| GT supervised loss | `1.0` |
| output KD | `0.05 ~ 0.15` |
| residual KD | `0.03 ~ 0.10` |
| edge/frequency KD | `0.02 ~ 0.08` |
| chroma KD | `0.03 ~ 0.08` |

Teacher disagreement가 큰 영역, edge 영역, small-target 후보 영역에서는 KD weight를 낮춰야 한다.

---

## 9. 추가 데이터 취득 후 업데이트 계획

추후 추가 3개 케이스 데이터를 확보하면 다음 순서로 진행한다.

1. 현재 분석 스크립트로 동일하게 분석한다.
2. 기존 `flat/dark/morning_cloudy` 결과와 비교한다.
3. scene/zoom/brightness profile을 다시 조정한다.
4. `denoise_mc_g105_v3_daylight.yaml` 초안을 작성한다.
5. 기존 `v1`, `v2` config 삭제 여부를 결정한다.
6. fixed MC-G105 synthetic validation set을 생성한다.
7. B1/C1/C2 실험을 시작한다.

추가 데이터가 들어오기 전까지는 YAML을 만들지 않는다. 현재 문서는 개념 설계와 판단 기준을 보존하기 위한 문서로 둔다.

---

## 10. 현재 결론

현재 추천 방향은 다음과 같다.

```text
Phase 1:
  약한 generic denoise prior 학습
  clean/detail 보존 중심
  MC-G105 특화 열화는 최소화

Phase 2:
  MC-G105 분석 결과 기반 조건부 profile 학습
  road_shaded green/chroma 강화
  7x는 detail/shading 중심
  hot pixel은 랜덤 density만 사용

Validation:
  fixed synthetic validation + real MC-G105 probe 병행
  real capture는 checkpoint 선택용 정답 validation이 아니라 현장성 점검용

KD:
  supervised Phase 1/2 결론 이후 진행
```

한 줄로 정리하면, 공개 HR 데이터셋은 clean scene prior로 쓰고, MC-G105 특성은 Phase 2에서 조건부 열화 profile로 주입하는 것이 현재 데이터와 가장 잘 맞는다.
