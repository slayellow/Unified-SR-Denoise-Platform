# MC-G105 Denoise Degradation Pipeline v1 설계

작성일: 2026-05-13

목적:
- `results/mc_g105_analysis/all` 통합 분석 결과를 기준으로 MC-G105 Denoise 학습용 degradation pipeline v1을 정의한다.
- 이전 `denoise_mc_g105_v1.yaml`, `denoise_mc_g105_v2.yaml`는 day/night + zoom 중심의 중간 실험 config로 보고 제거한다.
- 새 v1은 Phase 1 generic prior와 Phase 2 field sensor profile을 분리한다.
- Phase 2 이름은 `daylight` 대신 `field`를 사용한다. 현재 목적은 특정 날씨 subset이 아니라 cloudy afternoon을 제외한 실제 MC-G105 취득 조건 전체를 대표하는 field profile 구축이다.

---

## 1. 설계 결론

현재 분석 결과 기준으로 MC-G105 Denoise v1은 다음 구조가 가장 안전하다.

```text
Phase 1:
  clean/detail/edge 보존 중심의 generic denoise prior
  MC-G105 artifact는 약하게만 포함

Phase 2:
  road_shaded stress
  afternoon mixed green/chroma/clipping
  7x shading/detail/luma instability
  sparse random hot-pixel
  generic replay와 clean anchor 유지
```

중요한 판단:
- `road_shaded`가 dominant hard case다.
- `7x`는 green cast worst case가 아니라 shading/vignetting/detail attenuation/luma instability 축으로 분리한다.
- outdoor hot-pixel persistence는 핵심 문제가 아니므로 fixed mask를 쓰지 않는다.
- `temporal_noise`는 Gaussian sigma가 아니라 operational hard-case proxy다.
- sunny 조건의 green excess는 clipping/scene composition과 섞여 있으므로 전역 color cast로 처리하지 않는다.

---

## 2. 반영한 분석 근거

입력 분석:

```text
results/mc_g105_analysis/all/
```

주요 근거:
- 전체 2700 image, 42 burst 기준 통합 분석.
- `road_shaded` temporal/detail instability proxy가 오전/오후/흐림 조건에서 계속 가장 높다.
- `afternoon sunny mixed 5x/7x`는 green excess와 clipping risk가 증가한다.
- flat baseline은 1x/7x shading이 강하고, 7x는 chroma non-uniformity가 상대적으로 크다.
- dark frame에서는 hot-pixel persistence가 있으나 outdoor에서는 거의 고정 persistence가 없다.

---

## 3. 새 Config 세트

### Data Config

| 파일 | 용도 |
| --- | --- |
| `configs/data/denoise_mc_g105_phase1_generic_v1.yaml` | Phase 1 generic prior |
| `configs/data/denoise_mc_g105_phase2_field_v1.yaml` | Phase 2 field sensor profile |

### Train Config

| 파일 | 용도 |
| --- | --- |
| `configs/train/Denoise/svfocusdenoise_mc_g105_phase1_generic_v1_dim32.yaml` | dim32 Phase 1 scratch |
| `configs/finetune/Denoise/svfocusdenoise_mc_g105_phase2_field_v1_dim32.yaml` | dim32 Phase 2 finetune |

Phase 2는 `--resume`이 아니라 `pretrained_path` 기반 finetune으로 사용한다.

---

## 4. Phase 1 Profile

목표:
- clean scene prior 형성
- 약한~중간 denoise prior 학습
- identity/detail/edge 보존
- 색을 과도하게 중립화하지 않도록 clean anchor 유지

핵심 설정:
- `clean_prob: 0.12`
- `target_resize.enabled: false`
- `ir_noise.enabled: false`
- weak unprocess/read/shot noise
- Gaussian/Poisson은 약하게
- detail attenuation은 낮은 확률/낮은 강도
- EO shading, chroma mottle, hot pixel은 보조 수준

Phase 1에서는 hard-case sensor artifact를 강하게 넣지 않는다.

---

## 5. Phase 2 Profile

Phase 2는 `conditional_profiles.axes: [profile]` 구조를 사용한다.

Sampling:

| profile | ratio | 의미 |
| --- | ---: | --- |
| `generic_replay` | 0.40 | Phase 1 prior 유지 |
| `normal_building` | 0.08 | 안정적인 outdoor 기준 |
| `normal_mixed` | 0.10 | 일반 mixed scene |
| `road_shaded_1x_3x_5x` | 0.18 | 핵심 hard case |
| `road_shaded_7x` | 0.08 | 7x luma/detail/shading hard case |
| `afternoon_mixed_5x_7x` | 0.07 | green/chroma/clipping stress |
| `building_7x` | 0.04 | high-zoom building profile |
| `hot_pixel_stress` | 0.03 | sparse random hot-pixel |
| `near_clean` | 0.02 | oversmoothing 방지 |

`clean_prob: 0.08`이 별도로 있으므로 실제 clean/near-clean anchor는 약 10% 수준이다.

---

## 6. 새 Primitive

`src/data/datasets.py`에 다음 primitive를 추가했다.

| primitive | 이유 |
| --- | --- |
| `eo_shading` | IR noise와 분리된 EO shading/vignetting |
| `chroma_mottle` | YCrCb chroma-only low-frequency mottle |
| `highlight_clipping` | sunny highlight saturation proxy |
| generic `conditional_profiles.axes` | `profile` bucket sampler 지원 |

기존 `chroma_noise`는 RGB/BGR 전체 채널에 저해상도 noise를 더하는 구조라 MC-G105 color mottle에는 `chroma_mottle`을 우선 사용한다.

---

## 7. Training Commands

Phase 1:

```bash
python tools/train.py \
  --config configs/train/Denoise/svfocusdenoise_mc_g105_phase1_generic_v1_dim32.yaml \
  --data_config configs/data/denoise_mc_g105_phase1_generic_v1.yaml \
  --device 0
```

Phase 2:

```bash
python tools/train.py \
  --config configs/finetune/Denoise/svfocusdenoise_mc_g105_phase2_field_v1_dim32.yaml \
  --data_config configs/data/denoise_mc_g105_phase2_field_v1.yaml \
  --device 0
```

`--data_config`를 반드시 명시한다. `tools/train.py` 기본값은 `configs/data/denoise.yaml`이므로 명시하지 않으면 MC-G105 v1 profile이 적용되지 않는다.

---

## 8. Validation 설계

Fixed synthetic validation은 materialized pair로 만든다.

권장 구조:

```text
val_denoise_generic_fixed_v1/
  HR/
  LR/

val_denoise_mc_g105_field_fixed_v1/
  HR/
  LR/

real_probe_mc_g105_field_v1/
  input/
  output_by_checkpoint/
```

Real probe 고정 후보:
- `road_shaded/3x`
- `road_shaded/5x`
- `road_shaded/7x`
- `afternoon/mixed/5x`
- `afternoon/mixed/7x`
- `building/7x`
- `flat/1x`
- `flat/7x`
- `dark/1x`
- `dark/7x`

Real probe는 GT 평가가 아니라 input/output 변화와 부작용 검사에 사용한다.

---

## 9. Checkpoint 선택 기준

Phase 1:
- generic fixed val loss/PSNR
- near-clean identity
- edge/high-frequency retention

Phase 2:
- `best.pth`는 참고용
- 40/60/80/100/120/140 epoch snapshot 비교
- `road_shaded`, `mixed`, `building 7x` real probe 확인
- green/chroma 개선과 edge/detail 보존의 Pareto로 선택

최종 gate는 QCS8550/QNN/AIMET 이후 정성 출력이다.

---

## 10. B1/C1/C2 실험 정의

| ID | 학습 방식 | 목적 |
| --- | --- | --- |
| B1 | Phase 1 only | clean/detail 보존 기준선 |
| C1 | Phase 2 profile scratch | one-stage hard-profile 반례 |
| C2 | Phase 1 -> Phase 2 | v1 기본 후보 |

C1과 C2는 총 update 수를 맞춰 비교한다.

초기 기대값은 C2가 가장 높다. 다만 C2가 real probe에서 oversmoothing되거나 small target/edge를 지우면 Phase 2 profile 강도를 다시 낮춘다.
