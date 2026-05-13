# MC-G105 Sunny Outdoor Capture 분석 결과

작성일: 2026-05-13

목적:
- `morning/sunny`와 `afternoon/sunny` 조건에서 추가 수집한 MC-G105 캡처 결과를 정리한다.
- `--save_outdoor_maps`를 포함한 통합 분석 결과를 기준으로, 실제 Denoise Phase 2 degradation profile에 반영할 관찰을 정리한다.
- 본 문서는 RAW 센서 물리 파라미터 추정이 아니라, NV12 저장 이미지 기반 운용 출력 proxy 분석으로 해석한다.

---

## 1. 분석 입력과 산출물

실행 명령:

```bash
python3 examples/analysis/analyze_mc_g105_sensor_capture.py \
  --input_dir results/mc_g105_capture_frames/ \
  --output_dir results/mc_g105_analysis/all \
  --save_outdoor_maps
```

산출물 위치:

```text
results/mc_g105_analysis/all/
```

주요 산출물:

| 파일/폴더 | 내용 |
| --- | --- |
| `per_image_metrics.csv` | 이미지별 RGB/Y/CrCb, green excess, clipping, detail, hot-pixel 후보 |
| `burst_metrics.csv` | burst별 temporal noise, auto/detail instability, hot-pixel persistence |
| `group_summary.csv` | time/weather/scene/zoom group 통계 |
| `scene_contrast_summary.csv` | flat/dark baseline 대비 outdoor scene 차이 |
| `hot_pixel_overlap_summary.csv` | dark/flat hot-pixel 후보 overlap count/ratio |
| `summary.md` | 자동 생성 요약 |
| `plots/` | scene/zoom별 aggregate plot |
| `maps/` | flat/dark 및 outdoor temporal/hot-persistence heatmap |

이번 `all` 실행은 입력 root 전체를 대상으로 했기 때문에 `morning/cloudy`도 함께 포함한다. 본 문서의 해석은 오늘 추가 수집한 `morning/sunny`, `afternoon/sunny` outdoor 조건을 중심으로 한다.

전체 분석 규모:

| 항목 | 값 |
| --- | ---: |
| 전체 이미지 | 2700 |
| 전체 burst | 42 |
| map 이미지 | 92 |
| sunny outdoor scene | `building`, `mixed`, `road_shaded` |
| sunny outdoor zoom | `1x`, `3x`, `5x`, `7x` |

---

## 2. Sunny Outdoor 핵심 요약

가장 중요한 결론:
- Outdoor에서 고정 hot-pixel 문제는 거의 보이지 않는다.
- 가장 강한 문제 신호는 `road_shaded`의 temporal/detail instability다.
- 오후 sunny에서 green excess와 clipping이 오전보다 커지는 경향이 있다.
- `temporal_std` map은 순수 센서 temporal noise라기보다 차선, 차량, 수목, 그림자 경계, auto exposure/focus/detail 변화가 섞인 hard-case proxy로 봐야 한다.

Scene 평균:

| time | scene | images | mean_y | green_excess | chroma_mottle | high_freq | temporal_noise |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| morning | building | 200 | 130.65 | 2.00 | 1.72 | 364.34 | 8.32 |
| morning | mixed | 200 | 141.31 | 1.51 | 1.16 | 297.11 | 5.69 |
| morning | road_shaded | 400 | 131.49 | 2.69 | 2.19 | 903.25 | 40.20 |
| afternoon | building | 200 | 133.28 | 2.32 | 1.55 | 303.89 | 4.86 |
| afternoon | mixed | 200 | 134.90 | 3.23 | 2.18 | 931.90 | 9.58 |
| afternoon | road_shaded | 400 | 130.03 | 3.61 | 2.57 | 1357.89 | 42.68 |

해석:
- `road_shaded`는 오전/오후 모두 temporal noise proxy가 `40` 이상으로 높다.
- `building`은 temporal noise가 낮고, scene 안정성이 상대적으로 좋다.
- `mixed`는 오후에 green excess와 high-frequency energy가 크게 증가한다. 하늘/산/건물 조합의 highlight 및 색 편향 영향이 섞였을 가능성이 있다.
- `road_shaded`는 green excess, chroma mottle, temporal/detail instability가 동시에 큰 hard case다.

---

## 3. Zoom/Scene별 주요 관찰

### Building

| time | zoom | green_excess | temporal_noise | unstable_auto |
| --- | --- | ---: | ---: | --- |
| morning | 1x | 1.10 | 5.41 | false |
| morning | 3x | 1.35 | 6.25 | true |
| morning | 5x | 2.00 | 9.06 | false |
| morning | 7x | 3.53 | 12.56 | true |
| afternoon | 1x | 1.06 | 4.07 | false |
| afternoon | 3x | 1.72 | 4.75 | true |
| afternoon | 5x | 2.55 | 5.97 | true |
| afternoon | 7x | 3.93 | 4.67 | true |

관찰:
- 7x에서 green excess가 가장 커진다.
- 3x 이상에서 detail instability flag가 자주 켜진다.
- temporal noise 자체는 `road_shaded`에 비하면 낮다.

### Mixed

| time | zoom | green_excess | temporal_noise | unstable_auto |
| --- | --- | ---: | ---: | --- |
| morning | 1x | 1.27 | 3.75 | false |
| morning | 3x | 1.18 | 7.33 | true |
| morning | 5x | 1.44 | 5.74 | true |
| morning | 7x | 2.14 | 5.95 | true |
| afternoon | 1x | 2.20 | 7.51 | false |
| afternoon | 3x | 2.12 | 11.11 | true |
| afternoon | 5x | 4.52 | 10.78 | true |
| afternoon | 7x | 4.10 | 8.92 | true |

관찰:
- 오후 mixed는 green excess가 오전보다 확실히 커진다.
- `afternoon/mixed/5x`, `afternoon/mixed/7x`는 색 편향과 detail instability가 함께 커진다.
- 하늘/건물/산처럼 밝기 범위가 넓은 장면에서는 clipping이 색 통계를 왜곡할 수 있다.

### Road Shaded

| time | zoom | green_excess | temporal_noise | mean_y_range | unstable_luma |
| --- | --- | ---: | ---: | ---: | --- |
| morning | 1x | 4.20 | 38.22 | 6.69 | false |
| morning | 3x | 4.51 | 47.77 | 26.24 | true |
| morning | 5x | 3.03 | 45.55 | 36.32 | true |
| morning | 7x | -0.99 | 29.25 | 18.29 | true |
| afternoon | 1x | 5.21 | 38.95 | 3.76 | false |
| afternoon | 3x | 4.60 | 44.91 | 5.39 | false |
| afternoon | 5x | 4.21 | 45.30 | 6.59 | false |
| afternoon | 7x | 0.42 | 41.56 | 14.23 | true |

관찰:
- `road_shaded`는 sunny outdoor에서 가장 중요한 hard case다.
- temporal std map은 도로 차선, 차량, 수목, 그림자 경계가 강하게 나타난다.
- `morning/road_shaded/3x`, `morning/road_shaded/5x`는 luma range가 매우 커서 AE 또는 장면 변화 영향이 강하다.
- `7x`는 green excess가 줄지만, luma/detail instability는 여전히 크다. 따라서 7x를 단순히 green cast worst case로 보면 안 된다.

---

## 4. Worst Burst Ranking

Temporal noise proxy 상위 sunny burst:

| burst | frames | temporal_noise | p90 | mean_y_range | green_excess_range | detail_rel_range |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `morning/sunny/road_shaded/3x` | 100 | 47.77 | 74.90 | 26.24 | 3.16 | 1.24 |
| `morning/sunny/road_shaded/5x` | 100 | 45.55 | 64.89 | 36.32 | 6.63 | 1.41 |
| `afternoon/sunny/road_shaded/5x` | 100 | 45.30 | 71.03 | 6.59 | 5.10 | 1.33 |
| `afternoon/sunny/road_shaded/3x` | 100 | 44.91 | 74.76 | 5.39 | 4.11 | 0.69 |
| `afternoon/sunny/road_shaded/7x` | 100 | 41.56 | 67.31 | 14.23 | 1.33 | 3.15 |
| `afternoon/sunny/road_shaded/1x` | 100 | 38.95 | 70.62 | 3.76 | 0.62 | 0.70 |
| `morning/sunny/road_shaded/1x` | 100 | 38.22 | 78.48 | 6.69 | 1.45 | 0.81 |
| `morning/sunny/road_shaded/7x` | 100 | 29.25 | 55.78 | 18.29 | 4.20 | 3.37 |

해석:
- 상위 8개가 모두 `road_shaded`다.
- 이 결과는 Phase 2에서 `road_shaded`를 별도 stress bucket으로 두는 근거가 된다.
- 하지만 이 값을 Gaussian sigma처럼 직접 매핑하면 안 된다. map을 보면 scene edge, moving object, shadow boundary가 강하게 반영되어 있기 때문이다.

---

## 5. Flat/Dark Baseline

Flat baseline:

| zoom | G/R | G/B | green_excess | shading_range | center_corner_y | chroma_mottle |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1x | 1.0000 | 1.0000 | 0.0008 | 0.2021 | 1.0322 | 0.0040 |
| 3x | 1.0000 | 1.0000 | 0.0000 | 0.0961 | 1.0408 | 0.0000 |
| 5x | 0.9996 | 0.9993 | -0.0807 | 0.0756 | 1.0474 | 0.0293 |
| 7x | 0.9969 | 0.9968 | -0.4669 | 0.2000 | 1.1541 | 0.1181 |

Flat 해석:
- 흰 벽 기준으로는 green cast가 강하지 않다.
- `1x`와 `7x`에서 shading range가 약 `0.20`으로 크다.
- `7x`는 center/corner Y ratio가 `1.1541`이고 chroma mottle도 가장 크다.
- 따라서 7x degradation은 green cast보다 shading/vignetting, detail attenuation, color non-uniformity 쪽을 더 중요하게 본다.

Dark baseline:

| burst | hot >= 25% | hot >= 50% | hot >= 75% | ratio >= 50% | temporal_noise |
| --- | ---: | ---: | ---: | ---: | ---: |
| `morning/sunny/dark/1x` | 3435 | 1096 | 266 | 0.000529 | 7.33 |
| `morning/sunny/dark/7x` | 3068 | 975 | 261 | 0.000470 | 7.23 |

Dark/flat overlap:
- dark 1x와 7x의 `>= 50%` persistence 후보 intersection은 433개다.
- dark 1x 기준 intersection ratio는 `0.395`, dark 7x 기준은 `0.444`다.
- flat과 dark union의 overlap은 거의 0에 가깝다.
- outdoor sunny burst에서는 `hot_persistence_count_50`이 사실상 0이다.

해석:
- 고정 hot-pixel 후보는 dark frame에서 확인된다.
- 하지만 outdoor 품질 문제의 주 원인은 hot pixel이 아니다.
- 학습에는 특정 sensor mask를 쓰지 말고, density 범위만 random hot-pixel augmentation으로 사용한다.

---

## 6. Clipping과 Color 해석 주의

Sunny outdoor에서 highlight clipping이 일부 관측된다.

주요 예:

| condition | y_clip_high | rgb_clip_high |
| --- | ---: | ---: |
| `morning/sunny/building/5x` | 0.0473 | 0.1053 |
| `afternoon/sunny/building/7x` | 0.0001 | 0.0974 |
| `afternoon/sunny/road_shaded/7x` | 0.0436 | 0.0838 |
| `afternoon/sunny/road_shaded/5x` | 0.0344 | 0.0718 |
| `afternoon/sunny/mixed/5x` | 0.0206 | 0.0697 |

해석:
- strong sunny 조건에서는 RGB channel saturation이 color ratio와 green excess를 흔들 수 있다.
- `mixed`와 `road_shaded`의 green excess는 실제 색 편향과 clipping/scene composition이 섞여 있다.
- degradation profile에서는 color cast를 전역 고정값으로 넣지 말고, highlight/shadow scene 조건과 함께 조건부로 넣는 것이 안전하다.

---

## 7. Outdoor Map 해석

`--save_outdoor_maps`로 outdoor burst의 다음 map이 추가됐다.

```text
results/mc_g105_analysis/all/maps/*_temporal_std.png
results/mc_g105_analysis/all/maps/*_hot_persistence.png
```

관찰:
- `road_shaded` temporal std map은 도로 차선, 차량, 수목, 그림자 경계, 밝은 반사 영역에서 강하게 나타난다.
- `building`은 spatial structure가 안정적이고 temporal map 강도가 낮다.
- `mixed`는 하늘/산/건물 경계와 밝은 영역에서 변화가 커진다.
- outdoor hot-persistence map은 실제 고정 결함보다는 거의 빈 결과에 가깝다.

해석 원칙:
- outdoor temporal map은 sensor noise map이 아니다.
- auto exposure/focus/detail drift, 움직임, 그림자 변화가 합쳐진 real-domain instability map으로 사용한다.
- denoise 학습에서는 이 값을 직접 sigma로 쓰기보다 hard-case sampling weight와 low-frequency/chroma/detail instability augmentation 근거로 사용한다.

---

## 8. Phase 2 Degradation 반영안

### Bucket 우선순위

| bucket | 반영 강도 | 근거 |
| --- | --- | --- |
| `road_shaded` | 가장 높음 | temporal/detail instability, green excess, chroma mottle이 동시에 큼 |
| `mixed afternoon sunny` | 높음 | green excess와 clipping 증가 |
| `building 7x` | 중간 | green excess 증가, high zoom detail instability |
| `flat 7x` | 중간 | shading, center/corner ratio, chroma non-uniformity |
| `dark hot pixel` | 낮음~중간 | dark에서만 persistence 확인, outdoor 주 원인은 아님 |

### 권장 profile 수정

| 항목 | 수정 방향 |
| --- | --- |
| Green cast | scene/time/zoom conditional. `road_shaded 1x~5x`, `afternoon mixed 5x~7x`, `building 7x`에서 강화 |
| Chroma mottle | `road_shaded`, `afternoon mixed`에서 강화. flat 기준으로는 약함 |
| Shading/vignetting | flat 1x/7x 근거로 1x와 7x에 별도 profile 적용 |
| Detail attenuation | 7x와 `road_shaded`에 강화. 단 green cast와 독립 축으로 둠 |
| Signal instability | `road_shaded` stress bucket에서만 강화. 일반 noise sigma로 직접 매핑하지 않음 |
| Hot pixels | density `0.0001 ~ 0.0006` 유지. 위치 mask 사용 금지 |
| Clipping/highlight | sunny mixed/road_shaded에 highlight clipping profile 추가 검토 |

### 샘플링 비율 제안

현재 Phase 2 초안은 유지하되, MC-G105 scene-specific 내부 비율을 다음처럼 나눈다.

| subset | 비율 | 목적 |
| --- | ---: | --- |
| generic/mild replay | 40~50% | clean/detail 보존 |
| normal outdoor MC-G105 | 20~25% | building, mixed mild 조건 |
| road_shaded hard case | 15~20% | shadow/edge/temporal instability 대응 |
| high-zoom profile | 10~15% | 7x shading/detail attenuation |
| hot-pixel stress | 2~5% | random sparse outlier 대응 |
| clean/near-clean | 5~10% | oversmoothing 방지 |

---

## 9. 결론

이번 sunny morning/afternoon 수집으로 Phase 2 설계 방향은 더 명확해졌다.

1. Outdoor 화질 문제의 중심은 fixed hot pixel이 아니라 `road_shaded` hard case다.
2. `road_shaded`는 green/chroma/detail/temporal proxy가 동시에 크므로 별도 stress bucket이 필요하다.
3. `afternoon sunny`는 `mixed`, `road_shaded`에서 green excess와 clipping이 커지는 경향이 있다.
4. 7x는 green cast worst case로만 보면 안 된다. shading, vignetting, detail attenuation, luma instability를 별도 축으로 둬야 한다.
5. Hot pixel은 dark frame에서 확인되지만 outdoor persistence가 거의 없으므로 random sparse augmentation 정도로 제한한다.
6. `temporal_noise`는 sensor noise coefficient가 아니라 real-domain instability proxy로 사용해야 한다.

다음 작업:
- `denoise_mc_g105_v3.yaml` 또는 Phase 2 config를 만들 때 `road_shaded`, `afternoon mixed`, `7x` 조건부 degradation profile을 분리한다.
- real probe 평가에서는 `road_shaded/3x`, `road_shaded/5x`, `road_shaded/7x`, `mixed/5x`, `building/7x`를 고정 비교 샘플로 둔다.
- 추가 수집이 가능하면 `afternoon/cloudy`를 채워 sunny/cloudy/time interaction을 완성한다.
