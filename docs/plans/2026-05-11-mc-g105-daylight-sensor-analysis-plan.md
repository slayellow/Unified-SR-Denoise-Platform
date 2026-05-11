# MC-G105 주간 운용 조건 센서 분석 계획

작성일: 2026-05-11

목표: 드론 임무장비 EO Global Shutter 센서(MC-G105 계열)를 실제 운용 조건에 맞춰 분석한다. 현재 운용은 오전/오후, 맑음/흐림 조건을 중심으로 하고, 입력은 NV12 영상 스트리밍을 받아 저장하는 형태다. 따라서 목표는 RAW 기반 물리 센서 모델 추정보다는, 실제 배포 파이프라인에서 반복적으로 관측되는 green cast, hot pixel 후보, temporal noise, chroma/common noise, zoom별 detail 저하를 정량화하여 Denoise Degradation Pipeline에 반영하는 것이다.

---

## 1. 현재 가능한 조건

### 운용 환경

| 축 | 가능 여부 | 적용 |
| --- | --- | --- |
| 오전 맑음 | 가능 | 주요 운용 조건 |
| 오전 흐림 | 가능 | diffuse light / 저대비 조건 |
| 오후 맑음 | 가능 | 고조도/열 영향 가능 조건 |
| 오후 흐림 | 가능 | 낮은 daylight / gain 상승 가능 조건 |
| 야간 자연 장면 | 불필요 | 실제 임무 운용 분포가 아니면 제외 |

야간 자연 장면은 현재 운용 분포가 아니므로 필수 수집 대상에서 제외한다. 단, dark frame은 야간 장면과 다른 목적이므로 가능하면 별도로 수집한다.

### 촬영 장소/장면

야외 운용 장면은 회사 사무실 옥상에서 가능한 장면 기준으로 구성한다. 실내에서 촬영하는 flat/dark는 오전/오후, 맑음/흐림의 영향을 직접 받는 장면이 아니므로, weather/time 조건에 반복해서 펼치지 않고 calibration 기준 장면으로 분리한다.

| 장면 유형 | 실제 촬영 대상 | 위치 | 목적 |
| --- | --- | --- | --- |
| Uniform Flat | 흰 벽 | 실내 | 색 편향, shading, low-texture noise 관찰 |
| Dark Frame | 렌즈 가림 | 실내 | hot/stuck/dead pixel, black offset, fixed defect 후보 관찰 |
| Urban Detail | 옆 건물 | 옥상 | edge/detail, zoom별 detail 저하, texture 보존 확인 |
| Mixed Dynamic Range | 하늘/산/건물 | 옥상 | 노출 변화, highlight/shadow, 색 편향 확인 |
| Shaded Low-Texture | 도로, 그늘진 도로/벽면 | 옥상 | 어두운 주간 영역 noise, hot pixel 후보 확인 |
| Vegetation-Heavy | 촬영 어려움 | 옥상 | 필수에서는 제외. 가능하면 산/나무가 일부 포함된 mixed scene으로 대체 |

### Zoom

핵심 zoom은 다음 4개로 충분하다.

- 1x: wide baseline
- 3x: mid zoom
- 5x: high zoom 진입 구간
- 7x: worst-case high zoom

기존 결과에서 5x-7x 구간의 temporal/detail 불안정성이 더 컸으므로, 5x와 7x는 반드시 포함한다.

### 프레임 수

- 장면/조건/zoom별 20장 burst 촬영은 가능하다.
- 20장은 temporal noise와 hot-pixel persistence를 보기 위한 실용적 최소선으로 적절하다.
- 5장 burst는 quick check로는 가능하지만 profile 추정 근거로는 부족하다.

---

## 2. Flat Frame과 Dark Frame 의미

### Flat Frame

Flat frame은 화면 전체가 최대한 균일한 장면을 찍은 이미지다. 목적은 장면 texture를 줄이고 센서/렌즈/ISP에서 생기는 색 편향, vignetting, shading, low-frequency non-uniformity, chroma noise를 보기 위한 것이다.

가능한 촬영 방법:
- 흰 벽이 화면을 최대한 채우도록 촬영
- 벽에 강한 그림자, 반사, 글자, 패턴이 없는 위치를 선택
- 가능하면 focus가 벽 texture를 과하게 잡지 않도록 살짝 defocus 또는 충분히 균일한 면을 사용

주의:
- 흰 벽도 조명 색, 그림자, 벽 재질 영향을 받으므로 완전한 물리 flat은 아니다.
- 그래도 하늘보다 색이 중립적이고 시간 변화가 적어 green cast / low-texture noise 분석에는 더 적합하다.
- 벽면 texture가 선명하면 edge/high-frequency metric이 올라가므로 flat frame으로 쓸 patch는 texture가 적은 영역을 우선한다.
- 실내 촬영이므로 오전/오후, 맑음/흐림 조건에 반복 배치하지 않고 `morning/sunny/flat/...` 폴더를 calibration 기준 버킷으로 사용한다.

### Dark Frame

Dark frame은 센서에 빛이 들어오지 않도록 렌즈를 막고 찍은 이미지다. 목적은 hot/stuck/dead pixel, black offset, dark noise, DSNU, row/column fixed pattern 후보를 보기 위한 것이다.

가능한 촬영 방법:
- 렌즈 캡 또는 검은 천으로 완전히 가리고 촬영
- 가능하면 센서 전원 인가 후 몇 분 warm-up 뒤 촬영
- 세션 초반과 후반에 각각 촬영하면 온도/시간에 따른 hot pixel 변화 확인 가능

주의:
- Auto exposure 상태에서는 dark frame에서 카메라가 gain/exposure를 올릴 수 있다.
- 따라서 이 dark frame은 "운용 중 실제 dark scene"이라기보다 "고정 결함과 worst-case hot pixel 후보 확인" 용도다.
- 도로 그늘이나 어두운 벽은 dark frame이 아니다. 이는 shaded low-texture scene으로 별도 취급한다.
- 실내 촬영이므로 오전/오후, 맑음/흐림 조건에 반복 배치하지 않고 `morning/sunny/dark/...` 폴더를 calibration 기준 버킷으로 사용한다.

---

## 3. 메타데이터 제약과 분석 목표 변경

현재 조건:
- Exposure time 기록 불가
- Exposure는 Auto
- White Balance는 Auto
- Focus는 Auto
- R Gain, B Gain은 50 고정
- Zoom은 optical zoom만 사용
- Pixel format은 NV12 stream 저장 기반

이 제약 때문에 다음 물리량은 신뢰도 있게 추정하기 어렵다.

| 지표 | 상태 | 이유 |
| --- | --- | --- |
| absolute read noise | 어려움 | exposure/gain/black level 미기록 |
| shot noise coefficient | 어려움 | mean-variance curve에 필요한 gain/exposure 축 부족 |
| 물리적 PRNU/DSNU 정량값 | 제한적 | RAW/black level/flat calibration 부족 |
| Auto WB 변화량 | 어려움 | 실제 WB gain metadata 미기록 |
| focus별 MTF | 어려움 | Auto focus 위치 metadata 미기록 |

따라서 분석 목표를 다음처럼 바꾼다.

기존 목표:
- 센서 물리 파라미터를 정밀 추정

수정 목표:
- 실제 NV12 운용 출력에서 반복적으로 보이는 경험적 열화 profile 추정
- Denoise degradation range를 정하기 위한 robust proxy metric 산출
- Auto exposure/focus/WB 변동이 큰 burst를 걸러내거나 별도 표시

---

## 4. 유지할 지표

현재 분석 스크립트의 지표 중 다음은 계속 유지한다.

| 지표 | 유지 여부 | 용도 |
| --- | --- | --- |
| `mean_r/g/b` | 유지 | green dominance, channel bias |
| `std_r/g/b` | 유지 | 채널별 contrast/noise proxy |
| `mean_y`, `std_y` | 유지 | 밝기/대비 상태 |
| `tint_rg = R-G` | 유지 | green cast 방향성 |
| `tint_bg = B-G` | 유지 | green/blue cast 방향성 |
| `edge_density` | 유지 | detail/edge 보존 proxy |
| `high_freq_energy` | 유지 | zoom별 detail collapse proxy |
| `dark_region_ratio` | 유지 | 어두운 영역 비중 |
| `dark_noise_std` | 유지하되 해석 변경 | true dark noise가 아니라 shaded/dark scene noise proxy |
| `hot_pixel_ratio/count` | 유지하되 개선 필요 | single-frame 후보가 아니라 burst persistence 기준으로 개선 |
| `temporal_noise` | 유지하되 조건부 사용 | auto/focus 흔들림이 작은 burst에서만 신뢰 |

---

## 5. 추가하거나 변경할 지표

### 5-1. Auto 안정성 지표

Auto exposure/focus/WB를 사용하므로, burst 내부에서 카메라 제어가 흔들렸는지 확인해야 한다.

추가 지표:
- burst mean Y std
- burst mean Y max-min
- first frame vs last frame mean Y drift
- frame-to-frame global color drift
- edge/high-frequency energy drift

사용:
- mean Y drift가 큰 burst는 temporal noise 계산에서 제외하거나 `unstable_auto` flag 부여
- focus hunting이 의심되는 burst는 detail metric에서 제외

### 5-2. Green dominance 지표

R/B gain이 50 고정이라면 green dominance가 센서/ISP 출력 특성으로 반복될 가능성이 있다.

추가 지표:
- `G/R`
- `G/B`
- `(G - (R+B)/2)`
- flat/white-wall 영역 기준 channel ratio

사용:
- `color_cast.rg_shift`, `color_cast.bg_shift` 범위 설정
- clear/cloudy, 오전/오후 조건별 green cast 변화 확인

### 5-3. Burst Persistence Hot Pixel

현재 single-frame local residual 방식은 작은 밝은 표적/반사/edge를 hot pixel로 오탐할 수 있다.

추가 지표:
- 같은 pixel 위치가 20장 중 몇 장에서 outlier인지
- persistence ratio
- dark frame persistence map
- flat/white-wall frame hot pixel 후보와 자연 장면 후보의 교집합

사용:
- 반복 위치: fixed hot/stuck pixel 후보
- 반복되지 않는 위치: random bright outlier 또는 scene highlight 가능성
- degradation의 `hot_pixels.density_min/max`는 persistence 기반 통계로 조정

### 5-4. NV12/색 변환 관련 지표

NV12 stream을 저장하므로 색공간 변환이 분석값에 영향을 줄 수 있다.

추가 기록:
- NV12를 RGB/PNG로 변환한 방식
- BT.601/BT.709 여부
- full range / limited range 여부
- 저장 시 압축 여부

가능하면 Y plane 기반 지표도 별도 계산한다.

추가 지표:
- Y plane temporal noise
- UV plane low-frequency variance
- UV plane block/chroma mottling

### 5-5. Clipping 지표

Auto exposure 조건에서는 sky/highlight clipping이 green/tint/noise 통계를 왜곡할 수 있다.

추가 지표:
- pixel value near 0 비율
- pixel value near 255 비율
- channel별 saturation ratio

사용:
- clipping이 큰 frame은 color/noise 통계에서 제외하거나 별도 group으로 분리

---

## 6. 수정된 수집 프로토콜

### 기본 수집 방향

수집 데이터는 두 종류로 나눈다.

| 구분 | 장면 | Time/Weather 적용 | 목적 |
| --- | --- | --- | --- |
| 야외 운용 장면 | building, mixed, road_shaded | 오전/오후, 맑음/흐림 전체 적용 | 실제 드론 운용 조건의 detail/noise/color/auto 변화 확인 |
| 실내 calibration 장면 | flat, dark | `morning/sunny` 기준 버킷에만 저장 | white-wall flat과 lens-covered dark의 baseline 분석 |

기본 operational frame 수:

```text
야외 운용: 4 conditions x 3 scenes x 4 zooms x 20 frames = 960 frames
실내 flat: 1 condition x 1 scene x 4 zooms x 20 frames = 80 frames
실내 dark: 1 condition x 1 scene x 2 zooms x 20 frames = 40 frames
최소 합계 = 1080 frames
```

Vegetation-heavy는 필수에서 제외한다. 가능하면 mixed scene에 산/수목이 일부 포함되도록 한다.

### 폴더 기반 metadata 규칙

별도 metadata 파일 없이, 폴더 구조를 metadata로 사용한다. 순차 저장된 frame은 같은 burst로 간주한다.

기본 구조:

```text
mc_g105_daylight_capture/
  {time}/{weather}/{scene}/{zoom}/{frame_index}.png
```

예시:

```text
mc_g105_daylight_capture/morning/sunny/flat/1x/0000.png
mc_g105_daylight_capture/morning/sunny/dark/7x/0000.png
mc_g105_daylight_capture/afternoon/cloudy/building/7x/0019.png
```

폴더명 규칙:

| 축 | 값 | 의미 |
| --- | --- | --- |
| `time` | `morning`, `afternoon` | 오전/오후 |
| `weather` | `sunny`, `cloudy` | 맑음/흐림 |
| `scene` | `flat`, `building`, `mixed`, `road_shaded`, `dark` | 촬영 장면 |
| `zoom` | `1x`, `3x`, `5x`, `7x` | optical zoom |
| `frame_index` | `0000.png` ... `0019.png` | burst 내부 순서 |

이 구조만 있어도 time/weather/scene/zoom/frame index는 복원 가능하다. 현재 목적에서는 별도 metadata 파일 없이 이 폴더 구조를 기본 metadata로 사용한다. 다만 exposure time, gain, focus, white balance 등은 기록되지 않으므로 분석은 물리 센서 파라미터가 아니라 운용 출력 기반 proxy로 제한된다.

`flat`과 `dark`는 실내 calibration 장면이므로 실제 날씨/시간 조건을 의미하지 않는다. 폴더 구조를 단순하게 유지하기 위해 `morning/sunny` 아래에 넣는 기준 버킷으로만 사용한다.

### 통합 수집 표

| Time | Weather | Scene | 실제 대상 | Zoom | Frames | Path 예시 |
| --- | --- | --- | --- | --- | ---: | --- |
| morning | sunny | flat | 실내 흰 벽 | 1x, 3x, 5x, 7x | 각 20 이상 | `morning/sunny/flat/1x/0000.png` |
| morning | sunny | dark | 실내 렌즈 가림 | 1x, 7x | 각 20 이상 | `morning/sunny/dark/7x/0000.png` |
| morning | sunny | building | 옆 건물 | 1x, 3x, 5x, 7x | 각 20 | `morning/sunny/building/3x/0000.png` |
| morning | sunny | mixed | 하늘/산/건물 | 1x, 3x, 5x, 7x | 각 20 | `morning/sunny/mixed/5x/0000.png` |
| morning | sunny | road_shaded | 도로/그늘진 저텍스처 영역 | 1x, 3x, 5x, 7x | 각 20 | `morning/sunny/road_shaded/7x/0000.png` |
| morning | cloudy | building | 옆 건물 | 1x, 3x, 5x, 7x | 각 20 | `morning/cloudy/building/3x/0000.png` |
| morning | cloudy | mixed | 하늘/산/건물 | 1x, 3x, 5x, 7x | 각 20 | `morning/cloudy/mixed/5x/0000.png` |
| morning | cloudy | road_shaded | 도로/그늘진 저텍스처 영역 | 1x, 3x, 5x, 7x | 각 20 | `morning/cloudy/road_shaded/7x/0000.png` |
| afternoon | sunny | building | 옆 건물 | 1x, 3x, 5x, 7x | 각 20 | `afternoon/sunny/building/3x/0000.png` |
| afternoon | sunny | mixed | 하늘/산/건물 | 1x, 3x, 5x, 7x | 각 20 | `afternoon/sunny/mixed/5x/0000.png` |
| afternoon | sunny | road_shaded | 도로/그늘진 저텍스처 영역 | 1x, 3x, 5x, 7x | 각 20 | `afternoon/sunny/road_shaded/7x/0000.png` |
| afternoon | cloudy | building | 옆 건물 | 1x, 3x, 5x, 7x | 각 20 | `afternoon/cloudy/building/3x/0000.png` |
| afternoon | cloudy | mixed | 하늘/산/건물 | 1x, 3x, 5x, 7x | 각 20 | `afternoon/cloudy/mixed/5x/0000.png` |
| afternoon | cloudy | road_shaded | 도로/그늘진 저텍스처 영역 | 1x, 3x, 5x, 7x | 각 20 | `afternoon/cloudy/road_shaded/7x/0000.png` |

### 실내 Flat/Dark만 먼저 촬영했을 때의 1차 분석

내일 실내에서 `flat`과 `dark`만 먼저 촬영해도 1차 분석은 가능하다.

| 데이터 | 가능한 분석 | 주의 |
| --- | --- | --- |
| `morning/sunny/flat/{1x,3x,5x,7x}` | green dominance, channel ratio, shading/vignetting proxy, flat-region chroma/noise proxy, zoom별 균일면 변화 | 실내 조명 색과 벽 재질 영향이 섞이므로 야외 색 편향 전체를 대표하지는 않음 |
| `morning/sunny/dark/{1x,7x}` | hot/stuck pixel persistence, fixed defect 후보, black output, dark-frame outlier map | Auto exposure가 dark 상황에서 gain/exposure를 올릴 수 있으므로 물리 dark current로 해석하지 않음 |

이 1차 분석은 다음 용도로 충분하다.

1. 폴더 기반 metadata parser 검증
2. flat/white-wall 분석 모드 검증
3. dark-frame hot pixel persistence 검증
4. MC-G105 degradation profile의 초기 범위 설정

다만 야외 운용 장면을 찍기 전에는 오전/오후, 맑음/흐림에 따른 auto exposure/WB/focus 변화, mixed scene clipping, building detail 유지, road_shaded noise는 판단할 수 없다.

---

## 7. 분석 지표와 degradation mapping

| 관측 지표 | Degradation 반영 위치 | 주의 |
| --- | --- | --- |
| `R-G`, `B-G`, `G/R`, `G/B` | `stage2.color_cast` | 자연 장면 색과 섞이므로 흰 벽 flat 중심으로 robust percentile 사용 |
| burst temporal noise | `stage1.gaussian_noise`, `stage1.poisson_noise`, `stage2.common_noise` | auto instability가 큰 burst 제외 |
| UV/chroma variance | `stage2.chroma_noise` | NV12 변환 방식 고정 필요 |
| hot pixel persistence | `stage2.hot_pixels` | random outlier와 fixed defect 분리 |
| edge/high_freq 감소 | `stage2.detail_attenuation`, `stage2.blur` | Auto focus hunting 여부 확인 |
| brightness/color drift | `stage2.signal_instability` | auto exposure/WB 변화와 sensor noise를 구분 |
| clipping ratio | training sample filter 또는 highlight profile | clipping frame은 tint/noise 통계 왜곡 가능 |

---

## 8. 현재 분석 스크립트 수정 필요사항

`examples/analysis/analyze_mc_g105_sensor_capture.py`는 다음 분석을 수행하도록 확장한다.

1. 폴더 구조 metadata parser 추가
   - `{time}/{weather}/{scene}/{zoom}/{frame_index}.png`에서 condition, weather, scene, zoom, frame index 복원
   - manifest CSV는 선택 사항으로만 둔다.

2. burst 안정성 평가 추가
   - mean Y drift
   - color drift
   - focus/detail drift

3. hot pixel persistence 분석 추가
   - 20장 burst 내 반복 위치 계산
   - dark frame 기반 defect candidate map 생성

4. flat/white-wall 분석 모드 추가
   - shading map
   - center/corner color ratio
   - G/R, G/B map

5. NV12 직접 분석 가능성 검토
   - Y plane noise
   - UV plane chroma noise
   - RGB 변환 방식 기록

6. confidence interval 출력
   - 평균만이 아니라 median, p10, p90, std, sample count 출력

### 실행 방법

현재 환경에서는 `python`이 2.7을 가리킬 수 있으므로 `python3`로 실행한다.

```bash
python3 examples/analysis/analyze_mc_g105_sensor_capture.py \
  --input_dir /path/to/mc_g105_daylight_capture \
  --output_dir results/mc_g105_analysis/flat_dark_first
```

내일 실내 `flat/dark`만 먼저 수집한 경우에도 같은 명령으로 실행한다. 이후 야외 데이터까지 같은 root 아래에 추가한 뒤 다시 실행하면 된다.

이전 실행 결과와 같은 group key 기준 차이를 보고 싶으면 이전 `group_summary.csv`를 넘긴다.

```bash
python3 examples/analysis/analyze_mc_g105_sensor_capture.py \
  --input_dir /path/to/mc_g105_daylight_capture \
  --output_dir results/mc_g105_analysis/full_daylight \
  --previous_group_summary results/mc_g105_analysis/flat_dark_first/group_summary.csv
```

### 출력 파일

| 파일/폴더 | 내용 |
| --- | --- |
| `per_image_metrics.csv` | 이미지별 RGB/Y/CrCb, green dominance, clipping, edge/detail, dark-region, single-frame hot-pixel 후보 |
| `burst_metrics.csv` | burst별 temporal noise, auto 안정성, hot-pixel persistence, map 경로 |
| `group_summary.csv` | time/weather/scene/zoom별 mean, median, p10, p90, std |
| `scene_contrast_summary.csv` | 야외 scene이 추가된 경우 flat/dark baseline 대비 차이 |
| `previous_run_delta.csv` | `--previous_group_summary` 사용 시 이전 실행 대비 차이 |
| `summary.md` | 사람이 빠르게 읽기 위한 요약 report |
| `maps/` | flat/dark의 temporal std, hot persistence, fixed hot mask, flat shading, green excess heatmap |
| `plots/` | scene/zoom별 green excess, chroma mottle, high-frequency, hot-pixel, temporal noise plot |

---

## 9. 결론

현재 제약에서도 MC-G105 분석은 충분히 진행 가능하다. 다만 분석의 성격은 RAW 센서 물리 모델 추정이 아니라, 실제 NV12 운용 출력 기반의 empirical degradation profile 추정으로 정의해야 한다.

밤 자연 장면은 현재 운용 조건에서는 제외해도 된다. 대신 다음은 유지하거나 추가하는 것이 좋다.

1. 오전/오후, 맑음/흐림 4조건
2. 야외 운용 장면은 building, mixed, road_shaded 3장면
3. 실내 calibration 장면은 `morning/sunny` 기준 버킷의 flat, dark
4. 1x, 3x, 5x, 7x zoom. 단 dark는 1x, 7x 우선
5. 야외 운용 장면은 조건/장면/zoom별 20장 burst
6. 실내 flat/dark만 먼저 촬영해도 green dominance, shading proxy, hot-pixel persistence의 1차 분석 가능
7. burst 안정성, green dominance, hot-pixel persistence, NV12 color/chroma 지표 추가

분석 지표는 대부분 유지 가능하지만, `dark_noise_std`, `temporal_noise`, `hot_pixel_count`의 해석은 바뀐다. 특히 exposure/gain metadata가 없으므로 물리 noise 계수로 해석하지 말고, 운용 출력에서의 상대적 noise/defect proxy로 사용해야 한다.
