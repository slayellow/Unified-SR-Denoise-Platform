# MC-G105 Capture Analysis Summary

- Input: `/home/jshong/Code/unfied-sr-denoise-platform/results/mc_g105_capture_frames`
- Output: `/home/jshong/Code/unfied-sr-denoise-platform/results/mc_g105_analysis/flat_dark_first`
- Total images: 300
- Total bursts: 6
- Scenes: dark, flat

## Scene / Zoom Counts

| scene_label | zoom_label | image_count |
| --- | --- | --- |
| dark | 1x | 50 |
| dark | 7x | 50 |
| flat | 1x | 50 |
| flat | 3x | 50 |
| flat | 5x | 50 |
| flat | 7x | 50 |

## Flat Baseline

| time_label | weather_label | scene_label | zoom_label | image_count | g_over_r_mean | g_over_b_mean | green_excess_mean | flat_shading_range_norm_mean | chroma_mottle_std_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| morning | sunny | flat | 1x | 50 | 1.000000 | 1.0000 | 0.000776 | 0.202109 | 0.004005 |
| morning | sunny | flat | 3x | 50 | 1.000000 | 1.000000 | 0.000000 | 0.096069 | 0.000000 |
| morning | sunny | flat | 5x | 50 | 0.999620 | 0.999286 | -0.080749 | 0.075555 | 0.029260 |
| morning | sunny | flat | 7x | 50 | 0.996882 | 0.996829 | -0.466871 | 0.199966 | 0.118121 |

## Dark / Hot-Pixel Persistence

| burst_id | zoom_label | frame_count | hot_persistence_count_25 | hot_persistence_count_50 | hot_persistence_count_75 | hot_persistence_ratio_50 | burst_temporal_noise_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| morning/sunny/dark/1x | 1x | 50 | 3435 | 1096 | 266 | 0.000529 | 7.3283 |
| morning/sunny/dark/7x | 7x | 50 | 3068 | 975 | 261 | 0.000470 | 7.2272 |

## Group Summary Preview

| time_label | weather_label | scene_label | scene_kind | zoom_label | image_count | mean_y_mean | std_y_mean | green_excess_mean | hot_pixel_ratio_mean | burst_temporal_noise_mean_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| morning | sunny | dark | calibration_dark | 1x | 50 | 17.8619 | 12.4157 | -4.8619 | 0.001472 | 7.3283 |
| morning | sunny | dark | calibration_dark | 7x | 50 | 17.4156 | 12.1602 | -2.9144 | 0.001420 | 7.2272 |
| morning | sunny | flat | calibration_flat | 1x | 50 | 150.95 | 14.7479 | 0.000776 | 0.001024 | 9.5032 |
| morning | sunny | flat | calibration_flat | 3x | 50 | 147.28 | 8.5590 | 0.000000 | 0.001877 | 5.9543 |
| morning | sunny | flat | calibration_flat | 5x | 50 | 147.42 | 8.0335 | -0.080749 | 0.002008 | 5.9750 |
| morning | sunny | flat | calibration_flat | 7x | 50 | 148.31 | 12.5561 | -0.466871 | 0.000441 | 6.1099 |

## Top Single-Frame Hot-Pixel Candidates

| relative_path | scene_label | zoom_label | hot_pixel_count | hot_pixel_ratio | dark_noise_std | green_excess |
| --- | --- | --- | --- | --- | --- | --- |
| morning/sunny/flat/5x/frame_1678.png | flat | 5x | 4520 | 0.002180 | 0.000000 | -0.134041 |
| morning/sunny/flat/5x/frame_1698.png | flat | 5x | 4499 | 0.002170 | 0.000000 | 0.001450 |
| morning/sunny/flat/5x/frame_1656.png | flat | 5x | 4490 | 0.002165 | 0.000000 | -0.143196 |
| morning/sunny/flat/5x/frame_1682.png | flat | 5x | 4460 | 0.002151 | 0.000000 | 0.001480 |
| morning/sunny/flat/5x/frame_1666.png | flat | 5x | 4460 | 0.002151 | 0.000000 | -0.137894 |
| morning/sunny/flat/5x/frame_1687.png | flat | 5x | 4414 | 0.002129 | 0.000000 | 0.001480 |
| morning/sunny/flat/5x/frame_1676.png | flat | 5x | 4412 | 0.002128 | 0.000000 | -0.137726 |
| morning/sunny/flat/5x/frame_1662.png | flat | 5x | 4412 | 0.002128 | 0.000000 | -0.142052 |

## Auto-Stability Flags

| burst_id | scene_label | zoom_label | burst_mean_y_range | burst_green_excess_range | burst_high_freq_rel_range |
| --- | --- | --- | --- | --- | --- |
| morning/sunny/dark/1x | dark | 1x | 8.1273 | 5.2365 | 2.6064 |
| morning/sunny/dark/7x | dark | 7x | 10.1624 | 2.1019 | 2.3001 |
| morning/sunny/flat/1x | flat | 1x | 2.0467 | 0.004173 | 1.2625 |
| morning/sunny/flat/3x | flat | 3x | 0.961319 | 0.000000 | 0.983465 |
| morning/sunny/flat/5x | flat | 5x | 1.9302 | 0.152870 | 0.937829 |
| morning/sunny/flat/7x | flat | 7x | 1.0567 | 0.809921 | 0.945445 |

## Scene Contrast Preview

_No data._

## Previous Run Delta

_No data._

## Interpretation Notes

- `flat`은 실내 흰 벽 baseline이므로 야외 색 편향 전체를 대표하지 않는다.
- `dark`는 렌즈 가림 baseline이며 Auto exposure 때문에 물리 dark current로 해석하지 않는다.
- `hot_persistence_ratio_50`은 같은 위치가 burst의 50% 이상에서 hot-pixel 후보로 반복된 비율이다.
- `burst_mean_y_range`, `burst_green_excess_range`, `burst_high_freq_rel_range`가 크면 temporal noise 해석에 Auto exposure/WB/focus 변화가 섞였을 수 있다.
- NV12 원본 plane이 아니라 저장된 이미지 분석이면 Y/Cr/Cb 지표는 RGB 변환 이후 proxy다.