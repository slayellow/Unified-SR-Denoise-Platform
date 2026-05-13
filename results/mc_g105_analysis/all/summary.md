# MC-G105 Capture Analysis Summary

- Input: `/home/jshong/Code/unfied-sr-denoise-platform/results/mc_g105_capture_frames`
- Output: `/home/jshong/Code/unfied-sr-denoise-platform/results/mc_g105_analysis/all`
- Total images: 2700
- Total bursts: 42
- Scenes: building, dark, flat, mixed, road_shaded

## Scene / Zoom Counts

| scene_label | zoom_label | image_count |
| --- | --- | --- |
| building | 1x | 150 |
| building | 3x | 150 |
| building | 5x | 150 |
| building | 7x | 150 |
| dark | 1x | 50 |
| dark | 7x | 50 |
| flat | 1x | 50 |
| flat | 3x | 50 |
| flat | 5x | 50 |
| flat | 7x | 50 |
| mixed | 1x | 150 |
| mixed | 3x | 150 |
| mixed | 5x | 150 |
| mixed | 7x | 150 |
| road_shaded | 1x | 300 |
| road_shaded | 3x | 300 |
| road_shaded | 5x | 300 |
| road_shaded | 7x | 300 |

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
| afternoon | sunny | building | outdoor | 1x | 50 | 135.79 | 67.5630 | 1.0595 | 0.000000 | 4.0686 |
| afternoon | sunny | building | outdoor | 3x | 50 | 134.87 | 65.7205 | 1.7250 | 0.000000 | 4.7474 |
| afternoon | sunny | building | outdoor | 5x | 50 | 130.33 | 69.4589 | 2.5533 | 0.000000 | 5.9736 |
| afternoon | sunny | building | outdoor | 7x | 50 | 132.11 | 70.3973 | 3.9294 | 0.000000 | 4.6672 |
| afternoon | sunny | mixed | outdoor | 1x | 50 | 130.06 | 60.7892 | 2.1998 | 0.000000 | 7.5139 |
| afternoon | sunny | mixed | outdoor | 3x | 50 | 123.27 | 57.1995 | 2.1204 | 0.000000 | 11.1095 |
| afternoon | sunny | mixed | outdoor | 5x | 50 | 141.12 | 49.4244 | 4.5182 | 0.000000 | 10.7769 |
| afternoon | sunny | mixed | outdoor | 7x | 50 | 145.16 | 31.8228 | 4.1008 | 0.000000 | 8.9174 |
| afternoon | sunny | road_shaded | outdoor | 1x | 100 | 130.46 | 64.9752 | 5.2097 | 0.000000 | 38.9481 |
| afternoon | sunny | road_shaded | outdoor | 3x | 100 | 128.04 | 63.5296 | 4.6028 | 0.000000 | 44.9087 |
| afternoon | sunny | road_shaded | outdoor | 5x | 100 | 128.54 | 60.8670 | 4.2072 | 0.000000 | 45.3018 |
| afternoon | sunny | road_shaded | outdoor | 7x | 100 | 133.08 | 55.8606 | 0.421595 | 0.000000 | 41.5606 |
| morning | cloudy | building | outdoor | 1x | 50 | 134.74 | 55.7429 | 1.3628 | 0.000000 | 20.0249 |
| morning | cloudy | building | outdoor | 3x | 50 | 130.35 | 56.3954 | 1.6913 | 0.000000 | 3.2802 |
| morning | cloudy | building | outdoor | 5x | 50 | 129.50 | 58.7823 | 1.9375 | 0.000000 | 3.3906 |
| morning | cloudy | building | outdoor | 7x | 50 | 126.69 | 65.5976 | 2.4137 | 0.000000 | 4.4621 |
| morning | cloudy | mixed | outdoor | 1x | 50 | 135.26 | 69.0932 | 2.8215 | 0.000000 | 3.6278 |
| morning | cloudy | mixed | outdoor | 3x | 50 | 141.77 | 48.3993 | 2.5979 | 0.000000 | 5.9872 |
| morning | cloudy | mixed | outdoor | 5x | 50 | 141.55 | 42.3032 | 0.036839 | 0.000000 | 8.9693 |
| morning | cloudy | mixed | outdoor | 7x | 50 | 141.44 | 33.0603 | 0.617401 | 0.000110 | 12.0670 |

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

| burst_id | scene_label | zoom_label | unstable_luma | unstable_color | unstable_detail | burst_mean_y_range | burst_green_excess_range | burst_high_freq_rel_range |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| afternoon/sunny/building/3x | building | 3x | False | False | True | 1.8385 | 0.212746 | 0.608904 |
| afternoon/sunny/building/5x | building | 5x | False | False | True | 10.5249 | 0.347073 | 0.973888 |
| afternoon/sunny/building/7x | building | 7x | False | False | True | 3.7708 | 0.550949 | 1.4455 |
| afternoon/sunny/mixed/3x | mixed | 3x | False | False | True | 1.4159 | 0.615852 | 0.510736 |
| afternoon/sunny/mixed/5x | mixed | 5x | False | False | True | 6.1559 | 1.4730 | 0.928678 |
| afternoon/sunny/mixed/7x | mixed | 7x | False | False | True | 6.1539 | 0.839302 | 1.4651 |
| afternoon/sunny/road_shaded/1x | road_shaded | 1x | False | False | True | 3.7561 | 0.617043 | 0.702168 |
| afternoon/sunny/road_shaded/3x | road_shaded | 3x | False | False | True | 5.3860 | 4.1114 | 0.688692 |
| afternoon/sunny/road_shaded/5x | road_shaded | 5x | False | False | True | 6.5893 | 5.0986 | 1.3343 |
| afternoon/sunny/road_shaded/7x | road_shaded | 7x | True | False | True | 14.2334 | 1.3301 | 3.1467 |
| morning/cloudy/building/1x | building | 1x | False | False | True | 4.5084 | 0.470276 | 0.896666 |
| morning/cloudy/building/3x | building | 3x | False | False | True | 0.572891 | 0.253296 | 1.2600 |
| morning/cloudy/building/5x | building | 5x | False | False | True | 1.0906 | 0.329388 | 0.823165 |
| morning/cloudy/building/7x | building | 7x | False | False | True | 4.5140 | 0.826675 | 1.4817 |
| morning/cloudy/mixed/3x | mixed | 3x | False | False | True | 3.6859 | 0.481125 | 0.884281 |
| morning/cloudy/mixed/5x | mixed | 5x | False | False | True | 2.1222 | 0.347206 | 0.856656 |
| morning/cloudy/mixed/7x | mixed | 7x | False | False | True | 0.404953 | 0.617256 | 1.2416 |
| morning/cloudy/road_shaded/1x | road_shaded | 1x | False | False | True | 4.0784 | 0.637524 | 0.641879 |
| morning/cloudy/road_shaded/3x | road_shaded | 3x | False | False | True | 7.2717 | 5.2301 | 0.982265 |
| morning/cloudy/road_shaded/5x | road_shaded | 5x | True | False | True | 15.1002 | 6.1833 | 1.1622 |

## Hot-Pixel Candidate Overlap

| comparison_type | threshold | burst_a | burst_b | zoom_a | zoom_b | count_a | count_b | intersection_count | jaccard | intersection_ratio_a |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dark_vs_dark | 0.250000 | morning/sunny/dark/1x | morning/sunny/dark/7x | 1x | 7x | 3435 | 3068 | 1437 | 0.283656 | 0.418341 |
| dark_vs_dark | 0.500000 | morning/sunny/dark/1x | morning/sunny/dark/7x | 1x | 7x | 1096 | 975 | 433 | 0.264347 | 0.395073 |
| dark_vs_dark | 0.750000 | morning/sunny/dark/1x | morning/sunny/dark/7x | 1x | 7x | 266 | 261 | 96 | 0.222738 | 0.360902 |
| dark_vs_dark | 0.900000 | morning/sunny/dark/1x | morning/sunny/dark/7x | 1x | 7x | 107 | 92 | 29 | 0.170588 | 0.271028 |
| flat_vs_dark_union | 0.250000 | morning/sunny/flat/1x | dark_union | 1x | all | 444 | 5066 | 2 | 0.000363 | 0.004505 |
| flat_vs_dark_union | 0.250000 | morning/sunny/flat/3x | dark_union | 3x | all | 1377 | 5066 | 11 | 0.001710 | 0.007988 |
| flat_vs_dark_union | 0.250000 | morning/sunny/flat/5x | dark_union | 5x | all | 1596 | 5066 | 17 | 0.002558 | 0.010652 |
| flat_vs_dark_union | 0.250000 | morning/sunny/flat/7x | dark_union | 7x | all | 514 | 5066 | 4 | 0.000717 | 0.007782 |
| flat_vs_dark_union | 0.500000 | morning/sunny/flat/1x | dark_union | 1x | all | 83 | 1638 | 0 | 0.000000 | 0.000000 |
| flat_vs_dark_union | 0.500000 | morning/sunny/flat/3x | dark_union | 3x | all | 506 | 1638 | 1 | 0.000467 | 0.001976 |
| flat_vs_dark_union | 0.500000 | morning/sunny/flat/5x | dark_union | 5x | all | 586 | 1638 | 1 | 0.000450 | 0.001706 |
| flat_vs_dark_union | 0.500000 | morning/sunny/flat/7x | dark_union | 7x | all | 271 | 1638 | 1 | 0.000524 | 0.003690 |
| flat_vs_dark_union | 0.750000 | morning/sunny/flat/1x | dark_union | 1x | all | 35 | 431 | 0 | 0.000000 | 0.000000 |
| flat_vs_dark_union | 0.750000 | morning/sunny/flat/3x | dark_union | 3x | all | 259 | 431 | 0 | 0.000000 | 0.000000 |
| flat_vs_dark_union | 0.750000 | morning/sunny/flat/5x | dark_union | 5x | all | 321 | 431 | 1 | 0.001332 | 0.003115 |
| flat_vs_dark_union | 0.750000 | morning/sunny/flat/7x | dark_union | 7x | all | 169 | 431 | 1 | 0.001669 | 0.005917 |
| flat_vs_dark_union | 0.900000 | morning/sunny/flat/1x | dark_union | 1x | all | 16 | 170 | 0 | 0.000000 | 0.000000 |
| flat_vs_dark_union | 0.900000 | morning/sunny/flat/3x | dark_union | 3x | all | 175 | 170 | 0 | 0.000000 | 0.000000 |
| flat_vs_dark_union | 0.900000 | morning/sunny/flat/5x | dark_union | 5x | all | 209 | 170 | 0 | 0.000000 | 0.000000 |
| flat_vs_dark_union | 0.900000 | morning/sunny/flat/7x | dark_union | 7x | all | 108 | 170 | 1 | 0.003610 | 0.009259 |

## Scene Contrast Preview

| scene_label | zoom_label | image_count | green_excess_delta_vs_flat | chroma_mottle_std_delta_vs_flat | high_freq_energy_delta_vs_flat | hot_pixel_ratio_delta_vs_dark |
| --- | --- | --- | --- | --- | --- | --- |
| building | 1x | 150 | 1.1432 | 2.2007 | -644.87 | -0.001413 |
| building | 3x | 150 | 1.6766 | 0.982151 | -319.96 | nan |
| building | 5x | 150 | 2.1655 | 1.4872 | -429.24 | nan |
| building | 7x | 150 | 3.8636 | 1.2601 | -556.90 | -0.001375 |
| mixed | 1x | 150 | 2.1705 | 2.2708 | -715.76 | -0.001413 |
| mixed | 3x | 150 | 2.1232 | 1.3011 | -219.46 | nan |
| mixed | 5x | 150 | 1.5654 | 1.3828 | -120.95 | nan |
| mixed | 7x | 150 | 2.4789 | 1.3710 | -228.42 | -0.001375 |
| road_shaded | 1x | 300 | 5.1863 | 3.0010 | 487.16 | -0.001413 |
| road_shaded | 3x | 300 | 4.5662 | 2.3739 | 440.03 | nan |
| road_shaded | 5x | 300 | 3.3258 | 2.1724 | 27.7848 | nan |
| road_shaded | 7x | 300 | 0.790668 | 1.7484 | -371.40 | -0.001375 |

## Previous Run Delta

_No data._

## Interpretation Notes

- `flat`은 실내 흰 벽 baseline이므로 야외 색 편향 전체를 대표하지 않는다.
- `dark`는 렌즈 가림 baseline이며 Auto exposure 때문에 물리 dark current로 해석하지 않는다.
- `hot_persistence_ratio_50`은 같은 위치가 burst의 50% 이상에서 hot-pixel 후보로 반복된 비율이다.
- `unstable_luma`, `unstable_color`, `unstable_detail`은 밝기/색/디테일 변화를 분리해서 표시한다.
- hot-pixel overlap summary는 특정 센서 mask를 저장하지 않고 count/ratio만 저장한다.
- NV12 원본 plane이 아니라 저장된 이미지 분석이면 Y/Cr/Cb 지표는 RGB 변환 이후 proxy다.