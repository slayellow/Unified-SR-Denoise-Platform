# SR x4 Real Sensor 검증 - MC-G105 320x180 Crop

## 기술 요약

- 입력: `results/260602_mc_g105_probe_42/raw`
- 샘플 수: `42`장
- Crop 기준: 원본 `1920x1080` frame의 중앙 `320x180`
- 출력 기준: Bicubic x4와 Deploy GPU x4 모두 `1280x720`
- Deploy checkpoint: `checkpoints/csuav_deploy/finetune_svfocussrnet_eo_sr_x4_dim32_epoch100_bs_16_ga_2_lr_5e-5/best.pth`
- 실제 센서 입력에는 HR/GT가 없으므로 PSNR/SSIM은 계산하지 않고, no-reference IQA와 Deploy-vs-Bicubic proxy 지표 중심으로 판단한다.

## 전체 경향

| 지표 | 평균 |
|---|---:|
| Deploy / Bicubic sharpness ratio | 2.331 |
| Deploy / Bicubic edge density ratio | 1.951 |
| Deploy / Bicubic high-frequency ratio | 0.892 |
| Low-frequency luma MAE vs Bicubic | 0.453 |
| Chroma MAE vs Bicubic | 0.675 |
| 평균 failure risk score | 1.333 |

## No-Reference IQA 요약

NIQE, BRISQUE, PIQE는 `pyiqa` 구현으로 계산했다. 세 지표 모두 낮을수록 좋은 방향으로 해석한다.

| 지표 | Bicubic 평균 | Deploy 평균 | Deploy - Bicubic |
|---|---:|---:|---:|
| NIQE | 8.859 | 8.185 | -0.674 |
| BRISQUE | 67.223 | 48.718 | -18.505 |
| PIQE | 79.459 | 60.930 | -18.529 |

## Failure Signal Count

| failure_signal | frame_count |
| --- | --- |
| edge_oversharpening_or_ringing | 35 |
| noise_or_false_texture_amplification | 11 |
| large_local_deviation_from_bicubic | 10 |
| no_strong_failure_signal | 7 |

![Failure counts](metrics/failure_counts.png)

## Top Risk Frame

| relative_path | scene | zoom | risk_score | failure_labels | hf_energy_ratio | edge_density_ratio | ringing_ratio | deploy_niqe | deploy_brisque | deploy_piqe | deploy_minus_bicubic_niqe | deploy_minus_bicubic_brisque | deploy_minus_bicubic_piqe | deploy_minus_bicubic_luma_mean | deploy_bicubic_lowfreq_luma_mae | comparison_path |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| afternoon/sunny/mixed/1x/frame_0125.png | mixed | 1x | 3 | noise_or_false_texture_amplification;edge_oversharpening_or_ringing;large_local_deviation_from_bicubic | 1.205 | 1.065 | 1.196 | 6.753 | 35.941 | 77.903 | -1.012 | -22.270 | -10.764 | 0.545 | 0.590 | afternoon/sunny/mixed/1x/frame_0125.compare.jpg |
| afternoon/sunny/road_shaded/1x/frame_1725.png | road_shaded | 1x | 3 | noise_or_false_texture_amplification;edge_oversharpening_or_ringing;large_local_deviation_from_bicubic | 1.246 | 1.308 | 1.132 | 6.801 | 56.597 | 77.088 | -0.987 | -8.198 | -8.327 | 0.181 | 0.639 | afternoon/sunny/road_shaded/1x/frame_1725.compare.jpg |
| afternoon/sunny/road_shaded/5x/frame_2175.png | road_shaded | 5x | 3 | noise_or_false_texture_amplification;edge_oversharpening_or_ringing;large_local_deviation_from_bicubic | 1.301 | 1.668 | 1.213 | 7.359 | 56.669 | 80.598 | -1.315 | -4.972 | -6.277 | -0.203 | 0.358 | afternoon/sunny/road_shaded/5x/frame_2175.compare.jpg |
| morning/sunny/road_shaded/1x/frame_1925.png | road_shaded | 1x | 3 | noise_or_false_texture_amplification;edge_oversharpening_or_ringing;large_local_deviation_from_bicubic | 1.182 | 1.193 | 1.139 | 6.636 | 46.813 | 75.543 | -1.548 | -18.404 | -11.303 | 0.148 | 0.577 | morning/sunny/road_shaded/1x/frame_1925.compare.jpg |
| morning/sunny/road_shaded/5x/frame_2275.png | road_shaded | 5x | 3 | noise_or_false_texture_amplification;edge_oversharpening_or_ringing;large_local_deviation_from_bicubic | 1.305 | 1.721 | 1.191 | 7.160 | 52.102 | 78.473 | -1.120 | -9.559 | -4.444 | -0.082 | 0.261 | morning/sunny/road_shaded/5x/frame_2275.compare.jpg |
| morning/cloudy/building/1x/frame_4375.png | building | 1x | 3 | noise_or_false_texture_amplification;edge_oversharpening_or_ringing;large_local_deviation_from_bicubic | 1.271 | 1.560 | 1.210 | 7.199 | 42.879 | 70.509 | -1.307 | -23.010 | -8.233 | 0.246 | 0.341 | morning/cloudy/building/1x/frame_4375.compare.jpg |
| morning/cloudy/road_shaded/1x/frame_2275.png | road_shaded | 1x | 3 | noise_or_false_texture_amplification;edge_oversharpening_or_ringing;large_local_deviation_from_bicubic | 1.223 | 1.530 | 1.034 | 6.846 | 54.054 | 76.534 | -1.310 | -11.994 | -9.255 | 0.159 | 0.490 | morning/cloudy/road_shaded/1x/frame_2275.compare.jpg |
| morning/cloudy/road_shaded/5x/frame_2775.png | road_shaded | 5x | 3 | noise_or_false_texture_amplification;edge_oversharpening_or_ringing;large_local_deviation_from_bicubic | 1.233 | 1.432 | 1.149 | 6.987 | 43.827 | 76.861 | -1.199 | -19.211 | -8.960 | 0.128 | 0.325 | morning/cloudy/road_shaded/5x/frame_2775.compare.jpg |
| morning/sunny/building/1x/frame_4375.png | building | 1x | 3 | noise_or_false_texture_amplification;edge_oversharpening_or_ringing;large_local_deviation_from_bicubic | 1.246 | 1.322 | 1.216 | 6.981 | 37.090 | 71.033 | -0.985 | -24.171 | -14.535 | 0.278 | 0.397 | morning/sunny/building/1x/frame_4375.compare.jpg |
| afternoon/sunny/building/1x/frame_3825.png | building | 1x | 3 | noise_or_false_texture_amplification;edge_oversharpening_or_ringing;large_local_deviation_from_bicubic | 1.270 | 2.102 | 1.291 | 7.366 | 40.801 | 71.067 | -0.741 | -24.262 | -12.659 | 0.211 | 0.299 | afternoon/sunny/building/1x/frame_3825.compare.jpg |
| afternoon/sunny/building/3x/frame_3925.png | building | 3x | 2 | noise_or_false_texture_amplification;edge_oversharpening_or_ringing | 1.153 | 1.295 | 1.337 | 8.298 | 39.352 | 60.118 | -0.176 | -31.829 | -29.833 | 0.198 | 0.257 | afternoon/sunny/building/3x/frame_3925.compare.jpg |
| afternoon/sunny/road_shaded/3x/frame_2925.png | road_shaded | 3x | 1 | edge_oversharpening_or_ringing | 0.902 | 1.120 | 1.177 | 8.265 | 54.209 | 65.004 | -0.380 | -15.327 | -15.299 | 0.756 | 0.758 | afternoon/sunny/road_shaded/3x/frame_2925.compare.jpg |

## Scene별 요약

| time_of_day | weather | scene | sharpness_ratio | edge_density_ratio | hf_energy_ratio | ringing_ratio | local_contrast_ratio | deploy_minus_bicubic_luma_mean | deploy_bicubic_lowfreq_luma_mae | deploy_bicubic_chroma_mae | bicubic_niqe | deploy_niqe | deploy_minus_bicubic_niqe | bicubic_brisque | deploy_brisque | deploy_minus_bicubic_brisque | bicubic_piqe | deploy_piqe | deploy_minus_bicubic_piqe | risk_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| afternoon | sunny | building | 3.064 | 1.699 | 1.001 | 1.314 | 0.991 | 0.208 | 0.278 | 0.526 | 8.518 | 8.419 | -0.099 | 67.996 | 40.695 | -27.301 | 83.487 | 54.602 | -28.885 | 1.750 |
| afternoon | sunny | mixed | 2.168 | 1.514 | 0.895 | 1.015 | 0.939 | 0.411 | 0.429 | 0.732 | 8.773 | 7.774 | -0.998 | 63.923 | 45.941 | -17.982 | 82.705 | 65.999 | -16.706 | 1.250 |
| afternoon | sunny | road_shaded | 2.538 | 1.523 | 1.034 | 1.173 | 1.015 | 0.353 | 0.610 | 0.868 | 8.414 | 7.480 | -0.934 | 64.870 | 55.212 | -9.658 | 82.284 | 70.434 | -11.849 | 2.000 |
| morning | cloudy | building | 2.791 | 1.560 | 1.009 | 1.095 | 0.983 | 0.234 | 0.284 | 0.503 | 8.894 | 8.553 | -0.341 | 66.413 | 42.950 | -23.463 | 70.817 | 52.817 | -18.000 | 1.500 |
| morning | cloudy | mixed | 2.011 | 3.345 | 0.849 | 1.151 | 0.956 | 0.403 | 0.430 | 0.694 | 8.896 | 7.802 | -1.094 | 66.921 | 54.033 | -12.888 | 80.742 | 70.516 | -10.226 | 1.000 |
| morning | cloudy | road_shaded | 2.509 | 1.427 | 1.015 | 1.054 | 1.011 | 0.340 | 0.484 | 0.747 | 8.495 | 7.916 | -0.580 | 66.028 | 50.879 | -15.148 | 82.159 | 68.533 | -13.626 | 2.000 |
| morning | sunny | building | 3.608 | 1.322 | 1.131 | 1.265 | 1.007 | 0.213 | 0.288 | 0.552 | 9.083 | 8.520 | -0.563 | 69.437 | 39.733 | -29.704 | 78.585 | 52.387 | -26.198 | 1.500 |
| morning | sunny | dark | 2.168 | 3.138 | 1.002 | 0.897 | 1.027 | -0.678 | 0.679 | 1.150 | 8.799 | 8.907 | 0.108 | 65.248 | 45.012 | -20.236 | 83.827 | 52.303 | -31.524 | 1.000 |
| morning | sunny | flat | 0.960 | 1.703 | 0.420 | 0.566 | 0.467 | 0.668 | 0.669 | 0.686 | 10.069 | 8.549 | -1.520 | 71.679 | 55.626 | -16.053 | 76.100 | 58.627 | -17.473 | 0.250 |
| morning | sunny | mixed | 1.213 | 2.928 | 0.466 | 0.568 | 0.626 | 0.430 | 0.445 | 0.417 | 8.999 | 8.457 | -0.542 | 68.856 | 51.540 | -17.316 | 74.177 | 53.487 | -20.691 | 0.250 |
| morning | sunny | road_shaded | 2.527 | 1.351 | 1.046 | 1.153 | 1.024 | 0.303 | 0.503 | 0.792 | 8.474 | 8.019 | -0.455 | 67.096 | 52.424 | -14.672 | 81.351 | 66.213 | -15.138 | 2.000 |

## Zoom별 요약

| zoom | sharpness_ratio | edge_density_ratio | hf_energy_ratio | ringing_ratio | local_contrast_ratio | deploy_minus_bicubic_luma_mean | deploy_bicubic_lowfreq_luma_mae | deploy_bicubic_chroma_mae | bicubic_niqe | deploy_niqe | deploy_minus_bicubic_niqe | bicubic_brisque | deploy_brisque | deploy_minus_bicubic_brisque | bicubic_piqe | deploy_piqe | deploy_minus_bicubic_piqe | risk_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1x | 2.815 | 1.830 | 1.111 | 1.118 | 1.042 | 0.213 | 0.483 | 0.866 | 8.531 | 7.491 | -1.040 | 65.371 | 48.050 | -17.321 | 83.906 | 73.073 | -10.833 | 2.273 |
| 3x | 2.627 | 1.821 | 0.887 | 1.016 | 0.883 | 0.461 | 0.482 | 0.520 | 8.979 | 8.516 | -0.463 | 70.196 | 48.637 | -21.559 | 77.313 | 58.649 | -18.664 | 0.900 |
| 5x | 2.280 | 1.900 | 0.918 | 0.896 | 0.898 | 0.224 | 0.355 | 0.738 | 9.148 | 8.191 | -0.956 | 67.679 | 48.820 | -18.860 | 83.799 | 61.083 | -22.716 | 1.300 |
| 7x | 1.623 | 3.111 | 0.654 | 1.049 | 0.805 | 0.338 | 0.486 | 0.569 | 8.814 | 8.573 | -0.241 | 65.958 | 49.367 | -16.591 | 73.018 | 50.723 | -22.295 | 0.818 |

## 시각 비교 Overview

![Top risk overview](comparisons/overview_top_risk.jpg)

포함 frame:

- `afternoon/sunny/mixed/1x/frame_0125.png`
- `afternoon/sunny/road_shaded/1x/frame_1725.png`
- `afternoon/sunny/road_shaded/5x/frame_2175.png`
- `morning/sunny/road_shaded/1x/frame_1925.png`
- `morning/sunny/road_shaded/5x/frame_2275.png`
- `morning/cloudy/building/1x/frame_4375.png`
- `morning/cloudy/road_shaded/1x/frame_2275.png`
- `morning/cloudy/road_shaded/5x/frame_2775.png`

## 산출 파일

- 전체 frame 지표: `metrics/per_image_metrics.csv`
- Scene별 요약: `metrics/summary_by_scene.csv`
- Zoom별 요약: `metrics/summary_by_zoom.csv`
- 입력 crop: `input_crop_320x180/`
- Bicubic x4: `bicubic/`
- Deploy GPU x4: `pred_deploy_gpu/`
- Frame별 비교 이미지: `comparisons/`

## 해석 및 한계

- Failure label은 절대 품질 판정이 아니라 screening용 threshold signal이다.
- `color_or_tone_instability`는 HR target이 없기 때문에 Deploy-vs-Bicubic low-frequency luma/chroma drift로 근사했다.
- `noise_or_false_texture_amplification`, `edge_oversharpening_or_ringing`은 high-frequency, edge density, sharpness, ringing proxy를 함께 사용해 분류했다.
- NIQE/BRISQUE/PIQE도 no-reference 지표라서 실제 임무 관점의 정답은 아니며, top-risk frame을 정성적으로 같이 확인해야 한다.
- SR 학습이나 MTKD 설계 변경 전에 본 보고서의 top-risk frame을 직접 확인해 domain mismatch, edge/texture hallucination, tone drift, model capacity 이슈를 분리하는 것이 좋다.
