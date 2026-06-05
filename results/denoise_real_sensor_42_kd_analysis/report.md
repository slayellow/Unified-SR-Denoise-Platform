# Denoise Real Sensor 42장 KD 효과 분석

## 분석 기준

- 입력: `results/260602_mc_g105_probe_42/raw`
- 샘플 수: `42`장
- 비교 모델: Deploy, MTKD, STKD
- checkpoint 기준: 분석 시점의 각 모델 `best.pth`
- 실제 센서 입력에는 HR/GT가 없으므로 PSNR/SSIM/LPIPS 대신 no-reference IQA와 Raw-vs-Output proxy를 사용했다.

## 모델별 요약

낮을수록 좋은 지표: NIQE, BRISQUE, PIQE, flat_hf_ratio, raw_output_lowfreq_luma_mae, raw_output_chroma_mae, risk_score.

`flat_hf_ratio`는 Raw의 평탄 영역 고주파 대비 출력의 평탄 영역 고주파 비율이다. 낮을수록 평탄 영역 noise-like HF가 줄었다는 뜻이다.

`strong_edge_grad_ratio`는 Raw의 강한 edge 위치에서 출력 edge 강도가 얼마나 유지되는지를 보는 값이다. 1에 가까울수록 edge 보존, 너무 낮으면 smoothing, 너무 높으면 sharpening 가능성이 있다.

| model | niqe | brisque | piqe | flat_hf_ratio | strong_edge_grad_ratio | sharpness_ratio | edge_density_ratio | hf_energy_ratio | ringing_ratio | raw_output_lowfreq_luma_mae | raw_output_chroma_mae | raw_output_diff_p95 | risk_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| deploy | 5.453 | 33.421 | 28.691 | 0.828 | 0.923 | 0.610 | 0.771 | 0.874 | 1.068 | 0.249 | 0.556 | 7.524 | 0.524 |
| mtkd | 6.415 | 52.884 | 63.933 | 0.639 | 0.726 | 0.312 | 0.494 | 0.653 | 1.103 | 5.781 | 1.454 | 16.286 | 2.786 |
| stkd | 6.096 | 42.370 | 44.788 | 0.687 | 0.801 | 0.358 | 0.562 | 0.724 | 1.099 | 4.664 | 1.195 | 13.714 | 2.262 |

![Model summary](metrics/model_summary_bars.png)

## Failure Signal Count

| failure_signal | frame_count |
| --- | --- |
| tone_or_color_shift | 84 |
| oversmoothing_risk | 55 |
| oversharpening_or_ringing_risk | 51 |
| large_raw_deviation | 44 |
| balanced | 22 |

## STKD 개선 폭이 큰 Frame 예시

| relative_path | time_of_day | weather | scene | zoom | deploy_flat_hf_ratio | mtkd_flat_hf_ratio | stkd_flat_hf_ratio | deploy_strong_edge_grad_ratio | mtkd_strong_edge_grad_ratio | stkd_strong_edge_grad_ratio | deploy_niqe | mtkd_niqe | stkd_niqe | mtkd_minus_deploy_flat_hf_ratio | stkd_minus_deploy_flat_hf_ratio | mtkd_minus_deploy_niqe | stkd_minus_deploy_niqe | comparison_path |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| morning/sunny/dark/7x/frame_4275.png | morning | sunny | dark | 7x | 0.863 | 0.451 | 0.433 | 0.944 | 0.421 | 0.449 | 7.780 | 9.823 | 10.155 | -0.412 | -0.430 | 2.043 | 2.375 | morning/sunny/dark/7x/frame_4275.compare.jpg |
| morning/sunny/dark/1x/frame_3425.png | morning | sunny | dark | 1x | 0.870 | 0.463 | 0.465 | 0.901 | 0.445 | 0.473 | 7.189 | 9.628 | 9.601 | -0.407 | -0.405 | 2.439 | 2.412 | morning/sunny/dark/1x/frame_3425.compare.jpg |
| morning/sunny/mixed/3x/frame_0575.png | morning | sunny | mixed | 3x | 0.902 | 0.617 | 0.673 | 0.968 | 0.566 | 0.689 | 3.890 | 5.823 | 5.162 | -0.285 | -0.229 | 1.933 | 1.271 | morning/sunny/mixed/3x/frame_0575.compare.jpg |
| morning/sunny/mixed/1x/frame_0425.png | morning | sunny | mixed | 1x | 0.863 | 0.622 | 0.658 | 1.018 | 0.826 | 0.876 | 3.640 | 5.284 | 5.062 | -0.241 | -0.205 | 1.644 | 1.422 | morning/sunny/mixed/1x/frame_0425.compare.jpg |
| morning/sunny/mixed/7x/frame_1025.png | morning | sunny | mixed | 7x | 0.599 | 0.307 | 0.400 | 0.506 | 0.215 | 0.331 | 6.403 | 8.142 | 6.104 | -0.292 | -0.200 | 1.738 | -0.300 | morning/sunny/mixed/7x/frame_1025.compare.jpg |
| afternoon/sunny/mixed/7x/frame_0575.png | afternoon | sunny | mixed | 7x | 0.816 | 0.530 | 0.623 | 0.896 | 0.608 | 0.733 | 4.427 | 6.746 | 5.098 | -0.286 | -0.193 | 2.319 | 0.671 | afternoon/sunny/mixed/7x/frame_0575.compare.jpg |
| morning/sunny/mixed/5x/frame_0825.png | morning | sunny | mixed | 5x | 0.852 | 0.598 | 0.661 | 0.833 | 0.489 | 0.611 | 4.289 | 6.224 | 5.196 | -0.254 | -0.191 | 1.935 | 0.907 | morning/sunny/mixed/5x/frame_0825.compare.jpg |
| morning/sunny/road_shaded/7x/frame_3425.png | morning | sunny | road_shaded | 7x | 0.808 | 0.580 | 0.638 | 0.921 | 0.683 | 0.753 | 5.716 | 7.340 | 6.202 | -0.228 | -0.170 | 1.624 | 0.486 | morning/sunny/road_shaded/7x/frame_3425.compare.jpg |
| morning/sunny/flat/3x/frame_1375.png | morning | sunny | flat | 3x | 0.381 | 0.117 | 0.218 | 0.239 | 0.073 | 0.176 | 12.520 | 10.345 | 11.614 | -0.264 | -0.163 | -2.174 | -0.906 | morning/sunny/flat/3x/frame_1375.compare.jpg |
| morning/sunny/road_shaded/3x/frame_3075.png | morning | sunny | road_shaded | 3x | 0.877 | 0.695 | 0.718 | 1.033 | 0.897 | 0.967 | 3.927 | 5.161 | 4.833 | -0.182 | -0.159 | 1.234 | 0.906 | morning/sunny/road_shaded/3x/frame_3075.compare.jpg |
| morning/sunny/flat/1x/frame_0225.png | morning | sunny | flat | 1x | 0.499 | 0.191 | 0.341 | 0.441 | 0.158 | 0.329 | 8.538 | 6.019 | 6.973 | -0.308 | -0.159 | -2.519 | -1.565 | morning/sunny/flat/1x/frame_0225.compare.jpg |
| morning/sunny/flat/5x/frame_1675.png | morning | sunny | flat | 5x | 0.368 | 0.102 | 0.211 | 0.228 | 0.064 | 0.171 | 12.625 | 11.561 | 11.698 | -0.266 | -0.157 | -1.063 | -0.926 | morning/sunny/flat/5x/frame_1675.compare.jpg |

## 시각 비교 Overview

![Overview](comparisons/overview_stkd_improvement.jpg)

포함 frame:

- `morning/sunny/dark/7x/frame_4275.png`
- `morning/sunny/dark/1x/frame_3425.png`
- `morning/sunny/mixed/3x/frame_0575.png`
- `morning/sunny/mixed/1x/frame_0425.png`
- `morning/sunny/mixed/7x/frame_1025.png`
- `afternoon/sunny/mixed/7x/frame_0575.png`
- `morning/sunny/mixed/5x/frame_0825.png`
- `morning/sunny/road_shaded/7x/frame_3425.png`

## 산출 파일

- 모델별 long metrics: `metrics/per_model_metrics.csv`
- frame별 wide metrics: `metrics/per_frame_comparison.csv`
- scene별 요약: `metrics/summary_by_scene_model.csv`
- zoom별 요약: `metrics/summary_by_zoom_model.csv`
- 출력 이미지: `outputs/{deploy,mtkd,stkd}/`
- frame별 비교 이미지: `comparisons/`

## 해석 주의

- 이 분석은 real input 기반이라 GT 정답 비교가 아니다.
- NIQE/BRISQUE/PIQE는 자연 영상 품질 proxy이며, 임무 장비에서의 edge fidelity를 완전히 대변하지 않는다.
- 최종 판단은 flat noise 감소, edge 유지, tone/color shift, 비교 이미지를 함께 봐야 한다.
