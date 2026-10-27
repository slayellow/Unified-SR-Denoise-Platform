# EO Denoise v3 실험 레지스트리

현재 베이스라인: **BL1** = Denoise v2 출시본 (합성 val 34.43 / 0.924 / 0.1380은 카메라 정답 A 세트. C 런과 같은 정답 원본 세트(HR = val_denoise/HR)에서는 30.19 / 0.899 / 0.2168, 실기 P1.1 10배 노이즈 3.69 · 더한 넘침 −2.0 %p · MTF50 비 1.05/1.04) · 상태: **종료 (2026-10-13)**. EXP-09-1(HYP-07 절반 4층)을 BicubicPPDenoise[v1]로 배포(소유자 2026-10-13, `results/denoise_release_v1/MANIFEST.md`). EXP-08-1(C 처음부터) W8A8 + QAT는 10-12에 v2 대체 후보로 채택됐지만 배포되지 않고 v1로 대체됨
현재 프로토콜: **P1.1** (`docs/20261008-RESEARCH-EODenoiseV3.md` §2, 지표 정밀도: 더한 넘침 P2 0.7 · P3 1.2 %p, MTF 비 0.02) · 노이즈 하한(P1.1, 시드 42/43): A 레시피(EXP-01-1 vs 01-2) 10배 노이즈 0.40, 더한 넘침 P2 0.15 %p · C 레시피(EXP-04-1 vs 04-2) 노이즈 0.02, 넘침 0.1 %p
주 지표: 확정 카메라 설정 차트 클립의 Denoise 출력 평탄부 노이즈 · 더해진 오버슈트(카메라 윤곽 제거는 음수) · MTF50 유지율, 보조: 질감 비 · 점 보존 · 핫픽셀 잔차

| 일자 | EXP | HYP | 축 · 변경 (한 줄) | P | BL 기준 | 노이즈 · 오버슈트 · MTF50 | delta | 판정 | 문서 |
|---|---|---|---|---|---|---|---|---|---|
| 2026-10-08 | EXP-00-1 | – | BL1 채점 (학습 없음) | P1.1 | – | 10배 3.69 · −2.0 %p · 1.05/1.04; 질감 0.80 | – | 기준 | `results/261008_eo_capture/p1_scores/P1.1/BL1_v2.json` |
| 2026-10-08 | EXP-01-1 | HYP-01 | 카메라 모델 쌍, 정답 A, 시드 42 (ep24 best) | P1.1 | BL1 | 10배 4.82 · −0.1 %p · 1.01/1.00; 질감 0.61 | 노이즈 +1.13 | 기각 (노이즈) | `P1.1/EXP-01-1.json` |
| 2026-10-08 | EXP-01-2 | HYP-01 | EXP-01-1과 같음, 시드 43 (노이즈 하한) | P1.1 | BL1 | 10배 4.42 · +0.0 %p · 1.01/1.00 | 노이즈 +0.73 | 기각 | `P1.1/EXP-01-2.json` |
| 2026-10-08 | EXP-02-1 | HYP-02 | `camera_model.prob` 0.75 → 1.0 (replay 없음) | P1.1 | EXP-01-1 | 10배 4.59 · +0.0 %p · 1.02/1.00 | 하한 안 | 기각 | `P1.1/EXP-02-1.json` |
| 2026-10-08 | EXP-04-1 | HYP-04 | `camera_model.target` camera → source (C안) | P1.1 | EXP-01-1 | 10배 2.40 · −9.0 %p · 1.12/1.06; 1배 −8.2 %p; 질감 0.44; 점 1px 0.12 | 노이즈 −2.42, 윤곽 −8.8 %p | 1순위 후보 → 재현됨(04-2), 08-1로 대체 | `P1.1/EXP-04-1.json` |
| 2026-10-08 | EXP-05-1 | HYP-05 | `camera_model.target` camera → optics (B안) | P1.1 | EXP-01-1 | 10배 1.85 · −12.7 %p · 0.56/0.62; 질감 0.15 | MTF −0.45 | 기각 (흐림) | `P1.1/EXP-05-1.json` |
| 2026-10-08 | EXP-06-1 | HYP-06 | C안 + `hot_pixels` + 움직임 모드 `impulses` | P1.1 | EXP-04-1 | EXP-04-1과 같음 | 0 | 판정 불가 (같은 시드, 결함 희소) | `P1.1/EXP-06-1.json` |
| 2026-10-08 | EXP-04-2 | HYP-04 | EXP-04-1과 같음, 시드 43 (C 재현·하한) | P1.1 | EXP-04-1 | 10배 2.42 · −9.0 %p · 1.12/1.06; 1배 −8.3 %p; 질감 0.44; 점 1px 0.12 | 노이즈 +0.02, 윤곽 0.0 %p | 재현됨 (C 하한: 노이즈 0.02, 넘침 0.1 %p) | `P1.1/EXP-04-2.json` |
| 2026-10-08 | EXP-07-1 | HYP-04 | C안, lr 5e-5 → 1e-4 (학습량 축) | P1.1 | EXP-04-1 | 10배 2.33 · −9.1 %p · 1.14/1.09; 1배 −9.0 %p; 질감 0.44; 점 1px 0.10 | 노이즈 −0.07, 윤곽 −0.1~−0.8 %p | 소폭 개선 (하한 수준) | `P1.1/EXP-07-1.json` |
| 2026-10-08 | EXP-08-1 | HYP-04 | C안, v2 없이 처음부터 300에폭 lr 1e-4 (시작점 축, v2 원래 일정) | P1.1 | EXP-04-1 | 10배 2.32 · −12.6 %p · 1.15/1.18; 1배 −11.4 %p; 질감 0.46; 점 1px 0.20; 합성 val 35.80/0.1298 | 노이즈 −0.08, 윤곽 −3.6 %p (하한 0.1의 36배), 점·질감 개선 | **채택 (10-13 판정 후보, 소유자 승인 2026-10-12)** | `P1.1/EXP-08-1.json` |
| 2026-10-12 | EXP-08-1 보드 | HYP-04 | EXP-08-1을 W8A8 + QAT 30에폭 / W16A16 PTQ DLC로 보드 98에서 실행 (학습 없음) | P1.1 (보드 출력) | BL1 보드 | W8: 10배 2.45 · −12.8 %p · 1.15/1.17, 1배 넘침 8.1 %; W16: 2.32 · −12.6 %p · 1.15/1.18; v2: 3.66 · −1.6 %p · 1.05/1.04. 지연 burst W8 7.68 · W16 30.99 · v2 7.11 ms. 보드 체인 10배 x4 흰 윤곽 v2 17.3 → W8 5.3 % | W8 노이즈 −1.21, 넘침 −11.2 %p (v2 보드 대비) | **채택 W8A8 (소유자 보드 화면 판정 2026-10-12)**, W16 보류 (30 ms 초과) | `results/261012_eo_denoise_board_test/board_scores.json`, `chain_board.json` |
| 2026-10-12 | EXP-09-1 | HYP-07 | 모델만 `bicubicplusplus` scale 1 `legacy`(절반 해상도) 본체 4층 LeakyReLU, 그 밖은 EXP-08-1과 같음 (W16 벤치 burst 7.9 ms) | P1.1 | EXP-08-1 | ep287: 10배 1.99 · −11.5 %p · 1.05/1.06. 보드 W16A16(10-13): 10배 2.07, SR x4 체인 흰 윤곽 3.7 % / 노이즈 1.67, 지연 v3 C W8A8의 1.22배(burst 8.15 ms) / 1.29배(1분 15.2 ms) | 노이즈 −0.33, 넘침 +1.1 %p | **채택: BicubicPPDenoise[v1] 배포 (소유자 2026-10-13)** | `checkpoints/experiments/eo_denoise/20261012-EXP-09-1/` |
| 2026-10-12 | EXP-10-1 | HYP-07 | 같은 구조 본체 2층 LeakyReLU (W16 벤치 burst 5.1 ms) | P1.1 | EXP-08-1 | ep237: 10배 3.00 · −9.1 %p · 1.03/1.07. 보드 W16A16: 10배 3.05, 지연 0.85배 | 노이즈 +0.68, 넘침 +3.5 %p | 기각 (빠르지만 노이즈) | `checkpoints/experiments/eo_denoise/20261012-EXP-10-1/` |
| 2026-10-12 | EXP-11-1 | HYP-07 | `auto`(원 해상도) 본체 2층 ReLU (W16 벤치 burst 11.9 ms) | P1.1 | EXP-08-1 | ep254: 10배 2.46 · −11.3 %p · 1.18/1.13. 보드 W16A16: 지연 2.06배 | 노이즈 +0.14, 넘침 +1.3 %p | 기각 (지연) | `checkpoints/experiments/eo_denoise/20261012-EXP-11-1/` |

`문서` 열의 경로는 기록이다. 이 연구의 `checkpoints/experiments/eo_denoise/`와 `results/2610xx_*` 폴더는 소유자 결정(2026-10-13)으로 지워 지금은 없다. 지금 남은 사본은 다음과 같다.

| 무엇 | 지금 위치 |
|---|---|
| P1.1 채점 JSON과 로그 (`BL1_v2`, `EXP-01-1`~`EXP-11-1`, `EXP-01-1_ep50`) | `results/denoise_release_v1/analysis/p1_scores/P1.1/` |
| P1 채점 JSON (P1.1로 바꾸기 전, `BL1_v2`, `EXP-01-1`~`EXP-06-1`) | `results/denoise_release_v1/analysis/p1_scores/` |
| 보드 채점 (`v2`, `c08w8` = v3 C, `bppv1`, 요약 `board_summary.md`) | `results/denoise_release_v1/eval/scores/` |
| EXP-09-1 런 (배포 v1, 에폭별 체크포인트 포함) | `checkpoints/bicubicpp_deploy/train_bicubicppdenoise_eo_x1_dim32_half4_ep300_bs128_lr1e-4_v1/`. 배포 체크포인트는 `results/denoise_release_v1/checkpoint/best.pth` |
| 그 밖의 런 체크포인트 | 없음 |
| P1.1 채점 입력 (클립 3개, `P1_clips.sha256`, 핫픽셀 목록) | `results/denoise_release_v1/eval/`, `results/denoise_release_v1/analysis/noise_census/hot_pixels_a0.json` |

계획서 §2의 P1.1 입력 경로(`results/261008_eo_capture/...`)와 호스트 채점 명령도 기록이다. `score_eo_chart.py`의 기본값이 위 사본을 가리키므로, 지금 다시 채점할 때는 `docker/run.sh -- env PYTHONPATH=. python3 examples/evaluation/score_eo_chart.py --checkpoint <ckpt> --json <out.json>`을 쓴다([실험·릴리스 기록](../README.md) 2.7절).

프로토콜 변경 이력
| 버전 | 일자 | 이유 | 재평가한 행 | 새 노이즈 하한 |
|---|---|---|---|---|
| P1 | 2026-10-08 | 초안 → 고정(채점기 저장소 이관) | – | 노이즈 0.40, 넘침 0.4 %p (10배) |
| P1.1 | 2026-10-08 | P1 넘침·MTF 비가 노이즈 편향(완벽한 Denoise +5.6 %p, 1.73), 엣지 직선 입력 고정·프레임 평균 프로파일, 보조 지표 추가 | 전부(EXP-00-1~06-1) | 시드 차이(01-1 vs 01-2): 노이즈 0.40, 넘침 0.15 %p |
