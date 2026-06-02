# Denoise 체크리스트

- 기준 날짜: 2026-06-02 화요일
- 목적: MC-G105 Denoise Teacher 후보 판단, 학습 상태 인수인계, MTKD 적용 전 준비 절차 정리
- 상태 표기: `[x]` 완료, `[ ]` 예정, `[/]` 진행 중

---

## 1. 퇴근 동기화 기록

- [x] `/home/jshong/Code/commute-sync-workflow.md` 확인
- [x] GitLab 저장소 상태 확인
  - [x] 경로: `/home/jshong/Code/unfied-sr-denoise-platform`
  - [x] 브랜치: `master`
  - [x] 추적 중이던 `results/mc_g105_analysis/all` 결과물 105개 삭제 확인
  - [x] 실행 부산물 `__pycache__` 수정분은 커밋 범위에서 제외
- [x] GitHub 저장소 상태 확인
  - [x] 경로: `/home/jshong/Code/Unified-SR-Denoise-Platform`
  - [x] 브랜치: `main`
  - [x] 동기화 전 clean 상태 확인
- [x] GitLab 커밋 및 푸시
- [x] GitLab -> GitHub rsync 수행
- [x] GitHub 커밋 및 푸시

## 2. 고정 42장 Probe Set

- [x] 고정 42장 probe set 확정
  - [x] 경로: `results/260602_mc_g105_probe_42`
  - [x] 선택 기준: `time/weather/scene/zoom` leaf마다 deterministic frame 1장
  - [x] RAW: 42장
  - [x] Deploy: 42장
  - [x] 기존 NAFNet snapshot: 42장
  - [x] `manifest.csv` 생성
  - [x] `selection_policy.txt` 생성
- [x] 기존 2700장 raw frame 폴더 삭제
  - [x] 삭제 경로: `results/mc_g105_capture_frames`
  - [x] 삭제 전 크기: 약 8.9GB
- [x] probe 출력물은 Git ignore 대상임을 확인
  - [x] `results/260602_mc_g105_probe_42/`

## 3. 최신 Teacher Probe 추론

- [x] NAFNet 최신 checkpoint로 42장 probe 추론
  - [x] checkpoint: `checkpoints/train_nafnet_teacher_mc_g105_denoise_x1_width64_260527/last.pth`
  - [x] checkpoint epoch: 71
  - [x] 출력 경로: `results/260602_mc_g105_probe_42/nafnet_last_e071`
  - [x] 처리: 42장
  - [x] 실패: 0장
  - [x] 상태 CSV: `results/260602_mc_g105_probe_42/nafnet_last_e071_status.csv`
- [x] Restormer tone-safe 최신 checkpoint로 42장 probe 추론
  - [x] checkpoint: `checkpoints/train_restormer_teacher_mc_g105_denoise_x1_dim48_tonesafe_lr1e4_260601/last.pth`
  - [x] checkpoint epoch: 1
  - [x] 출력 경로: `results/260602_mc_g105_probe_42/restormer_last_e001`
  - [x] 처리: 42장
  - [x] 실패: 0장
  - [x] 상태 CSV: `results/260602_mc_g105_probe_42/restormer_last_e001_status.csv`
- [x] 비교 산출물 생성
  - [x] 리포트: `results/260602_mc_g105_probe_42/comparison_current/comparison_report.md`
  - [x] 이미지별 CSV: `results/260602_mc_g105_probe_42/comparison_current/comparison_per_image.csv`
  - [x] 모델별 요약 CSV: `results/260602_mc_g105_probe_42/comparison_current/summary_by_variant.csv`
  - [x] 장면별 요약 CSV: `results/260602_mc_g105_probe_42/comparison_current/summary_by_scene_variant.csv`
  - [x] 주요 사례 contact sheet: `results/260602_mc_g105_probe_42/comparison_current/contact_sheet_key_cases.jpg`
  - [x] center-crop contact sheet: `results/260602_mc_g105_probe_42/comparison_current/contact_sheet_center_crops.jpg`

## 4. Probe 판단 결과

- [x] RAW / Deploy / NAFNet e71 / Restormer e1 비교 완료
- [x] NAFNet e71 판단
  - [x] Deploy보다 detail 및 edge 보존이 좋음
  - [x] 여전히 tone-down 위험이 남아 있음
  - [x] probe 평균 `delta_y_mean`: `-6.619`
  - [x] probe `lowfreq_y_mae`: `6.757`
  - [x] probe `nonedge_hf_ratio`: `0.915`
  - [x] probe `strong_edge_ratio`: `0.984`
- [x] Restormer e1 판단
  - [x] tone-safe 방향은 더 안전해 보임
  - [x] denoise는 아직 약하고 RAW에 가까운 출력임
  - [x] probe 평균 `delta_y_mean`: `1.500`
  - [x] probe `lowfreq_y_mae`: `1.517`
  - [x] probe `nonedge_hf_ratio`: `0.983`
  - [x] probe `strong_edge_ratio`: `0.978`
- [x] 현재 Teacher 후보 순위
  - [x] 실질적 1순위 후보: `NAFNet e71`
  - [x] 관찰 후보: `Restormer tone-safe`, 후속 epoch 필요
  - [x] Deploy 모델은 생산 baseline이며, smoothing이 강해 primary Teacher로는 부적합

## 5. NAFNet Tone-Safe 판단

- [x] 현재 NAFNet은 `configs/data/denoise_mc_g105_val.yaml` 사용 중임을 확인
- [x] 현재 Restormer는 `configs/data/denoise_mc_g105_tone_safe.yaml` 사용 중임을 확인
- [x] 결정 사항
  - [x] 현재 NAFNet 학습을 즉시 중단하지 않음
  - [x] NAFNet은 epoch 80 또는 100까지 진행 후 재평가
  - [x] NAFNet tone-safe pipeline은 후속 실험안으로 유지
  - [x] 고정 probe에서 큰 음수 Y shift가 계속되면 별도 `save_name`으로 tone-safe NAFNet fine-tune 시작
- [/] NAFNet tone gate 유지
  - [/] 집중 확인 장면: `road_shaded`, `building`, `flat`
  - [/] Teacher 사용 전 목표: 평균 Y shift가 중립에 가까워져야 하며, 가능하면 `-3` 내외 또는 그보다 좋아야 함

## 6. NAFNet 학습 인수인계

- [/] NAFNet MC-G105 학습 진행 중
  - [/] 실행 방식: resume
  - [/] 명령어: `accelerate launch --num_processes=1 tools/train.py --resume checkpoints/train_nafnet_teacher_mc_g105_denoise_x1_width64_260527/last.pth`
  - [/] accelerate PID: `167977`
  - [/] main worker PID: `168952`
  - [/] child worker: `3690904`, `3690906`, `3690909`, `3690912`
  - [/] GPU: 1
- [x] 최신 완료 epoch 확인
  - [x] epoch: 71
  - [x] PSNR: `27.305063247680664`
  - [x] SSIM: `0.840343713760376`
  - [x] LPIPS: `0.20120041072368622`
  - [x] NIQE: `3.9625232219696045`
  - [x] lr: `8.646198293969953e-05`
- [x] 최신 checkpoint timestamp 확인
  - [x] `last.pth`: `2026-06-02 10:18 KST`
  - [x] `best.pth`: `2026-05-31 10:35 KST`
  - [x] `training_log.csv`: `2026-06-02 10:18 KST`

## 7. Restormer 학습 인수인계

- [/] Tone-safe Restormer MC-G105 학습 진행 중
  - [/] 실행 방식: scratch restart, resume 아님
  - [/] 명령어: `accelerate launch --num_processes=1 tools/train.py --config configs/train/Denoise/restormer_deploy_denoise_teacher.yaml --data_config configs/data/denoise_mc_g105_tone_safe.yaml`
  - [/] accelerate PID: `2225235`
  - [/] main worker PID: `2225703`
  - [/] child worker: `2751925`, `2751926`, `2751927`, `2751928`
  - [/] GPU: 0
  - [/] GPU memory가 A6000 한계에 가까움
- [x] Restormer config 확인
  - [x] `data_config_path`: `configs/data/denoise_mc_g105_tone_safe.yaml`
  - [x] `lr`: `1e-4`
  - [x] `batch_size`: 2
  - [x] `gradient_accumulation_steps`: 32
  - [x] effective batch: 64
  - [x] `save_name`: `train_restormer_teacher_mc_g105_denoise_x1_dim48_tonesafe_lr1e4_260601`
- [x] 최신 완료 epoch 확인
  - [x] epoch: 1
  - [x] PSNR: `24.355131149291992`
  - [x] SSIM: `0.8034186363220215`
  - [x] LPIPS: `0.26657748222351074`
  - [x] NIQE: `3.7979326248168945`
  - [x] lr: `0.0001`
- [x] 최신 checkpoint timestamp 확인
  - [x] `last.pth`: `2026-06-02 04:26 KST`
  - [x] `best.pth`: `2026-06-02 04:26 KST`
  - [x] `training_log.csv`: `2026-06-02 04:26 KST`

## 8. 다음 작업

- [ ] NAFNet과 tone-safe Restormer 학습을 계속 진행
- [ ] NAFNet이 epoch 80 또는 100에 도달하면 고정 42장 probe 재실행
- [ ] Restormer가 epoch 3에 도달하면 고정 42장 probe 재실행, epoch 10 전후에 다시 재실행
- [ ] NAFNet tone drift가 계속 크면 별도 tone-safe NAFNet fine-tune 실험 생성
- [ ] probe 비교 폴더는 lightweight report가 필요한 경우를 제외하고 local/ignored 상태 유지
- [ ] Teacher 후보가 real MC-G105 tone/detail gate를 통과하기 전까지 MTKD Student 학습 시작 금지

## 9. 2026-06-03 Denoise MTKD 방향성

- [ ] 현재 방향 유지: 큰 Teacher 모델의 복원 능력을 QCS8550 친화적인 작은 CNN Student로 압축
  - [ ] `NAFNet`: detail 및 edge Teacher
  - [ ] `Restormer tone-safe`: 후속 epoch 이후 tone/color safety reference
  - [ ] Deploy 모델: 생산 baseline 및 smoothing/regression reference, primary Teacher 아님
  - [ ] Student 목표: `SVFocusDenoiseNet` block3 -> block2 축소
  - [ ] Student 배포 조건: 학습은 advanced reparameterized block, 배포는 fused static Conv graph
- [ ] MTKD에서 단순 Teacher 평균 사용 금지
  - [ ] NAFNet의 tone-down을 Student가 그대로 학습하지 않도록 제한
  - [ ] 초기 Restormer의 RAW-like 약한 denoise를 Student가 그대로 학습하지 않도록 제한
  - [ ] attribute-aware distillation 선호: NAFNet detail + Restormer tone safety + supervised clean anchor
- [ ] KD loss 우선순위
  - [ ] P0: supervised clean HR anchor를 항상 중심에 둠
  - [ ] P1: residual/noise KD, `noisy - student` vs `noisy - teacher`
  - [ ] P2: tone-gated output KD, 낮은 weight로 적용
  - [ ] P3: edge/frequency KD, detail 보존용 보조 loss
  - [ ] P4: feature KD, output/residual KD 효과 확인 후 후순위로 검토
- [ ] Degradation refinement 방향
  - [ ] MC-G105 특성에 맞는 noise 성분은 충분히 강하게 유지
  - [ ] tone/color/gamma shift는 모델이 원치 않는 색 보정을 배우지 않도록 제어
  - [ ] tone drift가 계속 크면 tone-preserving pair 실험 진행: `target = tone_shift(HR)`, `noisy = noise(target)`
- [ ] Latency를 hard deployment gate로 승격
  - [ ] QCS8550에서 fused Student model-only latency 측정
  - [ ] QCS8550에서 4-5개 모델 concurrent workload 측정
  - [ ] p50/p90/p99, memory, fallback, sustained runtime behavior 기록
  - [ ] 저가형 Edge chip까지 고려하여 static CNN/Conv graph 우선

## 10. 2026-06-03 출근 후 진행 절차

- [ ] 1단계: Teacher 학습 상태 확인
  - [ ] NAFNet 최신 epoch 및 `training_log.csv` 갱신 여부 확인
  - [ ] Restormer tone-safe 최신 epoch 및 `training_log.csv` 갱신 여부 확인
  - [ ] OOM, stalled process, stale checkpoint 여부 확인
- [ ] 2단계: 의미 있는 checkpoint에서 고정 42장 probe 재실행
  - [ ] NAFNet: epoch 80 또는 100에서 재실행
  - [ ] Restormer: epoch 3에서 재실행, epoch 10 전후에서 재실행
  - [ ] 동일 probe set으로 RAW / Deploy / NAFNet / Restormer 비교
- [ ] 3단계: MTKD 전 Teacher acceptance gate 적용
  - [ ] NAFNet gate: Deploy보다 detail/edge가 좋고 심각한 tone-down이 없어야 함
  - [ ] NAFNet 목표: `delta_y_mean`이 중립에 가까워져야 하며, 가능하면 `-3` 내외 또는 그보다 좋아야 함
  - [ ] Restormer gate: RAW-like 상태에서 벗어나 denoise 효과가 확인되어야 함
  - [ ] Restormer gate: tone/color 안정성이 NAFNet보다 안전해야 함
  - [ ] failure bucket 정리: tone-down, oversmooth, residual noise, color shift, artifact
- [ ] 4단계: KD 전 Student baseline 준비
  - [ ] block2 basic이 아니라 block2 advanced Student config 생성 또는 확인
  - [ ] `block2 + use_advanced_rep: true`를 supervised loss로 먼저 학습
  - [ ] 가능하면 Deploy block3 advanced 및 block2 basic baseline과 비교
  - [ ] train graph -> fused deploy graph 출력 parity 확인
- [ ] 5단계: Student 후보 QCS8550 배포 gate 실행
  - [ ] fused Student 모델 export
  - [ ] single-model latency benchmark 실행
  - [ ] 4-5개 모델 concurrent workload benchmark 실행
  - [ ] p50/p90/p99 기록 후 full SW operation에서 실제 latency gain이 나타나는지 확인
- [ ] 6단계: Gate 통과 후 MTKD 시작
  - [ ] 1차 ablation: supervised + NAFNet residual KD
  - [ ] 2차 ablation: tone-gated output KD 추가
  - [ ] 3차 ablation: edge/frequency KD 추가
  - [ ] Restormer는 denoise 효과가 충분히 올라온 뒤 추가
  - [ ] random degradation 학습에서는 Teacher output cache 사용 금지, online Teacher inference 유지
- [ ] 7단계: 판단 결과 기록
  - [ ] `experiment.log`에 Teacher probe 요약 기록
  - [ ] `denoise_checklist.md`에 gate 완료 여부 반영
  - [ ] 무거운 probe 출력물은 lightweight report가 필요한 경우를 제외하고 local/ignored 상태 유지

## 11. 실행 부산물 주의

- [/] 추적 중인 `__pycache__` 수정분은 실행 부산물이므로 커밋에서 제외
  - [/] `src/engine/__pycache__/trainer.cpython-310.pyc`
  - [/] `src/engine/__pycache__/trainer.cpython-38.pyc`
  - [/] `src/models/__pycache__/__init__.cpython-38.pyc`
  - [/] `tools/__pycache__/train.cpython-38.pyc`
