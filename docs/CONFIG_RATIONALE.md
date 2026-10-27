# 설정 근거 기록 (CONFIG_RATIONALE)

`configs/` 아래의 YAML/JSON 에서 주석을 걷어냈다. 그 주석에 담겨 있던 "왜 이 값인가"의 근거는 전부 이 문서로 옮겼고, 앞으로는 여기가 유일한 근거 기록이다. 설정 파일은 값만 들고 있고, 값의 이유·실측 수치·폐기된 시도는 이 문서에서 찾는다.

주제별로 묶었다 — 디렉터리(`train/`, `finetune/`, `data/`, `aimet/`)가 아니라 "무엇을 왜 그렇게 정했는가"가 기준이다. 한 항목이 여러 디렉터리에 걸치면 적용 범위를 항목 안에 적었다. **폐기된 값은 "폐기"로 명시**했으니, 옛 파일에서 그 값을 보더라도 근거로 삼지 마라.

**옛 config 이름 → 현재 이름.** 출하 config 는 아래처럼 이름이 바뀌었다. 표에 없는 Bicubic++ `_v1`/`_v2` config 는 삭제됐고, 옛 내용은 `git show <커밋>^:<옛 경로>` 로 본다(지운 커밋은 표의 이름 변경 커밋과 같다: Phase-1 b49ff09, EO Phase-2 1817870, IR Phase-2 3fe100d). 구세대 모델 config 42개(`QuickSRNet/`, `SR/`, `MambaIR/`, `SVFocusSRNet/`, `Denoise/` 의 `bicubicppdenoise.yaml` 외)는 2026-10-16 지웠다(9). 본문에서 v1/v2/v3 이름은 이력을 말할 때만 쓴다.

| 옛 이름 | 현재 이름 | 커밋 |
|---|---|---|
| `configs/train/BicubicPlusPlus/bicubicplusplus_{2,3,4}x_v3.yaml` | `configs/train/BicubicPlusPlus/bicubicplusplus_{2,3,4}x.yaml` | b49ff09 (2026-10-13) |
| `configs/finetune/BicubicPlusPlus/bicubicplusplus_{2,3,4}x_eo_v3.yaml` | `configs/finetune/BicubicPlusPlus/bicubicplusplus_{2,3,4}x_eo.yaml` | 1817870 (2026-10-14) |
| `configs/finetune/BicubicPlusPlus/bicubicplusplus_{2,3,4}x_ir_v3.yaml` | `configs/finetune/BicubicPlusPlus/bicubicplusplus_{2,3,4}x_ir.yaml` (레시피도 바뀜, 5.6) | 3fe100d (2026-10-06) |
| `configs/data/sr_train_bicubic_only_eo_v2.yaml` | `configs/data/bicubicpp/sr_train_phase-1.yaml` | 1817870 |
| `configs/data/sr_train_bicubic_weak_noise_eo_v2.yaml` | `configs/data/bicubicpp/sr_finetune_eo_phase-2.yaml` | 1817870 |
| `configs/data/sr_train_bicubic_weak_noise_ir_v3.yaml` → `configs/data/sr_train_bicubic_weak_noise_ir.yaml` | `configs/data/bicubicpp/sr_finetune_ir_phase-2.yaml` (3fe100d 에서 열 노이즈 σ 도 바뀜, 8.7) | 3fe100d, 1817870 |
| `configs/data/denoise.yaml` | `configs/data/bicubicpp/denoise.yaml` | 1817870 |

---

## 1. 두 단계 학습 구조 (Phase-1 사전학습 / Phase-2 파인튜닝)

전체 파이프라인의 뼈대다. 아래 대부분의 값이 "지금 어느 Phase 인가"에 따라 갈린다.

| | Phase-1 (`configs/train/BicubicPlusPlus/`) | Phase-2 (`configs/finetune/BicubicPlusPlus/`) |
|---|---|---|
| 데이터 | `tmp/` 98,897장 (LSDIR 전체 + OST + 기타, 1.1) | 도메인 데이터 `train/` (EO/IR) |
| 열화 (data config) | `target_resize` 뿐 | 노이즈·블러·JPEG·핫픽셀·(IR)컬럼노이즈 |
| 배포 softening (trainer) | 배치의 50 % (7.2) | EO 만 50 %, IR 은 없음 (7.2) |
| 손실 | 순수 MSE | MSE + ssim/edge/(EO)perceptual/gradient_variance |
| lr · 스케줄러 | 5e-4, KneeLR 120ep (1.5) | 5e-5, CosineAnnealingLR `T_max` 100 |
| EMA · ECB(`use_rep`) | 둘 다 켬 (7.1, 7.3) | 둘 다 켬 |
| 출발점 | 스크래치 | v3 Phase-1 `best.pth` (1.3) |

### 1.1 Phase-1 데이터 소스: `train/` → `tmp/`

- **어디에**: `configs/train/BicubicPlusPlus/bicubicplusplus_{2x,3x,4x}.yaml` 의 `data.train_root`.
- **왜**: Phase-1 은 `tmp/` **98,897장**(LSDIR 84,991장 + OST 8,984장 + `train/` 의 `LSDIR_HR`·`OST_HR` 밖 출처 9개 4,922장, 2026-10-20 `/mnt/data2` 사본에서 셈)으로 사전학습한다. 기존 `train/` 은 출처 폴더 11개의 **10,821장**이고 그중 LSDIR 은 `LSDIR_HR` **5,000장**뿐이라, `tmp/` 로 **약 9.1배** 많은 표본을 추가 비용 없이 얻는다(장 수는 2026-10-19 sshfs 원본과 `/mnt/data2` 사본에서 센 값. 이전에 적혀 있던 98,520장·LSDIR_HR 10,444장·9.4배는 지금 폴더와 맞지 않아 고쳤다).
- **주의**: Phase-2 는 도메인 데이터(`train/`)로 되돌아간다. `tmp/` 는 Phase-1 전용이다.

### 1.2 epoch 수가 아니라 step 예산으로 본다 (v1 1000ep vs v2 120ep)

- **어디에**: `configs/train/BicubicPlusPlus/*.yaml` 의 `train.epochs: 120`, `early_stopping.patience: 120`, `scheduler.total_steps: 120`.
- **왜**: `tmp/` 는 epoch 당 **6,182 step**(98,897장 / batch 16, 올림)이라 120ep = **약 742k step**(741,840). v1(당시 `train/` × 1000ep = **652k step**)과 계산량은 거의 같고 데이터만 약 9배다. **1000 → 120 은 학습을 줄인 것이 아니라 epoch 의 정의가 바뀐 것.**
- **주의**: v2 의 `save_name` 에는 `epoch1000` 이 박혀 있었다(`..._epoch1000_bs16_lr5e-4_mse_v2`, 실제 120). v3 부터 `ep120` 으로 맞췄지만 옛 체크포인트 폴더 이름으로 설정을 추론하지 마라.

### 1.3 Phase-2 출발점을 v2 Phase-1 체크포인트로 교체

- **어디에**: `configs/finetune/BicubicPlusPlus` 6개 파일의 `pretrained_path`. 지금은 v3 Phase-1 `checkpoints/bicubicpp_deploy/train_bicubicplusplus_sr_x{2,3,4}_dim32_knee_ep120_bs16_lr5e-4_mse_v3/best.pth` 이다(1.5). 이 절은 v1 → v2 교체의 기록이다(당시 v1 은 `checkpoints/bicubicpp_v1_deploy/...`, v2 는 `..._mse_v2/best.pth`). v3 Phase-1 도 Phase-2 와 같은 커널 목록(`sr_train_phase-1.yaml`, 8.1)으로 학습한다.
- **왜**: v2 Phase-1 은 파인튜닝 레시피와 **같은 target_resize 커널 분포**(AREA ×2, CUBIC, LINEAR, LANCZOS4)로 학습됐고, v1 은 **INTER_CUBIC 만** 봤다. EO 에서 이 교체가 x3 그래디언트를 **64.1% → 73.3%(GT 대비)** 로 올렸다.
- **주의**: 요점은 **사전학습 커널 분포와 파인튜닝 커널 분포의 일치**다. v1 계보 체크포인트에서 v2 레시피로 파인튜닝하지 마라.

### 1.4 Phase-2 학습률 5e-5 = 이동 예산

- **어디에**: `configs/finetune/BicubicPlusPlus` 6개 파일 전부 (eo/ir × x2/x3/x4).
- **왜**: Phase-1 은 5e-4 × 1000ep 로 학습했다(v1 기준. v2 이후 `tmp/` 120ep 도 step 수는 비슷하다, 1.2). Phase-2 의 5e-5 는 Phase-1 대비 **이동 예산 0.2% → 1%** 를 주기 위한 값이다. 그래야 Phase-2 가 열화 분포에 실제로 적응한다. **하방 위험은 없다** — Phase-2 의 `best.pth` 는 Phase-1 baseline 을 이겨야만 채택되므로(trainer 의 baseline seed 메커니즘) 하방이 Phase-1 로 막혀 있다.
- **Phase-2 가 `best.pth` 를 못 만들면**: Phase-1 baseline 을 한 번도 넘지 못한 런에는 `best.pth` 가 없다(그 런 자체의 최선은 `best_attempt.pth`). **EO** 는 그때 Phase-1 `best.pth` 를 그 Phase-2 config 로 export 한다(지금은 제거된 `chain_v3_qat.sh` 의 규칙, 1b1ccb4). EO Phase-2 의 model 블록이 Phase-1 과 같아서 export 의 model 블록 검사를 통과한다. 명령은 `docker/run.sh --gpu <N> -- python3 examples/aimet_quantization/6_apply_qat.py --config <Phase-2 config> --data_config <그 config 의 레시피> --checkpoint <Phase-1 best.pth> --output_dir <export 폴더>/x<s>_eo/qat --mmp_config configs/aimet/mmp_noprune_w16a16.json --param_bw 16 --output_bw 16 --qat_epochs 0 --width <W> --height <H> --device 0` 이다. v3 EO x2 가 그 경우다 — Phase-2 가 baseline LPIPS 0.1672 를 넘지 못해 Phase-1 ep113 을 출하했다(`results/release_v3/MANIFEST.md`; `checkpoints/bicubicpp_deploy/finetune_bicubicplusplus_eo_sr_x2_dim32_v3/` 에 `best.pth` 가 없다). **IR 에는 이 규칙이 통하지 않는다.** IR Phase-2 config 에는 `achromatic_output: true` 가 더 있어서, `check_model_config`(`src/engine/checkpoint.py`)가 Phase-1 체크포인트를 `ValueError`(`was trained with another model config`)로 거부한다. IR 은 그 Phase-2 런의 `best_attempt.pth` 나 `last.pth` 를 그 런의 `train_config.yaml` 로 export 한다.

### 1.5 Phase-1 v3 — 데이터는 그대로, softening·EMA·ECB 는 trainer 와 모델 쪽, 스케줄러 KneeLR

- **어디에**: `configs/train/BicubicPlusPlus/bicubicplusplus_{2x,3x,4x}.yaml`.
- **구성**: 데이터 레시피 `configs/data/bicubicpp/sr_train_phase-1.yaml`(`target_resize` 뿐, 8.1·8.2) + `train.deploy_softening`(배치의 50 % 를 동결된 v2 denoise 에 통과, 그 앞에 σ 2~15 DN 가우시안 노이즈, 7.2) + `train.ema`(decay 0.999, `start_epoch` 2, 7.3) + `model.use_rep: true`(ECB, 7.1) + `scheduler.type: KneeLR`. 손실은 순수 MSE, 판정은 `psnr`/`max`.
- **왜 (c7be65d, 2026-09-08)**: v3 Phase-2 가 v2 Phase-1 체크포인트에서 출발하는 것은 두 번 틀렸다. ① v2 Phase-1 은 softening 없이 학습됐는데, 이미지 99k장과 학습의 대부분이 Phase-1 에 있어서 거기서 빼면 불일치가 대부분 남는다. v2 Phase-2 는 x2 에서 자기 baseline 을 0.0009 차로 넘지 못해, Phase-2 혼자서는 적응하지 못함을 보였다. ② v2 Phase-1 은 ECB 분기 이전 가중치라 `use_rep` 모델에 로드하면 키 이름에서 바로 실패한다. 그래서 Phase-1 에도 `use_rep`, 같은 0.5 softening, EMA 를 넣었다. softening 은 열화 레시피가 아니라 trainer(`Trainer._soften`)가 하므로 데이터 레시피는 바뀌지 않았다.
- **KneeLR**: 공식 Bicubic++ 구현(aselsan-research-imaging-team/bicubic-plusplus)의 "knee" 스케줄을 재현한 것이다(`src/engine/schedulers.py` `KneeLRScheduler`, b777628 "Align Bicubic++ scratch training with reference"). warmup·explore 가 0 이라 5e-4 에서 0 을 향해 직선으로 내리고, 하한은 `min_lr` 5e-6 이다. epoch 마다 한 번 step 하므로 `total_steps` = `epochs` = 120 이다.

---

## 2. 크롭 기하학 — 불변량은 "HR 크롭 576px"

Phase-1·Phase-2 공통 규칙이다. **`patch_size` 는 LR 기준**이고, 지켜야 할 것은 LR 숫자가 아니라 `patch_size × scale = 576` 이다.

| scale | patch_size | HR 크롭 |
|---|---|---|
| x2 | 288 | 576 |
| x3 | 192 | 576 |
| x4 | 144 | 576 |

- **어디에**: `configs/train/BicubicPlusPlus/bicubicplusplus_{2x,3x,4x}.yaml`, `configs/finetune/BicubicPlusPlus` 6개 전부. IR 은 v1 부터 이미 x3/x4 가 192/144, EO 는 v2 에서 바뀌었다.
- **왜 (Phase-1, LSDIR)**: 모든 스케일에 288 을 쓰면 x4 의 HR 크롭이 **1152px** 가 되는데 LSDIR 중앙값이 **1128×864** 라, 당시 코드에서는 표본의 **93%** 가 `ensure_min_image_size` 로 bicubic 확대됐다 — **정답(GT) 자체가 뭉개진 상태로 학습**된다. x3(HR 864px)에서도 **56%** 가 확대 대상이었다. 지금 `SRDataset.__getitem__` 은 HR 크롭보다 작은 이미지를 확대하지 않고 건너뛰어, 무작위로 다시 뽑은 충분히 큰 이미지를 쓴다(건너뛰기 fd86960, 무작위 재추첨 9770dce, 4.4). 그래서 576 을 어기면 GT 가 뭉개지는 대신 그만큼 표본이 버려진다(`ensure_min_image_size` 는 지웠다. `DenoiseDataset` 도 같은 `redraw_fitting_image` 로 건너뛴다).
- **왜 (Phase-2, EO)**: 288 이면 x3 의 HR 크롭 864px 가 **LSDIR train/ 의 40%** 를, x4 의 1152px 가 **60%** 를 건너뛴다 — 사전학습과 다른 크기로 파인튜닝하게 된다.
- **왜 (Phase-2, IR)**: 288 이면 HR 크롭이 864px(x3)/1152px(x4) 인데 **IR 소스가 최대 1280×1024** 라 전부 미달이었다.
- **폐기**: 전 스케일 288 방식, 그리고 v1 EO 의 x3/x4 `patch_size: 288`. v1 계열(`*_v1.yaml`)과 구세대 `SVFocusSRNet/*`, `QuickSRNet/*`(모두 삭제됨)의 `patch_size: 256` 은 이 보정 **이전** 값이다.

---

## 3. 배치 크기·메모리·데이터로딩

### 3.1 x4 배치 크기 — 43GB OOM 과 그 해소

| config | batch | grad accum | 유효 배치 |
|---|---|---|---|
| `bicubicplusplus_4x_{eo,ir}_v1` (삭제됨) | 8 | 2 | 16 |
| `bicubicplusplus_4x_eo.yaml` (v2·v3) | 16 | 1 | 16 |
| `bicubicplusplus_4x_ir.yaml` (v2·v3) | 8 | 2 | 16 |

- **왜**: x4 는 HR 패치가 **1152²** 이라 VGG 지각 손실을 켜면 batch 16 에서 **43GB** 를 먹고 OOM 이 났다(실측). batch 8 + `gradient_accumulation 2` 로 유효 배치는 같게, 메모리는 절반으로 우회했다.
- **왜 EO 만 되돌렸나**: v2 에서 `patch_size` 를 144 로 내려 **HR 크롭 1152 → 576, 메모리 1/4** 이 되자 EO 는 batch 16 + ga 1 로 복귀했다. ga=2 + batch 8 은 유효 배치가 같아도 **스텝 수가 2배**라 손해이기 때문.
- **주의(근거 주석 없음)**: IR x4 는 v2/v3 에서도 batch 8 + ga 2 를 그대로 뒀다. 되돌리지 않은 이유는 기록이 없다.
- **재현 제약 (EMA)**: 출하 v3 를 학습한 trainer 는 EMA 를 optimizer step 이 아니라 micro-batch 마다 갱신했다. 그래서 ga 2 인 IR x4 는 decay 0.999 가 step 기준 약 0.998(0.999²)이었다. 2026-10-16부터 optimizer step 마다 갱신하고(4.4), IR x4 config 의 decay 를 0.998 로 바꿔 같은 평균 길이를 유지한다. 비트 단위로 같지는 않다(예전에는 갱신 전 가중치도 한 번 더 섞였다). 나머지 출하 SR config 8개와 `bicubicppdenoise.yaml` 은 ga 1 이라 갱신 횟수가 그대로다.

### 3.2 `num_workers`

- **Phase-1 `= 8`** (v2 는 16): `configs/train/BicubicPlusPlus/*.yaml`. 16 은 서버 **64코어를 동시 3런**으로 나눈 몫이었다. `num_workers: 4`(v1 과 지운 구세대 config 대부분의 기본값)일 때 데이터 로딩이 병목이라 **GPU 사용률이 0~7% 로 굶었다**.
- **Phase-2 IR `= 8`**: `bicubicplusplus_{2,3,4}x_ir.yaml` (v2 부터. EO `bicubicplusplus_{2,3,4}x_eo.yaml` 는 4). **근거 주석 없음 — 값만 바뀌었다.**
- **주의**: **4 는 실측으로 부족하다고 판명된 값**이다. 다른 config 의 `num_workers: 4` 를 그대로 복제하지 마라.

---

## 4. 조기종료와 판정 지표

### 4.1 patience 30 → 50 (Phase-2 BicubicPlusPlus)

- **어디에**: `configs/finetune/BicubicPlusPlus` 전 파일. v1 EO 는 30, v1 IR x3/x4 는 이미 50, v2/v3 전부 50 (`2x_ir` 은 v2 에서 30 → 50 으로 마지막에 합류).
- **왜**: patience 30 은 **100에폭 코사인 스케줄과 맞지 않는다**. 30에폭 시점의 lr 이 아직 **시작값의 80%** 라, LPIPS 가 가장 많이 떨어지는 저학습률 구간에 진입조차 못 하고 잘린다. **미달 3런은 전부 마지막 에폭이 최고 기록**이었다 — 하강 중 절단된 것.
- **폐기**: patience 30.

### 4.2 판정 지표는 손실과 같은 성격으로 — `monitor`/`mode`

- **BicubicPlusPlus Phase-2 6개 전부**: `monitor: lpips`, `mode: min`. **지각 손실로 파인튜닝하면 지각 지표로 판정한다** (실험 14). Phase-1 3개는 순수 MSE 라 `psnr`/`max` 다.
- **규약**: `monitor` 는 `loss / psnr / ssim / lpips / niqe` 중 하나이고 **`mode` 는 지표 방향에 맞춰야 한다 — PSNR·SSIM 은 `max`, Loss·LPIPS 는 `min`**. `patience` = 개선 없는 epoch 허용 횟수, `min_delta` = 최소 개선 폭.

### 4.3 val LR 세트 — 게이트는 학습 레시피와 같은 열화로 만든 것만

- **어디에**: `configs/finetune/BicubicPlusPlus` 6개 전부의 `val_lr_root` (EO `val/LRx{2,3,4}_weaknoise_eo_v3`, IR `val/LRx{2,3,4}_weaknoise_ir`). Phase-1 3개의 `val/LRx{2,3,4}_v3` 도 같은 방식이다.
- **왜**: val LR 은 **학습 레시피 그대로** `examples/dataset_preparation/make_val_lr.py`, **seed 2026** 으로 생성했다. 학습에 softening 이 있는 Phase-1 과 EO Phase-2 는 같은 denoise 체크포인트를 `--denoise_checkpoint` 로 넘겨 짝수 인덱스(절반)를 softening 했다(c7be65d. 2026-10-15 x3 의 첫 두 장을 다시 만들어 Phase-1·EO 모두 비트 단위로 같음을 확인). 구 `LRx2/x3/x4` 는 **미기록 강열화**라 게이트로 부적합하다. IR 은 2026-10-06 합친 IR 레시피(지금 `sr_finetune_ir_phase-2.yaml`)로 다시 만들었고(8 머리말), 홀드아웃은 **OPENAI_RAYSENSE_SR 100장**.
- **주의**: 게이트 숫자를 비교할 때 **v1 val 셋과 v2 val 셋의 값을 섞지 마라** (열화 레시피가 다르다).

### 4.4 재현 기준 커밋 — 학습 분포를 바꾼 수정

출하 SR v3와 Denoise v1의 학습 데이터를 비트 단위로 다시 만들려면 아래 수정이 들어가기 전 커밋 `de502f6`에서 학습한다. 디스크에 있는 val 세트도 그 커밋의 `make_val_lr.py`로만 비트 단위로 같게 다시 만들어진다. 출하 모델과 비교할 때는 디스크의 val 세트를 그대로 쓴다.

- **2026-10-16 가우시안 노이즈 반올림**: `add_gaussian_noise`(와 출하 레시피에서는 꺼져 있다가 8.6에서 지운 `add_chroma_noise`)가 노이즈를 더한 뒤 uint8로 버림 변환을 해서, 화소가 평균 약 0.5 DN 어두워졌다(평탄 120 DN, σ 2~15에서 −0.49 DN). 반올림으로 바꿨다. EO·IR Phase-2와 Denoise 학습 입력이 바뀌고, Phase-1(resize만)은 그대로다. 기존 가드 테스트는 `.astype(np.uint8)` 표기만 찾아서 `.astype('uint8')`를 놓쳤다.
- **2026-10-16 작은 이미지 건너뛰기**: HR 크롭보다 작은 이미지를 만나면 바로 다음 파일로 넘어갔다. LSDIR은 출처별로 파일이 모여 있어 작은 이미지가 연달아 나오므로, 한 묶음이 그 뒤의 한 장으로 몰렸다(Phase-1 epoch당 `OST/tallgrass_019.png` 1,374회, EO Phase-2 `OST_HR/quokka_137.png` 228회). 무작위로 다시 뽑도록 바꿨다. Phase-1과 EO Phase-2의 샘플 분포가 바뀌고, 작은 이미지가 없는 IR은 그대로다.
- **2026-10-16 EMA 갱신 시점**: gradient accumulation 중에는 optimizer 가 누적 구간 끝에서만 step 하는데, EMA 는 micro-batch 마다 갱신해서 바뀌지 않은 가중치를 한 번 더 섞었다. step 마다 갱신하도록 바꿨다. ga 1 인 출하 config 9개는 그대로이고, ga 2 인 IR x4 는 decay 를 0.999 → 0.998 로 바꿔 평균 길이를 맞췄다(3.1).

---

## 5. 손실 함수

### 5.1 BicubicPlusPlus (현행) 레시피

| 항 | EO v2/v3 | IR (`bicubicplusplus_{2,3,4}x_ir.yaml`, 2026-10-06~) | 근거 |
|---|---|---|---|
| MSE | 1.0 | 1.0 | 앵커 |
| perceptual | 0.05, `gray: true` | **off** | 5.2 |
| ssim | 0.02 (warmup 10) | **0.05** (warmup 10) | 5.3, 5.6 |
| edge | 0.02, `mode: 'mean'` (warmup 10) | 0.02, `mode: 'mean'`, **`signed: true`** (warmup 10) | 5.4, 5.6 |
| gradient_variance (v3) | 0.05, patch 8, `mode: 'mean'`, warmup 10 | 동일 | 7.4 |

### 5.2 perceptual — EO 는 유지, IR 은 끈다

- **어디에**: EO 전 버전 `enabled`(weight 0.05). IR 은 v1 x2 만 켜져 있었고 **v1 x3/x4 부터 `enabled: false`**, v2/v3 IR 전부 false.
- **EO 유지 근거**:
  - 실험 11 — **다층** 지각 손실. relu2_2(미세구조)~relu4_4(의미)를 한 번의 VGG forward 로 함께 감독한다(`layers: {8: 0.25, 17: 0.5, 26: 0.25}`). 단층 relu5_4 는 **stride 16** 이라 머리카락 굵기 구조를 물리적으로 볼 수 없었다.
  - v1 Phase-2 에서 EO 는 이 항으로 **실제 이겼다**: x2 LPIPS **0.1325 → 0.1103**, x3 **0.2569 → 0.2329**.
- **IR 차단 근거**:
  - 22.7k 파라미터 / 수용영역 8~10px 모델에서 VGG 특징은 움직이지 않는다 (**23에폭에 0.341 → 0.334**).
  - 그래디언트 크기는 MSE 의 **22~30배**인데 방향이 거의 **직교(cos +0.098)** — 지각/충실도 맞바꿈이 아니라 그냥 옆으로 끄는 힘이고, **Adam 업데이트의 약 76%** 를 먹는다.
  - 외부 근거: PBVS 열화 SR 챌린지 2023~2025 상위팀 중 VGG perceptual 을 쓴 팀은 **하나뿐**, 그 팀은 **289M 파라미터 + 판별자**를 붙이고도 SSIM 꼴찌였다.
- **주의**: IR 을 끈 근거(직교 그래디언트, 구조적 바닥)는 **무채색 IR 에서 잰 값**이라 EO 로 옮길 수 없다. **EO 는 이 항으로 이겼으므로 같은 논리로 EO 에서 끄지 마라.**
- **죽은 설정**: v1 IR x3/x4 의 `perceptual` 블록은 `enabled: false` 인 채 값이 남아 있다. 그 안의 `warmup_epochs: 10` 근거만은 살릴 값 → 5.5.

### 5.3 ssim 가중치 0.1 → 0.02

- **어디에**: BicubicPlusPlus. v1 IR x2 는 0.1, **v1 IR x3/x4 부터 0.02**, v2/v3 전부 0.02 (+ `warmup_epochs: 10`).
- **왜**: PBVS **2025 우승(DRCT)** 과 **2024 2위(HAT-L)** 가 MSE 1.0 + SSIM 0.02 를 썼다. 그 모델들은 이 모델보다 **1000배 크다**. 0.1 은 **업계 관행의 2~5배**였다.
- **폐기**: ssim 0.1 (Bicubic++ SR 한정. Denoise v1 은 0.1 을 쓴다, 14.3).
- **IR 은 2026-10-06 부터 0.05**: SSIM 은 x4 바둑판을 억누르는 항으로 측정됐고(끄면 격자 63 → 538), 0.05 로 올리면 에지 대비와 저대비 질감이 오른다(5.6). EO 는 0.02 그대로.

### 5.4 edge — 형태는 그대로, 가중치만 내림

- **어디에**: BicubicPlusPlus 전 파일 `weight: 0.02` (v1 IR x2 만 0.05).
- **왜**: "**magnitude+threshold 형태가 할로에 둔감할 것**"이라는 가설을 세워 검증했으나, 합성 계단 엣지로 시험하니 **오히려 가장 크게 벌줬다 (+0.40 vs signed +0.17)**. 근거 없이 형태를 바꾸지 않기로 하고 **가중치만 ssim 과 같은 수준(0.02)** 으로 내렸다.
- **폐기(가설)**: "magnitude+threshold 를 signed 로 바꾸면 할로에 더 민감해질 것" — 합성 계단 엣지 실측으로 반증됨.
- **IR 은 2026-10-06 `signed: true` 로 바꿨다.** 위 결론은 합성 계단 엣지의 할로 벌점만 본 것이다. 실제 IR Phase-2 학습에서는 magnitude 형태가 x4 4×4 바둑판의 **최대 단일 원인**이었다(edge 를 끄면 격자 63 → 32). signed 로 바꾸면 격자를 만들지 않으면서 링잉과 평탄부 노이즈도 줄었다(5.6). EO 는 magnitude 그대로이고, EO 에서 바꿀 때는 EO 로 따로 검증한다.

### 5.5 `warmup_epochs: 10` — 손실 교체 충격 완화

- **어디에**: v1 IR x3/x4 의 ssim/edge(및 죽은 perceptual), v2/v3 전부의 ssim/edge/gradient_variance.
- **왜**: 손실 항을 0 → 목표 가중치로 **10에폭에 걸쳐** 올린다. 워밍업 없이 갈아끼운 이전 런은 **1에폭에서 이미 baseline 보다 나쁜 상태로 시작**해(x3_ir **+0.036**) 회복에만 **26에폭**을 썼다.

---


### 5.6 IR 바둑판 격자 수정 (2026-10-06) — 학습 설정 4가지

- **어디에**: `configs/finetune/BicubicPlusPlus/bicubicplusplus_{2,3,4}x_ir.yaml`, `configs/data/bicubicpp/sr_finetune_ir_phase-2.yaml`, 학습 데이터 `Finetunning_Dataset/train/lsdir_gray_922/`.
- **문제**: v3(09-14) IR x4 출력 전체에 4픽셀 주기 격자(특히 4열마다 세로선)가 깔렸다. PC FP32 출력에서도 같아서 양자화나 보드 문제가 아니라 가중치의 성질이다. Phase-1 에는 없고 Phase-2 에서 생겼다. x3 에도 3픽셀 격자가 있었다(Phase-1 의 4배).
- **바꾼 것과 근거** (실험 EXP-00~14, `docs/20260928-RESEARCH-X4IRGrid.md`):
  1. **흑백 LSDIR replay 922장을 학습 데이터에 섞는다** (`train/lsdir_gray_922`). IR 학습 사진은 x4 로 줄인 부드러운 영상뿐이라 그것만 배우면 세로선이 자란다. 열화를 Phase-1 식으로 바꿔도 그대로 자랐고(EXP-08-1), Phase-1 사진을 함께 보여 주면 사라졌다(EXP-06-1, 4열 세로선 31.6 → 2.5). Phase-1 학습 사진(LSDIR)에서 seed 2026 으로 섞어 짧은 변이 576 이상인 것을 흑백으로 저장했다(`Dataset_Replay/make_replay.py`, 데이터 서버에서 지워짐. 결과 폴더 `train/lsdir_gray_922`는 남아 있다).
  2. **edge `signed: true`** (5.4).
  3. **IR ssim 0.05** (5.3).
  4. **열 노이즈 σ 1~3 → 0.5~1.5 DN** (8.7).
- **결과 (x4, cam92)**: 격자 진폭 0.218 → 0.066 DN, 링잉 over_lr 8.3 → 3.8 %, 평탄부 노이즈 1.58 → 1.01 배, 에지 단차 0.990 → 1.013. 보드 화면에서 v3 만 세로 줄무늬가 보였다. 상세 `results/release_v3/V3_CHANGES_20261006.md`.
- **주의**: 모델 구조, 가중치 수, 그래프, DLC 이름은 v3 그대로다. 학습 파일 목록이 경로 순으로 정렬되므로 replay 폴더 위치를 바꾸면 같은 config 로도 가중치가 비트 단위로 같지 않다.

## 6. 무채색(achromatic) 문제

원인이 서로 다른 두 문제다. 하나는 **손실항이 색 방향으로 밀던 것**, 다른 하나는 **모델 출력 3채널 자체가 비대칭인 것**이다.

### 6.1 문제 1 — 손실항의 색 방향 제거 (EO·IR 공통)

- **어디에**: `configs/finetune/BicubicPlusPlus` 전 파일의 `edge.mode`, `perceptual.gray`, `gradient_variance.mode`.
- **`edge.mode: 'mean'` (`'gray'` 가 아니다)**: BT.601 루마 가중치(0.299/0.587/0.114) 자체가 **색 방향**이라, 무채색 면(R=G=B)에서 edge 그래디언트의 **50.4%** 가 색을 밀었다 — **4K EO 스트림의 회색 기둥에 크로마 얼룩**으로 나왔다. `'mean'` 은 세 채널에 같은 그래디언트를 줘서 크로마 성분이 **정확히 0**, 크기는 **0.0197 대 0.0228 로 86%** 남는다.
- **`perceptual.gray: true`**: VGG19 특징은 색에 민감해서 같은 무채색 테스트에서 그래디언트의 **77.8%** 가 색이었다. `gray=True` 는 두 이미지를 채널 평균으로 접고 VGG 에 넣어 크로마 성분이 0 이 된다(크기 **0.0332 대 0.0521 로 64% 유지**). IR v1 에서도 같은 이유로 측정 **+50% 색차** → gray 로 정확히 0.
- **`gradient_variance.mode: 'mean'` (v3)**: 같은 이유. **IR 프레임은 정의상 무채색이므로 BT.601 luma 는 절대 쓰지 않는다.**
- **주의**: **`'gray'` 를 새 설정에 쓰지 마라.** 단 **v1 EO 는 의도적으로 표준 BT.601 gray 를 썼고**, 그 주석은 "이 설정은 EO 재finetune 결정 시 적용됨 — 이미 완료된 v4 EO 런들은 perceptual-only 레시피였음"이라는 단서를 달고 있었다.

### 6.2 문제 2 — `achromatic_output: true` (IR 전용, v2 신설)

- **어디에**: `configs/finetune/BicubicPlusPlus/bicubicplusplus_{2,3,4}x_ir.yaml` 의 `model.achromatic_output` (v2 부터). **EO 에는 없다.**
- **왜**: IR 은 정의상 무채색 — 학습 GT 도 val HR 도 채널간 편차가 **0.000 DN** 이다. **모델은 아니었다**: 출력 3채널이 독립된 필터라서 컬러 사진으로 한 Phase-1 의 비대칭이 IR 파인튜닝 뒤에도 남는다(x3 에서 **100에폭 뒤에도 16.5 DN**).
  - **어떤 손실항도 원인이 아니다** — edge `mode:'mean'` 과 perceptual off 는 둘 다 크로마 기여 **0%** 로 측정됐다.
  - x2/x4 관찰: **W8 양자화가 그 편차를 크게 증폭한다.**
  - 해법: forward 에서 채널을 묶으면 공유 필터 하나가 학습되고 **학습·QAT·추론 전 구간**에서 유지된다.
- **폐기/주의**: **사후 접기(post-hoc fold)로는 안 된다** — 접은 뒤 **100스텝만 더 돌려도 0.0029 DN 이 다시 생겼고**, 당시 계획한 QAT 는 **30에폭**이었다. 출하 export 는 `--qat_epochs 0`(W16A16 PTQ)이라 지금은 접은 뒤 학습이 없지만, QAT 를 다시 돌리면 같은 문제가 생긴다. 비용 걱정도 불필요: 배포 시 최종 conv 가 선형이라 평균이 가중치로 접혀 사라지므로 **연산은 늘지 않는다.** 접기는 `BicubicPlusPlus.switch_to_deploy()` 가 한다(2026-10-17부터. 전에는 `6_apply_qat.py` 에만 있어(7단계는 그 함수를 불러 썼다) tools 와 AIMET 1~5단계는 평균 연산이 남은 그래프를 다뤘다).

---

## 7. v3 신설 항목 (BicubicPlusPlus Phase-1·Phase-2)

### 7.1 `use_rep: true` — 재파라미터화

- **어디에**: 출하 SR config 9개 전부의 `model.use_rep` (Phase-1 3개 + Phase-2 6개). Phase-1 에도 넣은 이유는 1.5.
- **왜**: 학습은 ECB 분기로 하고, **AIMET 전에 정확히 원래 conv 로 접힌다.** 따라서 **배포 그래프·레이턴시·DLC 는 변하지 않는다**(`ECBConv` 참조).
- **주의**: `upsample_mode: deconv`(x3)에서는 **head 가 ConvTranspose 라 손대지 않고**, body 만 접힌다.

### 7.2 `deploy_softening` — Phase-1 과 EO Phase-2

- **어디에**: `train.deploy_softening` — Phase-1 `bicubicplusplus_{2x,3x,4x}.yaml` 과 EO Phase-2 `bicubicplusplus_{2,3,4}x_eo.yaml`. **IR Phase-2 에는 의도적으로 없다**(IR 도 softening 으로 학습한 Phase-1 에서 출발한다, 13.17). 노이즈 σ 는 Phase-1 이 `noise_sigma: [2.0, 15.0]` 으로 적고, EO Phase-2 는 키가 없어 trainer 기본값 `[2.0, 15.0]` 을 쓴다 — 둘 다 2~15 DN.
- **왜(EO)**: 배포의 절반이 denoise 를 거치므로 **배치의 절반(`denoise_prob: 0.5`)** 을 그 denoise 망에 통과시킨다. **2026-09-08 실측**: denoise 경로의 SR 입력은 half-Nyquist 위 에너지가 **0.0065**, 학습용 LR 은 **0.0227** — denoise 단계 하나가 **3배 감소**를 설명한다. 나머지 절반은 denoise 단계가 없는 카메라를 위해 선명하게 둔다. 같은 날 같은 실기 프레임으로 내용을 맞춰 다시 재면(c7be65d) 학습식 축소 0.02077 → + denoise 0.01110(−47 %), 배포 0.00650 으로, 격차가 3.2배에서 1.7배로 준다.
- **왜(IR 제외)**: IR 경로에는 denoise 단계가 없다(`gst_compare_ir.sh`) — 재현할 대상 자체가 없다. IR 의 흐림은 **센서 자체**에서 오고(half-Nyquist 위 **0.0011**, EO 의 0.0065 대비), 그건 denoise pass 가 아니라 **downscale blur** 에 들어가야 한다.
- **주의**: `denoise_checkpoint` 는 SR v3 학습 당시 배포돼 있던 Denoise v2(SVFocusDenoiseNet, DLC `SVFocusDenoiseNet_1280x720_v2.dlc`)의 **FP32 원본**(260309, CSUAV 배포 계보)이다. **DLC 가 아니라 이 가중치를 쓰는 이유는 DLC 를 PyTorch 학습 루프 안에서 돌릴 수 없기 때문**이다(교사는 동결된 채 `no_grad`로 돈다).
- **softening 체크포인트 기록**: `checkpoints/csuav_deploy/train_svfocusdenoise_adv_x1_dim32_block3_epoch300_lr1e-4_charbonnier_edge_ssim_hotpixel_260309/best.pth`, md5 `0df4cc5174f9c8463ee28b0f036b42fb`, ep129. git 이 무시하는 파일이다(`checkpoints/`). **지우지 말고, v3 모델(Denoise v3 연구에서 나온 현 배포 BicubicPPDenoise[v1] 포함)로 바꾸지 마라** — 출하 SR v3 는 이 넷으로 연화한 분포에서 학습됐고, 바꾸면 같은 config 로 다시 학습해도 출하와 달라진다. trainer(`Trainer._load_denoise_net`)는 이 .pth 에 들어 있는 model config(`svfocussrnet`, `scale` 1, `dim` 32, `n_resblocks` 3, `use_advanced_rep` true)로 망을 만든다. 그래서 SVFocusSRNet 은 이 config 를 계속 만들 수 있어야 한다.

### 7.3 EMA

- **어디에**: 출하 SR config 9개 전부 (`decay: 0.999`, `start_epoch: 2`, ga 2 인 IR x4 만 `decay: 0.998`). Phase-1 에도 넣은 이유는 1.5.
- **왜**: 마지막 스텝이 아니라 **꼬리 구간의 평균**을 판정하고 출하한다.
- **주의**: EMA 는 optimizer step 마다 갱신된다(2026-10-16 전에는 micro-batch 마다). IR x4 의 decay 0.998 과 재현 제약은 3.1.

### 7.4 `gradient_variance` 손실

- **어디에**: Phase-2 6개 전부 (weight 0.05, patch 8, `mode: 'mean'`, warmup 10).
- **왜**: **게이트가 채점하는 양을 목적함수로 삼는 유일한 손실**이다. MSE 에는 그래디언트 확산 항이 없어서 엣지가 불확실한 곳에서 그냥 평균을 낸다 — 그게 지금 고치려는 softening 이다.
- **주의**: **Phase-1 은 순수 MSE 로 유지한다.** 이 항은 Phase-2 전용이다.

---

## 8. 열화 레시피 (`configs/data/`)

> **2026-10-06**: IR 레시피는 `sr_train_bicubic_weak_noise_ir.yaml` 하나로 합쳤다(`_ir_v1/v2/v3` 삭제). 이 파일은 2026-10-14 `configs/data/bicubicpp/sr_finetune_ir_phase-2.yaml` 로 이름이 바뀌었다(맨 앞 표). 아래에서 옛 `_ir_v3` 레시피로 적힌 값은 열 노이즈 σ(8.7)를 빼고 모두 이 파일에 그대로 있다. IR 검증 세트도 `val/LRx{2,3,4}_weaknoise_ir` 하나로 합치고 이 파일로 다시 생성했다(seed 2026). 그 이전 val 숫자와 섞지 마라.

### 8.1 `target_resize` 다운샘플 커널 다양화 — `[AREA, AREA, CUBIC, LINEAR, LANCZOS4]`

- **어디에**: `configs/data/bicubicpp/`의 `sr_train_phase-1.yaml`, `sr_finetune_eo_phase-2.yaml`, `sr_finetune_ir_phase-2.yaml`. 2026-10-14 배포판 정리 때 `sr_train_bicubic_only_eo_v2.yaml`, `sr_train_bicubic_weak_noise_eo_v2.yaml`, `sr_train_bicubic_weak_noise_ir.yaml`을 이 폴더로 옮기며 이름을 바꿨다(`denoise.yaml`도 같은 폴더).
- **왜**: cv2 의 축소는 안티에일리어싱을 하지 않는다. INTER_CUBIC 단독이면 학습 LR 이 "에일리어싱된 인위적으로 바삭한" 이미지가 되고, 모델은 **실제 센서에 없는 지글거림을 지우도록** 배운다. 실측: **실제 카메라 크롭 고주파 0.048 vs 학습 LR 0.379**. INTER_AREA 는 네 커널 중 유일하게 픽셀 면적 적분처럼 동작하므로 목록에 **두 번** 넣어 가중한다.
- **IR 에도 같은 값**: IR 은 EO 만큼 심하진 않았지만(비교 스트립에서 bicubic 대비 **0.89~1.03배**) 같은 편향을 공유하므로 함께 맞춘다.
- **폐기**: v1 계열(`sr_train_bicubic_only_eo_v1.yaml`, `..._weak_noise_eo_v1.yaml`, `..._ir_v1.yaml`, 모두 삭제됨)의 `interpolations: ["cv2.INTER_CUBIC"]` 단독 설정이 **위 편향의 원인**이다. 새 레시피에서 재사용 금지.

### 8.2 `clean_prob = 0.3` — 열화를 통째로 건너뛰는 비율

- **어디에**: Phase-2 (`sr_finetune_eo_phase-2.yaml`, `sr_finetune_ir_phase-2.yaml`). Phase-1(`sr_train_phase-1.yaml`)은 **0.0**.
- **왜**: 샘플의 30%는 노이즈/블러/JPEG 를 전부 건너뛰고 축소만 거친다. "**깨끗한 입력이 오면 그대로 두라**"를 명시적으로 가르치는 항이고, 현장에서 관측된 뿌연 출력에 대한 대응책이다. 실측(PDF 비교): EO 세 스케일 모두 bicubic 대비 **그래디언트 0.76~0.81배, 국소대비 0.81~0.84배**. IR 도 동일 근거로 EO 와 같은 값을 쓴다.
- **왜 Phase-1 은 0 인가**: Phase-1 은 열화가 `target_resize` 뿐이라 clean 경로와 일반 경로가 하는 일이 같다. 올려도 효과가 없다. 의미가 생기는 곳은 노이즈가 붙는 Phase-2 다. (같은 이유로 Phase-1 은 `target_resize` 외 모든 열화가 `enabled: false`, 확률·강도 0 이다. v1→v2 의 유일한 실질 변경은 커널 다양화다.) v3 Phase-1 도 데이터 레시피는 이것 그대로이고, 배치의 50 % softening(σ 2~15 DN 노이즈 + 동결된 v2 denoise)은 열화 레시피가 아니라 trainer 가 한다. EMA·ECB·KneeLR 와 함께 1.5, 7.2 참고.
- **주의**: clean 경로도 **8.1 의 커널 목록을 써야 한다**(`datasets.py` 수정 완료). 여기를 INTER_CUBIC 으로 하드코딩하면 clean 비율이 "모델이 과적합한 것으로 측정된" 옛 단일 커널 분포의 주머니가 되어 커널 랜덤화를 부분적으로 무효화한다.

### 8.3 IR 컬럼(수직선) 노이즈 — 세 번 바뀐 값

- **어디에**: `sr_finetune_ir_phase-2.yaml` 의 `stage2.ir_noise.vertical_line`. 값의 이력: `..._ir_v1.yaml`(폐기, 삭제됨) / `..._ir_v2.yaml`(폐기, 삭제됨) / `..._ir_v3.yaml`(현행 값, 지금 파일).
- **현행(v3) 근거**: **2026-09-08**, `/dev/video2` 600프레임을 **센서 자체 NUC 이후에** 측정한 실측 형태로 교체했다. 실제 잔차는 고립된 몇 개 열이 아니라 **모든 열이 조금씩 어긋난 것**이다 — 로버스트 진폭 **0.867 DN**, **56%의 열이 0.5 DN 초과**, **26%가 1.0 DN 초과**. 시간적으로 고정도 아니다(**18초 간격 프로파일 상관 0.046**). 고정 패턴이 아니므로 캘리브레이션 테이블로 들고 있지 않고 **샘플마다 새로 뽑는다**. (실측 상세: `src/data/datasets.py::add_vertical_line_noise` docstring)
- **v3 강도 파라미터** (`sigma_dn 0.75 / strong_frac 0.03 / strong_dn 6.0`): `sigma_dn` 은 가정값이 아니라 **캘리브레이션 값**이다. RAYSENSE 학습 프레임에 주입한 뒤 센서에 쓴 것과 같은 추정기로 재측정했을 때 0.75 가 센서 수치(진폭 0.87 DN, 56% > 0.5 DN, 26% > 1.0 DN)를 이미지 간 산포 이내로 재현한다. 센서 자체 sigma(0.867)보다 낮은 이유는 **uint8 반올림이 sub-DN 오프셋의 일부를 삼키기** 때문이고, 추정기가 주입량이 아니라 **엔드투엔드로 캘리브레이션**되어 있기 때문이다.
- **폐기 1 (v1)**: `prob: 0.02, intensity 1~3`. 현장에서 컬럼 노이즈 낀 입력에 모델이 대처하지 못한다는 보고가 있었는데, **2% 확률에 선 1~3개면 학습에서 사실상 본 적이 없는 것과 같다**. 같은 config 의 `hot_pixels` 가 0.4 인 것과 20배 차이라 의도한 값으로 보이지 않는다.
- **폐기 2 (v2)**: `prob: 0.35, intensity 2~8`. 확률은 살렸지만 **형태 자체가 틀렸다** — 2~8개 고립 열을 ±38 DN(풀스케일의 alpha 0.15)만큼 밀어내는 아티팩트인데 센서에는 그런 것이 없다. 이 형태로 학습한 모델은 실측 형태를 주입했을 때 **8%밖에 지우지 못했다**.
- **주의**: v1/v2 의 `intensity_min`/`intensity_max`(및 `alpha`)는 **현재 코드가 받아만 두고 무시한다**. 옛 config 도 로드는 되지만 실제로는 실측 형태가 주입되므로, **옛 파일의 강도 숫자를 근거로 삼지 마라.** 지금 레시피에도 이 키가 남아 있다: `sr_train_phase-1.yaml`·`sr_finetune_eo_phase-2.yaml` 의 꺼진 `vertical_line`(값 0)과 `sr_train.yaml` 의 `vertical_line`(2~10, 자기 `enabled` 는 true 지만 상위 `ir_noise` 가 꺼짐). 어느 쪽도 아무 일을 하지 않는다. 하위 키는 모르는 이름이어도 경고 없이 무시되고(`keys not read by the pipeline` 경고는 연산 이름에만 뜬다, `src/data/datasets.py` `log_active_degradations`), `vertical_line` 은 `enabled`·`prob`·`sigma_dn`·`strong_frac`·`strong_dn` 만 읽는다.

### 8.4 노이즈 강도 상한 원칙 — "노이즈를 더 주면 디노이징, 디노이징은 블러"

- **어디에**: IR 레시피 전반의 강도 결정 근거(`sr_finetune_ir_phase-2.yaml`, 옛 `_ir_v2`·`_ir_v3` 부터). Phase-2 EO 레시피의 낮은 확률값도 같은 방향.
- **왜**: 노이즈를 많이 볼수록 모델은 디노이징 쪽으로 기울고, 디노이징은 곧 블러다. **EO 에서 확인한 함정**이라 IR 에서 반복하지 않는다. 그래서 컬럼 노이즈는 **확률만 학습에 보이는 수준(0.35)으로 올리고 강도는 실측값에 고정**한다.

### 8.5 `hot_pixels` — SR 이 한 번도 본 적 없던 열화

- **어디에**: Phase-2 두 파일의 `stage2.hot_pixels`. `sr_finetune_eo_phase-2.yaml` prob 0.2, `sr_finetune_ir_phase-2.yaml` prob 0.4. 밀도는 전부 `density 0.0005~0.002`, `blob_prob: 0.0`, `blob_radius 1~1`.
- **왜**: **실험 13** — SR 이 핫픽셀을 학습에서 본 적이 없었다(denoise 학습에만 켜져 있었음). 밀도는 `denoise.yaml` 의 튜닝값(`density_min 0.0005 / density_max 0.002`)을 그대로 가져다 쓴다.
- **주의**: SR 쪽은 **blob 을 끈 상태**(`blob_prob 0.0`, 반경 1)로 쓴다. `denoise.yaml` 원본은 `blob_prob 0.1`, 반경 1~2, `sr_train.yaml`(비활성)은 `blob_prob 0.4`, 반경 1~4 로 서로 다르다 — **값을 옮길 때 공유되는 것은 밀도뿐이다.**

### 8.6 지운 열화 연산 (2026-10-16)

- **무엇을**: 현재 어떤 config 에서도 켜지지 않는 연산 11개(`unprocess_noise`, `turbulence`, `eo_shading`, `detail_attenuation`, `signal_instability`, `color_cast`, `highlight_clipping`, `chroma_noise`, `chroma_mottle`, `row_noise`, `periodic_grid_noise`. `eo_shading`·`chroma_mottle` 은 2026-05 MC-G105 deploy-hybrid config 에서 켜졌지만 그 config 들은 이미 지워졌다)와 `conditional_profiles`, `src/data/unprocess.py`, 쓰이지 않던 BasicSR 복사본을 지웠다. sr 계열 config 4개에 남아 있던 꺼진 `unprocess_noise`·`chroma_noise` 블록도 지웠다.
- **왜 결과가 같은가**: 꺼진 연산은 `enabled` 를 먼저 보고 난수를 뽑지 않으므로, 지워도 학습 쌍이 비트 단위로 같다(TC-REL-04-1 20개, `tests/test_make_val_lr.py` 3개 레시피로 확인).
- **주의**: 옛 저장 config(런 폴더, 릴리스 사본)에는 이 블록들이 남아 있어 로그에 'keys not read' 경고가 뜬다. 꺼진 블록이라 결과에는 영향이 없다. `unprocess_noise` 범위의 출처(Brooks et al. 2019, `lambda_shot ~ LogUniform[1e-4, 1.2e-2]`)는 다시 쓸 때 git 이력에서 찾는다.


### 8.7 IR 열 노이즈 `thermal_noise sigma 0.5~1.5` (2026-10-06)

- **어디에**: `sr_finetune_ir_phase-2.yaml` `stage2.ir_noise.thermal_noise` (이전 1~3, 근거 없는 값이었다 — 12).
- **왜**: 보드 IR 카메라(92·98) LR 평탄부에서 실측했다(`results/release_v3/eval/sensor_noise.py`). 프레임 안 공간 σ 는 0.8~1.3 DN(컬럼 성분 제외), 프레임 간 시간 σ 는 0.25~0.47 DN 이다. 1~3 DN 으로 학습하면 모델이 그 진폭 아래의 저대비 질감까지 노이즈로 보고 지운다(8.4 와 같은 함정). 낮추자 FLIR 저대비 질감이 0.716 → 0.814, 다른 IR 영상의 대각 격자가 257 → 64(Phase-1 수준)로 돌아왔다(EXP-12-1, seed 43 재현 EXP-12-2).
- **참고**: 컬럼 σ 는 1.5~1.9 DN 으로 컬럼 증강(`sigma_dn 0.75`)보다 크게 측정됐다. 추정기가 달라 직접 비교할 수 없어 바꾸지 않았다(8.3).
---

## 9. 지운 구세대 모델과 KD (2026-10-16)

- **무엇을**: 소유자 결정으로 BicubicPlusPlus 이전 모델(QuickSRNet·QuickDenoiseNet, LRCSR, SVSRNet, DDRNet, MambaIR/MambaIRv2, NAFNet, Restormer), KD 전체(`tools/train_kd.py`, `src/engine/kd_trainer.py`, `src/losses/kd_losses.py`), 출하 config 밖의 `train/`·`finetune/` config 42개, 그 config 들만 쓰던 ReduceLROnPlateau 스케줄러를 지웠다. 이 절에 있던 근거(공통 손실 레시피, 경량 모델 epoch, KD 실행 순서, teacher config 데이터 경로)와 옛 config 는 지운 커밋 a667e92 의 부모에서 본다(`git show a667e92^:docs/CONFIG_RATIONALE.md` 9절, `git show a667e92^:<옛 경로>`).
- **남긴 것**: `SVFocusSRNet`. 배포 열화 매칭(7.2)의 연화 넷이 이 구조이고, 센서가 바뀔 때의 후보다. 다시 학습할 때는 Bicubic++ config 를 바탕으로 새로 만든다.
- **연화 넷 구조 값의 근거**(체크포인트 사본 `checkpoints/csuav_deploy/train_svfocusdenoise_adv_x1_dim32_block3_epoch300_lr1e-4_charbonnier_edge_ssim_hotpixel_260309/train_config.yaml` 의 `model`): `scale: 1` — 해상도 유지. `dim: 32` — **Crouton Layout 최적 채널(32 정렬)**. `n_resblocks: 3` — 당시 비교 대상 **QuickDenoiseNet Medium 과 같은 연산량**. `use_advanced_rep: true` — **Hot Pixel(스파이크 노이즈) 방어**용 `RepBlock_Advanced`.
- **왜 출하 결과가 같은가**: 출하 config 10개는 지운 모델·KD·스케줄러를 부르지 않고, 연화 넷은 .pth 안에 저장된 model config 로 만든다(`src/engine/trainer.py` `_load_denoise_net`). 출하 체크포인트·export·데이터 레시피 golden 테스트로 확인했다(`docs/modules/20261015-MOD-ReleaseGolden.md`).

---

## 10. 공통 규약 (스케줄러 · 옵티마이저 · 체크포인트)

- **스케줄러**: 지원 타입은 KneeLR / CosineAnnealingLR / StepLR 이다. Phase-2 6개와 `Denoise/bicubicppdenoise.yaml` 은 CosineAnnealingLR, Phase-1 3개는 KneeLR(1.5)다. **`T_max` 는 전체 epoch 수와 맞추는 것이 규약**이고(지금 config 는 전부 지킨다), `eta_min` 은 하한 LR(`1e-7`).
- **Optimizer**: 지원 타입은 Adam / AdamW / SGD. **`momentum: 0.9` 는 SGD 를 쓸 때만 필요한 키**다.
- **체크포인트 이름과 재개**: `save_name` 은 선택 항목 — 설정하면 `checkpoints/{save_name}/` 에 저장되고, 비우면 **날짜시간으로 자동 생성**된다. 재개 방식 **두 가지는 의미가 다르다**: `--resume`(CLI 인자)은 **epoch·optimizer 상태까지 포함해 이어가고**, `pretrained_path`(config 항목)는 **가중치만 불러와 epoch 1 부터 새로 시작**한다.
- **`--epochs` 는 스케줄 길이를 바꾸지 않는다**: CLI `--epochs` 는 `train.epochs` 만 바꾼다(`tools/train.py` `apply_arg_overrides`). 출하 config 는 `T_max`·`total_steps` 를 적어 두었으므로 LR 곡선은 원래 길이대로 진행하다 중간에 끝난다. 짧은 런을 제대로 하려면 config 사본에서 `T_max`·`total_steps` 도 줄인다.
- **같은 `save_name`·`--work_dir` 로 다시 돌리면 폴더를 그대로 쓴다**: 에러 없이 `training_log.csv` 에 이어 쓰고(헤더를 다시 쓰지 않는다), `train_config.yaml` 과 체크포인트를 덮어쓴다. 실험과 스모크 테스트마다 새 `save_name` 이나 `--work_dir` 를 쓴다.

---

## 11. 양자화 설정 (`configs/aimet/`)

주석이 있던 JSON 은 `mmp_noprune_w16a16.json` 하나(`_comment` 키)뿐이다. 나머지는 근거가 코드에 있어 위치를 함께 적는다.

### 11.1 빈 `layers: {}` 를 일부러 넘긴다 (W16A16)

- **어디에**: `mmp_noprune_w16a16.json`, 사실상 같은 형태인 `mmp_noprune_w8a8.json`.
- **왜**: 층별 예외 없이 전부 기본 비트폭을 쓴다는 뜻이고, W16A16 은 `--param_bw 16 --output_bw 16` 으로 지정한다. **2026-10-16 전에는 빈 layers 라도 반드시 넘겨야 했다**: `mmp_config` 를 주지 않으면 `6_apply_qat.py` 가 SVSRNet 용 하드코딩 MMP 경로로 떨어졌다. SVFocusSRNet(v2 W16A16)에서는 **최종층 출력을 INT8 로 되돌렸고**, Bicubic++ 에서는 `head` 가 없어 AttributeError 로 멈췄다. 그 폴백은 SVSRNet 과 함께 지웠고(5·7단계의 같은 폴백도), 지금은 `mmp_config` 가 없으면 모든 층이 기본 비트폭이다. 출하 명령과 같게 두려고 계속 넘긴다. 주어진 파일이 없으면 `FileNotFoundError` 로 멈춘다.

### 11.2 빌트인 `htp_v73` 대신 `no_leakyrelu_supergroup` 파일을 쓴다

- **어디에**: `htp_v73_no_leakyrelu_supergroup.json`(기본값), `htp_v73_no_leakyrelu_supergroup_perchannel.json`. 근거는 `src/aimet/quant_sim.py` 의 `DEFAULT_QUANTSIM_CONFIG` 주석과 `examples/aimet_quantization/6_apply_qat.py` 의 `--quantsim_config` 도움말.
- **왜**: 빌트인 `htp_v73` 은 Conv+LeakyRelu 를 3개 supergroup 으로 융합하고, 그 결과 **Conv 출력 인코딩이 생략되어 Bicubic++(LeakyReLU 0.1) 8개 op 중 6개가 DLC 변환 때 float 으로 떨어진다.** 이 저장소 config 는 빌트인에서 **정확히 그 3개 LeakyRelu supergroup 만 뺀 것**이다(이 파일의 supergroup 은 11개). '8개 op 중 6개 float' 은 2026-08 qairt-converter 2.41.0 변환 로그에서 관측한 값이다(`examples/README.md` 3.1절). 빌트인은 supergroup 14개이고, 이 파일은 그중 LeakyRelu 를 담은 3개만 빠졌을 뿐 나머지 supergroup 과 `defaults`·`op_type`·`params`·`model_input`·`model_output` 이 빌트인과 같다(이미지 안 `aimet_torch/common/quantsim_config/htp_quantsim_config_v73.json` 과 비교, 2026-10-20). Conv+Clip 은 남겨두었으므로 Hardtanh 모델은 빌트인과 동일한 quantizer 집합을 얻는다.
- **주의**: **타깃 컨버터가 Conv+LeakyRelu 를 융합하지 않는 한 빌트인 이름(`htp_v73`)을 그대로 넘기지 마라.**
- **두 파일의 차이**: `defaults.per_channel_quantization` 이 `"False"` vs `"True"` 인 것 **하나뿐**이다(둘 다 `op_type` 수준에서는 Conv/ConvTranspose/Gemm/MatMul/PRelu 를 per-channel 로 둔다). 어느 쪽을 언제 쓰는지에 대한 근거는 없다.

### 11.3 지운 W4A8 MMP (2026-10-16)

- 이전 모델용 `mmp_quicksrnet-large_eo_{norpune,prune20}_w4a8.json`, `mmp_quicksrnet-large_ir_{norpune,prune50}_w4a8.json`, `mmp_svsrnet.json` 을 모델과 함께 지웠다. 공통 구조는 입력단과 말단부, `clip_output`·`module_add` 출력을 8bit 로 두는 것이었고, 층 선택의 실측 근거는 원래 남아 있지 않았다. 새 모델의 MMP 는 `3_analyze_quant_sensitivity.py` / `4_apply_quant_analyzer.py` 결과로 다시 만든다.

---

## 12. 근거가 없는 값 · 어긋난 설정 (이관 시 확인 필요)

주석이 없어 옮길 지식이 아예 없는 값들이다. **주석 제거로 잃는 것도 없지만, 반대로 "실측 근거가 있다"고 오해해서도 안 된다.**

**근거 주석이 없는 값**
- `configs/data/bicubicpp/denoise.yaml` 의 replay 열화(`degradation`) 확률·강도(주석 0줄). `camera_model` 블록과 `clean_prob` 의 근거는 14 에 있고, 14.3 은 replay 에서 무엇을 남겼는지만 말한다.
- Phase-2 확률 배분 — EO: blur 0.08 / gaussian·poisson 0.10 / jpeg 0.12 / stage2 blur 0.05 / common_noise 0.05 / final_jpeg 0.08. IR: gaussian·poisson 0.08 / jpeg 0.10.
- `gray_prob`(EO 0.5, IR 1.0), `kernel_probs [0.45, 0.25, 0.12, 0.03, 0.12, 0.03]`, `non_uniformity prob 0.03 / amp 0.02~0.06`, Phase-2 JPEG 품질 95~100. (`thermal_noise sigma` 는 2026-10-06 실측값 0.5~1.5 로 바꿨다 — 8.7)
- Phase-2 IR x4 가 batch 8 + ga 2 를 유지한 이유(3.1), IR Phase-2 의 `num_workers 8`(3.2), `perchannel` 파일 선택 기준(11.2).

**읽는 코드가 없는 죽은 키**
- 레시피 5개(`configs/data/bicubicpp/` 4개와 `sr_train.yaml`)의 `crop_size 256` 은 근거가 없는 값이 아니라 어떤 코드도 읽지 않는 키다(`src/`, `tools/`, `examples/` 검색). 학습 크롭은 `train.patch_size`(SR 은 LR 기준, HR 크롭 = `patch_size × scale`)가 정한다.

---

## 13. 설정 값을 바꾸기 전에 볼 것

이 프로젝트가 실제로 데인 함정만 모았다. 각 항목 뒤 괄호는 근거 절.

1. **노이즈 강도를 올리지 마라.** 노이즈를 더 주면 모델은 디노이징으로 기울고, 디노이징은 곧 블러다. EO 에서 이미 당했다. 확률만 올리고 강도는 실측값에 고정한다. (8.4)
2. **무채색 IR/EO 에 `mode: 'gray'` 를 쓰지 마라.** BT.601 루마는 그 자체가 색 방향이라 회색 면에 크로마 얼룩이 생긴다. `'mean'` 을 써라. (6.1)
3. **IR 무채색은 손실이 아니라 모델 구조로 고친다.** `achromatic_output: true`. 사후 접기는 100스텝 만에 되살아난다. 출하 export 는 QAT 없이(`--qat_epochs 0`) 접지만, QAT 를 다시 돌리면 그 안에서 되살아난다. (6.2)
4. **`patch_size` 를 만질 때는 LR 값이 아니라 `patch × scale = 576` 을 지켜라.** 어기면 HR 크롭보다 작은 이미지를 `SRDataset` 이 건너뛰어 표본이 버려진다(288 고정이면 x4 에서 LSDIR 표본의 93%). 예전 코드는 건너뛰지 않고 bicubic 확대해 GT 를 뭉갰다. (2)
5. **epoch 숫자로 학습량을 판단하지 마라.** `tmp/` 는 epoch 당 6,182 step 이다. 비교는 step 예산으로. (1.2)
6. **`num_workers: 4` 를 복제하지 마라.** 실측에서 GPU 가 0~7% 로 굶었다. (3.2)
7. **게이트 숫자를 버전 간에 섞지 마라.** v1 val 셋과 v2 val 셋은 열화 레시피가 다르다. 옛 `LRx2/x3/x4` 는 미기록 강열화라 게이트로 쓸 수 없다. (4.3)
8. **`monitor` 와 `mode` 는 짝이다.** PSNR·SSIM 은 `max`, Loss·LPIPS 는 `min`. 그리고 지각 손실로 학습했으면 지각 지표로 판정한다. (4.2)
9. **손실 항을 갈아끼울 때 `warmup_epochs: 10` 을 빼지 마라.** 워밍업 없이 바꾼 런은 1에폭에서 baseline 보다 나쁜 상태로 시작해(x3_ir +0.036) 회복에만 26에폭을 썼다. (5.5)
10. **IR 에서 perceptual 을 껐다고 EO 에서 끄지 마라.** 그 근거는 무채색 IR 에서 잰 값이고, EO 는 이 항으로 실제로 이겼다. (5.2)
11. **IR edge 는 `signed: true` 다(2026-10-06).** 합성 계단 엣지 할로 테스트(+0.40 vs +0.17)만 보고 magnitude 를 유지했다가, 그 형태가 x4 바둑판의 최대 원인이었다. magnitude 로 되돌리지 마라. EO 에서 바꿀 때는 EO 로 따로 검증한다. (5.4, 5.6)
12. **v1 계보 체크포인트에서 v2 레시피로 파인튜닝하지 마라.** 사전학습 커널 분포와 파인튜닝 커널 분포가 일치해야 한다. (1.3)
13. **`interpolations` 를 INTER_CUBIC 단독으로 되돌리지 마라.** clean 경로도 마찬가지다. 학습 LR 이 실제 센서에 없는 에일리어싱을 갖게 된다(고주파 0.379 vs 실카메라 0.048). (8.1, 8.2)
14. **옛 IR 컬럼 노이즈 config 의 강도 숫자를 근거로 쓰지 마라.** 코드가 받아만 두고 무시한다. v2 형태로 학습한 모델은 실측 노이즈를 8%밖에 못 지웠다. (8.3)
15. **AIMET 에 빌트인 `htp_v73` 을 그대로 넘기지 마라.** Conv+LeakyRelu supergroup 때문에 Bicubic++ 8개 op 중 6개가 DLC 에서 float 으로 떨어진다. (11.2)
16. **`mmp_config` 없이 export 한 2026-10-16 이전의 SVFocusSRNet W16 결과는 의심하라.** 그때의 하드코딩 폴백이 최종층을 INT8 로 되돌렸다(Bicubic++ 는 그 경로에서 멈췄다). 지금은 폴백이 없다. (11.1)
17. **IR Phase-2 에 `deploy_softening` 을 넣지 마라.** IR 경로에는 denoise 단계가 없다. IR 의 흐림은 센서에서 오고, 그건 downscale blur 로 다룬다. IR Phase-2 는 softening 으로 학습한 Phase-1 에서 출발하므로 그 영향은 이미 물려받았다. 이것이 맞는지는 다음 SR 설계 때 다시 본다. (1.5, 7.2)
18. **`gradient_variance` 를 Phase-1 에 넣지 마라.** Phase-1 은 순수 MSE 로 유지한다. (7.4)
19. **`--resume` 과 `pretrained_path` 를 혼동하지 마라.** 전자는 epoch·optimizer 까지 이어가고, 후자는 가중치만 불러와 epoch 1 부터 시작한다. (10)
20. **존재하지 않는 data config 를 가리키면 멈춰야 한다.** 지운 `tools/train_kd.py` 는 에러 없이 `data_config={}` 로 퇴화해 teacher config 에서 한 번 당했다(9). `tools/train.py` 는 `FileNotFoundError` 로 멈추고, 2026-10-16부터는 데이터셋(`SRDataset`, `DenoiseDataset`)이 빈 config 를 `ValueError` 로 거부하므로 AIMET 단계도 멈춘다.
21. **IR 학습 데이터에서 `train/lsdir_gray_922` 를 빼지 마라.** 빼면 x4 4열 세로선이 다시 자란다. (5.6)
22. **IR 학습 노이즈를 실측 센서보다 세게 주지 마라.** 저대비 질감이 지워지고 다른 장면에서 격자가 다시 생긴다. (8.7)

---

## 14. EO 카메라 모델과 Denoise 배포 v1 (BicubicPPDenoise[v1])

- **어디에**: `configs/data/bicubicpp/denoise.yaml`(`camera_model` 블록과 replay 열화), `configs/train/Denoise/bicubicppdenoise.yaml`(배포 v1 학습, 2026-10-13 소유자 배포 결정). 릴리스는 `results/denoise_release_v1/MANIFEST.md`. 코드는 `src/data/degradations.py`의 카메라 모델 절, 설계는 `docs/modules/20261008-MOD-CameraModel.md`, 모델 구조는 `docs/modules/20261012-MOD-BicubicPPDenoise.md`, 연구 계획은 `docs/20261008-RESEARCH-EODenoiseV3.md`.
- **왜**: EO 카메라(MC-G105)를 운용 설정 **Aperture 9 / 3D NR 5**로 유지하기로 했다(소유자 2026-10-08, 보드 화면 비교 후). 이 설정의 출력은 작은 신호를 최대 약 6배 키운 무늬 있는 노이즈를 가진다. v2 Denoise는 이런 노이즈를 본 적이 없고, "흐린 입력 → 선명한 원본"으로 학습해 스스로 샤프닝을 해서 헤일로를 더했다. 그래서 학습 입력을 카메라 출력과 같은 경로로 만든다. 정답은 `camera_model.target`으로 고른다. 기본 `camera`(A안: 같은 경로에서 노이즈만 뺀 영상)는 실기 채점에서 v2와 차이가 없어 기각했다(RESEARCH §5, 2026-10-08). 소유자 목표는 **Denoise 출력 = 우리가 가진 고품질 원본 형태**(`source`, C안)이고, 비교용 `optics`(B안: 렌즈 흐림만 남김)는 실험(EXP-05-1)에서 기각했다. 이 파일의 `target`은 배포 v1과 같은 `source`다. 실험 설정은 배포 결정(2026-10-13) 뒤 지웠다.

### 14.1 측정 조건
- 2026-10-08 회의실. 보드에서 인코딩 없는 NV12 1920×1080을 `capture_eo_raw.sh`로 저장. A4 차트(글자 줄, 5° 경사 사각형, 막대, 회색 패치 5단).
- 10배 3 m(실측 0.166 mm/px, 화소당 55 µrad): Aperture 0 / 2 / 5 / 9, Aperture 9 + 3D NR 0. 1배 1 m(0.482 mm/px): Aperture 0 / 5 / 9. 나머지 설정은 운용값(NR 2D 6, 3D 5, 노출 보정 10, Gamma 2, Contrast 11).
- 각 클립 프레임 20번부터 사용(시작 직후 노출 안정화 구간 제외). 값 단위는 카메라 출력 그대로 **BT.601 제한 범위 DN**(Y 16~235).
- 분석 스크립트와 결과: `results/denoise_release_v1/analysis/scripts/`(`cam_model.py` 노이즈 측정, `fit_camera_v2.py` 커널·밝기 세기·비대칭 맞춤, `cam_config_build.py` 설정 블록 생성, `cam_validate.py` 실측 대조), `results/denoise_release_v1/analysis/camera_model/`(측정 json, 맞춤 json, 대조 결과, 비교 그림). 이 분석 스크립트와 결과는 git 밖이다. `results/` 에서 git 이 추적하는 것은 릴리스 폴더의 기록 텍스트(MANIFEST 와 변경 기록 `.md`, `convert_all.sh`, `export.sh`)뿐이고, 바이너리(`.pth`, `.onnx`, `.dlc`)는 추적하지 않는다.

### 14.2 `camera_model` 값의 근거
| 키 | 값 | 근거 |
|---|---|---|
| `optics_sigma` | [0.55, 0.9] px | Aperture 0 경사 엣지의 Gaussian 등가 σ: 10배 0.83, 1배 0.58~0.64. 줌 범위 전체를 덮는다 |
| `noise_scale` | [0.7, 1.4], 로그 균등 | 1.0 = 회의실 조명에서 잰 수준. 실외 조도·장면에 따른 변화를 ±40 %로 둔 판단값(실외 측정 없음) |
| `gain_jitter` | 0.1 | 샤프닝 세기 ±10 %. 1배에서 엣지 고주파가 10배 맞춤값보다 약했던 점(14.4)을 일부 덮는다 |
| `noise_modes[0]` `static_3dnr5`, `prob` 0.7 | 휘도 스펙트럼·밝기별 표준편차(NLF)·색 스펙트럼·색 표준편차 | Aperture 0 클립(샤프닝 없음)의 잔차 = 샤프닝 전 노이즈. NLF는 기울기 작은 화소의 밝기 구간별 중앙값(Y 45: 1.73, 80: 2.27, 120: 3.02, 160: 2.77, 208: 2.52). 색은 반해상도 NV12 평면에서 잼(1.86 / 1.55) |
| `noise_modes[1]` `motion_3dnr_inactive`, `prob` 0.3 | 같은 항목 | 움직이는 화소에서는 3D NR이 꺼진다고 보고, Aperture 9 / 3D NR 0 클립에서 유도: 휘도 스펙트럼을 맞춘 샤프닝 증폭²으로 나누고, NLF를 같은 스펙트럼에서의 증폭으로 나눔. 색은 그 클립 값(3.21 / 2.43). **클립 1개 기반이고, 같은 Aperture에서 3D NR 0의 밝기 노이즈가 3D NR 5보다 낮게 나온 점(6.2 vs 12.8 DN)이 설명되지 않아** 확률을 낮게 뒀다 |
| `nps_bands` | 0.025~0.675 cycles/px, 0.05 간격(14개) | 측정 고리 중심. 2-D 구석(대각, 0.707까지)까지 잰다: 처음 0.475에서 끊었더니 그 밖을 마지막 값으로 고정해 대각 구석 노이즈가 실제의 1.4~3배가 됐다(검토 지적, 2026-10-08). 0.675 밖은 마지막 값 |
| `sharpen.kernel` | 9×9, D4 대칭, 합 0 | 밝은 종이(Y 약 155) 2-D 노이즈 스펙트럼 비 √(NPS_a9/NPS_a0)에 최소제곱 맞춤(오차 rms 0.03). 등방 커널로는 Aperture 9 노이즈의 2화소 상관 −0.44를 재현하지 못했다(−0.17). 같은 주파수 0.25 cycles/px에서 증폭이 가로·세로 3.0배, 대각 6.0배(실측 2.96 / 5.75) |
| `sharpen.brightness_y` / `brightness_gain` | Y 45 / 80 / 120 / 160 / 208 → 0.29 / 0.41 / 0.85 / 1.00 / 0.96 | 밝기 구간별 노이즈 표준편차 비(a9/a0: 1.97 / 2.39 / 3.94 / 4.51 / 4.38)를 위 커널과 Aperture 0 스펙트럼으로 역산. 카메라가 어두운 곳의 샤프닝을 줄인다 |
| `sharpen.brightness_sigma` | 2.0 | 밝기 세기를 국소 평균 밝기로 정한다(노이즈 자체가 세기를 흔들지 않게) |
| `sharpen.gain_pos` / `gain_neg` | 1.049 / 0.951 | Aperture 0 평균 영상을 Aperture 9 평균 영상에 어파인 정렬하고 엣지 근처 화소로 맞춘 넘침/꺼짐 비대칭. 평균은 1로 고정(작은 신호 증폭은 커널이 정함) |
| 빠진 항: 엣지 세기 억제 | – | 맞춰 보았으나 기준점(σ1.5 흐린 영상에서 81 DN/px)에 8-bit 영상의 엣지가 닿지 않아 작용하지 않았다(차트 최대 33 DN/px). 실제 카메라도 차트의 가장 강한 엣지에서 증폭을 줄이지 않았다 |
| `prob` | 0.75 | 나머지 0.25는 아래 replay. 대조 실험 축으로 바꿀 수 있다(RESEARCH §4) |
| `target` | `source` (배포 v1 = EXP-09-1, C안은 EXP-08-1에서 먼저 채택 2026-10-12). 기각: `camera`(A안), `optics`(B안). 키가 없으면 모듈 기본값 `camera` | 위 '왜' 참고. LR은 모드와 무관하다(같은 시드 → 같은 LR) |
| `hot_pixels` (EXP-06-1에서만 시험, 배포 설정에 없음) | `density` [20, 60] 개/Mpx, `amplitude` [8, 25] DN(샤프닝 전), `footprint` 십자(중심 1, 이웃 0.35) | Aperture 0 클립 1배·10배에서 같은 위치 49개(약 33개/Mpx, 10 DN 이상만 셈 → 범위를 그 아래위로), 세기 11~18 DN(10~90 %), 모양은 49개 평균. 샤프닝 전에 넣어 실제처럼 흰 점 + 어두운 테가 된다: 모델 13 DN → 62 DN·테 −0.23, 실측 52.5~57 DN·테 −0.22 |
| `noise_modes[1].impulses` (EXP-06-1에서만 시험, 배포 설정에 없음) | `density` [200, 600] 개/Mpx, `amplitude` [30, 55] DN(샤프닝 후), `bright_share` 0.31 | 3D NR 0 클립의 고립 점(6σ = 35 DN 넘는 것) 약 380개/Mpx, 세기 중앙 38·90 % 46 DN, 밝은 점 31 %. 3D NR 5 정지 장면은 1~1.5개/Mpx라 정지 모드에는 없다. 그 클립은 샤프닝이 운용 설정과 달라 보여(핫픽셀 11 DN, 테 없음) 샤프닝 후 실측값으로 넣는다 |

### 14.3 replay와 clean, 학습 값
- **replay(`degradation`)**: v2 레시피의 노이즈·JPEG·핫픽셀만 남기고 **블러·리사이즈·sinc는 끔**. 블러 열화가 "선명하게 만들기"를 가르쳐 헤일로를 만든다는 진단(RESEARCH HYP-01) 때문이다. 목적은 카메라 밖 노이즈(강한 노이즈, JPEG, 핫픽셀)에 대한 견고함 유지이고, 이 경로의 정답은 HR이다.
- **`clean_prob` 0.1**: 카메라 쌍에서는 입력과 정답이 모두 `target`이 고른 정답이다(`camera`면 카메라 정답, `source`면 원본. 깨끗한 영상은 건드리지 않게). v2는 0.2였는데, 실제 입력에는 항상 노이즈가 있으므로 줄였다.
- **학습**(`bicubicppdenoise.yaml`, 배포 v1 = EXP-09-1): Bicubic++ 배율 1, `dim` 32, 절반 해상도 본체 4층(`n_body: 4`, `upsample_mode: legacy` = stride-2 stem → PixelShuffle(2)), LeakyReLU, 재파라미터화(`use_rep`). 구조를 고른 이유는 W16A16 지연이다(MOD-BicubicPPDenoise, RESEARCH HYP-07). 처음부터 학습(`pretrained_path: null`), 300에폭, lr 1e-4, AdamW(weight decay 1e-4), Cosine(T_max 300, eta_min 1e-7), grad clip 1.0. 손실은 v2와 같다(charbonnier 1.0, edge 0.05 gray, ssim 0.1). 배치 128, 패치 256, EMA 0.999(시작 2에폭), 판정 `lpips`(patience 50). 배포 체크포인트는 ep287. 작업자 12: 표본당 약 71 ms(대부분 2K PNG 디코딩, 카메라 모델 14 ms)라 8개면 데이터 대기가 생긴다.
- **데이터 위치**: 학습 `/mnt/data2/jshong/SuperResolution/Pretrained_Dataset/train/`(sshfs 원본의 로컬 사본, 10,821장, 2026-10-08 복사, 소유자 승인). sshfs에서 읽으면 v4 Phase-1에서 6~17배 느렸다. 검증 `val_denoise_cam_a9nr5/` 는 이 로컬 사본에만 있고 sshfs 의 `Pretrained_Dataset/` 에는 없다(2026-10-19 확인). 로컬 사본의 `val/` 에는 Phase-1 게이트 `LRx{2,3,4}_v3` 와 지운 v4 실험의 `LRx{2,3,4}_weaknoise_eo_v4` 가 있고, EO Phase-2 게이트 `LRx{2,3,4}_weaknoise_eo_v3` 는 sshfs 에만 있다.
- **val**: `val_denoise_cam_a9nr5/`(HR = 카메라 정답, LR = 입력. 정답 C 런(`target: source`)은 이 폴더의 LR만 쓰고 정답은 `val_denoise/HR` 원본이다). `val_denoise/HR` 100장을 `examples/dataset_preparation/make_val_camera.py`로 시드 2026(+인덱스)에서 생성(14대역 설정으로 다시 생성, 2026-10-08 17:35). 파일 목록 sha256은 폴더의 `MANIFEST.sha256`, 목록 자체의 sha256 `dec62497…`. 학습 중 검증과 같은 잣대(`src/metrics`: 휘도 PSNR·SSIM, RGB LPIPS)로: 입력 그대로 31.97 / 0.803 / 0.243, v2 출시본 34.43 / 0.924 / 0.1380. (처음 적었던 '입력 그대로 29.69 dB'는 RGB PSNR이라 잣대가 달랐다.)

### 14.4 실측 대조 (모델 → 실제 Aperture 9)
| 항목 | 10배 실제 | 10배 모델(실제 a0 노이즈) | 10배 모델(합성 노이즈) | 1배 실제 | 1배 모델(실제 a0 노이즈) |
|---|---|---|---|---|---|
| 노이즈 스펙트럼 비(모델/실제), 14대역 | 1 | 0.95~1.03 | 0.86~1.03 | 1 | 0.90~0.98 (10대역) |
| 밝기별 휘도 σ (Y 100~140 / 140~180) | 11.69 / 12.28 | 11.98 / 11.92 | 11.51 / 11.76 | 12.16 / 10.53 | 12.40 / 10.80 |
| 2화소 상관 | −0.44 | −0.44 | −0.42 | −0.42 | −0.42 |
| 색 σ Cb / Cr | 1.83 / 1.51 | – | 1.86 / 1.55 | 2.10 / 1.55 | – |
| 회색 패치 엣지 넘침 P2 / P3 | 4.8 % / 0 % | 5.3 % / 0.9 % | – | 0 % / 1.8 % | 3.7 % / 2.0 % |
| 엣지 근처 고주파 rms | 17.03 | 17.20 | – | 20.19 | 24.28 |
- 1배에서는 엣지 근처 고주파가 실제보다 크다(24.3 vs 20.2). 운용이 망원 위주라 10배에 맞췄다.
- 낮은 대비 엣지(P4, 약 10 DN)의 어두운 쪽 꺼짐은 실제가 더 크다(22 % vs 0 %). 평균 영상의 남은 노이즈와 섞인 값이라 확정하지 않았다.

### 14.5 한계
- 카메라 톤 곡선(검정 뭉개짐), 디모자이크 가짜 색은 모델에 없다. 핫픽셀·점 노이즈는 배포 설정에 없다(14.2, EXP-06-1에서만 시험). 가로·세로 줄무늬 노이즈는 측정 한계(0.1~0.2 DN) 수준이라 넣지 않았다(측정 `results/denoise_release_v1/analysis/noise_census/`).
- 움직임 모드 값은 클립 1개에서 유도했다(14.2).
- 저조도·실외 장면에서 다시 재지 않았다. 이 레시피의 판정은 회의실 클립과 합성 val로만 한다.
