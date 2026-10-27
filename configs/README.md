# Configs

이 폴더에는 학습·파인튜닝 설정(YAML), 데이터 레시피(YAML), AIMET 설정(JSON)이 있다. 이 문서는 각 키가 코드에서 실제로 무엇을 하는지와 새 실험용 config 를 만드는 방법을 다룬다. 값을 왜 그렇게 정했는지(실측 근거, 폐기된 값)는 [docs/CONFIG_RATIONALE.md](../docs/CONFIG_RATIONALE.md)에 있다. 값을 바꾸기 전에 그 문서의 13절(설정 값을 바꾸기 전에 볼 것)을 먼저 읽는다.

명령은 모두 호스트 셸에서 저장소 루트를 현재 폴더로 두고 실행한다. Python 작업은 `docker/run.sh` 가 띄우는 컨테이너 안에서 돈다. 호스트의 `python3`(3.8)에는 jshong 의 `~/.local` 에 다른 프로젝트용 torch 2.4.1 만 있고 AIMET·accelerate·pyiqa 가 없으므로 작업 환경이 아니다. `docker/run.sh` 의 옵션과 거부 메시지는 [docker/README.md](../docker/README.md)에, `tools/train.py` 등 CLI 의 전체 옵션과 출력 폴더 구성은 [tools/README.md](../tools/README.md)에 있다. 컨테이너의 작업 폴더가 저장소 루트이므로 config 안의 상대 경로(`checkpoints/...`, `configs/...`)는 저장소 루트 기준이다.

## 1. 무엇이 어디에 있나

### 1.1 출하 학습 config (10개)

| 파일 (`configs/` 아래) | 용도 | `task` | 데이터 레시피 (`data_config_path`) |
|---|---|---|---|
| `train/BicubicPlusPlus/bicubicplusplus_{2,3,4}x.yaml` | SR Phase-1 사전학습. EO·IR Phase-2 의 공통 출발점 | `sr` | `configs/data/bicubicpp/sr_train_phase-1.yaml` |
| `finetune/BicubicPlusPlus/bicubicplusplus_{2,3,4}x_eo.yaml` | SR Phase-2 EO 파인튜닝 | `sr` | `configs/data/bicubicpp/sr_finetune_eo_phase-2.yaml` |
| `finetune/BicubicPlusPlus/bicubicplusplus_{2,3,4}x_ir.yaml` | SR Phase-2 IR 파인튜닝 | `sr` | `configs/data/bicubicpp/sr_finetune_ir_phase-2.yaml` |
| `train/Denoise/bicubicppdenoise.yaml` | BicubicPPDenoise[v1], EO Denoise (배율 1) | `denoise` | `configs/data/bicubicpp/denoise.yaml` |

이 10개는 출하 모델(SR v3, Denoise v1)을 학습한 기록이다. 어떤 체크포인트가 어떤 config 에서 나왔는지는 [results/release_v3/MANIFEST.md](../results/release_v3/MANIFEST.md)와 [results/denoise_release_v1/MANIFEST.md](../results/denoise_release_v1/MANIFEST.md)에 있다. 출하 config 는 고치지 않고, 실험은 사본으로 한다(7절).

### 1.2 데이터 레시피 (`configs/data/`)

| 파일 | 쓰는 config | 요약 |
|---|---|---|
| `bicubicpp/sr_train_phase-1.yaml` | Phase-1 | `target_resize` 만 켬, `clean_prob` 0 |
| `bicubicpp/sr_finetune_eo_phase-2.yaml` | EO Phase-2 | 약한 노이즈·블러·JPEG·핫픽셀, `clean_prob` 0.3 |
| `bicubicpp/sr_finetune_ir_phase-2.yaml` | IR Phase-2 | EO 와 같은 틀에 IR 컬럼·열 노이즈를 더함, `clean_prob` 0.3 |
| `bicubicpp/denoise.yaml` | Denoise | EO 카메라 모델(`camera_model`) 75 % + replay 열화 25 %, `clean_prob` 0.1 |
| `sr_train.yaml` | 출하 config 는 쓰지 않음 | 옛 Real-ESRGAN 식 전체 열화(블러·리사이즈·노이즈·JPEG·sinc). SR config 가 레시피를 지정하지 않으면 이것이 쓰인다 |

### 1.3 AIMET JSON (`configs/aimet/`)

| 파일 | 쓰는 곳 |
|---|---|
| `htp_v73_no_leakyrelu_supergroup.json` | AIMET quantsim config 기본값. `src/aimet/quant_sim.py` 의 `DEFAULT_QUANTSIM_CONFIG` 이고 AIMET 3~7단계 `--quantsim_config` 의 기본값이다 |
| `htp_v73_no_leakyrelu_supergroup_perchannel.json` | 위 파일과 `defaults.per_channel_quantization` 하나만 다르다. 기본으로 쓰는 코드는 없고 `--quantsim_config` 로 직접 넘긴다 |
| `mmp_noprune_w16a16.json` | 출하 W16A16 export 에 넘기는 `--mmp_config`. 층별 예외가 없는 빈 MMP 다 |
| `mmp_noprune_w8a8.json` | 같은 형태의 빈 MMP |

### 1.4 지운 config

출하 config 밖의 `train/`·`finetune/` config 42개(QuickSRNet, SVFocusSRNet, MambaIR, `SR/`, `Denoise/` 의 나머지)는 2026-10-16 커밋 `a667e92` 에서 지웠다. 옛 내용은 `git show a667e92^:<옛 경로>` 로 본다. 이름이 바뀐 출하 config·레시피의 옛 이름은 [CONFIG_RATIONALE](../docs/CONFIG_RATIONALE.md) 맨 앞 표에 있다. SVFocusSRNet 을 다시 학습할 때는 Bicubic++ config 를 복사해 model 블록을 바꾼다(3.2).

## 2. config 가 학습에 들어가는 순서

`tools/train.py` 는 아래 순서로 config 를 읽는다. 학습이 이상하면 이 순서대로 로그를 확인한다.

1. `--config` 의 YAML 을 읽는다. `--resume` 이면 그 대신 체크포인트가 있는 폴더의 `train_config.yaml` 과 `data_config.yaml` 을 읽는다.
2. 데이터 레시피를 고른다. 우선순위는 `--data_config`, config 의 `data_config_path`, task 기본값(`denoise` 는 `configs/data/bicubicpp/denoise.yaml`, `sr` 은 `configs/data/sr_train.yaml`) 순이다. 지정한 경로가 없으면 `FileNotFoundError` 로 멈춘다.
3. CLI 덮어쓰기(`--model`, `--task`, `--scale`, `--epochs`, `--batch_size`, `--lr`)를 적용한다. 값이 0 이면 적용되지 않는다.
4. `task` 로 데이터셋을 고른다. `sr` 은 `SRDataset`, `denoise` 는 `DenoiseDataset` 이다. 다른 값이나 빈 값은 `ValueError: unknown task` 로 멈춘다. `denoise` 인데 `model.scale` 이 1 이 아니면 `ValueError` 다.
5. 모델을 만들고, `train.pretrained_path` 가 있으면 그 가중치를 읽는다. `--resume` 일 때는 읽지 않는다.
6. 출력 폴더를 정한다. 우선순위는 `--work_dir`, `checkpoints/<train.save_name>`, `checkpoints/<model>_<task>_x<scale>_<YYYYmmdd_HHMMSS>` 순이다. 폴더가 이미 있어도 멈추지 않는다(9절).
7. 새 런이면 출력 폴더에 설정 사본 네 개를 쓴다.

| 파일 | 내용 |
|---|---|
| `train_config.orig.yaml`, `data_config.orig.yaml` | 넘긴 파일 그대로 (주석과 키 순서 유지) |
| `train_config.yaml` | CLI 덮어쓰기를 적용한 실제 설정. 레시피가 `data_config` 키로 들어 있고, `data_config_path` 는 실제로 쓴 레시피 경로다 |
| `data_config.yaml` | 실제로 쓴 레시피 |

체크포인트 `.pth` 의 `config` 키에도 `train_config.yaml` 과 같은 실제 설정이 들어간다. `--resume` 은 `train_config.yaml` 을 다시 쓰지 않고 `train_config.resume_ep<E>_<시각>.yaml` 을 따로 남긴다. 그래서 재개 전에 런 폴더의 `train_config.yaml` 을 고치면 그 값이 재개에 적용된다. 학습률만은 예외로, optimizer·scheduler 상태에서 복원된다(`--resume` 과 `--lr` 은 같이 쓸 수 없다).

## 3. 학습 config 의 키

### 3.1 최상위 키

| 키 | 필수 | 뜻 |
|---|---|---|
| `task` | 예 | `sr` 또는 `denoise`. 데이터셋을 고르는 유일한 키다 |
| `data_config_path` | 아니요 | 데이터 레시피 경로. 출하 10개는 모두 적는다. 새 SR config 에서 빠뜨리면 무거운 `sr_train.yaml` 로 학습하게 되므로 반드시 적는다 |
| `model` | 예 | 모델 구조 (3.2) |
| `data` | 예 | 학습·검증 이미지 경로 (3.3) |
| `train` | 예 | 학습 루프 (3.4) |
| `loss` | 예 | 손실 항 (3.5) |

### 3.2 `model`

`src/models/__init__.py` 의 `build_model` 이 `name` 으로 모델을 고른다. 지원하는 이름은 `bicubicplusplus` 와 `svfocussrnet` 이고, 다른 이름은 `ValueError` 다.

`bicubicplusplus` 는 `scale`(기본 2)을 `scaling_factor` 로, `dim` 을 `n_feats` 로 바꾸고 나머지 키를 모두 `BicubicPlusPlus.__init__` 에 넘긴다. 그래서 키 이름을 틀리면 기본 모델로 학습하는 대신 `TypeError` 로 바로 멈춘다.

| 키 | 기본값 | 값과 제약 |
|---|---|---|
| `scale` | 2 | 1 이상의 정수. 1 이면 Denoise 모델 |
| `dim` | 32 | 특징 채널 수 |
| `in_channels`, `out_channels` | 3, 3 | |
| `bias` | `false` | |
| `negative_slope` | 0.1 | LeakyReLU 기울기 |
| `act` | `leaky_relu` | `leaky_relu` 또는 `relu` |
| `clamp_output` | `true` | 마지막에 Hardtanh `final_act` 를 둔다. DLC 변환(`--set_output_layers`)과 보드 gst(`layers=`)가 이 이름을 출력 층으로 쓰므로 끄지 않는다 |
| `upsample_mode` | `legacy` | `legacy`(stride-2 stem 뒤 PixelShuffle), `auto`, `deconv`. 출하 SR 은 x2·x4 가 `auto`, x3 가 `deconv`, Denoise 는 `legacy` |
| `mid_feats` | 16 | |
| `icnr_init` | `true` | |
| `use_rep` | `false` | 학습 때 ECB 분기를 쓰고 배포 전에 conv 하나로 접는다. SR 에서는 `upsample_mode: legacy` 와 같이 쓸 수 없다 |
| `achromatic_output` | `false` | IR 전용. 출력 3채널을 평균해 무채색으로 묶는다. 배포 때 마지막 층 가중치로 접혀 연산이 늘지 않는다. 배율 1 에서는 쓸 수 없다 |
| `n_body` | 2 | 본체 conv 수. SR 은 2 만, 배율 1 은 2 이상 (Denoise v1 은 4) |

`svfocussrnet` 은 `scale`, `dim`(기본 32), `n_resblocks`(기본 8), `use_advanced_rep`(기본 `false`)만 읽고 다른 키는 조용히 무시한다. 지금 이 구조는 softening 넷(3.4 `deploy_softening`)으로만 쓰인다.

**학습한 뒤에는 model 블록을 바꾸지 않는다.** `tools/evaluate.py`, `tools/inference.py`, `tools/export.py`, `examples/aimet_quantization/6_apply_qat.py` 는 `--config` 의 model 블록을 체크포인트에 저장된 model 블록과 비교하고(`src/engine/checkpoint.py` 의 `check_model_config`), 하나라도 다르면 `ValueError: <경로> was trained with another model config; differing keys: ...` 로 멈춘다. 기본값과 같은 키를 새로 적어 넣기만 해도 다르다고 본다. 이 검사가 있는 이유는 `achromatic_output` 처럼 가중치가 없는 키는 strict load 로도 잡히지 않아서, IR 체크포인트를 EO config 로 열면 에러 없이 회색 입력에 색이 생기기 때문이다. 평가와 export 에는 그 런 폴더의 `train_config.yaml` 을 넘기면 항상 맞는다.

### 3.3 `data`

| 키 | 뜻 |
|---|---|
| `train_root` | 문자열 또는 문자열 목록. 하위 폴더까지 `.png`, `.jpg`, `.jpeg`(대소문자 무관)를 모아 전체 경로 순으로 정렬한다 |
| `val_hr_root`, `val_lr_root` | 문자열 또는 목록. 하위 폴더는 보지 않는다. HR 과 LR 을 확장자를 뺀 파일 이름으로 짝짓고, 짝이 안 맞는 파일이나 겹치는 이름이 하나라도 있으면 `ValueError` 로 멈춘다 |
| `type` | `eo`/`ir` 라벨일 뿐 읽는 코드가 없다 |

두 val 경로는 같이 적거나 같이 비운다. 둘 다 비우면 `train.allow_no_validation: true` 가 있어야 학습이 시작되고, 그때는 `best.pth` 가 만들어지지 않고 조기 종료도 일어나지 않는다.

`train_root` 의 파일 목록은 경로 순으로 정렬되므로, 같은 config 라도 하위 폴더 위치를 바꾸면 학습 순서가 달라져 가중치가 비트 단위로 같지 않다.

**데이터 위치 (2026-10-19 확인).** `/mnt/data_server` 는 데이터 서버의 sshfs 마운트(공유 1 GbE)이고, `/mnt/data2/jshong/SuperResolution` 은 로컬 디스크 사본이다. 아래 경로는 `/mnt/data_server/etc/jshong/SuperResolution/` 아래다.

| 데이터 | 경로 | 장 수 | 로컬 사본 |
|---|---|---|---|
| Phase-1 학습 | `Pretrained_Dataset/tmp/` | 98,897 | 있음 |
| Phase-1 검증 | `Pretrained_Dataset/val/HR`, `val/LRx{2,3,4}_v3` | 각 100 | 있음 |
| EO Phase-2 학습 | `Pretrained_Dataset/train/` (출처 폴더 11개) | 10,821 | 있음 |
| EO Phase-2 검증 | `Pretrained_Dataset/val/HR`, `val/LRx{2,3,4}_weaknoise_eo_v3` | 각 100 | HR 만 있음 |
| IR Phase-2 학습 | `Finetunning_Dataset/train/` (`OPENAI_RAYSENSE_SR` 1,844 + `lsdir_gray_922` 922) | 2,766 | 없음 |
| IR Phase-2 검증 | `Finetunning_Dataset/val/HR`, `val/LRx{2,3,4}_weaknoise_ir` | 각 100 | 없음 |
| Denoise 학습 | `Pretrained_Dataset/train/` | 10,821 | Denoise config 는 로컬 사본을 읽는다 |
| Denoise 검증 | `Pretrained_Dataset/val_denoise/HR`, `val_denoise_cam_a9nr5/LR` | 각 100 | `val_denoise_cam_a9nr5` 는 로컬에만 있다 |

로컬 사본 `/mnt/data2/jshong/SuperResolution/Pretrained_Dataset/val/` 에 있는 `LRx{2,3,4}_weaknoise_eo_v4` 는 지운 v4 실험의 것이다. EO Phase-2 게이트(`_weaknoise_eo_v3`)와 다르므로 대신 쓰지 않는다.

출하 Phase-1·Phase-2 config 는 sshfs 에서 읽는다. sshfs 는 느려서 2026-10-06 Phase-1 이 1.1~3.2 s/step 이었고, 로컬 사본에서는 약 0.2 s/step(한 epoch 약 21분)이었다. 새 Phase-1 실험은 복사한 config 의 경로를 로컬 사본으로 바꾸는 것이 좋다(7.2). 로컬 사본의 장 수는 sshfs 원본과 같다(2026-10-19). Phase-1 데이터 사본(2026-10-06 복사)은 경로와 파일 크기가 원본과 같음을 확인했고, 내용(md5)은 300장 표본만 비교했다.

`docker/run.sh` 는 config 가 `/mnt/data2` 만 읽더라도 `/mnt/data_server` 가 읽을 수 있는 sshfs 마운트가 아니면 시작을 거부한다. 호스트를 재부팅한 뒤에는 root 가 `-o allow_other` 로 다시 마운트해야 한다([docker/README.md](../docker/README.md)).

### 3.4 `train`

필수 키는 `epochs`, `batch_size`, `lr`, `num_workers`, `patch_size`, `save_interval` 이다. 빠지면 `KeyError` 로 멈춘다. `save_interval` 만은 첫 epoch 의 학습과 검증이 끝난 뒤에 읽으므로 그때 멈춘다.

| 키 | 기본값 | 뜻 |
|---|---|---|
| `epochs` | 필수 | 마지막 epoch 번호. `--epochs` 가 덮어쓴다 |
| `batch_size` | 필수 | |
| `lr` | 필수 | 시작 학습률 |
| `num_workers` | 필수 | DataLoader 작업자 수. 4 는 실측으로 부족했다(CONFIG_RATIONALE 3.2) |
| `patch_size` | 필수 | SR 은 **LR 크롭**의 한 변이고 HR 크롭은 `patch_size × scale` 이다. 출하 SR 은 HR 크롭이 모두 576(288/192/144). Denoise 는 크롭 그대로(256) |
| `save_interval` | 필수 | `epoch_<N>.pth` 와 `plots/probe_ep<N>.png` 를 남기는 주기 |
| `gradient_accumulation_steps` | 1 | |
| `optimizer.type` | `Adam` | `Adam`, `AdamW`, `SGD`. 다른 값은 `ValueError` |
| `optimizer.betas` | `[0.9, 0.999]` | Adam·AdamW 만 읽는다 |
| `optimizer.momentum` | 0.9 | SGD 만 읽는다 |
| `weight_decay` | 0.0 | `optimizer` 블록 안이 아니라 `train` 바로 아래에 둔다 |
| `grad_clip` | 0.0 | 0 보다 크면 gradient norm 을 자른다 |
| `scheduler` | StepLR | 아래 표 |
| `ema` | 끔 | `enabled`(`false`), `decay`(0.999), `start_epoch`(2). 켜면 `start_epoch` 부터 검증과 모든 체크포인트의 `model_state_dict` 가 EMA 가중치다. optimizer step 마다 갱신한다 |
| `early_stopping` | | `monitor`(`loss`), `mode`(`min`), `patience`(10), `min_delta`(0.0). 아래 설명 |
| `deploy_softening` | 끔 | 아래 설명 |
| `save_name` | 없음 | 출력 폴더 `checkpoints/<save_name>/`. `--work_dir` 를 주면 쓰이지 않는다 |
| `pretrained_path` | 없음 | 새 런의 시작 가중치 (4.3) |
| `seed` | 42 | `--seed` 가 이긴다 |
| `prefetch_factor` | 4 | `num_workers` 가 0 보다 클 때만 |
| `val_batch_size`, `val_num_workers` | 1, 0 | |
| `plots` | `true` | `plots/` 그림을 그린다 |
| `allow_no_validation` | `false` | 3.3 |

**scheduler.** 모든 스케줄러는 batch 마다가 아니라 epoch 마다 한 번 step 한다.

| `type` | 키 (기본값) | 출하 사용 |
|---|---|---|
| `KneeLR` | `peak_lr`(`lr`), `warmup_steps`(0), `explore_steps`(0), `total_steps`(`epochs`), `min_lr`(0) | Phase-1: 5e-4 에서 직선으로 내려 5e-6 에서 멈춘다. `total_steps` 120 |
| `CosineAnnealingLR` | `T_max`(`epochs`), `eta_min`(0) | Phase-2 `T_max` 100, Denoise `T_max` 300, 둘 다 `eta_min` 1e-7 |
| `StepLR` | `step_size`(50), `gamma`(0.5). `scheduler` 블록과 `train` 블록 어느 쪽에 적어도 읽는다 | 없음 |

다른 `type` 은 `ValueError` 다. 출하 config 는 `T_max`·`total_steps` 를 적어 두었으므로 `--epochs` 로 epoch 수만 바꾸면 LR 곡선은 원래 길이 그대로 진행하다 중간에 끝난다. 짧은 실험을 제대로 하려면 config 사본의 `T_max` 나 `total_steps` 도 같이 바꾼다.

**early_stopping.** `monitor` 는 `loss`, `psnr`, `ssim`, `lpips`, `niqe` 중 하나이고 `mode` 는 `min` 이나 `max` 다. 다른 값은 `ValueError` 다. 그러나 `mode` 가 지표의 방향과 맞는지는 검사하지 않는다. `psnr`·`ssim` 은 `max`, `loss`·`lpips`·`niqe` 는 `min` 으로 적는다. 틀리면 가장 나쁜 epoch 가 `best.pth` 가 된다. 손실 항 중 하나라도 `warmup_epochs` 가 있으면 `monitor: loss` 는 `ValueError` 다. 워밍업 동안에는 epoch 마다 손실 식이 달라 검증 손실을 서로 비교할 수 없기 때문이다.

**deploy_softening** (Phase-1, EO Phase-2). 배치의 일부를 동결된 denoise 넷에 통과시켜, 배포에서 SR 앞에 Denoise 가 붙는 경로의 흐려진 입력을 학습에 넣는다. 근거는 CONFIG_RATIONALE 7.2 에 있다.

| 키 | 기본값 | 뜻 |
|---|---|---|
| `denoise_prob` | 0.0 | 표본마다 이 확률로 softening 한다. 0 이면 넷을 읽지 않는다 |
| `noise_sigma` | `[2.0, 15.0]` | 넷에 넣기 전에 더하는 가우시안 노이즈 σ(DN). `null` 이면 노이즈를 더하지 않는다 |
| `denoise_checkpoint` | 없음 | `denoise_prob` 가 0 보다 크면 필수. 파일이 없으면 `FileNotFoundError` |
| `denoise_model` | 체크포인트에 저장된 model 블록 | |

출하 값은 `denoise_prob: 0.5` 와 `checkpoints/csuav_deploy/train_svfocusdenoise_adv_x1_dim32_block3_epoch300_lr1e-4_charbonnier_edge_ssim_hotpixel_260309/best.pth`(md5 `0df4cc5174f9c8463ee28b0f036b42fb`)다. `checkpoints/` 는 git 이 무시하므로 이 파일은 이 서버에만 있다. 다른 PC 에서는 이 파일과 Phase-1 `best.pth` 를 복사하기 전까지 Phase-1·EO Phase-2 학습이 `FileNotFoundError` 로 멈춘다(IR Phase-2 도 Phase-1 `best.pth` 가 없으면 멈춘다). 출하 SR v3 가 이 넷으로 학습됐으므로 다른 Denoise(현 배포 BicubicPPDenoise[v1] 포함)로 바꾸지 않는다. 로그에 `[Trainer] Deployment softening: 50% of each batch through best.pth` 가 찍히면 켜진 것이다.

### 3.5 `loss`

항마다 `{enabled, weight, ...}` 블록이고 `enabled: true` 인 항만 쓴다. 블록이 없거나 `enabled: false` 면 그 블록의 다른 값은 읽지 않는다. 예를 들어 Phase-1 config 의 꺼진 `edge` 블록에 적힌 `weight: 0.5` 는 아무 뜻이 없다. 켜진 항이 하나도 없으면 `ValueError` 다.

| 키 | 기본 `weight` | 다른 키 (기본값) |
|---|---|---|
| `mse` | 1.0 | |
| `l1` | 1.0 | |
| `charbonnier` | 1.0 | `eps`(1e-3) |
| `edge` | 0.05 | `mode`(`gray` = BT.601 루마; `mean` = 세 채널 평균), `threshold`(0.2), `signed`(`false`; `true` 면 `threshold` 를 쓰지 않는다) |
| `ssim` | 0.1 | |
| `perceptual` | 0.01 | `layers`(`{34: 1.0}` = VGG19 relu5_4), `gray`(`false`). torchvision VGG19 ImageNet 가중치를 쓴다 |
| `color_consistency` | 0.05 | `loss_type`(`l1`) |
| `hot_pixel_masked` | 0.1 | `threshold`(0.15), `base_loss`(`charbonnier`), `kernel_size`(3) |
| `tv` | 0.1 | |
| `gradient_variance` | 0.05 | `patch`(8), `mode`(`mean`) |

모든 항에 `warmup_epochs: N` 을 줄 수 있다. 가중치는 `목표 × min(1, (epoch − 1) / N)` 으로 올라가서 epoch 1 에는 0, epoch N+1 부터 목표값이다. 로그에는 `[Trainer] Epoch 1 loss warm-up: edge=0, gradient_variance=0, ssim=0` 처럼 찍힌다. 손실 항별 값은 TensorBoard(`logs/`)에만 기록되고 `training_log.csv` 에는 합계만 들어간다.

출하 레시피는 이렇다. 근거는 CONFIG_RATIONALE 5~7절에 있다.

| | 손실 |
|---|---|
| Phase-1 | `mse` 1.0 만 |
| EO Phase-2 | `mse` 1.0, `gradient_variance` 0.05, `ssim` 0.02, `edge` 0.02(`mode: mean`), `perceptual` 0.05(`gray: true`, `layers: {8: 0.25, 17: 0.5, 26: 0.25}`) |
| IR Phase-2 | `mse` 1.0, `gradient_variance` 0.05, `ssim` 0.05, `edge` 0.02(`mode: mean`, `signed: true`), `perceptual` 끔 |
| Denoise | `charbonnier` 1.0, `edge` 0.05(`mode: gray`), `ssim` 0.1 |

Phase-2 의 `gradient_variance`·`ssim`·`edge` 에는 `warmup_epochs: 10` 이 있지만 EO `perceptual` 에는 없다. 그래서 EO Phase-2 의 epoch 1 은 `mse` + `perceptual` 0.05 로, IR Phase-2 의 epoch 1 은 `mse` 만으로 돈다.

`perceptual` 의 VGG19 가중치는 호스트의 `~/.cache/torch/hub/checkpoints/vgg19-dcbb9e9d.pth` 에서 읽는다(`docker/run.sh` 가 `~/.cache/torch` 를 컨테이너에 연결한다). 캐시에 없는 가중치를 쓰는 손실을 처음 돌릴 때는 `--offline` 없이 실행해야 받을 수 있다.

## 4. Phase-1, Phase-2, Denoise

### 4.1 출하 config 비교

| | Phase-1 | EO Phase-2 | IR Phase-2 | Denoise |
|---|---|---|---|---|
| config | `train/BicubicPlusPlus/bicubicplusplus_{s}x.yaml` | `finetune/BicubicPlusPlus/bicubicplusplus_{s}x_eo.yaml` | `finetune/BicubicPlusPlus/bicubicplusplus_{s}x_ir.yaml` | `train/Denoise/bicubicppdenoise.yaml` |
| model 차이 | 기준 | Phase-1 과 같음 | + `achromatic_output: true` | 배율 1, `legacy`, `n_body: 4` |
| 학습 데이터 | `Pretrained_Dataset/tmp/` | `Pretrained_Dataset/train/` | `Finetunning_Dataset/train/` | 로컬 `Pretrained_Dataset/train/` |
| 검증 LR | `val/LRx{s}_v3` | `val/LRx{s}_weaknoise_eo_v3` | `val/LRx{s}_weaknoise_ir` | `val_denoise_cam_a9nr5/LR` |
| 시작 가중치 | 없음 | 출하 Phase-1 `best.pth` | 출하 Phase-1 `best.pth` | 없음 |
| epochs · lr | 120 · 5e-4 | 100 · 5e-5 | 100 · 5e-5 | 300 · 1e-4 |
| scheduler | KneeLR | Cosine `T_max` 100 | Cosine `T_max` 100 | Cosine `T_max` 300 |
| optimizer | Adam | Adam | Adam | AdamW, `weight_decay` 1e-4, `grad_clip` 1.0 |
| batch | 16 | 16 | 16 (x4 는 8 × `gradient_accumulation_steps` 2) | 128 |
| `patch_size` | 288 / 192 / 144 | 288 / 192 / 144 | 288 / 192 / 144 | 256 |
| `num_workers` | 8 | 4 | 8 | 12 |
| softening | 0.5, `noise_sigma` [2, 15] | 0.5 (`noise_sigma` 기본값 [2, 15]) | 없음 | 없음 |
| EMA | 0.999, 시작 2 | 0.999, 시작 2 | 0.999 (x4 는 0.998), 시작 2 | 0.999, 시작 2 |
| 판정 | `psnr`/`max`, patience 120 | `lpips`/`min`, patience 50, `min_delta` 1e-4 | 같음 | `lpips`/`min`, patience 50, `min_delta` 1e-4 |
| `save_interval` | 10 | 20 | 20 | 10 |
| `save_name` | `train_bicubicplusplus_sr_x{s}_dim32_knee_ep120_bs16_lr5e-4_mse_v3` | `finetune_bicubicplusplus_eo_sr_x{s}_dim32_v3` | `finetune_bicubicplusplus_ir_sr_x{s}_dim32_v3` | `train_bicubicppdenoise_eo_x1_dim32_half4_ep300_bs128_lr1e-4_v1` |

IR x4 의 EMA 0.998 은 gradient accumulation 2 에서 출하 당시와 같은 평균 길이를 맞춘 값이다(CONFIG_RATIONALE 3.1). Phase-1 한 epoch 는 6,182 step(98,897 / 16), EO Phase-2 x2 한 epoch 는 677 step(10,821 / 16)이다. 2026-10-19 EO x2 스모크에서 677 step 학습 루프는 약 8분이 걸렸다(검증 별도).

### 4.2 Phase-2 의 `best.pth` 는 없을 수 있다

`pretrained_path` 로 가중치를 읽은 런은 학습 전에 시작 가중치를 한 번 검증하고 그 점수를 기준으로 삼는다. 로그에는 다음처럼 찍힌다.

```
[Validation][Epoch 0] Loss: 0.033917 | PSNR: 31.3186 | SSIM: 0.8761 | LPIPS: 0.1672 | NIQE: 4.4021
[Trainer] Baseline from pretrained weights: lpips=0.167224 (loss=0.033917). best.pth is only written if this is beaten.
```

`best.pth` 는 이 기준을 `min_delta` 이상 넘은 epoch 에서만 쓰인다. 그래서 Phase-2 결과가 Phase-1 보다 나빠지는 일은 없지만, 기준을 넘지 못한 런에는 `best.pth` 가 없다. 그 런 자체의 최선은 `best_attempt.pth`, 마지막 epoch 는 `last.pth` 에 있다. 2026-10-19 EO x2 1 epoch 스모크는 LPIPS 0.1772 로 기준 0.1672 를 넘지 못해 `best.pth` 없이 끝났다. 출하 EO x2 Phase-2(`checkpoints/bicubicpp_deploy/finetune_bicubicplusplus_eo_sr_x2_dim32_v3/`)에도 `best.pth` 가 없다. 이 기준 점수는 `training_log.csv` 에 들어가지 않으므로(첫 행이 epoch 1) 콘솔 로그에서 본다. `--resume` 은 기준 검증을 다시 하지 않고 체크포인트에 저장된 기준을 이어 쓴다.

처음부터 학습하는 런(Phase-1, Denoise)은 기준이 없으므로 epoch 1 이 항상 `best.pth` 가 된다.

### 4.3 Phase-2 가 Phase-1 가중치를 찾는 법

`train.pretrained_path` 가 가리키는 파일의 `model_state_dict`(EMA 를 쓴 런이면 EMA 가중치)를 `strict=False` 로 읽는다. optimizer 와 epoch 는 새로 시작한다. 로그의 `[Finetune] matched=32 missing=0 unexpected=0` 줄을 확인한다(`matched` 는 출하 구조에서 x2 32, x3 25, x4 40). 같은 구조끼리는 `missing=0 unexpected=0` 이어야 한다. 하나도 안 맞으면(`matched=0`) `RuntimeError`, 파일이 없으면 `FileNotFoundError` 로 멈춘다. 일부만 맞으면 멈추지 않고 나머지를 무작위 초기화한 채 학습하므로 이 줄을 꼭 본다.

출하 Phase-2 6개는 출하 Phase-1 `checkpoints/bicubicpp_deploy/train_bicubicplusplus_sr_x{2,3,4}_dim32_knee_ep120_bs16_lr5e-4_mse_v3/best.pth` 를 가리킨다. EO 와 IR 이 같은 Phase-1 에서 출발한다.

**새로 학습한 런은 `checkpoints/bicubicpp_deploy/` 가 아니라 `checkpoints/<save_name>/`(또는 `--work_dir`)에 쓰인다.** 출하 Phase-1 config 를 다시 돌려도 Phase-2 config 는 여전히 출하 Phase-1 을 읽는다. 새 Phase-1 에서 파인튜닝하려면 Phase-2 config 사본의 `pretrained_path` 를 새 Phase-1 런의 `best.pth` 로 바꾼다(7.2). `checkpoints/bicubicpp_deploy/` 는 출하 기록이므로 거기에 출력하지 않는다.

### 4.4 Phase-2 가 `best.pth` 를 못 만들었을 때 export 할 가중치

EO 는 Phase-1 `best.pth` 를 그 Phase-2 config 로 export 할 수 있다. EO Phase-2 의 model 블록이 Phase-1 과 같아서 `check_model_config` 를 통과하기 때문이다. 출하 EO x2 가 이렇게 나왔다(Phase-2 가 기준 LPIPS 0.1672 를 넘지 못함).

IR 은 이 방법을 쓸 수 없다. IR Phase-2 config 에는 `achromatic_output: true` 가 더 있으므로 Phase-1 체크포인트를 IR config 로 열면 `ValueError` 로 멈춘다. IR 은 그 Phase-2 런의 `best_attempt.pth` 나 `last.pth` 를 그 런의 `train_config.yaml` 로 export 한다. 출하 IR x2·x3·x4 는 모두 Phase-2 `best.pth` 였다. export 명령은 [examples/README.md](../examples/README.md) 1.6절(새로 학습한 체크포인트를 export하기)과 [README.md](../README.md)의 "새 모델을 보드까지"에 있다.

## 5. 데이터 레시피 (`configs/data/`)

### 5.1 최상위 키

| 키 | 뜻 |
|---|---|
| `dataset_type` | 시작 로그 `Loading datasets for task=sr dataset_type=sr resolved=sr...` 에 찍힐 뿐 데이터셋을 고르지 않는다. 고르는 것은 학습 config 의 `task` 다 |
| `clean_prob` | 이 확률로 열화를 통째로 건너뛴다. SR 은 그때도 `stage2.target_resize.interpolations` 목록에서 뽑은 커널로 축소는 한다(목록이 없으면 `cv2.INTER_CUBIC`). Denoise 는 입력이 정답과 같아진다 |
| `crop_size` | 읽는 코드가 없다. 크롭 크기는 학습 config 의 `train.patch_size` 다 |
| `use_flip`, `use_rot` | SR 만 읽는다. 기본 `false`. 열화를 마친 뒤 HR·LR 에 같은 뒤집기·90° 회전을 적용한다. `DenoiseDataset` 은 뒤집지 않는다 |
| `camera_model` | `DenoiseDataset` 만 쓴다. SR 에서 켜면 `ValueError` |
| `degradation.stage1`, `degradation.stage2` | 열화 연산 (5.2) |

빈 레시피는 `SRDataset`·`DenoiseDataset` 이 `ValueError` 로 거부한다. 빈 레시피는 "열화 없음"이 아니라 "모든 연산이 기본값으로 켜짐"이기 때문이다.

### 5.2 열화 연산과 기본값

연산은 아래 표의 순서대로 돈다. 각 연산은 먼저 `enabled` 를 보고, 켜져 있으면 `prob` 확률로 적용된다. **레시피에 없는 연산은 꺼지는 것이 아니라 기본값으로 켜진다.** 기본으로 꺼진 연산은 `hot_pixels` 하나뿐이다. 그래서 새 레시피는 출하 레시피를 복사해 모든 연산 블록을 남기고, 쓰지 않는 연산은 `enabled: false` 로 끈다.

| 단계 | 연산 | 기본 `enabled` | 기본 `prob` | 다른 키 (기본값) |
|---|---|---|---|---|
| stage1 | `blur` | true | 0.5 | `kernel_probs` [0.45, 0.25, 0.12, 0.03, 0.12, 0.03], `kernel_sizes` [7, 9, …, 21], `sigma` [0.2, 3.0], `betag` [0.5, 4.0], `betap` [1.0, 2.0] |
| stage1 | `resize` | true | 0.7 | `scale_min` 0.5, `scale_max` 1.0, `interpolations` [LINEAR, CUBIC, AREA] |
| stage1 | `gaussian_noise` | true | 0.5 | `sigma_min` 1, `sigma_max` 5, `gray_prob` 0.5 |
| stage1 | `poisson_noise` | true | 0.5 | `scale_min` 0.05, `scale_max` 3.0, `gray_prob` 0.5 |
| stage1 | `jpeg` | true | 0.5 | `quality_min` 85, `quality_max` 95 |
| stage1 | `sinc` | true | 0.1 | `kernel_sizes` |
| stage2 | `blur` | true | 0.5 | stage1 `blur` 와 같은 키, `sigma` [0.2, 1.5] |
| stage2 | `target_resize` | true | 항상 | `interpolations` [CUBIC, LINEAR, LANCZOS4]. HR 크롭을 `1/scale` 크기로 줄인다 |
| stage2 | `ir_noise` | true | 하위 연산별 | `non_uniformity`(`enabled` true, `prob` 0.5, `amp_min` 0.1, `amp_max` 0.3), `vertical_line`(`enabled` true, `prob` 0.5, `sigma_dn` 0.75, `strong_frac` 0.03, `strong_dn` 6.0), `thermal_noise`(`enabled` true, `prob` 1.0, `sigma_min` 5, `sigma_max` 20, `gray_prob` 0.9) |
| stage2 | `common_noise` | true | 0.0 | `sigma_min` 1, `sigma_max` 10, `gray_prob` 0.2 |
| stage2 | `hot_pixels` | **false** | 0.5 | `density_min` 0.001, `density_max` 0.01, `blob_prob` 0.4, `blob_radius_min` 1, `blob_radius_max` 8 |
| stage2 | `final_jpeg` | true | 0.8 | `quality_min` 50, `quality_max` 90 |

`interpolations` 는 `"cv2.INTER_LINEAR"`, `"cv2.INTER_CUBIC"`, `"cv2.INTER_AREA"`, `"cv2.INTER_LANCZOS4"` 네 문자열만 알아듣고, 다른 문자열은 경고 없이 `INTER_CUBIC` 이 된다. 목록에 같은 값을 두 번 넣으면 그만큼 자주 뽑힌다(출하 SR 레시피는 `INTER_AREA` 를 두 번 넣었다, CONFIG_RATIONALE 8.1).

모르는 연산 이름은 시작 로그에 `[Degradation] SRDataset(train) WARNING: stage2 keys not read by the pipeline: <이름>` 경고만 찍고 넘어간다. 모르는 하위 키(예: `vertical_line` 의 `intensity_min`)는 경고도 없이 무시되고 기본값이 쓰인다.

`camera_model` 블록(Denoise)의 키와 값의 근거는 CONFIG_RATIONALE 14절과 [docs/modules/20261008-MOD-CameraModel.md](../docs/modules/20261008-MOD-CameraModel.md)에 있다. 이 블록은 학습 시작 때 검사되고(`validate_camera_model_cfg`), 잘못된 키는 그 이름을 담은 `ValueError` 로 멈춘다. `target` 은 `camera`, `optics`, `source` 중 하나이고 키가 없으면 `camera` 다. 출하 v1 은 `source` 다. 카메라 모델을 거치지 않는 나머지 `1 − prob` 표본은 `degradation` 의 replay 열화를 거친다.

### 5.3 출하 레시피 요약

| | Phase-1 | EO Phase-2 | IR Phase-2 | Denoise |
|---|---|---|---|---|
| `clean_prob` | 0.0 | 0.3 | 0.3 | 0.1 |
| `use_flip`, `use_rot` | 켬 | 켬 | 켬 | 키 없음 |
| stage1 | 전부 끔 | `blur` 0.08 (σ 0.2~0.8), `gaussian_noise` 0.10 (σ 1~2), `poisson_noise` 0.10 (0.05~0.3), `jpeg` 0.12 (q 95~100) | `blur` 0.08, `gaussian_noise`·`poisson_noise` 0.08 (`gray_prob` 1.0), `jpeg` 0.10 | `gaussian_noise` 0.5 (σ 1~5), `poisson_noise` 0.5, `jpeg` 0.5 (q 85~95) |
| stage2 | `target_resize` 만 | `blur` 0.05 (σ 0.2~0.6), `target_resize`, `common_noise` 0.05 (σ 1~2), `hot_pixels` 0.2, `final_jpeg` 0.08 (q 95~100) | `blur` 0.05, `target_resize`, `ir_noise`, `hot_pixels` 0.4, `final_jpeg` 0.08 | `common_noise` 0.5 (σ 2~15), `hot_pixels` 0.5 (`blob_prob` 0.1), `final_jpeg` 0.8 (q 30~85) |
| `target_resize` 커널 | AREA, AREA, CUBIC, LINEAR, LANCZOS4 | 같음 | 같음 | 끔 |
| `camera_model` | 없음 | 없음 | 없음 | 켬, `prob` 0.75, `target: source` |

IR 의 `ir_noise` 는 `non_uniformity` 0.03(진폭 0.02~0.06), `vertical_line` 0.35(`sigma_dn` 0.75, `strong_frac` 0.03, `strong_dn` 6.0), `thermal_noise` σ 0.5~1.5(`gray_prob` 1.0)다. `thermal_noise` 에는 `prob` 키가 없어 기본값 1.0 이 적용되므로, `clean_prob` 로 건너뛴 30 % 를 뺀 모든 표본에 들어간다. Phase-2 SR 레시피 두 개의 `hot_pixels` 밀도는 모두 0.0005~0.002 이고 SR 은 blob 을 끈다(`blob_prob` 0.0).

### 5.4 시작 로그로 확인하기

데이터셋은 시작할 때 실제로 쓰는 열화를 한 번 찍는다. 기본값으로 켜진 연산도 여기에 나오므로, 레시피를 새로 만들었으면 학습을 시작한 직후 이 줄을 반드시 본다. 2026-10-19 EO x2 Phase-2 스모크의 출력은 이렇다.

```
[Degradation] SRDataset(train) stage1 active: blur(p=0.08), gaussian_noise(p=0.1), poisson_noise(p=0.1), jpeg(p=0.12)
[Degradation] SRDataset(train) stage1 off: resize, sinc
[Degradation] SRDataset(train) stage2 active: blur(p=0.05), target_resize(always), common_noise(p=0.05), hot_pixels(p=0.2), final_jpeg(p=0.08)
[Degradation] SRDataset(train) stage2 off: ir_noise
[Degradation] SRDataset(train) clean_prob=0.3
[Degradation] SRDataset(train) camera_model off
```

런 시작 부분의 확인 줄은 호스트에서 이렇게 모아 본다. 로그에는 진행 막대의 `\r` 이 섞여 있어 먼저 줄바꿈으로 바꾼다.

```bash
tr '\r' '\n' < logs/<run>.log | grep -a -e '\[Degradation\]' -e '\[Finetune\]' -e 'Deployment softening' -e 'Baseline from' -e 'loss warm-up'
```

`plots/train_batch.png` 에는 학습 쌍 8개가 그려지므로 열화가 의도대로 들어갔는지 눈으로도 확인한다.

### 5.5 검증 세트는 레시피와 짝을 맞춘다

출하 게이트의 검증 LR 은 학습 레시피 그대로 `examples/dataset_preparation/make_val_lr.py`(seed 2026)로 만들었다. softening 이 있는 Phase-1·EO Phase-2 는 같은 softening 체크포인트를 `--denoise_checkpoint` 로 넘겨 절반(짝수 번째)을 softening 했다. 레시피를 바꾼 실험은 그 레시피로 새 검증 LR 을 **새 폴더에** 만들고 `data.val_lr_root` 를 그리로 바꾼다. 그래야 검증 점수가 학습한 열화를 잰다. 그 대신 서로 다른 검증 세트에서 잰 점수는 비교할 수 없다(CONFIG_RATIONALE 4.3). 출하 모델과 비교할 때는 두 체크포인트를 같은 세트에서 `tools/evaluate.py --hr_dir ... --lr_dir ...` 로 다시 잰다([tools/README.md](../tools/README.md)).

```bash
docker/run.sh -- python3 examples/dataset_preparation/make_val_lr.py \
  --data_config configs/experiments/<topic>/recipe_eo.yaml --scale 2 \
  --hr_root /mnt/data2/jshong/SuperResolution/Pretrained_Dataset/val/HR \
  --out_dir /mnt/data2/jshong/SuperResolution/Pretrained_Dataset/val/LRx2_<topic> \
  --denoise_checkpoint checkpoints/csuav_deploy/train_svfocusdenoise_adv_x1_dim32_block3_epoch300_lr1e-4_charbonnier_edge_ssim_hotpixel_260309/best.pth
```

이 스크립트는 CPU 로 돌고, 같은 이름의 파일을 확인 없이 덮어쓴다. 출하 검증 폴더를 `--out_dir` 로 주지 않는다. IR 은 softening 이 없으므로 `--denoise_checkpoint` 를 빼고, Denoise 검증 세트는 `examples/dataset_preparation/make_val_camera.py` 로 만든다. 옵션 전체는 [examples/README.md](../examples/README.md) 6절에 있다.

## 6. AIMET JSON (`configs/aimet/`)

**quantsim config.** `htp_v73_no_leakyrelu_supergroup.json` 은 AIMET 빌트인 `htp_v73` 에서 Conv+LeakyRelu supergroup 만 뺀 것이다(supergroup 11개). 빌트인을 그대로 쓰면 AIMET 이 Conv 출력 quantizer 를 끄는데 qairt 컨버터는 그 둘을 융합하지 않아서, Bicubic++ 의 Conv·LeakyReLU 가 DLC 에서 조용히 float 으로 돈다. 변환 로그의 `Following OPs fallback to float:` 뒤에 op 이름이 나열되는 것이 그 증상이다. 그래서 `--quantsim_config` 에 빌트인 이름 `htp_v73` 을 넘기지 않는다. 상세는 [examples/README.md](../examples/README.md) 3.1절(Conv+LeakyRelu supergroup 함정)과 CONFIG_RATIONALE 11.2 에 있다.

**MMP config.** 형식은 `{"layers": {"<sim.model 안의 모듈 경로>": {"output_bw": n, "input_bw": n, "param_bw": n}}}` 이다. 출하 두 파일은 `layers` 가 비어 있어 모든 층이 `--param_bw`·`--output_bw` 의 기본 비트폭을 쓴다. W16A16 은 `--param_bw 16 --output_bw 16 --mmp_config configs/aimet/mmp_noprune_w16a16.json` 이다. 파일이 없으면 `FileNotFoundError` 로 멈추지만, 찾지 못한 모듈 이름은 `[Warning] Could not find module` 만 찍고 건너뛴다. 층별 예외를 넣을 때는 이 경고가 없는지 확인한다.

## 7. 새 실험 config 만들기

### 7.1 원칙

- **출하 config 와 레시피는 고치지 않는다.** 출하 모델의 기록이고, 테스트가 고정해 두었다. `tests/test_release_golden.py` 는 출하 config 로 만든 모델과 출하 체크포인트를 대조하고, `tests/test_data_recipe_golden.py` 는 출하 레시피 4개로 만든 학습 쌍의 해시를, `tests/test_make_val_lr.py` 는 SR 레시피 3개를, `tests/test_config_guards.py` 는 EO Phase-2 의 `pretrained_path` 가 출하 Phase-1 출력인지를 확인한다. model 블록, 레시피, EO `pretrained_path` 를 고치면 이 테스트가 실패한다. `lr`·`epochs` 같은 학습 루프 값은 대부분 테스트가 잡지 않는다.
- **새 파일로 만든다.** 선례는 config 를 `configs/experiments/<topic>/` 에, 출력을 `checkpoints/experiments/<topic>/<run>/` 에 둔 것이다. `checkpoints/` 는 git 이 무시하지만 `configs/experiments/` 는 무시하지 않는다. 지금까지 실험 config 는 커밋하지 않았고, 실험 기록은 [docs/experiments/](../docs/experiments/README.md) 레지스트리에, 채택된 레시피는 출하 config 로 남겼다.
- **런마다 새 출력 폴더를 쓴다.** 같은 `save_name` 이나 `--work_dir` 로 다시 돌리면 에러 없이 `training_log.csv` 에 이전 런의 행 뒤로 이어 쓰고(헤더도 다시 쓰지 않는다), `train_config.yaml` 과 체크포인트를 덮어쓴다. 처음부터 학습하는 런이면 epoch 1 이 다시 `best.pth` 를 덮는다.
- **다른 배율은 그 배율의 config 를 복사한다.** `--scale` 은 `model.scale` 만 바꾸고 `patch_size`, `val_lr_root`, `save_name`, `pretrained_path` 는 그대로 둔다. 그러면 HR 크롭이 576 에서 벗어나고 검증 LR 배율이 맞지 않는다. 검증은 `[Validator] Shape mismatch detected ...` 경고를 한 번 찍고 잘라서 계산하므로 멈추지 않지만 점수는 의미가 없다.

### 7.2 예: EO x2, 새 Phase-1 레시피에서 Phase-2 까지

1) 출하 config 와 레시피를 복사한다.

```bash
mkdir -p configs/experiments/<topic>
cp configs/train/BicubicPlusPlus/bicubicplusplus_2x.yaml configs/experiments/<topic>/p1_x2.yaml
cp configs/finetune/BicubicPlusPlus/bicubicplusplus_2x_eo.yaml configs/experiments/<topic>/p2_x2_eo.yaml
cp configs/data/bicubicpp/sr_train_phase-1.yaml configs/experiments/<topic>/recipe_p1.yaml
```

2) `recipe_p1.yaml` 에서 바꿀 연산만 바꾼다. 블록을 지우지 말고 `enabled` 로 켜고 끈다(5.2).

3) `p1_x2.yaml` 에서 레시피 경로, 출력 이름, 데이터 경로를 바꾼다. 아래는 바꾸는 키만 보인 것이고, 나머지 키는 그대로 둔다. 데이터 경로는 sshfs 대신 로컬 사본으로 바꾼다(3.3).

```yaml
data_config_path: configs/experiments/<topic>/recipe_p1.yaml

data:
  train_root: /mnt/data2/jshong/SuperResolution/Pretrained_Dataset/tmp/
  val_hr_root: /mnt/data2/jshong/SuperResolution/Pretrained_Dataset/val/HR
  val_lr_root: /mnt/data2/jshong/SuperResolution/Pretrained_Dataset/val/LRx2_v3

train:
  save_name: <topic>_p1_x2
```

레시피를 바꿨으면 검증 LR 도 그 레시피로 새로 만들어 `val_lr_root` 를 바꿀지 정한다(5.5). 바꾸지 않으면 출하 Phase-1 과 같은 잣대로 비교할 수 있다.

4) Phase-1 을 학습한다. 쉬는 GPU 번호를 고르는 법과 작업 관리는 [docker/README.md](../docker/README.md)에 있다.

```bash
docker/run.sh --gpu <N> --name <topic>_p1_x2 -- python3 tools/train.py --config configs/experiments/<topic>/p1_x2.yaml --work_dir checkpoints/experiments/<topic>/p1_x2
```

`--name` 을 주면 컨테이너 `usdp-<topic>_p1_x2` 가 백그라운드로 돌고 로그는 `logs/<topic>_p1_x2.log` 에 쌓인다. `run.sh` 는 컨테이너가 시작되자마자 끝나므로, 학습 결과는 로그 마지막 줄 `[run.sh] exit 0` 으로 확인한다. GPU 는 `--gpu` 로만 고르고 `tools/train.py --device` 는 주지 않는다(컨테이너 안에는 고른 GPU 하나만 `cuda:0` 으로 보인다). 시작하고 1~2분 뒤 5.4 의 명령으로 `[Degradation]` 줄을 확인한다.

5) Phase-1 이 끝나면 `p2_x2_eo.yaml` 의 출발점을 새 Phase-1 로 바꾼다. 처음부터 학습한 Phase-1 은 항상 `best.pth` 를 남긴다(4.2).

```yaml
train:
  save_name: <topic>_p2_x2_eo
  pretrained_path: checkpoints/experiments/<topic>/p1_x2/best.pth
```

Phase-1 에서 model 블록을 바꿨다면 Phase-2 사본의 model 블록도 똑같이 바꾼다. IR Phase-2 사본이면 여기에 `achromatic_output: true` 만 더 있다. 연결 스크립트는 없으므로 Phase-1 이 끝난 것을 확인하고 Phase-2 를 직접 시작한다.

6) Phase-2 를 1 epoch 만 돌려 연결을 확인한다.

```bash
docker/run.sh --gpu <N> --name <topic>_p2_smoke -- python3 tools/train.py --config configs/experiments/<topic>/p2_x2_eo.yaml --epochs 1 --work_dir checkpoints/experiments/<topic>/p2_x2_eo_smoke
```

로그에서 `[Finetune] matched=32 missing=0 unexpected=0`, `[Trainer] Deployment softening: 50% ...`, `[Trainer] Baseline from pretrained weights: ...`, `[Degradation]` 줄을 확인한다. 스모크는 파이프라인 연결만 확인한다. epoch 1 에서는 워밍업 항의 가중치가 0 이고, `save_interval` 이 20 이라 `epoch_<N>.pth` 가 없으며, `best.pth` 도 없을 가능성이 크다. step 수를 줄이는 옵션은 없어서 1 epoch 도 `train_root` 전체(EO x2 는 677 step)를 돈다.

7) 본 학습을 돌린다.

```bash
docker/run.sh --gpu <N> --name <topic>_p2_x2_eo -- python3 tools/train.py --config configs/experiments/<topic>/p2_x2_eo.yaml --work_dir checkpoints/experiments/<topic>/p2_x2_eo
```

8) 평가와 export 에는 그 런의 `train_config.yaml` 을 넘긴다. 결과는 `<save_dir>/<체크포인트 이름에서 .pth 를 뺀 것>/metrics.csv`(아래 명령이면 `results/<topic>/eval_p2_x2_eo/best/metrics.csv`)에 남는다. `best.pth` 가 없으면 4.4 를 따른다.

```bash
docker/run.sh --gpu <N> --name <topic>_p2_eval -- python3 tools/evaluate.py --config checkpoints/experiments/<topic>/p2_x2_eo/train_config.yaml --checkpoint checkpoints/experiments/<topic>/p2_x2_eo/best.pth --save_dir results/<topic>/eval_p2_x2_eo
```

이후 W16A16 export, DLC 변환, 보드 확인은 [README.md](../README.md)의 "새 모델을 보드까지"를 따른다.

### 7.3 레시피만 바꿔 볼 때

config 를 복사하지 않고 `--data_config` 로 레시피만 바꿀 수 있다. 런 폴더의 `train_config.yaml` 에는 실제로 쓴 레시피 경로가 남는다.

```bash
docker/run.sh --gpu <N> --name <topic>_exp -- python3 tools/train.py --config configs/finetune/BicubicPlusPlus/bicubicplusplus_2x_eo.yaml --data_config configs/experiments/<topic>/recipe_eo.yaml --work_dir checkpoints/experiments/<topic>/exp
```

이때도 `--work_dir` 를 꼭 준다. 빠뜨리면 출하 config 의 `save_name` 인 `checkpoints/finetune_bicubicplusplus_eo_sr_x2_dim32_v3/` 에 쓰인다.

### 7.4 새 모델 구조나 Denoise 실험

새 모델이나 새 손실·열화 연산을 코드에 추가하는 방법은 [src/README.md](../src/README.md)에 있다. config 쪽에서 지킬 것은 세 가지다. model 블록은 학습 뒤 고치지 않는다(3.2). 마지막 층 이름은 `final_act` 여야 DLC 변환과 보드 스크립트가 그대로 동작한다. DLC 이름 규칙은 [examples/deployment/README.md](../examples/deployment/README.md)에 있다.

Denoise 실험은 `configs/train/Denoise/bicubicppdenoise.yaml` 을 복사한다. 데이터와 검증 세트가 모두 로컬 사본에 있다. `pretrained_path` 가 없으므로 `best.pth` 는 항상 생긴다. 출하 체크포인트와 기록은 [results/denoise_release_v1/MANIFEST.md](../results/denoise_release_v1/MANIFEST.md)에 있다.

## 8. 아무 일도 하지 않는 키, 조용히 무시되는 값

| 키 또는 경우 | 위치 | 실제 동작 |
|---|---|---|
| `data.type` | 학습 config | 읽는 코드가 없다 |
| `dataset_type` | 레시피 | 시작 로그에 찍기만 한다. 데이터셋은 `task` 가 고른다 |
| `crop_size` | 레시피 | 읽는 코드가 없다. 크롭은 `train.patch_size` |
| `vertical_line.intensity_min`, `intensity_max` | `sr_train_phase-1.yaml`, `sr_finetune_eo_phase-2.yaml`, `sr_train.yaml` | 옛 형식의 키라 무시된다. 경고도 없다 |
| 모르는 하위 키 | 레시피의 연산 블록 안 | 경고 없이 무시되고 기본값이 쓰인다 |
| 모르는 연산 이름 | 레시피의 `stage1`/`stage2` 바로 아래 | `keys not read by the pipeline` 경고만 찍는다 |
| `interpolations` 의 오타 | 레시피 | 경고 없이 `cv2.INTER_CUBIC` |
| 모르는 손실 이름 | `loss` | 경고 없이 무시된다. `gradient_varience` 처럼 틀리면 그 항 없이 학습한다 |
| `enabled: false` 인 블록의 나머지 값 | `loss`, 레시피 | 읽지 않는다 |
| `optimizer.momentum` / `optimizer.betas` | `train.optimizer` | 각각 SGD 일 때만 / Adam·AdamW 일 때만 읽는다 |
| `svfocussrnet` 의 다른 model 키 | `model` | `scale`, `dim`, `n_resblocks`, `use_advanced_rep` 외에는 무시한다 |
| `pretrained_path` | `--resume` 할 때 | 읽지 않는다 |
| CLI 값 0 | `--epochs 0`, `--lr 0` 등 | 적용되지 않는다 |
| `--epochs` | CLI | `T_max`·`total_steps` 를 바꾸지 않는다 (3.4) |
| `early_stopping.mode` 와 지표 방향 | `train.early_stopping` | 짝이 틀려도 검사하지 않는다 (3.4) |

## 9. 자주 밟는 함정

- **레시피에서 지운 연산이 켜진다.** 블록을 지우면 기본값으로 켜진다. 예를 들어 `ir_noise` 블록이 없으면 σ 5~20 열 노이즈가 모든 표본에 들어가 학습 LR 이 심하게 거칠어진다. 증상은 `[Degradation] ... active:` 줄에 의도하지 않은 연산이 보이는 것이다(5.2).
- **같은 출력 폴더로 다시 돌렸다.** 에러 없이 `training_log.csv` 에 두 런이 섞이고 체크포인트가 덮인다. 증상은 CSV 의 epoch 번호가 1 부터 다시 시작하는 것이다(7.1).
- **새 Phase-1 을 학습했는데 Phase-2 결과가 그대로다.** Phase-2 config 가 여전히 출하 Phase-1 을 가리킨다. 로그의 `Finetuning: Loading pretrained weights from ...` 경로를 확인한다(4.3).
- **Phase-2 런에 `best.pth` 가 없다.** 실패가 아니라 시작 가중치를 넘지 못한 것이다. `best_attempt.pth` 와 로그의 `Baseline` 줄을 본다(4.2).
- **평가·export 가 `was trained with another model config` 로 멈춘다.** `--config` 의 model 블록이 체크포인트와 다르다. 그 런의 `train_config.yaml` 을 넘긴다(3.2).
- **파인튜닝이 일부 가중치만 읽었다.** `strict=False` 라 멈추지 않는다. `[Finetune] ... missing=` 이 0 인지 본다(4.3).
- **`monitor` 와 `mode` 짝이 틀렸다.** 가장 나쁜 epoch 가 `best.pth` 가 되고, 조기 종료가 엉뚱한 때 일어난다(3.4).
- **`--epochs` 로 줄인 학습이 덜 수렴했다.** LR 이 config 의 원래 길이로 내려가다 중간에 끝났기 때문이다. `T_max`·`total_steps` 도 줄인다(3.4).
- **다른 PC 에서 Phase-1·Phase-2 가 `FileNotFoundError` 로 멈춘다.** softening 체크포인트와 출하 Phase-1 `best.pth` 는 git 밖에 있다(3.4).
- **`/mnt/data2` 만 쓰는데 `run.sh` 가 거부한다.** `/mnt/data_server` sshfs 마운트 검사는 모든 작업에 적용된다(3.3).

## 10. config·레시피를 바꾼 뒤 돌릴 테스트

config 가드, 데이터셋 선택, 출하 config·레시피 고정 테스트를 컨테이너에서 돌린다. GPU 는 필요 없다.

```bash
docker/run.sh --offline -- env PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q -rs -p no:cacheprovider tests/test_config_guards.py tests/test_dataset_routing.py tests/test_data_recipe_golden.py tests/test_release_golden.py tests/test_make_val_lr.py tests/test_augmentation.py
```

pytest 요약 줄이 `<N> passed` 로 시작하고(뒤에 `, <M> warnings in ...` 이 붙고, 그 아래에 `sys:1: DeprecationWarning ...` 한 줄이 더 찍힐 수 있다) `failed`, `error`, `skipped` 가 없으면 통과다. 출하 config 나 레시피를 고쳤다면 golden 테스트가 실패하는 것이 정상이다(7.1). `tests/test_config_guards.py` 의 `test_eo_phase2_starts_from_the_phase1_output` 는 `checkpoints/bicubicpp_deploy/` 의 출하 Phase-1 파일이 있어야 통과하므로 이 서버에서만 통과한다. 새 테스트를 추가했다면 `docker/accept.sh` 의 기대 테스트 수(`BASE_TESTS`, `tests/test_docker_*.py` 는 따로 세므로 그 밖의 테스트 수)도 같이 바꿔야 한다([docker/README.md](../docker/README.md)).
