# Tools

이 폴더에는 진입점 네 개가 있습니다.

| 스크립트 | 하는 일 |
|---|---|
| `train.py` | 학습. 처음부터(Phase-1, Denoise), 앞 단계 가중치에서 파인튜닝(Phase-2), 끊긴 런 이어서 학습(`--resume`) |
| `evaluate.py` | 짝지어진 검증 세트(HR/LR)에서 체크포인트 하나를 채점(PSNR, SSIM, LPIPS, NIQE) |
| `inference.py` | 이미지 파일 하나나 폴더 하나를 추론해 결과 이미지를 저장 |
| `export.py` | 그래프 확인용 FP32 ONNX와 연산량·파라미터 수. 출하 파일이 아님 |

보드에 올라가는 W16A16 export는 이 폴더에 없고 `examples/aimet_quantization/6_apply_qat.py`가 합니다([examples/README.md](../examples/README.md) 1절). 학습부터 보드까지 이어지는 한 바퀴는 루트 [README.md](../README.md)의 "새 모델을 보드까지"를 봅니다. config 키는 [configs/README.md](../configs/README.md), Trainer 내부(EMA, 손실, 데이터셋)는 [src/README.md](../src/README.md)가 다룹니다.

## 목차

1. [실행 방식 (네 도구 공통)](#1-실행-방식-네-도구-공통)
2. [train.py](#2-trainpy)
3. [evaluate.py](#3-evaluatepy)
4. [inference.py](#4-inferencepy)
5. [export.py](#5-exportpy)
6. [체크포인트와 config 짝 (세 도구 공통 로더)](#6-체크포인트와-config-짝-세-도구-공통-로더)
7. [다음 단계와 관련 스크립트](#7-다음-단계와-관련-스크립트)
8. [컨테이너에서 확인된 범위](#8-컨테이너에서-확인된-범위)

---

## 1. 실행 방식 (네 도구 공통)

모든 도구는 작업 컨테이너 안에서 `docker/run.sh`로 돌립니다. 호스트의 `python3`(3.8)에는 다른 프로젝트용 torch 2.4.1만 있고 aimet이 없으므로 작업 환경이 아닙니다. `run.sh`의 옵션 전체, 거부 메시지, GPU 고르기, 작업 멈추기는 [docker/README.md](../docker/README.md)에 있고, 여기에는 도구를 쓸 때 알아야 하는 것만 적습니다.

```bash
# from the repository root
docker/run.sh [--gpu <N>] [--name <run> [--log <file>]] [--offline] -- python3 tools/<tool>.py <options>
```

- **경로.** 컨테이너의 작업 디렉터리는 저장소 루트입니다. 그래서 `--` 뒤 명령에 쓴 상대 경로(`configs/...`, `checkpoints/...`, `results/...`)는 호스트 저장소 기준으로 풀립니다. `--log`의 상대 경로만은 호스트 셸의 현재 디렉터리 기준이므로 `run.sh`는 저장소 루트에서 실행합니다.
- **보이는 폴더.** 컨테이너에는 저장소, `/mnt`(`/mnt/data_server`, `/mnt/data2` 포함), `~/.cache/torch`만 붙습니다. 그 밖에 있는 이미지 폴더는 `run.sh --mount`로 붙입니다.
- **GPU.** `--gpu <N>`은 호스트 GPU N 한 장을 붙이고, 컨테이너 안에서는 그 GPU가 `cuda:0`입니다. 도구의 `--device`는 기본값 그대로 두고, `CUDA_VISIBLE_DEVICES`는 쓰지 않습니다. 빈 GPU는 `nvidia-smi`로 고릅니다. 이 서버의 A6000 5장은 다른 사용자와 k3s 파드가 같이 쓰고, `run.sh`는 사용 중인 GPU를 거부합니다.
- **백그라운드.** `--name <run>`을 주면 컨테이너가 `usdp-<run>`이라는 이름으로 뒤에서 돌고, `run.sh`는 시작하자마자 0으로 끝납니다. 출력은 `logs/<run>.log`(또는 `--log <file>`)에 쌓입니다. 작업의 실제 결과는 그 로그의 마지막 줄 `[run.sh] exit <코드>`입니다. 멈출 때는 `docker stop usdp-<run>`을 씁니다.
- **오프라인.** `--offline`은 컨테이너의 네트워크를 끊습니다. LPIPS·NIQE(pyiqa)와 perceptual loss의 VGG19 가중치를 `~/.cache/torch/hub`에서 읽으므로, 캐시가 차 있을 때만 됩니다. 이 서버의 jshong 캐시에는 `vgg19`, `alexnet`, `resnet18`, `convnext_small`, `LPIPS_v0.1_alex`, `niqe_modelparameters.mat`이 있습니다.
- **환경 변수.** 이미지에 `NUMPY_MADVISE_HUGEPAGE=0`과 `PYTHONUNBUFFERED=1`이 기본으로 들어 있고, 네 도구는 저장소 루트를 `sys.path`에 스스로 넣습니다. 그래서 `env` 접두어나 `PYTHONPATH=.`이 필요 없습니다. `NUMPY_MADVISE_HUGEPAGE=0`이 빠지면 메모리가 파편화된 이 서버에서 데이터 워커가 커널 시간 97 %로 묶여 학습이 6분/epoch에서 3시간/epoch로 느려졌습니다(2026-09-10). `train.py`는 이 값을 numpy import 전에 `setdefault`로 한 번 더 넣습니다.

### GPU 유무에 따른 동작

| 도구 | `--device` 기본값 | `--gpu <N>`을 줬을 때 | `--gpu`를 뺐을 때 |
|---|---|---|---|
| `train.py` | `cuda` | `cuda:0`에서 학습 | `Warning: CUDA not available, falling back to CPU.`를 찍고 CPU에서 학습 |
| `evaluate.py` | `cuda` | `cuda:0` | 경고 없이 CPU. 로그의 `Using device: cpu`로만 알 수 있음 |
| `inference.py` | `cuda` | `cuda:0` | 경고 없이 CPU. `--fp16`은 무시됨 |
| `export.py` | `cpu` | `--device cuda`를 줄 때만 GPU | CPU. `--device cuda`를 주면 실패 |

`train.py`, `evaluate.py`, `inference.py`의 `--device`는 `cpu`, 숫자 `N`(`cuda:N`으로 바뀜), 또는 `cuda`·`cuda:0` 같은 torch 장치 문자열을 받습니다. 컨테이너 안에는 `cuda:0`만 있으므로 `--device 1` 이상은 실패합니다. `export.py`는 숫자 축약을 해석하지 않고 문자열을 그대로 `torch.device`에 넘기므로 `cpu`나 `cuda`로 줍니다.

---

## 2. train.py

### 자주 쓰는 명령

```bash
# Phase-1 SR from scratch (x3/x4: bicubicplusplus_3x.yaml / bicubicplusplus_4x.yaml)
docker/run.sh --gpu <N> --name <run> -- python3 tools/train.py \
    --config configs/train/BicubicPlusPlus/bicubicplusplus_2x.yaml --work_dir checkpoints/<new_dir>

# Phase-2 SR finetune from the Phase-1 weights in train.pretrained_path
docker/run.sh --gpu <N> --name <run> -- python3 tools/train.py \
    --config configs/finetune/BicubicPlusPlus/bicubicplusplus_<2|3|4>x_<eo|ir>.yaml --work_dir checkpoints/<new_dir>

# Denoise (BicubicPPDenoise v1 recipe)
docker/run.sh --gpu <N> --name <run> -- python3 tools/train.py \
    --config configs/train/Denoise/bicubicppdenoise.yaml --work_dir checkpoints/<new_dir>

# 1-epoch pipeline check (this exact form ran on 2026-10-19 with --gpu 3)
docker/run.sh --gpu <N> --name smoke_x2_eo_p2 --log results/smoke_<date>/x2_eo_p2.log -- python3 tools/train.py \
    --config configs/finetune/BicubicPlusPlus/bicubicplusplus_2x_eo.yaml --epochs 1 --work_dir results/smoke_<date>/x2_eo_p2

# Continue an interrupted run (config comes from the run folder)
docker/run.sh --gpu <N> --name <run> -- python3 tools/train.py --resume <run_dir>/last.pth
```

출하 config는 모두 `data_config_path`를 담고 있어서 `--data_config`가 필요 없습니다. 시드를 주지 않으면 42입니다(출하 config에는 `train.seed`가 없음). `--work_dir`는 매번 새 폴더로 줍니다. 이유는 [함정](#trainpy-함정)의 첫 항목에 있습니다.

### 옵션

| 옵션 | 기본값 | 뜻 | `--resume`과 함께 |
|---|---|---|---|
| `--config` | 없음 | 학습 config. `--resume`이 없으면 필수 | 거부 |
| `--data_config` | config의 `data_config_path`, 없으면 task 기본값(sr `configs/data/sr_train.yaml`, denoise `configs/data/bicubicpp/denoise.yaml`) | 데이터(열화) 레시피 | 거부 |
| `--model` | config 값 | `model.name`만 바꿈. `bicubicplusplus`, `svfocussrnet`만 받음 | 적용되지만 쓰지 않음(strict 로드가 깨짐) |
| `--task` | config 값 | `task`(`sr` 또는 `denoise`)만 바꿈 | 적용되지만 쓰지 않음(데이터셋이 바뀜) |
| `--scale` | config 값 | `model.scale`만 바꿈 | 적용되지만 쓰지 않음(strict 로드가 깨짐) |
| `--work_dir` | `checkpoints/<train.save_name>`, `save_name`이 없으면 `checkpoints/<model>_<task>_x<scale>_<YYYYmmdd_HHMMSS>` | run 폴더 | 거부(run 폴더는 체크포인트가 있는 폴더) |
| `--resume` | 없음 | 이어서 학습할 체크포인트. 없는 파일이면 시작하지 않고 exit 2 | |
| `--epochs` | `train.epochs` | 마지막 epoch 번호(총 epoch 수) | 허용 |
| `--batch_size` | `train.batch_size` | 배치 크기 | 허용 |
| `--lr` | `train.lr` | 옵티마이저의 시작 LR | 거부(LR은 체크포인트의 옵티마이저·스케줄러 상태에서 옴) |
| `--device` | `cuda` | 모델을 처음 만들고 pretrained 가중치를 올릴 장치. 학습 장치는 Accelerate가 정함 | 허용 |
| `--seed` | `train.seed`, 없으면 42 | python·numpy·torch 시드. 로그에 `Seed: N` | 허용 |

`--resume`과 함께 거부되는 옵션을 주면 `--resume reads its config from the run directory and writes there; drop --config` 같은 메시지와 함께 exit 2로 끝납니다.

### config를 읽는 순서

1. `--config` YAML을 읽습니다(재개라면 `<run_dir>/train_config.yaml`).
2. 데이터 레시피를 `--data_config` → `data_config_path` → task 기본값 순서로 고릅니다. 경로를 적었는데 파일이 없으면 `FileNotFoundError`로 멈춥니다. 빈 레시피는 "열화 없음"이 아니라 "기본값 전부 켜짐"이라서, 오타 하나로 엉뚱한 레시피를 며칠 학습하지 않도록 조용히 넘어가지 않습니다.
3. 명령줄 덮어쓰기(`--model`, `--task`, `--scale`, `--epochs`, `--batch_size`, `--lr`)를 마지막에 적용합니다.
4. 데이터셋은 `task`만으로 정해집니다(`sr` → `SRDataset`, `denoise` → `DenoiseDataset`, 그 밖은 `ValueError`). `denoise`는 `model.scale: 1`이어야 합니다.
5. 검증은 `data.val_hr_root`와 `data.val_lr_root`를 둘 다 적어야 합니다. 하나만 적으면 `ValueError`입니다. 둘 다 없으면 `train.allow_no_validation: true`일 때만 돌고, 그때는 `best.pth`가 생기지 않고 조기 종료도 일어나지 않습니다.

필수 키와 각 키의 뜻은 [configs/README.md](../configs/README.md)에 있습니다.

### run 폴더 구조

```
<work_dir>/
├── train_config.orig.yaml       # verbatim copy of --config (comments kept)
├── data_config.orig.yaml        # verbatim copy of the data recipe
├── train_config.yaml            # effective config, overrides applied; --resume reads this
├── data_config.yaml             # effective data recipe; --resume reads this
├── train_config.resume_ep<N>_<stamp>.yaml   # one per resume, history only
├── training_log.csv             # epoch,train_loss,val_loss,psnr,ssim,lpips,niqe,lr (appended)
├── logs/events.out.tfevents.*   # TensorBoard
├── plots/
│   ├── train_batch.png          # written at start
│   ├── dataset_stats.png        # written at start
│   ├── results.png              # rewritten every epoch
│   └── probe_ep<N>.png          # every train.save_interval epochs
├── best.pth                     # only when the monitored metric beats the best so far
├── best_attempt.pth             # this run's own best, always written
├── epoch_<N>.pth                # every train.save_interval epochs
└── last.pth                     # end of every epoch
```

콘솔 출력은 run 폴더가 아니라 `run.sh`의 로그(`logs/<run>.log` 또는 `--log`)에 갑니다. `train.plots: false`면 `plots/`를 만들지 않습니다.

체크포인트(`.pth`) 하나에는 `model_state_dict`, `live_state_dict`, `ema_used`, `optimizer_state_dict`, `scheduler_state_dict`, `epoch`, `best_val_loss`, `best_metric_val`, `best_attempt`, `early_stop_counter`, `metric_spec`(`y-ycbcr/crop=scale`), `git_sha`(짧은 sha, 추적 파일이 바뀌었으면 `-dirty`), `config`(덮어쓰기까지 적용된 전체 config)가 들어 있습니다. `train.ema.start_epoch`(출하 config는 2)부터 `model_state_dict`는 EMA 가중치이고, 검증도 그 EMA 가중치로 채점합니다. `evaluate.py`, `inference.py`, `export.py`, `6_apply_qat.py`는 모두 `model_state_dict`를 읽습니다. EMA가 왜 이렇게 저장되는지는 [src/README.md](../src/README.md)를 봅니다.

### best.pth, best_attempt.pth, last.pth 중 무엇을 쓰나

- **`best.pth`**: `train.early_stopping.monitor`(출하 config: Phase-1 `psnr`, Phase-2와 Denoise `lpips`)가 지금까지의 최고를 `min_delta`보다 크게 넘을 때만 씁니다.
- **Phase-2에는 `best.pth`가 없을 수 있습니다.** `train.pretrained_path`를 읽은 런은 학습 전에 시작 가중치를 한 번 채점하고(`[Validation][Epoch 0] ...`, `[Trainer] Baseline from pretrained weights: lpips=... best.pth is only written if this is beaten.`), 그 점수를 최고값의 출발점으로 삼습니다. 파인튜닝이 Phase-1보다 나빠졌는데도 `best.pth`로 출하되는 일을 막기 위해서입니다. 2026-10-19 EO x2 1 epoch 확인 런은 기준 LPIPS 0.167224를 넘지 못해(epoch 1: 0.1772) `best.pth` 없이 `last.pth`와 `best_attempt.pth`만 남겼습니다. 출하 EO x2 Phase-2 런 폴더에도 `best.pth`가 없고, 그래서 출하 `EO_x2.pth`는 Phase-1 모델입니다.
- **`best_attempt.pth`**: 시작점 기준과 상관없이 이 런 안에서의 최고입니다. 항상 생깁니다. Phase-2에서 `best.pth`가 없다면 `best_attempt.pth`는 Phase-1보다 지표가 나쁜 가중치라는 뜻이므로, 출하 후보가 아니라 비교·분석용입니다.
- **`last.pth`**: 매 epoch 끝에 씁니다. 파이프라인 확인(1 epoch smoke)의 export와 평가에는 `last.pth`나 `best_attempt.pth`를 씁니다.
- 조기 종료가 걸린 epoch에는 `last.pth`와 `epoch_<N>.pth`를 쓰지 않고 멈춥니다. 그래서 그 경우 `last.pth`는 `training_log.csv`의 마지막 행보다 한 epoch 앞입니다.
- Phase-2의 기준 점수(epoch 0)는 `training_log.csv`에 없고 콘솔 로그와 TensorBoard에만 있습니다.

### 이어서 학습 (`--resume`)

- 재개의 기준은 `<run_dir>/train_config.yaml`과 `<run_dir>/data_config.yaml`입니다(`<run_dir>`은 `--resume` 파일이 있는 폴더). `train_config.yaml`이 없으면 `FileNotFoundError: Cannot find train_config.yaml in <run_dir>`로 멈춥니다. `.pth` 안에 든 config는 다시 읽지 않습니다.
- 그래서 설정을 바꿔 재개하려면 `<run_dir>/train_config.yaml`을 고칩니다. 데이터 경로, 손실 가중치, `early_stopping.patience`, `save_interval` 같은 값은 반영됩니다. LR과 스케줄러(`T_max`, `total_steps` 포함)는 반영되지 않습니다. 옵티마이저·스케줄러 상태를 체크포인트에서 복원하면서 덮어쓰기 때문입니다.
- 복원하는 것: 옵티마이저가 쓰던 가중치(`live_state_dict`, 없으면 `model_state_dict`), EMA 상태, 옵티마이저, 스케줄러, 시작 epoch(저장된 epoch + 1), `best_val_loss`, `best_metric_val`, `best_attempt`, `early_stop_counter`. 마지막 두 개는 2026-10-16 이후 체크포인트에만 있습니다.
- 복원하지 않는 것: 난수 상태. 재개한 런은 끊김 없이 돈 런과 데이터 순서가 달라집니다. Phase-2의 시작 가중치 채점도 다시 하지 않고, 체크포인트에 저장된 최고값을 이어 씁니다.
- `--epochs`는 "더 돌릴 epoch 수"가 아니라 마지막 epoch 번호입니다. 다 끝난 런을 `--epochs`를 올리지 않고 재개하면 아무것도 학습하지 않고 끝납니다.
- 조기 종료로 멈춘 런은 카운터가 복원되므로, 개선이 없으면 한 epoch 만에 다시 멈춥니다. 더 돌리려면 `<run_dir>/train_config.yaml`의 `early_stopping.patience`를 올립니다.
- run 폴더의 어떤 체크포인트로도 재개할 수 있습니다(`epoch_<N>.pth`, `best_attempt.pth` 등). 그 epoch + 1부터 다시 돌고 `training_log.csv`에 이어 쓰므로, 예전 체크포인트에서 재개하면 같은 epoch 번호의 행이 두 번 생깁니다.
- 재개할 때마다 `train_config.resume_ep<N>_<stamp>.yaml`을 기록으로 남기고, `train_config.yaml`은 건드리지 않습니다.
- `docker stop`이나 호스트 재부팅으로 멈추면 진행 중이던 epoch는 잃습니다. Trainer에는 신호 처리기가 없고, 체크포인트는 epoch 끝에만 씁니다. `last.pth`를 쓰는 도중에 멈춰 파일이 깨졌다면 `epoch_<N>.pth`나 `best_attempt.pth`로 재개합니다.

### 걸린 시간 (참고)

| 런 | 스텝 | 시간 | 출처 |
|---|---|---|---|
| EO x2 Phase-2, 1 epoch, 배치 16 | 677 | 학습 7분 56초 + 검증 2회(각 약 13초), 전체 약 9분 | `results/smoke_20261019/x2_eo_p2.log`, 2026-10-19 컨테이너 |
| Denoise, 1 epoch, 배치 128 | | 약 2분 | `results/docker_accept_20261020_final/train_dn_1ep/`, 2026-10-20 컨테이너 |
| Phase-1 x2, 1 epoch, 배치 16 | 6,182 | 로컬 사본에서 약 21분, sshfs에서는 1.1~3.2 s/step | 2026-10-06 기록(컨테이너 도입 전, 호스트 실행) |

`--epochs 1`도 학습 데이터 전체를 한 번 돕니다(스텝 수 제한 옵션은 없음). Phase-1 config의 `data.train_root`는 sshfs(`/mnt/data_server/.../Pretrained_Dataset/tmp/`)를 가리키고, 같은 데이터의 로컬 사본이 `/mnt/data2/jshong/SuperResolution/Pretrained_Dataset/`에 있습니다. 사본을 쓰려면 config를 복사해 경로를 바꿉니다([configs/README.md](../configs/README.md)).

### 로그에서 확인할 줄

| 줄 | 뜻 |
|---|---|
| `Seed: 42`, `Using device: cuda` | 시드와 장치 |
| `[Degradation] SRDataset(train) stage1 active: ...` / `... off: ...` | 실제로 켜진 열화. 레시피 오타를 여기서 잡음 |
| `[Finetune] matched=32 missing=0 unexpected=0` | Phase-2 시작 가중치 로드(`matched`는 x2 32, x3 25, x4 40). `missing`이나 `unexpected`가 0이 아니면 일부만 올라간 것 |
| `[Trainer] Deployment softening: 50% of each batch through best.pth` | 배포 softening이 켜짐(Phase-1, EO Phase-2) |
| `[Validator] Metrics: y-ycbcr/crop=scale, scale=2 ...` | 채점 기준 |
| `[Trainer] Baseline from pretrained weights: ...` | Phase-2 기준 점수 |
| `[Validation][Epoch N] Loss: ... \| PSNR: ... \| SSIM: ... \| LPIPS: ... \| NIQE: ...` | epoch별 검증 |
| `Saved Best Checkpoint: ...` / `[Trainer] EarlyStopping counter: k out of P (Monitor: lpips)` | 최고 갱신 여부 |
| `[Trainer] Early stopping triggered.` | 조기 종료 |
| `[run.sh] exit 0` | 작업 정상 종료(`run.sh --name`일 때 마지막 줄) |

tqdm 진행 표시가 캐리지 리턴(`\r`)을 써서 로그가 한 줄에 몰립니다. 따라가기는 `tail -f logs/<run>.log`로 되고, 검색할 때는 줄바꿈으로 바꿔서 봅니다.

```bash
tr '\r' '\n' < logs/<run>.log | grep -a -e "Validation\]" -e "Baseline" -e "Saved" -e "run.sh"
```

TensorBoard는 호스트에 설치돼 있어 컨테이너 없이 봅니다(`tensorboard --logdir <work_dir>/logs`). 손실 항목별 값(`Loss/<name>`)과 Phase-2 기준 점수(epoch 0)는 CSV에 없고 여기에만 있습니다.

### train.py 함정

- **같은 run 폴더를 다시 쓰면 기록이 섞입니다.** `--work_dir` 없이 같은 config를 두 번 돌리면 둘 다 `checkpoints/<save_name>/`을 씁니다. 두 번째 런이 `train_config.yaml`과 `.pth`를 덮어쓰고, `training_log.csv`에는 헤더 없이 epoch 1부터 행을 이어 붙입니다. CSV 중간에 epoch 번호가 1로 돌아가 있으면 이 경우입니다. 매번 새 `--work_dir`를 줍니다.
- **`--device`로 GPU를 고를 수 없습니다.** 학습 장치는 Accelerate가 보이는 첫 GPU로 정합니다. `--device cpu`를 줘도 GPU가 붙어 있으면 GPU에서 학습하고, `--device 1` 이상은 컨테이너 안에서 실패합니다. CPU 학습이 필요하면 `--gpu`를 뺍니다.
- **0인 덮어쓰기는 조용히 무시됩니다.** `--epochs 0`, `--lr 0`, `--batch_size 0`, `--scale 0`은 아무것도 바꾸지 않습니다.
- **`--scale`은 `model.scale`만 바꿉니다.** 검증 LR 폴더, 데이터 레시피, `save_name`은 원래 배율 그대로라 검증이 크기가 맞지 않는 LR로 돕니다. Validator가 `[Validator] Shape mismatch detected ...` 경고를 한 번 찍고 공통 크기로 잘라 계속 채점하므로 지표가 의미 없어집니다. 다른 배율은 그 배율의 config로 돌립니다.
- **`--task`는 기본 데이터 레시피를 바꾸지 못합니다.** 레시피는 YAML에 적힌 원래 `task`로 먼저 고르고, `--task`는 그 뒤에 적용됩니다. 출하 config는 `data_config_path`를 적어 두어 상관없습니다.
- **Phase-1 config에서는 `--lr`이 효과가 없습니다.** KneeLR은 `scheduler.peak_lr`로 LR을 직접 계산하고, 출하 Phase-1 config는 `peak_lr: 5e-4`를 적어 두었습니다. CosineAnnealingLR(Phase-2, Denoise)에서는 `--lr`이 시작 LR이 됩니다.
- **`--epochs`는 스케줄 길이를 바꾸지 않습니다.** Cosine의 `T_max`와 KneeLR의 `total_steps`는 config 값 그대로입니다. `T_max`를 넘겨 돌리면 cosine LR이 다시 올라가고, KneeLR은 `total_steps` 뒤로 `min_lr`에 머뭅니다. 그래서 `--epochs 1` 확인 런은 거의 최대 LR로 한 epoch를 돕니다.
- **Phase-2 시작 가중치는 `strict=False`로 올립니다.** 이름이 하나도 맞지 않을 때(`matched=0`)와 크기가 다른 텐서가 있을 때만 `RuntimeError`로 멈추므로, `[Finetune] matched=... missing=0 unexpected=0`을 직접 확인합니다. `train.pretrained_path` 파일이 없으면 랜덤 초기화로 넘어가지 않고 `FileNotFoundError`(`pretrained_path 를 찾을 수 없습니다`)로 멈춥니다. `--resume`일 때는 `pretrained_path`를 읽지 않습니다.
- **git 밖의 파일에 기대는 런이 있습니다.** Phase-1과 EO Phase-2는 `train.deploy_softening.denoise_checkpoint`(`checkpoints/csuav_deploy/train_svfocusdenoise_adv_x1_..._260309/best.pth`)가 있어야 하고, Phase-2는 `train.pretrained_path`(`checkpoints/bicubicpp_deploy/train_bicubicplusplus_sr_x<s>_..._v3/best.pth`)가 있어야 합니다. `checkpoints/`는 git에 없으므로 다른 PC에서는 `FileNotFoundError`로 멈춥니다. 이 서버에는 모두 있습니다.
- **새 런은 `checkpoints/bicubicpp_deploy/`에 생기지 않습니다.** 출하 런은 `checkpoints/bicubicpp_deploy/<save_name>/`에 있고, 기본 run 폴더는 한 단계 위인 `checkpoints/<save_name>/`입니다. Phase-2 config의 `pretrained_path`는 계속 출하 Phase-1을 가리키므로, 새로 학습한 Phase-1에서 파인튜닝하려면 config를 복사해 `pretrained_path`를 새 런의 `best.pth`로 바꿉니다. Phase-1이 끝나면 Phase-2를 자동으로 시작하는 스크립트는 없습니다(Phase-2 → QAT 자동 체인은 사고 이력 때문에 2026-09-14에 제거했습니다).
- **런끼리의 검증 점수는 바로 비교할 수 없습니다.** Phase-1(x2 `val/LRx2_v3`), EO Phase-2(`val/LRx2_weaknoise_eo_v3`), IR Phase-2(`Finetunning_Dataset/val/...`), Denoise(`val_denoise_cam_a9nr5/LR`)는 서로 다른 LR 세트로 검증합니다. 비교는 `evaluate.py`에 같은 `--hr_dir`/`--lr_dir`를 주고 합니다.
- **재현성.** `cudnn.deterministic`은 처리량 때문에 일부러 켜지 않았습니다. 그래도 같은 config와 시드 42로 다시 학습했을 때 출하 v3와 가중치가 비트 단위로 같았던 기록이 있습니다(호스트 실행, [docs/experiments/20260928-EXP-00-1.md](../docs/experiments/20260928-EXP-00-1.md)). 컨테이너에서는 확인하지 않았고, `--resume`을 한 번이라도 거친 런은 난수 상태가 달라 재현되지 않습니다.
- **단일 프로세스.** 도구는 `accelerate launch` 없이 돌아 한 프로세스로 학습하고, `run.sh`는 GPU를 한 장만 붙입니다.

---

## 3. evaluate.py

체크포인트를 배포 그래프(`switch_to_deploy()` 뒤 FP32)로 만들어 짝지어진 HR/LR 세트에서 채점합니다. 학습 Validator와 같은 지표 정의(`src/metrics`)를 쓰므로 `training_log.csv`와 숫자가 그대로 비교됩니다. 2026-10-19 확인 런에서 epoch 1의 로그 값(PSNR 31.0827, SSIM 0.8686, LPIPS 0.1772, NIQE 4.547)이 `evaluate.py` 결과와 같았습니다.

```bash
# verified form (2026-10-19, --gpu 2)
docker/run.sh --gpu <N> --name <run> --log <dir>/<run>.log -- python3 tools/evaluate.py \
    --config configs/finetune/BicubicPlusPlus/bicubicplusplus_2x_eo.yaml \
    --checkpoint <run_dir>/last.pth --save_dir <eval_dir>

# general form: the run's own config always matches its checkpoint
docker/run.sh --gpu <N> -- python3 tools/evaluate.py \
    --config <run_dir>/train_config.yaml --checkpoint <run_dir>/best.pth \
    --save_dir <eval_dir> [--hr_dir <HR> --lr_dir <LR>] [--save_images]
```

100장 검증 세트 채점은 GPU에서 10초 남짓 걸렸습니다(2026-10-19).

### 옵션

| 옵션 | 기본값 | 뜻 |
|---|---|---|
| `--config` | 필수 | `model` 블록으로 모델을 만들고, `data.val_hr_root`/`data.val_lr_root`를 기본 데이터로 씀 |
| `--checkpoint` | 필수 | 채점할 `.pth` |
| `--save_dir` | 필수(빠지면 exit 2) | 결과 상위 폴더. `<save_dir>/<체크포인트 파일 이름(확장자 제외)>/`에 씀 |
| `--hr_dir` | config `data.val_hr_root` | HR 폴더 |
| `--lr_dir` | config `data.val_lr_root` | LR 폴더 |
| `--device` | `cuda` | [1절 표](#gpu-유무에-따른-동작) 참고 |
| `--save_images` | 꺼짐 | 결과 이미지도 저장 |

HR과 LR 경로가 명령줄과 config 어디에도 없으면 `ValueError: Dataset paths must be provided via CLI arguments or Config file.`로 멈춥니다.

### 데이터와 지표

- 폴더는 맨 위 단계만 읽고(하위 폴더 안 봄) `.png`, `.jpg`, `.jpeg`만 씁니다(대소문자 무관). HR과 LR은 파일 이름(확장자 제외)으로 짝짓습니다.
- 한쪽에만 있는 이름이 하나라도 있으면 `PairedDataset stem mismatch: HR=..., LR=..., matched=...`로, 같은 이름이 두 번 나오면 `duplicate stem`으로, 두 폴더 모두 이미지가 없으면 `FileNotFoundError`로 멈춥니다. 예전에는 정렬 순서로 짝지어 파일 하나가 빠지면 뒤의 짝이 전부 밀린 채 그럴듯한 숫자가 나왔기 때문에 멈추게 했습니다.
- 출력과 HR 크기가 다르면(x3에서 1280폭 → 1278) 학습 Validator처럼 둘 다 공통 크기로 자릅니다. 출력은 [0, 1]로 자릅니다.
- PSNR과 SSIM은 휘도(BT.601 ycbcr) 채널에서 테두리 `scale` 픽셀을 깎고 잽니다. LPIPS는 RGB, NIQE는 출력만 보는 무참조 지표입니다. 이 기준(`y-ycbcr/crop=scale`)이 생기기 전의 로그보다 PSNR이 2 dB 안팎 높게 나오는데, 자가 바뀐 것이지 모델이 바뀐 것이 아닙니다.
- 채점 대상은 FP32 배포 그래프입니다. W16A16 양자화 결과나 DLC가 아닙니다. 양자화 그래프 채점은 [examples/README.md](../examples/README.md)를 봅니다.

### 출력

```
<save_dir>/<checkpoint stem>/
├── metrics.csv        # filename,PSNR,SSIM,LPIPS,NIQE  (one row per image, no mean row)
└── <HR file name>     # only with --save_images
```

평균은 CSV에 없고 콘솔의 `=== Evaluation Summary ===` 아래에만 찍힙니다. `--save_images` 이미지는 `inference.py`와 같은 반올림으로 저장되어 두 도구의 결과 파일이 바이트 단위로 같습니다.

### 두 런 비교

```bash
docker/run.sh --gpu <N> -- python3 tools/evaluate.py --config <runA>/train_config.yaml --checkpoint <runA>/best.pth \
    --hr_dir <HR> --lr_dir <LR> --save_dir results/compare/<A>
docker/run.sh --gpu <N> -- python3 tools/evaluate.py --config <runB>/train_config.yaml --checkpoint <runB>/best.pth \
    --hr_dir <HR> --lr_dir <LR> --save_dir results/compare/<B>
```

같은 `--hr_dir`/`--lr_dir`를 주고, `--save_dir`는 런마다 따로 줍니다. 그러면 이미지별 행을 서로 맞대어 볼 수 있습니다.

### evaluate.py 함정

- **결과 폴더 이름은 체크포인트 파일 이름입니다.** 두 런의 `best.pth`를 같은 `--save_dir`로 채점하면 둘 다 `<save_dir>/best/`에 쓰고, 뒤의 것이 `metrics.csv`를 덮어씁니다.
- **`--gpu`를 빼면 경고 없이 CPU로 돕니다.** LPIPS와 NIQE가 CPU에서 느려 오래 걸립니다.
- **`--config`의 `model` 블록이 체크포인트와 달라도 멈춥니다.** [6절](#6-체크포인트와-config-짝-세-도구-공통-로더)을 봅니다.
- **HR이 `.jpg`면 `--save_images` 결과도 `.jpg`(손실 압축)입니다.** 저장 파일은 HR 파일 이름과 확장자를 그대로 씁니다.

---

## 4. inference.py

```bash
docker/run.sh --gpu <N> -- python3 tools/inference.py \
    --config configs/finetune/BicubicPlusPlus/bicubicplusplus_4x_eo.yaml \
    --checkpoint results/release_v3/checkpoints/EO_x4.pth \
    --input <image_or_dir> --output <out_dir> [--fp16]
```

### 옵션

| 옵션 | 기본값 | 뜻 |
|---|---|---|
| `--input` | 필수 | 이미지 파일 하나, 또는 이미지가 든 폴더 |
| `--output` | `results/inference` | 결과 폴더(저장소 루트 기준). 없으면 만듦 |
| `--config` | 필수 | `model` 블록으로 모델을 만듦 |
| `--checkpoint` | 필수 | `.pth` |
| `--device` | `cuda` | [1절 표](#gpu-유무에-따른-동작) 참고 |
| `--fp16` | 꺼짐 | 모델과 입력을 half로 바꿈. 장치가 `cuda`일 때만 적용 |

### 동작과 출력

- 폴더를 주면 맨 위 단계의 파일을 이름순으로 읽고 `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tiff`만 씁니다(대소문자 무관). 파일 하나를 주면 확장자를 보지 않고 그대로 읽습니다.
- 이미지는 `cv2.imread`로 8비트 3채널로 읽습니다. 읽지 못하면 `ValueError: Failed to read image: <path>`로 멈춥니다.
- 결과는 `<output>/<입력 파일 이름>`으로, 입력과 같은 확장자로 저장합니다. 값은 [0, 1]로 자른 뒤 반올림해 8비트로 씁니다. 같은 이름의 파일이 있으면 덮어씁니다.
- 끝에 `Average Inference Time (Images): X ms / image`를 찍습니다. forward만 잰 값(`cuda.synchronize` 포함)이고, 첫 호출의 워밍업이 들어가며 파일 읽기·쓰기는 빠집니다. 보드 지연 시간이 아닙니다. 보드 속도는 `bench_dlc.sh`로 잽니다([examples/deployment/README.md](../examples/deployment/README.md)).

### inference.py 함정

- `.tif`(f 하나)는 폴더 입력에서 빠집니다. `.tiff`만 받습니다.
- `.jpg` 입력의 결과는 `.jpg`로 저장되어 손실 압축이 한 번 더 들어갑니다. 화질 비교에는 PNG 입력을 씁니다.
- 출력 폴더가 저장소와 `/mnt` 밖(예: `/tmp`)이면 결과가 컨테이너 안에만 쓰이고 컨테이너와 함께 사라집니다. 저장소나 `/mnt` 아래로 주거나 `run.sh --mount`로 붙입니다.

---

## 5. export.py

FP32 ONNX를 만들어 그래프 구조를 확인하고, 연산량과 파라미터 수를 봅니다. 양자화가 없으므로 DLC 변환에 쓰는 파일이 아닙니다. IR의 무채색 접기는 `switch_to_deploy()`가 하므로 그래프 구조는 출하 그래프와 같습니다.

```bash
docker/run.sh -- python3 tools/export.py \
    --config configs/finetune/BicubicPlusPlus/bicubicplusplus_4x_eo.yaml \
    --checkpoint results/release_v3/checkpoints/EO_x4.pth \
    --width 320 --height 180 --output <onnx_dir>
# -> <onnx_dir>/EO_x4_180x320.onnx
```

### 옵션

| 옵션 | 기본값 | 뜻 |
|---|---|---|
| `--config` | 필수 | `model` 블록 |
| `--checkpoint` | 없음 | 없으면 랜덤 초기화 가중치로 export하고 `[Info] No checkpoint provided. ...` 한 줄만 찍음 |
| `--output` | 없음 → `results/` | ONNX를 둘 폴더 |
| `--height` | 256 | 입력 세로 |
| `--width` | 256 | 입력 가로 |
| `--opset` | 17 | ONNX opset |
| `--device` | `cpu` | `torch.device`에 그대로 넘김(`cpu` 또는 `cuda`) |

### 동작과 출력

- 입력은 `(1, 3, H, W)` 고정 크기이고, 입력 이름은 `input`, 출력 이름은 `output`입니다. `torch.onnx.export(dynamo=False)` 뒤에 `onnx.checker.check_model`을 돌립니다. export가 실패하면 0이 아닌 코드로 끝납니다. onnxsim은 쓰지 않습니다.
- 파일 이름은 `<체크포인트 파일 이름>_<H>x<W>.onnx`(세로가 먼저, 예 `EO_x4_180x320.onnx`)이고, `--checkpoint`가 없으면 `<model.name>_init_<H>x<W>.onnx`입니다. DLC 이름(`BicubicPP_EO_320x180_x4_<ver>.dlc`)은 가로가 먼저라 순서가 반대입니다.
- 콘솔에 `GFLOPs: x.xxxx`, `Params: x.xxxx M`, `ONNX export successful.`, `Verification passed.`가 찍힙니다. `GFLOPs` 값은 thop이 센 곱셈-누산(MAC) 수를 1e9로 나눈 것이라 실제로는 GMACs입니다. thop 계산이 실패해도 메시지만 찍고 export는 계속합니다.

### 배포 입력 크기 (`--width` x `--height`)

보드 그래프는 크기가 고정이라, 배포 크기와 같은 크기로 export해야 출하 그래프와 비교가 됩니다. 아래 값은 `tests/test_release_golden.py`의 출하 프로파일입니다.

| 프로파일 | `--width` | `--height` |
|---|---|---|
| x2 EO, x2 IR | 640 | 360 |
| x3 EO, x3 IR | 428 | 240 |
| x4 EO, x4 IR | 320 | 180 |
| x2 IR 1024 | 640 | 512 |
| x3 IR 1024 | 428 | 342 |
| x4 IR 1024 | 320 | 256 |
| Denoise | 1280 | 720 |

### export.py 함정

- `--output`을 빼면 저장소의 `results/`에 바로 씁니다.
- `--checkpoint`를 빼면 학습된 가중치가 아니라 랜덤 가중치입니다. 구조와 연산량 확인에는 맞지만 출력값은 의미가 없습니다.
- `--device cuda`는 `--gpu`를 줬을 때만 됩니다. 기본값 `cpu`로 충분합니다.

---

## 6. 체크포인트와 config 짝 (세 도구 공통 로더)

`evaluate.py`, `inference.py`, `export.py`는 같은 로더 `src/engine/checkpoint.py`의 `load_model`을 씁니다.

1. `torch.load(weights_only=False)`로 읽습니다.
2. 파일이 통째로 저장된 `nn.Module`이면(`2_compress_svd_pruning.py`의 `compressed_model.pth`) 그대로 씁니다.
3. 아니면 체크포인트에 저장된 `config['model']`과 `--config`의 `model` 블록을 비교해 다르면 `ValueError: <path> was trained with another model config; differing keys: <key> (checkpoint ..., --config ...)`로 멈춥니다. config를 저장하지 않은 옛 체크포인트는 비교하지 않습니다.
4. `--config`로 모델을 만들고, 이름 앞의 `module.`, `_orig_mod.`를 떼고, strict로 가중치를 올립니다.
5. `switch_to_deploy()`(재매개변수화 가지 합치기, IR 무채색 접기) 뒤 `eval()`로 돌려줍니다.

3번 비교가 있는 이유: `achromatic_output`처럼 가중치가 없는 설정은 strict 로드로 잡히지 않습니다. IR 체크포인트를 EO config로 열면 오류 없이 올라간 뒤 회색 입력에 색을 입힌 출력을 냅니다. 비교는 단순한 dict 동등 비교라서 기본값과 같은 값을 config에 명시로 적기만 해도 다르다고 판정합니다. 가장 안전한 짝은 그 런의 `<run_dir>/train_config.yaml`입니다. `6_apply_qat.py`도 같은 비교를 합니다.

출하 체크포인트의 짝(`tests/test_release_golden.py`가 검사하는 쌍):

| 체크포인트 | `--config` |
|---|---|
| `results/release_v3/checkpoints/EO_x2.pth` | `configs/train/BicubicPlusPlus/bicubicplusplus_2x.yaml`(Phase-1 모델. EO x2 Phase-2 config도 `model` 블록이 같아 통과) |
| `results/release_v3/checkpoints/EO_x3.pth`, `EO_x4.pth` | `configs/finetune/BicubicPlusPlus/bicubicplusplus_<3\|4>x_eo.yaml` |
| `results/release_v3/checkpoints/IR_x2.pth`, `IR_x3.pth`, `IR_x4.pth` | `configs/finetune/BicubicPlusPlus/bicubicplusplus_<2\|3\|4>x_ir.yaml` |
| `results/denoise_release_v1/checkpoint/best.pth` | `configs/train/Denoise/bicubicppdenoise.yaml` |

`.pth`는 git에 없으므로 이 파일들은 이 서버에만 있습니다.

---

## 7. 다음 단계와 관련 스크립트

학습과 평가 다음은 출하 export와 DLC 변환입니다. 출하 레시피는 W16A16, QAT 0 epoch(PTQ)입니다. 2026-10-19에 아래 형태로 돌았습니다. 옵션과 함정은 [examples/README.md](../examples/README.md) 1절(옵션 전체는 1.9절)과 3절, 변환과 보드는 [examples/deployment/README.md](../examples/deployment/README.md)가 다룹니다.

```bash
docker/run.sh --gpu <N> --name <run> --log <dir>/<run>.log -- python3 examples/aimet_quantization/6_apply_qat.py \
    --config <train_config> --data_config <data_config> --checkpoint <run_dir>/<file>.pth \
    --output_dir <export_root>/x<s>_<eo|ir>/qat --mmp_config configs/aimet/mmp_noprune_w16a16.json \
    --param_bw 16 --output_bw 16 --qat_epochs 0 --width <W> --height <H> --device 0
```

관련 스크립트(사용법은 [examples/README.md](../examples/README.md)):

- `examples/dataset_preparation/make_val_lr.py`: 학습 레시피로 SR 검증 LR 세트를 만듭니다.
- `examples/dataset_preparation/make_val_camera.py`: 카메라 모델로 Denoise 검증 세트(`<out_dir>/HR`, `<out_dir>/LR`)를 만듭니다.
- `examples/evaluation/score_grid_artifact.py`(IR 격자), `score_eo_chart.py`(Denoise 차트): 도메인 게이트 채점기입니다. 이 둘과 `make_val_camera.py`는 `sys.path`를 건드리지 않으므로 `docker/run.sh -- env PYTHONPATH=. python3 examples/evaluation/...`처럼 돌립니다.

---

## 8. 컨테이너에서 확인된 범위

2026-10-19, 이미지 `usdp:20261019` 기준입니다.

| 대상 | 실제 체크포인트로 돌린 것 | 그 밖의 확인 |
|---|---|---|
| `train.py` | EO x2 Phase-2 1 epoch(`results/smoke_20261019/x2_eo_p2`), Denoise 1 epoch(`docker/accept.sh`). 둘 다 `[run.sh] exit 0` | pytest |
| `train.py --resume` | 없음 | pytest(`tests/test_trainer.py`, `tests/test_config_guards.py`) |
| `evaluate.py` | EO x2 smoke `last.pth`(`results/smoke_20261019/x2_eo_eval/last/metrics.csv`) | pytest |
| `inference.py`, `export.py` | 없음 | pytest(`tests/test_tools_evaluate.py`, CPU의 작은 x3 dim-8 모델) |

SR Phase-1 전체 학습은 아직 컨테이너에서 돌린 적이 없습니다.
