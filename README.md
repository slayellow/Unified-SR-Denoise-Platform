# Unified-SR-Denoise-Platform

Super Resolution(SR, 초해상화)과 Denoise(노이즈 제거) 모델을 한 코드베이스에서 학습하고, 평가하고, Qualcomm AIMET으로 W16A16 양자화하고, QAIRT 2.41로 DLC로 바꿔 QCS8550 보드(Hexagon HTP)에 올리는 플랫폼입니다. SR과 Denoise는 같은 `tools/train.py`, 같은 `Trainer`, 같은 config 구조를 씁니다(`task: sr | denoise`).

이 문서는 입구입니다. 처음 한 번 하는 환경 준비, 새 모델을 보드까지 가져가는 한 바퀴, 매일 쓰는 명령, 산출물 위치, 문제 해결 색인을 담고, 세부는 각 문서로 연결합니다([문서 지도](#문서-지도)).

---

## 전체 흐름과 장비

| 단계 | 도구 | 도는 곳 |
|---|---|---|
| 학습 (Phase-1, Phase-2) | `tools/train.py` | 서버 작업 컨테이너 (GPU) |
| 평가 | `tools/evaluate.py`, `examples/evaluation/*.py` | 작업 컨테이너 |
| W16A16 export | `examples/aimet_quantization/6_apply_qat.py` (`--qat_epochs 0` = PTQ) | 작업 컨테이너 (GPU 또는 CPU) |
| 변환 스크립트 생성 | `examples/deployment/gen_convert_all.py` | 작업 컨테이너 |
| DLC 변환 | 생성된 `convert_all.sh` (`qairt-env` 안에서) | 작업 컨테이너 (CPU) |
| 보드에 올리기 | `scp`, `ssh` | 호스트 (jshong) |
| 지연 측정, 화면 비교 | `bench_dlc.sh`, `gst_compare_*.sh` | 보드 98 |

| 장비 | 역할 |
|---|---|
| 이 서버 (호스트) | RTX A6000 5장. k3s 노드이기도 해서 다른 사용자와 파드가 GPU를 같이 씁니다. 모든 작업은 `usdp` Docker 이미지 안에서 `docker/run.sh`로 돌립니다. 호스트의 `python3`(3.8. jshong의 `~/.local`에 torch 2.4.1이 있지만 AIMET, accelerate, pyiqa가 없음)는 작업 환경이 아닙니다. |
| 192.168.100.98 | QCS8550 테스트 보드. DLC는 `/data/dlc/`, 벤치 입력은 `/data/bench/`, 보드 스크립트는 `/data/*.sh`에 있습니다(2026-10-17 기록). 호스트의 jshong 키로 `ssh root@192.168.100.98`이 됩니다. |
| 192.168.100.87 | 운용 SW 장비. 병렬 부하 측정에 씁니다. |
| 회사 PC | QAIRT SDK가 있는 변환 PC. 2026-10-19부터 변환은 서버 컨테이너에서 하므로 선택 사항입니다. |

보드의 gst 파이프라인 실행과 화면 확인은 소유자가 직접 합니다. 자동화 작업은 DLC를 올리고 `bench_dlc.sh`로 재는 데까지만 합니다.

---

## 출하 모델

| 모델 | 도메인·배율 | 보드 입력 (W x H) | 체크포인트 | 평가·export에 쓰는 config | 보드 DLC (사본) |
|---|---|---|---|---|---|
| Bicubic++ SR v3 | EO x2 | 640x360 | `results/release_v3/checkpoints/EO_x2.pth` | `configs/finetune/BicubicPlusPlus/bicubicplusplus_2x_eo.yaml` (학습 config는 Phase-1 `configs/train/BicubicPlusPlus/bicubicplusplus_2x.yaml`, `model` 블록이 같음) | `results/release_v3/dlc/BicubicPP_EO_640x360_x2_v3.dlc` |
| Bicubic++ SR v3 | EO x3 / x4 | 428x240 / 320x180 | `.../EO_x3.pth`, `EO_x4.pth` | `configs/finetune/BicubicPlusPlus/bicubicplusplus_3x_eo.yaml`, `bicubicplusplus_4x_eo.yaml` | `.../dlc/BicubicPP_EO_<W>x<H>_x<s>_v3.dlc` |
| Bicubic++ SR v3 (IR 수정판, 10-06) | IR x2 / x3 / x4 | 720 프로필 640x360 / 428x240 / 320x180, 1024 프로필 640x512 / 428x342 / 320x256 | `.../IR_x<s>.pth` | `configs/finetune/BicubicPlusPlus/bicubicplusplus_<s>x_ir.yaml` | `.../dlc/BicubicPP_IR_<W>x<H>_x<s>_v3.dlc` (배율마다 2개) |
| BicubicPPDenoise[v1] (Bicubic++ `scale: 1`) | EO x1 | 1280x720 | `results/denoise_release_v1/checkpoint/best.pth` | `configs/train/Denoise/bicubicppdenoise.yaml` | `results/denoise_release_v1/BicubicPPDenoise_EO_1280x720_v1.dlc` |

- 전부 W16A16 PTQ이고, 출력층은 `final_act`이며, `snpe-dlc-graph-prepare --htp_socs qcs8550`로 미리 컴파일돼 있습니다. 수치, md5, 알려진 약점, 다시 만드는 법은 [results/release_v3/MANIFEST.md](results/release_v3/MANIFEST.md)와 [results/denoise_release_v1/MANIFEST.md](results/denoise_release_v1/MANIFEST.md)에 있습니다. IR 수정판의 변경 내용은 [results/release_v3/V3_CHANGES_20261006.md](results/release_v3/V3_CHANGES_20261006.md)에 있습니다.
- EO x2만 Phase-1 가중치입니다. Phase-2가 Phase-1의 LPIPS 0.1672를 넘지 못해서입니다(MANIFEST의 "x2 Phase-2 가 게이트를 못 넘은 이유"). 그래서 `EO_x2.pth`를 학습한 config는 Phase-1 config입니다. `model` 블록이 Phase-2 config와 같으므로, 출하 수치(합성 val LPIPS 0.167)와 출하 export는 Phase-2 config `bicubicplusplus_2x_eo.yaml`과 EO 레시피로 냈습니다(MANIFEST의 "재-export" 절, `docker/accept.sh` TC-DOCK-12-1).
- 보드에 출하본을 다시 올릴 때는 위 표의 DLC 사본만 씁니다. `results/release_v3/convert_all.sh`는 HTP prepare 단계가 생기기 전의 스크립트라, 그 출력을 올리면 보드의 미리 컴파일된 DLC가 같은 이름의 prepare 안 된 DLC로 바뀝니다. 에러는 나지 않고 보드가 실행할 때마다 그래프를 다시 준비하므로 CMA 절감이 사라집니다. 규칙은 [examples/deployment/README.md](examples/deployment/README.md)에 있습니다.
- `SVFocusSRNet`은 출하 모델이 아니지만 코드에 남아 있습니다. Phase-1과 EO Phase-2의 배포 열화 매칭(`train.deploy_softening`)이 옛 디노이즈 넷(SVFocusSRNet `scale: 1`)을 불러 쓰기 때문입니다(`src/engine/trainer.py`의 `_load_denoise_net`). 그 밖의 이전 모델(QuickSRNet, LRCSR, SVSRNet, DDRNet, MambaIR, NAFNet, Restormer)과 KD는 2026-10-16 `a667e92`에서 지웠습니다. 옛 파일은 `git show a667e92^:<옛 경로>`로 볼 수 있습니다.

---

## 처음 한 번: 환경 준비

자세한 내용은 [docker/README.md](docker/README.md)가 맡습니다. 여기에는 순서와 이유만 적습니다.

1. **Docker 권한.** jshong은 `docker` 그룹에 들어 있습니다(2026-10-19). 다른 계정으로 작업하려면 root가 그 계정을 `docker` 그룹에 넣어야 합니다.
2. **이미지 빌드.** `docker/build.sh`를 한 번 돌립니다. QAIRT tarball 위치는 환경 변수 `QAIRT_TARBALL`이고 기본값은 `$HOME/Code/qualcomm-vlm/env/QAIRT_2.41.tar.gz`입니다. 스크립트가 sha256을 확인하고 `usdp:<YYYYMMDD>`와 `usdp:latest` 두 태그를 붙입니다. 이미지에는 Python 3.10.12, `requirements.txt`의 178개 `==` 고정 패키지, QAIRT 2.41(`qairt-env` 안에서만 보임), ffmpeg 7.0.2가 들어 있습니다. 이미지는 빌드한 사람의 uid:gid를 라벨로 갖고, `docker/run.sh`는 root와 다른 uid의 호출을 거부합니다. 결과 파일이 jshong 소유로 남게 하려는 것이므로, 계정마다 자기 이미지를 빌드합니다. 지금 이미지는 `usdp:20261019`입니다.
3. **데이터 마운트.** SR 데이터는 sshfs 마운트 `/mnt/data_server/etc/jshong/SuperResolution`에 있고, 로컬 디스크 사본이 `/mnt/data2/jshong/SuperResolution`에 있습니다(sshfs는 공유 1 GbE라 느립니다). sshfs 마운트(`/mnt/data_server`, `/mnt/data_server2`)는 fstab에 없어서 호스트가 재부팅되면 사라집니다. `docker/run.sh`는 `/mnt/data_server`가 읽을 수 있는 sshfs 마운트가 아니면, 곧 root가 `-o allow_other`로 다시 마운트하기 전까지 모든 작업을 거부합니다(`/mnt/data_server2`는 확인하지 않습니다). CPU 작업과 `/mnt/data2`만 읽는 작업도 마찬가지입니다. 빈 마운트 지점을 데이터로 착각하고 학습하는 사고를 막으려는 장치입니다.
4. **이 서버에만 있는 파일.** 아래 파일은 git이 추적하지 않습니다(`checkpoints/`와 `results/**`는 git-ignore). 다른 PC로 옮기면 함께 복사해야 합니다. 없으면 해당 학습이 `FileNotFoundError`로 멈추고, golden 테스트는 실패 대신 skip이 됩니다.

   | 파일 | 쓰는 곳 |
   |---|---|
   | `checkpoints/bicubicpp_deploy/` | 출하 런 폴더. Phase-2 config 6종의 `train.pretrained_path`(Phase-1 `best.pth`)가 여기를 가리킵니다. |
   | `checkpoints/csuav_deploy/train_svfocusdenoise_adv_x1_dim32_block3_epoch300_lr1e-4_charbonnier_edge_ssim_hotpixel_260309/best.pth` | Phase-1과 EO Phase-2의 배포 열화 매칭용 디노이즈 넷 |
   | `results/release_v3/`, `results/denoise_release_v1/`의 바이너리 | 출하 체크포인트, qat export, DLC, 채점기 입력, golden 테스트 기준값 |
   | `~/.cache/torch/hub/` | vgg19, alexnet, LPIPS, NIQE 가중치. 컨테이너가 이 폴더를 붙이므로 `--offline` 작업이 내려받기 없이 돕니다. |

5. **인수 시험.** `docker/accept.sh --gpu <N>`이 이미지와 환경을 15개 항목으로 확인합니다. 약 14분 걸리고 항목마다 `PASS`/`FAIL`/`NOT RUN` 한 줄을 찍으며, 모두 PASS일 때만 0으로 끝납니다. 2026-10-19 실행은 15/15 PASS, pytest `3499 passed`, skip 0이었습니다(그 전 이미지 `112b82a375dd`. 그 뒤 같은 태그로 다시 빌드한 지금 이미지 `7fd2f0ce9454`에서는 아직 돌리지 않았습니다, [docker/README.md](docker/README.md) 10.3). 이미지를 다시 빌드했을 때, `docker/`를 고쳤을 때, 다른 호스트로 옮겼을 때 돌립니다.

---

## 작업 환경에서 꼭 지킬 것

전체 옵션과 거부 메시지는 [docker/README.md](docker/README.md)에 있습니다. 매번 영향을 주는 규칙만 적습니다.

- **저장소 루트에서 실행합니다.** `--log`와 `--mount` 경로는 호스트 셸의 현재 폴더 기준으로 풀리고, `--` 뒤 명령 안의 경로는 컨테이너의 작업 폴더(저장소 루트) 기준으로 풀립니다. 다른 폴더에서 상대 `--log`를 주면 로그가 엉뚱한 곳에 생깁니다.
- **GPU는 `--gpu <N>` 하나로 고릅니다.** 빈 GPU는 `nvidia-smi`로 찾습니다(아래 [매일 쓰는 명령](#매일-쓰는-명령)). 호스트 GPU N은 컨테이너 안에서 `cuda:0`이 됩니다. 그래서 `CUDA_VISIBLE_DEVICES`를 쓰지 않고, `--device`를 받는 스크립트에는 `0`이나 기본값 `cuda`를 줍니다. `--device 2`처럼 호스트 번호를 주면 컨테이너 안에 없는 GPU를 가리켜 실패합니다. 작업당 GPU는 한 장이고, 다중 GPU 학습은 이 환경에서 지원하지 않습니다.
- **다른 작업이 쓰는 GPU는 거부됩니다.** `nvidia-smi`에 프로세스가 보이거나 다른 `usdp` 컨테이너가 붙어 있으면 시작하지 않습니다. 같이 써도 된다고 확신할 때만 `--` 앞에 `--force`를 줍니다.
- **`--gpu`가 없으면 CPU로 돕니다.** `tools/evaluate.py`와 `tools/inference.py`는 경고 없이 CPU로 넘어가므로, GPU를 쓰려면 반드시 `--gpu <N>`을 붙입니다. 반대로 `tools/train.py`는 `--device cpu`를 줘도 GPU가 붙어 있으면 GPU로 학습합니다(Accelerate가 장치를 고름). CPU 학습은 `--gpu`를 빼서 합니다.
- **긴 작업은 `--name <run>`으로 백그라운드에서 돌립니다.** `run.sh`는 컨테이너가 뜨자마자 `run.sh: started usdp-<run>, log <경로> (stop: docker stop usdp-<run>)`을 찍고 0으로 끝납니다. 진짜 결과는 로그(`logs/<run>.log` 또는 `--log FILE`)의 마지막 줄 `[run.sh] exit <rc>`입니다. 0이 성공, 143은 `docker stop`으로 멈춘 것입니다. 같은 `--name`을 다시 쓰면 옛 로그 뒤에 이어 붙으므로 작업마다 새 이름을 씁니다.
- **컨테이너는 저장소, `/mnt`, `~/.cache/torch`만 봅니다**(그 밖은 `--mount SRC[:DST[:ro]]`). 호스트의 `/tmp`는 보이지 않으므로 출력 경로는 저장소나 `/mnt` 아래에 둡니다. 호스트 환경 변수도 넘어가지 않으므로 `-- env VAR=값 명령` 형태로 줍니다. `NUMPY_MADVISE_HUGEPAGE=0`과 `PYTHONUNBUFFERED=1`은 이미지 기본값이라 따로 줄 필요가 없습니다.
- **`--offline`은 네트워크를 끕니다**(`--network none`). 가중치를 내려받지 않는 작업(평가, export, 변환, pytest)에 붙여 숨은 다운로드를 막습니다.
- **작업은 호스트 재부팅이나 Docker 재시작을 넘기지 못합니다.** `docker stop`을 해도 아무것도 저장되지 않아 진행 중인 epoch는 잃습니다. `last.pth`는 epoch가 끝날 때마다 쓰이므로 거기서 `--resume`으로 이어 갑니다([tools/README.md](tools/README.md)).

---

## 새 모델을 보드까지

아래 순서는 2026-10-19에 이 서버에서 이미지 `usdp:20261019`로 끝까지 돌려 본 사슬(EO x2 Phase-2 1 epoch smoke run → DLC, 산출물 `results/smoke_20261019/`)을 일반화한 것입니다. 각 단계의 "기대 출력"은 그때 실제로 나온 값입니다. 보드 단계(7)는 그날 실행하지 않았습니다.

| 자리표시 | 뜻 | 2026-10-19 검증 때 값 |
|---|---|---|
| `<N>` | 빈 GPU 번호 | 3 (학습), 2 (평가·export) |
| `<out>` | 이 실험의 산출물 폴더 (저장소나 `/mnt` 아래) | `results/smoke_20261019` |
| `<cfg>` | 학습 config | `configs/finetune/BicubicPlusPlus/bicubicplusplus_2x_eo.yaml` |
| `<recipe>` | 그 config의 `data_config_path` (데이터 레시피) | `configs/data/bicubicpp/sr_finetune_eo_phase-2.yaml` |
| `<s>`, `<dom>` | 배율 2/3/4, 도메인 `eo`/`ir` | `2`, `eo` |
| `<W> <H>` | 보드 입력 크기 (4단계 표) | `640 360` |
| `<ver>` | 새 DLC 버전 이름 (영문자, 숫자, `_`만) | `smoke` |

### 0. 실험 config 준비

출하 config는 golden 테스트(`tests/test_release_golden.py`, `tests/test_data_recipe_golden.py`)가 고정하고 있으므로 고치지 않습니다. 출하 config를 새 파일로 복사해서 고칩니다. 실험 config는 `configs/experiments/<topic>/`에 두고 커밋하지 않는 것이 지금까지의 방식인데, 이 폴더는 git-ignore가 아니므로 `git add -A`를 쓰지 않습니다. 키 설명과 새 config 만드는 법은 [configs/README.md](configs/README.md), 값의 근거는 [docs/CONFIG_RATIONALE.md](docs/CONFIG_RATIONALE.md)에 있습니다.

### 1. (선택) Phase-1 사전 학습

배율별 Phase-1부터 다시 학습할 때만 합니다.

```bash
docker/run.sh --gpu <N> --name <exp>_p1 -- python3 tools/train.py \
    --config configs/train/BicubicPlusPlus/bicubicplusplus_<s>x.yaml --work_dir <out>/p1_x<s>
```

- 출하 Phase-2 config의 `train.pretrained_path`는 출하 Phase-1(`checkpoints/bicubicpp_deploy/...`)을 가리킵니다. 새 Phase-1에서 이어 가려면 복사한 Phase-2 config의 `pretrained_path`를 `<out>/p1_x<s>/best.pth`로 바꿉니다.
- Phase-1은 120 epoch이고 `tmp/`(98,897장)를 읽습니다. 2026-10-06 기록으로 sshfs에서는 1.1~3.2 s/step, 로컬 사본(`/mnt/data2`)에서는 약 21분/epoch(3개 동시 실행)였습니다. 로컬 사본을 쓰는 config는 [configs/README.md](configs/README.md)를 봅니다.
- Phase-1 → Phase-2를 자동으로 잇는 스크립트는 없습니다. Phase-1 로그가 `[run.sh] exit 0`으로 끝난 것을 보고 Phase-2를 시작합니다.

### 2. Phase-2 파인튜닝

```bash
docker/run.sh --gpu <N> --name <exp>_p2 --log <out>/x<s>_<dom>_p2.log -- python3 tools/train.py \
    --config <cfg> --work_dir <out>/x<s>_<dom>_p2
```

검증 때는 `--epochs 1`을 더해 1 epoch만 돌렸습니다. 기대 출력은 다음과 같습니다.

- 로그 앞부분: `[Finetune] matched=32 missing=0 unexpected=0`, EO면 `[Trainer] Deployment softening: 50% of each batch through best.pth`, 그리고 `[Degradation] ... active/off` 줄. `missing=0`이 아니면 다른 구조의 가중치가 일부만 실린 것입니다(이 줄만 찍고 계속 돕니다. `matched=0`이면 `RuntimeError`로 멈춥니다).
- 출발점 검증: `[Validation][Epoch 0] ... PSNR: 31.3186 | ... LPIPS: 0.1672 ...`와 `[Trainer] Baseline from pretrained weights: lpips=0.167224 (loss=0.033917). best.pth is only written if this is beaten.`
- epoch마다 `[Validation][Epoch N] ...`. 검증 때 epoch 1은 PSNR 31.0827, LPIPS 0.1772였습니다.
- 끝: `Training finished.` 다음 줄에 `[run.sh] exit 0`.
- 런 폴더: `last.pth`, `best_attempt.pth`, `training_log.csv`, `train_config.yaml`, `train_config.orig.yaml`, `data_config.yaml`, `data_config.orig.yaml`, `logs/`(TensorBoard), `plots/`. `epoch_<N>.pth`는 `save_interval`(Phase-2는 20) epoch마다 생깁니다.

**`best.pth`가 없을 수 있습니다.** 파인튜닝은 출발 가중치를 먼저 검증하고, 그 점수를 넘은 epoch만 `best.pth`로 저장합니다. 검증 때 epoch 1의 LPIPS 0.1772가 출발점 0.1672보다 나빠서 `best.pth`가 생기지 않았습니다. 에러가 아닙니다. `best_attempt.pth`는 이 런 안에서 가장 좋았던 epoch입니다. 배포 후보에서 `best.pth`가 끝내 없다면 Phase-2가 Phase-1보다 나아지지 않았다는 뜻이고, v3 EO x2처럼 Phase-1 가중치를 그대로 내는 것이 선례입니다. 파이프라인 점검이 목적이면 `last.pth`나 `best_attempt.pth`로 다음 단계를 이어 가도 됩니다.

출발점 점수(epoch 0)는 `training_log.csv`에 없고 콘솔 로그(`[Trainer] Baseline` 줄)와 TensorBoard에만 남습니다. 런을 비교할 때 이 줄을 따로 챙깁니다.

### 3. 평가

```bash
docker/run.sh --gpu <N> --name <exp>_eval --log <out>/x<s>_<dom>_eval.log -- python3 tools/evaluate.py \
    --config <cfg> --checkpoint <out>/x<s>_<dom>_p2/<best.pth|last.pth> --save_dir <out>/x<s>_<dom>_eval
```

- `--config`의 `model` 블록이 체크포인트에 저장된 것과 한 키라도 다르면 `ValueError: ... was trained with another model config`로 멈춥니다. 학습에 쓴 config나 런 폴더의 `train_config.yaml`을 줍니다.
- HR/LR 폴더는 기본으로 config의 `data.val_hr_root`, `data.val_lr_root`입니다. 다른 세트는 `--hr_dir`, `--lr_dir`로 줍니다.
- 출력: `<save_dir>/<체크포인트 파일 이름(확장자 제외)>/metrics.csv`(이미지마다 한 줄, 평균 줄 없음). 평균은 콘솔의 `=== Evaluation Summary ===` 아래에만 찍힙니다. 검증 때 출력은 `<out>/x2_eo_eval/last/metrics.csv`(100장)였고, 콘솔 평균 PSNR 31.0827, SSIM 0.8686, LPIPS 0.1772, NIQE 4.547이 학습 로그의 epoch 1 줄과 같았습니다(같은 지표 코드를 씁니다).
- Phase-1, EO Phase-2, IR Phase-2, Denoise는 서로 다른 val LR 세트로 검증합니다. 다른 config의 best 값끼리는 비교할 수 없으므로, 같은 `--hr_dir`/`--lr_dir`로 각각 평가하고 `--save_dir`을 런마다 다르게 줍니다(두 `best.pth`가 같은 `<save_dir>/best/`에 겹쳐 쓰이기 때문입니다).
- 도메인별 게이트도 있습니다. IR은 격자 채점(`examples/evaluation/score_grid_artifact.py`, 프로토콜 P2), Denoise는 차트 채점(`examples/evaluation/score_eo_chart.py`, P1.1)이고, 둘 다 `docker/run.sh -- env PYTHONPATH=. python3 ...`로 돌립니다. 사용법과 함정은 [examples/README.md](examples/README.md)에 있습니다.

### 4. W16A16 export

출하 레시피는 QAT 0 epoch(= 캘리브레이션만 하는 PTQ), `--param_bw 16 --output_bw 16`, 빈 MMP `configs/aimet/mmp_noprune_w16a16.json`입니다.

```bash
docker/run.sh --gpu <N> --name <exp>_export --log <out>/x<s>_<dom>_export.log -- \
    python3 examples/aimet_quantization/6_apply_qat.py \
    --config <cfg> --data_config <recipe> --checkpoint <out>/x<s>_<dom>_p2/<best.pth|last.pth> \
    --output_dir <out>/export/x<s>_<dom>/qat \
    --mmp_config configs/aimet/mmp_noprune_w16a16.json --param_bw 16 --output_bw 16 --qat_epochs 0 \
    --width <W> --height <H> --device 0
```

| 대상 | `--width --height` | export 런 폴더 |
|---|---|---|
| x2 (EO, IR 720) | 640 360 | `x2_eo`, `x2_ir` |
| x3 (EO, IR 720) | 428 240 | `x3_eo`, `x3_ir` |
| x4 (EO, IR 720) | 320 180 | `x4_eo`, `x4_ir` |
| IR 1024 프로필 x2 / x3 / x4 | 640 512 / 428 342 / 320 256 | `x<s>_ir_1024` |
| Denoise | 1280 720 | 아래 참고 |

- 기대 출력: 로그 끝의 `FINAL SUMMARY`에서 FP32 ≈ Pre-QAT = Post-QAT(검증 때 31.0827 / 31.0832 / 31.0832 dB). 폴더에는 `qat_model.onnx`, `qat_model.encodings`, `qat_model.pth`, `qat_model_torch.encodings`, `temp_onnx_model_with_all_markers.onnx`가 생깁니다. 다음 단계가 읽는 것은 앞의 두 파일입니다.
- export 런 폴더 이름은 반드시 `x<s>_<eo|ir>[_1024]`이고 그 안에 `qat/`가 있어야 합니다. 5단계 생성기는 이 이름만 찾습니다.
- `--data_config`는 필수이고 config 안의 `data_config_path`를 읽지 않습니다. 도메인에 맞는 레시피를 직접 줍니다. 틀린 레시피를 줘도 에러 없이 엉뚱한 열화로 캘리브레이션합니다.
- `--output_dir`가 이미 있으면 경고 없이 덮어씁니다. 출하 폴더(`results/release_v3/x*/qat` 등)를 가리키지 않습니다.
- GPU 없이 돌리려면 `--gpu`를 빼고 `--device cpu`를 줍니다. 출하 EO는 GPU로, IR과 Denoise는 CPU로 export했고, CPU와 GPU 결과는 W16 1단계만큼 달라 ONNX md5가 다릅니다. 새 DLC에는 어느 쪽이든 괜찮습니다.
- IR은 두 프로필을 한 번에 내는 스크립트가 있습니다: `docker/run.sh -- env SCALE=<s> results/release_v3/eval/export.sh <checkpoint> <out>/export [config]`(CPU). `<out>/export/x<s>_ir/qat`와 `x<s>_ir_1024/qat`를 씁니다. 이 스크립트는 export가 실패해도 0으로 끝나므로 출력의 `x<s>_ir exit 0`, `x<s>_ir_1024 exit 0` 두 줄을 확인합니다. config를 생략하면 출하 IR Phase-2 config를 쓰므로, `model` 블록이 다르면 런의 `train_config.yaml`을 셋째 인자로 줍니다.
- W16A16에서 QAT 학습(`--qat_epochs` > 0)은 쓰지 않습니다. 2026-09-10 측정에서 합성 PSNR은 올랐지만 실기 프레임에서는 더 흐리고 링잉이 늘었습니다. 근거와 AIMET 함정(Conv+LeakyRelu supergroup 등)은 [examples/README.md](examples/README.md)에 있습니다.
- Denoise(scale 1)는 export 런 폴더를 `x1_eo`로 두면 같은 생성기가 받습니다. 코드로만 확인했고 컨테이너에서 돌린 적은 없습니다. 출하 Denoise의 변환 절차는 [examples/deployment/README.md](examples/deployment/README.md)에 있습니다.

### 5. 변환 스크립트 생성

```bash
docker/run.sh --offline -- python3 examples/deployment/gen_convert_all.py <out>/export --version <ver>
```

- 기대 출력: `<out>/export/convert_all.sh: 1 runs`와 런마다 한 줄. 검증 때는 `x2_eo        in=t.1    out=37   -> BicubicPP_EO_640x360_x2_smoke.dlc`였습니다.
- DLC 이름은 `BicubicPP_<EO|IR>_<W>x<H>_x<s>_<ver>.dlc`입니다. `--version`의 기본값이 `v3`이므로 반드시 새 이름을 줍니다. 기본값으로 만들면 출하 DLC와 이름이 같아져, 보드에 올리는 순간 출하본을 덮어씁니다. 보드 스크립트는 영문자, 숫자, `_`만 받으므로 `v3.1`이 아니라 `v3_1`처럼 씁니다.
- `<out>/export/convert_all.sh`가 이미 있으면 거부하고 1로 끝납니다. 출하 변환 기록을 지우지 않으려는 장치입니다.
- 생성기는 각 런의 ONNX에서 입출력 텐서 이름과 크기를, encodings에서 비트폭을 읽습니다. HTP prepare 대상(`--htp_socs qcs8550`)과 출력층(`--output_layer final_act`)은 기본값을 씁니다.

### 6. DLC 변환

```bash
docker/run.sh --offline -- qairt-env bash <out>/export/convert_all.sh > <out>/export/convert.log 2>&1
grep -c 'Following OPs fallback to float\.$' <out>/export/convert.log    # = 런 수
grep -A1 'Following OPs fallback to float:' <out>/export/convert.log     # 아무것도 안 나와야 정상
grep '생성 .*개 / 건너뜀' <out>/export/convert.log
```

- 기대 출력: 런마다 `[ WARNING ] Following OPs fallback to float.`(마침표로 끝나고 뒤에 아무것도 없음 = float으로 떨어진 op 없음), `SNPE HTP Offline Prepare: Successfully created cache for QCS8550`, 그리고 끝에 `생성 1개 / 건너뜀 0개`. 검증 때 DLC는 `<out>/export/BicubicPP_EO_640x360_x2_smoke.dlc`(281,715 bytes)였습니다.
- op가 float으로 떨어지면 그 줄이 콜론(`float:`)으로 끝나고 다음 줄에 op 목록이 나옵니다. 변환은 성공으로 끝나므로 이 검사를 건너뛰면 느린 DLC를 모르고 올리게 됩니다.
- `WARNING_OP_VERSION_NOT_SUPPORTED`(LeakyRelu, Clip), `Restrict_quantization_steps ... Value will be ignored.`, `--float_fallback flag is deprecated` 같은 줄은 출하 DLC를 다시 만들 때도 나오는 무해한 줄입니다. 목록은 [examples/deployment/README.md](examples/deployment/README.md)에 있습니다.
- DLC md5는 변환할 때마다 다릅니다. QAIRT 2.41이 metadata에 초기화하지 않은 값과 임시 폴더 경로를 써 넣기 때문입니다. 같은 DLC인지는 md5가 아니라 `docker/accept.sh`의 `same_dlc` 방식(두 필드를 가리고 항목별 비교)으로 봅니다.
- 출하 릴리스 폴더(`results/release_v3`, `results/denoise_release_v1`) 안에서 `convert_all.sh`를 돌리지 않습니다. Denoise 폴더에서는 출하 DLC를 묻지 않고 덮어쓰고, v3 폴더에서는 HTP prepare가 없는 DLC 9개가 출하 이름으로 생깁니다. 출하본을 다시 변환하려면 복사본 폴더에서 합니다.

### 7. 보드

호스트에서 올리고 잽니다.

```bash
scp <out>/export/BicubicPP_<EO|IR>_<W>x<H>_x<s>_<ver>.dlc root@192.168.100.98:/data/dlc/
ssh root@192.168.100.98 'cd /data && ./bench_dlc.sh <ver>'
```

- `bench_dlc.sh`의 인자는 DLC 파일 이름의 부분 문자열입니다. 결과는 `<모델> <평균> ms <fps> fps (min~max)` 한 줄이고, 그 크기의 입력 폴더가 `/data/bench`에 없으면 `입력없음`을 찍습니다. 입력 만들기, 전력 설정(`PERF`), 보드 온도의 영향은 [examples/deployment/README.md](examples/deployment/README.md)에 있습니다.
- 화면 비교(소유자): 보드에서 `cd /data && BICUBICPP_VER=<ver> ./gst_compare_eo.sh <s>` 또는 `BICUBICPP_VER=<ver> ./gst_compare_ir.sh <s> <720|1024>`. 스크립트는 한 번에 Bicubic++ 버전 하나만 보여 주므로, v3와 비교하려면 두 번 차례로 돌립니다. 다른 gst나 `bench_dlc.sh`가 돌고 있으면 시작을 거부합니다.
- 출하 DLC와 같은 이름으로 올리지 않습니다. 실수로 덮었다면 [출하 모델](#출하-모델) 표의 사본으로 되돌립니다.

### 8. 기록

- 실험은 [docs/experiments/README.md](docs/experiments/README.md)의 방식으로 남깁니다(등록부 한 줄, `EXP-nn-k` 문서).
- 출하로 정하면 이름에 `release`가 든 폴더(`results/<...release...>/`)를 만들고 `MANIFEST.md`를 씁니다. 이 이름이어야 `MANIFEST.md`와 `*.sh`가 git에 잡힙니다. 무엇을 담는지는 두 출하 MANIFEST가 선례입니다.
- 출하 체크포인트와 export는 golden 테스트(`tests/test_release_golden.py`)에 항목을 더합니다. 테스트를 더하면 `docker/accept.sh`의 `BASE_TESTS=280`도 같은 커밋에서 올립니다. 올리지 않으면 인수 시험의 pytest 항목이 실패합니다.

### 걸린 시간

| 단계 | 2026-10-19 실측 (EO x2, 1 epoch, `docker events` 기준) |
|---|---|
| Phase-2 학습 1 epoch | 9분 8초. epoch 677 step 7분 56초 + 검증 100장 약 13초 x 2 + 시작 준비 |
| 평가 (100장, GPU) | 17초 |
| W16A16 export (GPU) | 43초 |
| 변환 스크립트 생성 + DLC 변환 (1런) | 수 초 |
| `docker/accept.sh` | 약 14분 |

전체 Phase-2는 최대 100 epoch이고 조기 종료(patience 50)로 더 일찍 끝날 수 있습니다. 위 속도로 곱하면 약 13시간입니다(추정). Denoise v1 학습은 300 epoch에 약 10시간이었습니다(MANIFEST 기록).

### 새 구조를 넣을 때

가중치가 아니라 모델 구조를 바꿀 때는 `src/` 바깥에 묶인 곳이 있습니다. 하나라도 빠지면 학습은 되는데 변환이나 보드 단계에서 막힙니다. 목록과 절차는 [src/README.md](src/README.md)에 있습니다.

- `src/models/__init__.py`의 `build_model`에 분기를 더하고, `forward`는 텐서 하나를 돌려줍니다(Trainer와 Validator가 튜플을 풀지 않습니다).
- 마지막 모듈 이름을 `final_act`로 둡니다. HTP prepare(`--set_output_layers final_act`)와 gst의 출력층 기본값이 이 이름입니다. 다르면 미리 컴파일한 캐시가 조용히 쓰이지 않습니다.
- DLC 이름 접두어 `BicubicPP_`가 `gen_convert_all.py`와 보드 스크립트에 박혀 있습니다. 접두어를 유지하고 새 버전 이름을 쓰거나, 두 곳을 함께 고칩니다.
- export 입력 크기는 고정이고 보드 크기(4단계 표)와 같아야 합니다.
- 새 Python 패키지는 그 의존성까지 전부 `==`로 `requirements.txt`에 고정하고(이미지는 `--no-deps`로 설치), `tests/test_docker_image.py`의 178을 고친 뒤 이미지를 다시 빌드합니다([docker/README.md](docker/README.md)).

---

## 매일 쓰는 명령

모두 호스트의 저장소 루트에서 실행합니다.

| 할 일 | 명령 | 확인할 것 |
|---|---|---|
| 실행 중인 작업 보기 | `docker ps --filter name=usdp- --format '{{.Names}}\t{{.Status}}\tgpu={{.Label "usdp.gpu"}}'` | 작업마다 이름, 가동 시간, GPU |
| 로그 따라보기 | `tail -f logs/<run>.log` | 진행 표시 |
| 끝났는지 보기 | `tail -n 1 logs/<run>.log` | `[run.sh] exit 0` |
| 작업 멈추기 | `docker stop usdp-<run>` | 로그 끝 `[run.sh] exit 143` |
| 학습 | `docker/run.sh --gpu <N> --name <run> -- python3 tools/train.py --config <cfg> --work_dir <dir>` | 런 폴더, 로그 끝 |
| 이어서 학습 | `docker/run.sh --gpu <N> --name <run> -- python3 tools/train.py --resume <dir>/last.pth` | `--config`, `--data_config`, `--work_dir`, `--lr`는 함께 줄 수 없음 |
| 학습 기록 보기 | `column -s, -t < <dir>/training_log.csv` | epoch별 손실과 지표 |
| 평가 | `docker/run.sh --gpu <N> -- python3 tools/evaluate.py --config <cfg> --checkpoint <ckpt> --save_dir <dir>` | `<dir>/<ckpt 이름>/metrics.csv` |
| 추론 | `docker/run.sh --gpu <N> -- python3 tools/inference.py --config <cfg> --checkpoint <ckpt> --input <파일 또는 폴더> --output <dir>` | `<dir>/<입력 파일 이름>` |
| FP32 ONNX 확인 | `docker/run.sh -- python3 tools/export.py --config <cfg> --checkpoint <ckpt> --height <H> --width <W> --output <dir>` | `<dir>/<ckpt 이름>_<H>x<W>.onnx`, 출하 파일 아님 |
| W16A16 export | [4단계](#4-w16a16-export) 명령 | `qat_model.onnx`, `qat_model.encodings` |
| 변환 스크립트 생성 | `docker/run.sh --offline -- python3 examples/deployment/gen_convert_all.py <root> --version <ver>` | `<root>/convert_all.sh: N runs` |
| DLC 변환 | `docker/run.sh --offline -- qairt-env bash <root>/convert_all.sh > <root>/convert.log 2>&1` | [6단계](#6-dlc-변환) 검사 |
| DLC 올리기 | `scp <dlc> root@192.168.100.98:/data/dlc/` | 출하본과 다른 이름 |
| 보드 지연 | `ssh root@192.168.100.98 'cd /data && ./bench_dlc.sh <필터>'` | ms, fps |
| 출하 Denoise 재-export | `docker/run.sh -- results/denoise_release_v1/export.sh <새 폴더>` | `<새 폴더>/qat/`, 폴더가 이미 있으면 거부 |
| 출하 IR 재-export | `docker/run.sh -- env SCALE=<s> results/release_v3/eval/export.sh <ckpt> <새 폴더> [config]` | `x<s>_ir exit 0`, `x<s>_ir_1024 exit 0` 두 줄 |
| 전체 pytest | `docker/run.sh --offline -- env PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q -rs -p no:cacheprovider` | 2026-10-19 기준 `3499 passed`, skip 0 |
| golden 테스트만 | 위 명령 끝에 `tests/test_release_golden.py tests/test_data_recipe_golden.py` | `56 passed`, skip 0 |
| 환경 인수 시험 | `docker/accept.sh --gpu <N>` | 15개 모두 `PASS` |
| 컨테이너 셸 | `docker/run.sh -- bash` (GPU: `--gpu <N>`, QAIRT 도구: `-- qairt-env bash`) | 저장소 루트의 셸 |

빈 GPU 찾기와 로그 거르기는 파이프가 들어가 따로 적습니다.

```bash
# GPUs with no compute process (the same test docker/run.sh applies)
APPS=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader); nvidia-smi --query-gpu=index,uuid --format=csv,noheader | while IFS=', ' read -r I U; do grep -qxF "$U" <<< "$APPS" || echo "GPU $I idle"; done

# Tagged log lines only; tqdm writes carriage returns, so plain grep sees one huge line
tr '\r' '\n' < logs/<run>.log | grep -a '^\[' | tail -n 20
```

- pytest는 `--gpu` 없이 돌립니다. `tests/test_docker_image.py`가 GPU가 붙지 않은 이미지 기본값(`NVIDIA_VISIBLE_DEVICES=void`)을 확인하므로, GPU를 붙이면 그 시험이 실패합니다. `docker/accept.sh`와 같이 포그라운드(`--name` 없이)로 돌립니다. skip이 하나라도 있으면 기준 파일(서버에만 있는 체크포인트나 채점 입력)이 빠진 것이므로 `-rs`가 찍는 이유를 읽습니다.
- 위 GPU 한 줄 명령은 아직 CUDA를 열지 않은 `usdp` 컨테이너는 보지 못합니다. `docker/run.sh`는 그런 컨테이너도 따로 확인하므로, 실제 거부 여부는 `run.sh`가 정합니다.

---

## 산출물 위치

`checkpoints/`, `logs/`, `tmp/`는 git-ignore입니다. `results/`도 전부 git-ignore이고, 이름에 `release`가 든 폴더의 맨 위 `*.md`, `*.sh`와 `eval/export.sh`만 추적합니다. `.pth`, `.onnx`, `.dlc`, `.nv12`, `.mp4`는 어디에 있든 추적하지 않습니다.

| 무엇 | 어디에 쓰나 |
|---|---|
| `tools/train.py` 런 폴더 | `--work_dir`. 없으면 `checkpoints/<train.save_name>/`. 같은 폴더로 다시 돌리면 에러 없이 `training_log.csv`에 이어 쓰고 config와 체크포인트를 덮으므로, 실험마다 새 `--work_dir`을 줍니다. |
| 백그라운드 작업의 콘솔 로그 | `logs/<NAME>.log` 또는 `--log FILE` (저장소나 `/mnt` 아래) |
| `tools/evaluate.py` | `<save_dir>/<체크포인트 파일 이름(확장자 제외)>/metrics.csv` (`--save_images`면 이미지도) |
| `tools/inference.py` | `--output` (기본 `results/inference/`) 아래 입력과 같은 파일 이름 |
| `tools/export.py` | `--output` (기본 `results/`) 아래 `<체크포인트 이름>_<H>x<W>.onnx` |
| `6_apply_qat.py` | `--output_dir` (기본 `results/qat`) |
| `gen_convert_all.py` | `<root>/convert_all.sh` |
| `convert_all.sh` | 스크립트와 같은 폴더의 `BicubicPP_<DOM>_<W>x<H>_x<s>_<ver>.dlc` (`<DOM>`은 `EO` 또는 `IR`) |
| `docker/accept.sh` | `results/docker_accept_<YYYYMMDD>/` (이미 있으면 거부, 다른 이름은 `--out`) |
| 출하 기록 | `results/release_v3/`, `results/denoise_release_v1/` (읽기만 합니다) |
| 출하 런 폴더 | `checkpoints/bicubicpp_deploy/`, 연화 넷 `checkpoints/csuav_deploy/` |
| 비교 영상, 프레임 덤프 | 저장소 밖 `/mnt/data_server/etc/jshong/SuperResolution/results/`. 서버 디스크가 차서 2026-10-13부터 `results/`에 영상을 만들지 않습니다. |
| 보드 벤치 입력(`make_bench_raw.py`) | 기본값 `bench/`는 git-ignore가 아니므로 `--out tmp/bench`를 줍니다. |

---

## 문제 해결 색인

| 증상 | 원인과 조치 | 자세히 |
|---|---|---|
| `run.sh: refused: /mnt/data_server is not a readable sshfs mount ...` | 호스트 재부팅 뒤 sshfs가 사라졌습니다. root가 `-o allow_other`로 다시 마운트합니다. | [docker/README.md](docker/README.md) |
| `run.sh: refused: GPU <N> is in use by ...` | 다른 프로세스나 `usdp` 컨테이너가 그 GPU를 씁니다. 빈 GPU를 고릅니다. | [docker/README.md](docker/README.md) |
| `run.sh: refused: run as your own user, not root` 또는 `... was built for uid:gid ...` | root로 실행했거나 다른 계정의 이미지입니다. 자기 계정으로 실행하고, 필요하면 `docker/build.sh`로 다시 빌드합니다. | [docker/README.md](docker/README.md) |
| `run.sh: refused: image usdp:latest not found ...` | 이미지가 없습니다. `docker/build.sh`. | [docker/README.md](docker/README.md) |
| `run.sh: refused: container name usdp-<NAME> is in use` | 같은 이름의 작업이 아직 돕니다. 다른 이름을 씁니다. | [docker/README.md](docker/README.md) |
| `run.sh: refused: --log ... is outside the mounted paths ...` | 로그 경로가 저장소와 `/mnt` 밖입니다(`/tmp` 등). | [docker/README.md](docker/README.md) |
| `entry.sh: refused: CUDA does not see exactly one GPU ...` (exit 3) | 컨테이너 안에서 GPU가 보이지 않습니다. `--gpu`와 NVIDIA 런타임을 확인합니다. | [docker/README.md](docker/README.md) |
| 로그 끝이 `[run.sh] exit 143` | `docker stop`으로 멈췄습니다. `last.pth`에서 `--resume`. | [tools/README.md](tools/README.md) |
| Phase-2 런 폴더에 `best.pth`가 없음 | 출발 가중치 점수를 넘지 못했습니다. 정상 동작입니다. [2단계](#2-phase-2-파인튜닝) 참고. | [tools/README.md](tools/README.md) |
| `ValueError: ... was trained with another model config; differing keys: ...` | `--config`의 `model` 블록이 체크포인트와 다릅니다. 런의 `train_config.yaml`을 줍니다. | [tools/README.md](tools/README.md) |
| `FileNotFoundError: pretrained_path 를 찾을 수 없습니다` 또는 연화 넷이 없음 | 서버에만 있는 `checkpoints/` 파일이 없습니다(다른 PC, 지운 폴더). | [configs/README.md](configs/README.md) |
| `--device 1` 이상이 컨테이너 안에서 실패 | 컨테이너 안에는 `cuda:0`만 있습니다. `--gpu <N>`으로 고르고 `--device`는 `0`이나 기본값. | [tools/README.md](tools/README.md) |
| 평가·추론이 이상하게 느림 | `--gpu` 없이 돌려 CPU로 넘어갔습니다(경고 없음). | [tools/README.md](tools/README.md) |
| 학습이 30~90배 느려지고 워커 CPU가 커널 시간으로 참 | NumPy hugepage madvise가 파편화된 호스트에서 커널 compaction 폭주를 일으킨 사례입니다(2026-09-10, 6분/epoch → 3시간/epoch). 이미지가 `NUMPY_MADVISE_HUGEPAGE=0`을 기본으로 두므로, 이 값을 덮어쓰지 않았는지 확인합니다. | [results/release_v3/MANIFEST.md](results/release_v3/MANIFEST.md) 인프라 절 |
| 학습 시작 로그의 `[Degradation] ... keys not read by the pipeline` | 데이터 레시피에 틀린 op 이름이 있습니다. 레시피에서 빠진 op는 꺼지는 것이 아니라 기본값(대부분 켜짐)으로 돕니다. | [configs/README.md](configs/README.md) |
| `ModuleNotFoundError: No module named 'src'` (채점기 등) | `examples/evaluation/*.py`와 `make_val_camera.py`는 `env PYTHONPATH=.`가 필요합니다. | [examples/README.md](examples/README.md) |
| 변환 로그에 `Following OPs fallback to float:`와 op 목록 | op가 float으로 떨어졌습니다. AIMET quantsim 설정(Conv+LeakyRelu supergroup)부터 봅니다. | [examples/README.md](examples/README.md), [examples/deployment/README.md](examples/deployment/README.md) |
| 변환 로그의 `WARNING_OP_VERSION_NOT_SUPPORTED`, `Restrict_quantization_steps ... ignored` | 출하 DLC를 다시 만들 때도 나오는 무해한 경고입니다. | [examples/deployment/README.md](examples/deployment/README.md) |
| 같은 입력으로 다시 변환했는데 DLC md5가 다름 | QAIRT 2.41의 정상 동작입니다. `same_dlc` 방식으로 비교합니다. | [examples/deployment/README.md](examples/deployment/README.md) |
| `... convert_all.sh already exists -- move it away first ...` | 그 폴더에 변환 스크립트가 이미 있습니다. 새 폴더를 쓰거나 옮깁니다. | [examples/deployment/README.md](examples/deployment/README.md) |
| 보드 스크립트의 `bad version name` / `model not found: ...` | 버전 이름에 영문자, 숫자, `_` 밖의 문자가 있거나, 그 이름의 DLC가 `/data/dlc`에 없습니다. | [examples/deployment/README.md](examples/deployment/README.md) |
| pytest에 `SKIPPED`가 나옴, 인수 시험의 pytest 항목 FAIL | 기준 파일이 빠졌거나(skip), 테스트 수가 `BASE_TESTS=280` + docker 테스트 수와 다릅니다. | [docker/README.md](docker/README.md), [docs/modules/20261015-MOD-ReleaseGolden.md](docs/modules/20261015-MOD-ReleaseGolden.md) |
| Phase-1과 Phase-2 런의 PSNR을 비교하니 이상함 | 둘은 다른 val LR 세트로 검증합니다. 같은 `--hr_dir`/`--lr_dir`로 각각 평가합니다. | [tools/README.md](tools/README.md) |

---

## 작업 규칙

- 새 작업은 브랜치를 만들어서 합니다. 회사 GitLab의 `master`는 보호 브랜치입니다.
- 커밋 제목만 쓰고 본문은 쓰지 않습니다: `[YYYYMMDD] <Component> <Verb> <what>`. Verb는 `Add`, `Update`, `Fix`, `Remove`, `Refactor`, `Init`, `Finish` 중 하나입니다. 예: `[20261019] DockerEnv Add the migration and acceptance results`.
- 문서와 그 문서가 설명하는 코드는 같은 커밋에서 바꿉니다.
- 새 모듈이나 기능은 모듈 문서 `docs/modules/<YYYYMMDD>-MOD-<Module>.md`(요구사항 → 설계 → 상세 설계 → 시험 항목 → 결과)부터 씁니다. 시험 함수 이름에 TC id를 넣어(`TC_REL_01_1` 등) 추적합니다.
- 모델이나 학습 레시피 실험은 평가 프로토콜과 잡음 수준을 먼저 고정하고, 실험마다 한 축만 바꿔 `EXP-nn-k`로 기록합니다([docs/experiments/README.md](docs/experiments/README.md)).
- 출하 config, 출하 데이터 레시피, 출하 릴리스 폴더는 고치지 않습니다. golden 테스트가 고정하고 있고, 출하 모델의 기록이기 때문입니다.

---

## 문서 지도

| 문서 | 담는 것 |
|---|---|
| [README.md](README.md) | 이 문서. 입구, 한 바퀴, 매일 쓰는 명령, 색인 |
| [docker/README.md](docker/README.md) | 작업 환경: `docker/run.sh` 옵션과 거부 메시지, GPU 고르기, 백그라운드 작업과 로그, 멈추고 이어 가기, VSCode 붙이기, `qairt-env`, 이미지 다시 빌드, `docker/accept.sh`, 다른 호스트로 옮기기 |
| [configs/README.md](configs/README.md) | config 구조와 키, Phase-1과 Phase-2, 데이터 레시피, AIMET JSON, 새 실험 config 만들기 |
| [docs/CONFIG_RATIONALE.md](docs/CONFIG_RATIONALE.md) | 설정값 하나하나의 근거, 값을 바꾸기 전에 볼 함정 |
| [src/README.md](src/README.md) | 코드 지도: 모델, 손실, 데이터셋과 열화 추가, Trainer 동작, 체크포인트, 지표 |
| [tools/README.md](tools/README.md) | `train`, `evaluate`, `inference`, `export` CLI: 모든 옵션, 출력, 재개 규칙, 함정 |
| [examples/README.md](examples/README.md) | AIMET 1~7단계(6단계가 출하 export), 평가 스크립트, 데이터셋 준비, AIMET 함정 |
| [examples/deployment/README.md](examples/deployment/README.md) | export 폴더 → DLC → 보드: 변환, DLC 이름, 로그 읽기, md5, 보드 스크립트, 출하 DLC 다시 올리기 |
| [docs/experiments/README.md](docs/experiments/README.md) | 실험과 릴리스를 기록하는 법, IR x4 격자 실험 등록부 |
| [docs/experiments/eo_denoise/README.md](docs/experiments/eo_denoise/README.md) | EO Denoise 실험 등록부 |
| [docs/20260928-RESEARCH-X4IRGrid.md](docs/20260928-RESEARCH-X4IRGrid.md), [docs/20261008-RESEARCH-EODenoiseV3.md](docs/20261008-RESEARCH-EODenoiseV3.md) | 끝난 연구 두 건: 프로토콜, 기준선, 가설, 결론 |
| [docs/modules/](docs/modules/) | 모듈 문서: GridScore, CameraModel, ChartScore, BicubicPPDenoise, ReleaseGolden, DockerEnv, DockerRun |
| [results/release_v3/MANIFEST.md](results/release_v3/MANIFEST.md), [results/denoise_release_v1/MANIFEST.md](results/denoise_release_v1/MANIFEST.md) | 출하 기록: 구성, md5, 실측, 약점, 다시 만들기 |
