# examples — export(양자화), 채점, 검증 세트

학습이 끝난 체크포인트를 장비(QCS8550 HTP)용 ONNX + encodings로 내보내고, 모델 출력을 채점하고, 검증(val) 세트를 만드는 스크립트 모음입니다. 학습·평가·추론 CLI(`tools/`)는 [tools/README.md](../tools/README.md), ONNX → DLC 변환과 보드 작업은 [deployment/README.md](deployment/README.md)에 있습니다.

| 폴더 | 내용 |
| --- | --- |
| `aimet_quantization/` | AIMET 1~7단계. 출하 export는 **6단계(`6_apply_qat.py`) 하나만** 씁니다. 나머지는 W8A8 분석·압축·QAT 도구입니다 |
| `deployment/` | `convert_all.sh` 생성기, 보드 비교·속도 측정 스크립트 ([deployment/README.md](deployment/README.md)) |
| `evaluation/` | 채점기: IR 격자 점수(P2), Denoise 차트 점수(P1.1), 옛 보드 텐서 비교 |
| `dataset_preparation/` | val 세트 생성기: SR용 `make_val_lr.py`, Denoise용 `make_val_camera.py` |

## 0. 공통 규칙

여기 있는 모든 명령은 저장소 루트에서, 작업 이미지 안에서 `docker/run.sh`로 돌립니다. 호스트의 `python3`(3.8, AIMET·onnx 없음)는 작업 환경이 아닙니다. `run.sh`의 옵션과 거부 메시지는 [docker/README.md](../docker/README.md)에 있습니다. 이 폴더의 스크립트에 해당하는 것만 추리면 다음과 같습니다.

- **작업 디렉터리**: 컨테이너는 저장소를 같은 경로로 마운트하고 저장소 루트에서 시작합니다. 명령의 상대 경로는 모두 저장소 루트 기준입니다. `/mnt`도 그대로 보이므로 config의 `/mnt/data_server/...`, `/mnt/data2/...` 경로가 그대로 읽힙니다. 그 밖의 호스트 경로(`/tmp` 등)는 보이지 않으므로 입력·출력(`<run_root>`, `<out>`, `<dir>` 등)은 저장소나 `/mnt` 아래에 두거나 `--mount`로 붙입니다.
- **GPU**: `docker/run.sh --gpu <N>`의 `<N>`은 호스트 GPU 번호(`nvidia-smi` 기준)입니다. 컨테이너 안에서는 그 GPU 하나만 `cuda:0`으로 보이므로 스크립트에는 항상 `--device 0`(또는 `cuda`)을 줍니다. `--device 2`처럼 호스트 번호를 주면 컨테이너에 없는 GPU를 가리켜 실패합니다.
- **`--gpu` 없이 돌리면 CPU입니다.** `run.sh`가 `run.sh: GPU: none (CPU run)`을 찍습니다. 이때 스크립트에 `--device cuda`나 숫자를 주면 경고 없이 CPU로 돕니다(6단계 기준). 의도한 CPU 실행이면 `--device cpu`를 적어 둡니다.
- **오래 걸리는 작업**(2단계 압축, 3·4단계 분석, QAT)은 `--name <run>`으로 백그라운드로 띄우고 `logs/<run>.log`(또는 `--log <파일>`)로 봅니다. 마지막 줄이 `[run.sh] exit 0`이면 정상 종료입니다. export(1~2분)는 그냥 앞에서 돌려도 됩니다.
- **`--offline`**: 컨테이너 네트워크를 끕니다(`--network none`). export와 채점은 네트워크가 필요 없어 인수 시험(`docker/accept.sh`)도 이 옵션으로 돌립니다.
- **데이터 마운트**: `/mnt/data_server`가 읽을 수 있는 sshfs 마운트가 아니면 `run.sh`가 모든 작업을 거부합니다. CPU export와 채점도 마찬가지입니다. 호스트 재부팅 뒤에는 root가 `-o allow_other`로 다시 마운트해야 합니다([docker/README.md](../docker/README.md)).
- **`PYTHONPATH`**: 이미지에는 `PYTHONPATH`가 설정돼 있지 않습니다. AIMET 1~7단계와 `make_val_lr.py`는 스스로 저장소를 `sys.path`에 넣고, `score_sr_tensors.py`는 저장소 코드를 쓰지 않습니다. 나머지 셋(`score_eo_chart.py`, `score_grid_artifact.py`, `make_val_camera.py`)은 `docker/run.sh -- env PYTHONPATH=. python3 ...`로 돌려야 합니다. 빼면 `ModuleNotFoundError: No module named 'src'`(또는 `'examples'`)로 멈춥니다. pytest는 `pytest.ini`가 `pythonpath = .`을 주므로 상관없습니다.
- **`NUMPY_MADVISE_HUGEPAGE=0`은 이미지에 이미 들어 있습니다.** 스크립트 머리말이나 옛 기록에 이 변수가 붙어 있는 것은 호스트에서 돌리던 시절의 흔적입니다. 이 값이 없으면 이 서버에서는 NumPy의 hugepage 요청 때문에 커널 메모리 압축이 폭주해 작업이 30~90배 느려집니다(2026-09-10, 워커 CPU 대부분이 커널 시간, GPU 0 %). 이미지 밖에서 돌릴 일이 생기면 꼭 붙입니다.
- **출력 폴더는 새로 만듭니다.** 6단계, 두 val 생성기, 7단계는 기존 파일을 묻지 않고 덮어씁니다. 출하 폴더(`results/release_v3/x*/qat`, `results/denoise_release_v1/dn_bpp_half4_w16a16/qat`)나 쓰고 있는 val 폴더를 출력으로 주면 기록이 사라집니다. `results/**`는 git에 없으므로(MANIFEST, `*.sh`만 추적) 되돌릴 방법도 없습니다.

## 1. 출하 export — W16A16 PTQ (`6_apply_qat.py`)

출하본(Bicubic++ SR v3 EO/IR x2·x3·x4, BicubicPPDenoise[v1])은 모두 **W16A16 PTQ**입니다. 즉 `--param_bw 16 --output_bw 16 --qat_epochs 0`이고, 층별 예외가 없는 빈 MMP(`configs/aimet/mmp_noprune_w16a16.json`)와 저장소 기본 quantsim config(`configs/aimet/htp_v73_no_leakyrelu_supergroup.json`)를 씁니다. 학습은 하지 않고, 보정(calibration)으로 양자화 범위를 정한 뒤 export만 합니다. 이름에 QAT가 들어 있지만 출하 경로에서는 QAT를 하지 않습니다(이유는 [3.2절](#32-w16a16에서는-qat를-하지-않는다)).

### 1.1 스크립트가 하는 일

1. `--checkpoint`를 읽고, 체크포인트에 저장된 `model` 블록이 `--config`의 `model` 블록과 다르면 `ValueError`로 멈춥니다(`check_model_config`).
2. `switch_to_deploy()`: ECB 갈래를 3x3 Conv 하나로 접고, IR 모델(`model.achromatic_output: true`)이면 출력 채널 평균을 마지막 층 가중치에 접습니다. 접기 뒤 회색 입력에 색이 1e-3 DN보다 많이 남으면 `RuntimeError`로 멈춥니다(`_check_achromatic_fold`).
3. `prepare_model` → eval → config의 `data.val_hr_root`/`data.val_lr_root`로 FP32 val PSNR을 잽니다.
4. QuantSim을 만듭니다(내부에서 BN fold + CLE). `--mmp_config`를 적용하고, `--encodings`가 있으면 읽고 없으면 보정합니다. 보정은 config의 `data.train_root`에 `--data_config` 열화를 입힌 학습 패치 `--calib_batches`개(기본 500, batch 1)를 무작위로 뽑아 씁니다. 그 뒤 Pre-QAT PSNR을 잽니다.
5. `--qat_epochs` > 0일 때만 Trainer로 QuantSim 모델을 학습합니다.
6. `<output_dir>/qat_model.*`로 export합니다.

모델 준비 순서(switch_to_deploy → prepare_model → eval → BN fold → CLE → QuantSim)는 [src/aimet/quant_sim.py](../src/aimet/quant_sim.py) 한 곳에만 정의돼 있고 3~7단계가 모두 그것을 부릅니다. 순서를 바꾸면 `.encodings`와 `.onnx`가 오류 없이 서로 어긋나므로 고치지 않습니다.

### 1.2 입력 크기와 폴더 이름

`--width`/`--height`는 export 그래프의 고정 입력 크기이며 보드가 모델에 넣는 크기와 같아야 합니다. 출력 폴더 이름은 변환 스크립트 생성기(`gen_convert_all.py`)가 찾는 규칙 `x<scale>_<eo|ir>[_1024]/qat/`을 따릅니다.

| 모델 | 프로파일 | `--width` x `--height` | export 폴더 | 생성기가 붙이는 DLC 이름 |
| --- | --- | --- | --- | --- |
| SR EO x2 | 720 | 640 x 360 | `x2_eo/qat` | `BicubicPP_EO_640x360_x2_<ver>.dlc` |
| SR EO x3 | 720 | 428 x 240 | `x3_eo/qat` | `BicubicPP_EO_428x240_x3_<ver>.dlc` |
| SR EO x4 | 720 | 320 x 180 | `x4_eo/qat` | `BicubicPP_EO_320x180_x4_<ver>.dlc` |
| SR IR x2 | 720 / 1024 | 640 x 360 / 640 x 512 | `x2_ir/qat`, `x2_ir_1024/qat` | `BicubicPP_IR_640x360_x2_<ver>.dlc` 등 |
| SR IR x3 | 720 / 1024 | 428 x 240 / 428 x 342 | `x3_ir/qat`, `x3_ir_1024/qat` | `BicubicPP_IR_428x240_x3_<ver>.dlc` 등 |
| SR IR x4 | 720 / 1024 | 320 x 180 / 320 x 256 | `x4_ir/qat`, `x4_ir_1024/qat` | `BicubicPP_IR_320x180_x4_<ver>.dlc` 등 |
| Denoise | — | 1280 x 720 | 출하 기록은 `dn_bpp_half4_w16a16/qat` | 손으로 쓴 `convert_all.sh` ([deployment/README.md](deployment/README.md)) |

### 1.3 SR EO x2 / x3 / x4

EO 출하 ONNX는 GPU에서 export했습니다. 출하 파일과 같은 바이트를 원하면 GPU(`--gpu <N> ... --device 0`)로 돌립니다(이유는 [1.8절](#18-gpu-export와-cpu-export)). `<run_root>`는 새 폴더입니다.

```bash
docker/run.sh --gpu <N> -- python3 examples/aimet_quantization/6_apply_qat.py \
    --config configs/finetune/BicubicPlusPlus/bicubicplusplus_2x_eo.yaml \
    --data_config configs/data/bicubicpp/sr_finetune_eo_phase-2.yaml \
    --checkpoint results/release_v3/checkpoints/EO_x2.pth --output_dir <run_root>/x2_eo/qat \
    --param_bw 16 --output_bw 16 --mmp_config configs/aimet/mmp_noprune_w16a16.json \
    --qat_epochs 0 --width 640 --height 360 --device 0

docker/run.sh --gpu <N> -- python3 examples/aimet_quantization/6_apply_qat.py \
    --config configs/finetune/BicubicPlusPlus/bicubicplusplus_3x_eo.yaml \
    --data_config configs/data/bicubicpp/sr_finetune_eo_phase-2.yaml \
    --checkpoint results/release_v3/checkpoints/EO_x3.pth --output_dir <run_root>/x3_eo/qat \
    --param_bw 16 --output_bw 16 --mmp_config configs/aimet/mmp_noprune_w16a16.json \
    --qat_epochs 0 --width 428 --height 240 --device 0

docker/run.sh --gpu <N> -- python3 examples/aimet_quantization/6_apply_qat.py \
    --config configs/finetune/BicubicPlusPlus/bicubicplusplus_4x_eo.yaml \
    --data_config configs/data/bicubicpp/sr_finetune_eo_phase-2.yaml \
    --checkpoint results/release_v3/checkpoints/EO_x4.pth --output_dir <run_root>/x4_eo/qat \
    --param_bw 16 --output_bw 16 --mmp_config configs/aimet/mmp_noprune_w16a16.json \
    --qat_epochs 0 --width 320 --height 180 --device 0
```

- CPU로 돌리려면 `--gpu <N>`을 빼고 `--device cpu`로 바꿉니다. 결과는 새 DLC로 쓰기에 문제없지만 ONNX md5가 출하 파일과 다릅니다.
- `EO_x2.pth`는 Phase-1 가중치입니다. EO x2 Phase-2가 Phase-1 기준을 넘지 못해 Phase-1 결과를 Phase-2 config로 export했습니다([results/release_v3/MANIFEST.md](../results/release_v3/MANIFEST.md)). EO는 Phase-1과 Phase-2의 `model` 블록이 같아서 이것이 가능합니다.
- 보정 데이터는 sshfs(`/mnt/data_server/.../Pretrained_Dataset/train/`)에서 읽습니다. EO x2 GPU export는 2026-10-19에 43초(컨테이너 시작부터 끝까지) 걸렸습니다.

### 1.4 SR IR — 한 스케일의 두 프로파일을 한 번에

[results/release_v3/eval/export.sh](../results/release_v3/eval/export.sh)가 출하 레시피(CPU, 스레드 4, IR Phase-2 data config 고정)로 720·1024 두 프로파일을 차례로 export합니다.

```bash
docker/run.sh --offline -- env SCALE=4 bash results/release_v3/eval/export.sh results/release_v3/checkpoints/IR_x4.pth <run_root>
```

- 사용법: `[SCALE=2|3|4] export.sh <checkpoint> <run_root> [config]`. `SCALE`의 기본값은 4이고, `config`를 생략하면 `configs/finetune/BicubicPlusPlus/bicubicplusplus_<s>x_ir.yaml`입니다.
- 출력: `<run_root>/x<s>_ir/qat/`, `<run_root>/x<s>_ir_1024/qat/`, 로그 `<run_root>/x<s>_ir.log`, `<run_root>/x<s>_ir_1024.log`. 화면에는 프로파일마다 `x4_ir exit <rc>` 한 줄이 찍힙니다.
- `<run_root>/x<s>_ir`(또는 `_1024`)가 이미 있으면 `... exists -- move it away first`로 멈춥니다. 덮어쓰기를 막는 장치입니다.
- **스크립트 자체는 export가 실패해도 0으로 끝납니다**(마지막 명령이 `echo`). `x<s>_ir exit 0`, `x<s>_ir_1024 exit 0` 두 줄과 각 로그 끝의 `FINAL SUMMARY`를 확인합니다.
- 프로파일 하나만 직접 돌리는 형태는 다음과 같습니다(export.sh가 실행하는 것과 같은 플래그).

```bash
docker/run.sh --offline -- python3 examples/aimet_quantization/6_apply_qat.py \
    --config configs/finetune/BicubicPlusPlus/bicubicplusplus_4x_ir.yaml \
    --data_config configs/data/bicubicpp/sr_finetune_ir_phase-2.yaml \
    --checkpoint results/release_v3/checkpoints/IR_x4.pth --output_dir <run_root>/x4_ir/qat \
    --param_bw 16 --output_bw 16 --mmp_config configs/aimet/mmp_noprune_w16a16.json \
    --qat_epochs 0 --width 320 --height 180 --device cpu
```

IR 로그에는 `[Info] Achromatic output folded into the final layer (residual chroma ... DN); the averaging op stays out of the graph.`가 찍혀야 합니다. 채널 평균이 그래프에 남으면 ONNX에 노드 6개가 붙고 HTP에서 float으로 돌아 x3 기준 2.746 ms가 10.401 ms가 됩니다(W8A8 측정). 접기가 실패하면 스크립트가 멈추므로 조용히 깨지지는 않습니다.

### 1.5 Denoise (BicubicPPDenoise[v1])

[results/denoise_release_v1/export.sh](../results/denoise_release_v1/export.sh)가 릴리스 폴더의 학습 당시 config 사본(`checkpoint/train_config.yaml`, `checkpoint/data_config.yaml`)과 `checkpoint/best.pth`로 출하 레시피(CPU, 1280x720, `--calib_batches 500`)를 그대로 돌립니다. 저장소 config가 나중에 바뀌어도 영향을 받지 않도록 사본을 씁니다.

```bash
docker/run.sh --offline -- bash results/denoise_release_v1/export.sh <out_dir>
```

- 출력: `<out_dir>/qat/qat_model.*`, `<out_dir>/export.log`, 화면에 `export exit <rc>`. 이 스크립트는 6단계의 종료 코드를 그대로 돌려줍니다.
- `<out_dir>`을 생략하면 `results/denoise_release_v1/re_export`입니다. 이미 있는 폴더면 멈춥니다.
- 2026-10-19 컨테이너에서 약 2분 걸렸고, PSNR 35.0118 → 35.0125 dB, ONNX md5는 출하 파일과 같았습니다.
- 직접 형태:

```bash
docker/run.sh --offline -- python3 examples/aimet_quantization/6_apply_qat.py \
    --config results/denoise_release_v1/checkpoint/train_config.yaml \
    --data_config results/denoise_release_v1/checkpoint/data_config.yaml \
    --checkpoint results/denoise_release_v1/checkpoint/best.pth --output_dir <out_dir>/qat \
    --param_bw 16 --output_bw 16 --mmp_config configs/aimet/mmp_noprune_w16a16.json \
    --qat_epochs 0 --calib_batches 500 --width 1280 --height 720 --device cpu
```

### 1.6 새로 학습한 체크포인트를 export하기

2026-10-19에 이 서버에서 끝까지 돈 사슬(EO x2 Phase-2 1에폭 → 평가 → export → 변환)의 export 단계를 일반화한 것입니다. 학습과 평가 명령은 [tools/README.md](../tools/README.md), 이어지는 변환과 보드는 [deployment/README.md](deployment/README.md)에 있습니다.

```bash
docker/run.sh --gpu <N> --name <run>_export --log results/<run>/export.log -- \
    python3 examples/aimet_quantization/6_apply_qat.py \
    --config <학습에 쓴 config> \
    --data_config <그 config의 data_config_path가 가리키는 data config> \
    --checkpoint <work_dir>/<best.pth | best_attempt.pth | last.pth> \
    --output_dir results/<run>/export/x<s>_<eo|ir>/qat \
    --mmp_config configs/aimet/mmp_noprune_w16a16.json --param_bw 16 --output_bw 16 \
    --qat_epochs 0 --width <W> --height <H> --device 0
```

실제로 돈 형태(2026-10-19, `FP32 31.0827 dB`, `Pre-QAT 31.0832`, `Post-QAT 31.0832`, `[run.sh] exit 0`):

```bash
docker/run.sh --gpu 2 --name smoke_x2_eo_export --log results/smoke_20261019/x2_eo_export.log -- \
    python3 examples/aimet_quantization/6_apply_qat.py \
    --config configs/finetune/BicubicPlusPlus/bicubicplusplus_2x_eo.yaml \
    --data_config configs/data/bicubicpp/sr_finetune_eo_phase-2.yaml \
    --checkpoint results/smoke_20261019/x2_eo_p2/last.pth \
    --output_dir results/smoke_20261019/export/x2_eo/qat \
    --mmp_config configs/aimet/mmp_noprune_w16a16.json --param_bw 16 --output_bw 16 \
    --qat_epochs 0 --width 640 --height 360 --device 0
```

고를 것과 그 이유:

- **체크포인트**: Phase-2 학습은 시작할 때 Phase-1 가중치를 먼저 채점하고(`[Trainer] Baseline from pretrained weights: ...`), 그 기준을 넘을 때만 `best.pth`를 씁니다. 그래서 짧은 런이나 기준을 못 넘은 런에는 `best.pth`가 없습니다. 이때는 그 런의 최고 에폭인 `best_attempt.pth`나 마지막 에폭 `last.pth`를 씁니다. 모든 Trainer 체크포인트에는 `model_state_dict`(EMA를 쓰면 EMA 가중치)와 전체 config가 들어 있어 어느 것이든 6단계에 넣을 수 있습니다.
- **Phase-1 가중치를 Phase-2 config로 export하는 것은 EO에서만 됩니다.** IR Phase-2의 `model` 블록에는 `achromatic_output: true`가 더 있어서, 저장된 config가 있는 IR Phase-1 체크포인트를 IR Phase-2 config로 열면 `ValueError ... differing keys: achromatic_output`로 멈춥니다.
- **`--config`**: 학습에 쓴 저장소 config나 `<work_dir>/train_config.yaml`(덮어쓴 옵션까지 반영된 실효 config)을 줍니다. 학습 뒤 저장소 config가 바뀌었으면 `<work_dir>`의 사본이 안전합니다. Denoise 출하 export가 이 방식(릴리스 폴더의 사본)입니다.
- **`--data_config`**: 6단계는 `--config` 안의 `data_config_path`를 읽지 않고 이 인자만 씁니다. 둘이 맞는지도 검사하지 않으므로, IR 모델에 EO 레시피를 주면 다른 열화로 보정한 채 오류 없이 끝납니다. config의 `data_config_path`와 같은 파일, 또는 `<work_dir>/data_config.yaml`을 줍니다.
- **`--output_dir`**: 변환 스크립트 생성기는 `<export root>/x<s>_<eo|ir>[_1024]/qat/`만 찾습니다(`RUN_RE = x(\d)_(eo|ir)(_1024)?$`). 폴더 이름이 다르면 변환 대상에서 빠집니다.
- **Denoise 새 체크포인트**: 생성기 규칙으로는 `<root>/x1_eo/qat`가 잡히고 `BicubicPP_EO_1280x720_x1_<ver>.dlc`로 이름이 붙는다고 코드에서 읽히지만, 이 경로는 실행해 보지 않았습니다(확인 필요). 보드 스크립트가 기대하는 Denoise 이름과 다르므로 [deployment/README.md](deployment/README.md)를 함께 봅니다.

다음 단계(같은 export root에서):

```bash
docker/run.sh --offline -- python3 examples/deployment/gen_convert_all.py results/<run>/export --version <ver>
docker/run.sh --offline -- qairt-env bash results/<run>/export/convert_all.sh > results/<run>/export/convert.log 2>&1
```

`<ver>`는 영문자·숫자·`_`만 씁니다(보드 스크립트가 다른 문자를 거부합니다). 변환 로그 읽는 법과 보드 업로드는 [deployment/README.md](deployment/README.md)에 있습니다.

### 1.7 결과 확인

출력 폴더에 생기는 파일:

| 파일 | 내용 | 쓰는 곳 |
| --- | --- | --- |
| `qat_model.onnx` | 접힌 배포 그래프 | 변환기 입력 |
| `qat_model.encodings` | 양자화 인코딩, **ONNX 텐서 이름** 키 | 변환기 `--quantization_overrides` |
| `qat_model_torch.encodings` | 같은 인코딩, **PyTorch 모듈 이름** 키 | 6단계 `--encodings`, 7단계 |
| `qat_model.pth` | quantizer를 뺀 일반 PyTorch 모델(모듈 통째, AIMET export) | 직접 쓰는 곳 없음 |
| `temp_onnx_model_with_all_markers.onnx` | AIMET 중간 산출물 | 직접 쓰는 곳 없음 |
| `qat_checkpoints/` | `--qat_epochs` > 0일 때만: `best.pth`, `logs/`(TensorBoard) | 7단계 |

로그 끝의 요약에서 세 PSNR을 봅니다. W16A16에서는 Pre-QAT가 FP32와 1e-3 dB 안에서 같아야 합니다(2026-10-19 컨테이너: EO x2 −0.0002, IR x4 +0.0001, Denoise +0.0007 dB). 차이가 크면 비트폭이나 quantsim config가 출하 레시피와 다른 것입니다.

```
 1. FP32 Model PSNR       : 31.0827 dB
 2. Pre-QAT PSNR          : 31.0832 dB
 3. Post-QAT PSNR         : 31.0832 dB
```

로그에 찍히지만 문제가 아닌 줄:

- `DeprecationWarning: QuantizationSimModel.export is deprecated ...`, `Quantsim export will stop exporting encodings ... in a future AIMET release`: aimet-torch 2.29.0(`requirements.txt`에 고정)의 경고입니다. AIMET을 올리면 export 단계가 바뀌거나 깨질 수 있다는 뜻이므로 버전은 그대로 둡니다.
- `High Bias folding is not supported for models without BatchNorm Layers`: BN이 없는 모델이라 CLE의 한 단계를 건너뛴다는 뜻입니다.
- `[SRDataset] HR 크롭 576px 보다 작은 이미지는 건너뜁니다 ...`: 보정 패치보다 작은 학습 이미지를 다시 뽑는다는 안내입니다.
- `>> Training Data: Using N samples out of M (30.0%)`: QAT용 부분집합 크기입니다. `--qat_epochs 0`이면 쓰이지 않습니다.

### 1.8 GPU export와 CPU export

| 프로파일 | 출하 ONNX md5 | 출하 파일을 만든 장치 | 같은 바이트 재현 | CPU export md5 |
| --- | --- | --- | --- | --- |
| x2_eo | `f4a757f3…` | GPU | `--gpu <N> ... --device 0`, 컨테이너 확인(2026-10-19) | `7e2fe59f…` |
| x3_eo | `c93864cd…` | GPU | GPU로 같을 것으로 보지만 컨테이너에서 돌려 보지 않음 | `411a63ef…` |
| x4_eo | `3fce4ecf…` | GPU | 위와 같음 | `eb299ebb…` |
| IR 6종 | 예: x4_ir `5de6458e…`, x4_ir_1024 `37d8f6ff…` | CPU | `--device cpu`, x4 두 프로파일 컨테이너 확인 | 출하와 같음 |
| Denoise | `81ac40c7…` | CPU | `--device cpu`, 컨테이너 확인 | 출하와 같음 |

- EO는 CPU에서 export하면 W16 scale이 1 ulp 달라져 md5가 바뀝니다. 새 DLC를 만들 때는 GPU든 CPU든 상관없습니다. 출하 파일과 바이트 단위로 맞춰 볼 때만 위 규칙을 따릅니다.
- 테스트 `TC_REL_02_1`은 EO를 CPU md5로, IR·Denoise를 출하 파일로 고정합니다. 2026-10-19 컨테이너 pytest에서 10개 프로파일이 모두 통과했습니다.
- 보정 데이터는 시드 없이 무작위로 뽑으므로 activation 인코딩은 실행할 때마다 조금씩 바뀝니다(Denoise는 scale이 최대 5 % 정도). `qat_model.encodings` 파일의 md5는 비교 기준이 되지 못합니다. 재현성은 ONNX md5와 `param_encodings`가 같은지로 판단합니다.

### 1.9 `6_apply_qat.py` 옵션 전체

| 옵션 | 기본값 | 설명 |
| --- | --- | --- |
| `--config` | (필수) | 학습 config. `model` 블록이 체크포인트에 저장된 것과 다르면 `ValueError`. `data.train_root`(보정), `data.val_*_root`(PSNR), `train.patch_size`·`batch_size`·`num_workers`를 읽습니다 |
| `--data_config` | (필수) | 보정 패치에 입힐 열화 레시피. 파일이 없으면 `FileNotFoundError`. 빈 설정은 데이터셋이 거부합니다(빈 설정은 '열화 없음'이 아니라 '모든 열화가 기본값'이라서) |
| `--checkpoint` | (필수) | FP32 체크포인트. Trainer 체크포인트(`model_state_dict` + `config`)나 2단계의 `compressed_model.pth`(모듈 통째) |
| `--output_dir` | `results/qat` | 출력 폴더. 있으면 그대로 덮어씁니다 |
| `--scheme` | `training_range_learning_with_tf_init` | `post_training_tf`, `post_training_tf_enhanced`, `training_range_learning_with_tf_init` 중 하나. 출하 PTQ도 기본값을 씁니다 |
| `--param_bw` | 8 | 가중치 비트폭. 출하 16 |
| `--output_bw` | 8 | 활성화 비트폭. 출하 16 |
| `--quantsim_config` | `configs/aimet/htp_v73_no_leakyrelu_supergroup.json` | AIMET 내장 이름(예: `htp_v73`)이나 JSON 경로. 바꾸지 않습니다([3.1절](#31-convleakyrelu-supergroup-함정)) |
| `--qat_epochs` | 0 | 0 = 보정과 export만(W16A16 출하). 0보다 크면 W8A8 QAT |
| `--qat_lr` | 5e-5 | QAT 학습률 |
| `--train_data_ratio` | 0.3 | QAT 학습에 쓸 학습 데이터 비율(시드 없는 무작위 부분집합) |
| `--encodings` | 없음 | 미리 구한 `*_torch.encodings`. **파일이 있을 때만** 읽고, 경로가 틀리면 아무 말 없이 보정합니다 |
| `--mmp_config` | 없음 | 혼합 정밀도 JSON. 없으면 모든 층이 기본 비트폭. 출하 명령은 빈 `mmp_noprune_w16a16.json`을 넘깁니다(2026-10-16부터는 빼도 결과가 같고, 기록과 같은 명령을 유지하려고 넘깁니다). 주어진 파일이 없으면 `FileNotFoundError` |
| `--width` / `--height` | 640 / 360 | export 그래프의 입력 크기([1.2절](#12-입력-크기와-폴더-이름)) |
| `--calib_batches` | 500 | 보정 패치 수(`--encodings`를 읽으면 안 씀) |
| `--device` | `cuda` | `cpu`, 숫자(`0` → `cuda:0`), 그 밖의 값. CUDA가 없으면 CPU로 떨어집니다 |

## 2. 변환으로 넘기는 것

6단계 산출물에서 DLC를 만드는 일은 `gen_convert_all.py`가 만든 `convert_all.sh`를 작업 컨테이너의 `qairt-env`로 돌리는 것입니다. 명령, 레이아웃 플래그, 텐서 이름(ONNX에서 읽음), W16 옵션, HTP prepare(`qcs8550`, 출력 레이어 `final_act`)가 스크립트에 들어 있으므로 손으로 적지 않습니다. 절차, 폴백 목록 읽는 법, DLC md5가 매번 다른 이유, 출하 DLC를 보드에 다시 올릴 때의 규칙은 [deployment/README.md](deployment/README.md)에 있습니다. 이 폴더와 맞물리는 점만 적습니다.

- 변환기의 `--quantization_overrides`에는 **`qat_model.encodings`**(ONNX 텐서 이름 키)를 줍니다. `qat_model_torch.encodings`는 PyTorch 모듈 이름 키라서 넣으면 텐서 이름이 하나도 맞지 않습니다. `convert_all.sh`는 `qat_model.encodings`를 씁니다.
- 5단계의 `mmp_model.*`는 생성기가 찾지 않습니다(`*/qat/qat_model.*`만 찾음). 6단계를 거쳐야 변환할 수 있습니다.
- 마지막 `DepthToSpace`(PixelShuffle) 출력에는 인코딩이 붙지 않습니다. 데이터 이동 연산이라 정상이며 폴백 목록에도 나오지 않습니다.
- `--restrict_quantization_steps "-0x8000 0x7F7F"`: 생성기는 encodings의 `quantizer_args.param_bitwidth`가 16이면 이 옵션을 붙입니다. 그러나 QAIRT 2.41에서는 `--float_fallback`과 함께 쓰면 로그에 `Quantization is disabled as --enable_float_fallback flag is provided`와 `Restrict_quantization_steps is only supported for --param_quantizer = symmetric ... Value will be ignored.`가 찍힙니다(2026-10-19 서버 변환 로그). 즉 지금 레시피에서는 이 옵션이 무시되고 AIMET encodings가 그대로 쓰입니다. 예전 README의 "빠지면 16비트 가중치가 조용히 잘린다"는 QAIRT 문서에 근거한 설명이며 2.41 레시피에서 다시 재 보지 않았습니다(확인 필요). 레시피를 바꿀지는 소유자 판단입니다.

## 3. AIMET 함정

### 3.1 Conv+LeakyRelu supergroup 함정

AIMET 내장 `htp_v73` config에는 Conv+LeakyRelu supergroup이 3개 있습니다. AIMET은 HTP가 이 둘을 융합한다고 보고 **Conv의 출력 quantizer를 끕니다.** 그래서 `qat_model.encodings`에 Conv 중간 출력 인코딩이 빠지는데, QAIRT 2.41 변환기는 이 융합을 하지 않아 해당 텐서가 인코딩 없이 남습니다. Conv가 float이 되면 그 출력을 받는 활성화도 입력 인코딩이 없어 연쇄적으로 float이 됩니다.

증상: 변환은 `CONVERSION_SUCCESS`로 끝나지만 그 바로 앞의 폴백 목록(`Following OPs fallback to float:` 다음 줄. 로그 끝이 아니라 런마다 변환기 출력 중간)에 op 이름이 나옵니다. Bicubic++에서는 연산 8개 중 6개가 float이었습니다(`module_add`, `conv_out`만 입출력 인코딩이 있어 남음).

```
[ WARNING ] Following OPs fallback to float:
conv0, module_act_2, conv2, conv1, module_act_1, act.
```

정상일 때는 같은 줄이 콜론이 아니라 마침표로 끝나고 뒤에 아무 op도 없습니다.

```
[ WARNING ] Following OPs fallback to float.
```

해결: 저장소 config [configs/aimet/htp_v73_no_leakyrelu_supergroup.json](../configs/aimet/htp_v73_no_leakyrelu_supergroup.json)을 씁니다. 내장 `htp_v73`에서 LeakyRelu supergroup 3개만 뺀 것(14개 → 11개, Conv+Clip은 유지)이고, 3~7단계 `--quantsim_config`의 기본값입니다(`src/aimet/quant_sim.py`의 `DEFAULT_QUANTSIM_CONFIG`). `--quantsim_config htp_v73`처럼 내장 이름을 주면 함정이 되살아납니다. `htp_v73_no_leakyrelu_supergroup_perchannel.json`은 `per_channel_quantization`만 `True`로 바꾼 변형입니다.

알아 둘 것:

- 적용하면 activation 인코딩이 6개에서 9개로 늘고 폴백이 사라집니다. 대신 W8A8에서는 늘어난 activation quantizer 3개 때문에 PSNR이 0.07~0.23 dB 떨어졌습니다. W16A16에서는 QuantSim과 FP32의 차이가 1e-3 dB 안이라 비용이 없습니다.
- 저장소 config에도 Conv+Relu supergroup은 남아 있습니다. 2026-09-30 ReLU판 IR x4 시험에서는 Conv 출력 인코딩이 빠졌지만 변환기가 Conv+ReLU를 처리해 폴백 목록이 비었습니다(소유자 회사 PC 변환). 새 활성화 함수를 쓰는 모델은 첫 변환 때 폴백 줄을 반드시 확인합니다.
- `svfocussrnet`은 이 문제가 없습니다. 활성화가 `nn.Hardtanh(0, 1)`이고 AIMET 내부 op 타입이 `HardTanh`라 `["Conv", "Clip"]`에 매칭되지 않습니다(ONNX로는 `Clip`으로 나가지만 매칭은 AIMET 내부 이름 기준). aimet-torch 2.29.0으로 다시 만들어도 Conv 출력 quantizer가 살아 있었습니다.
- 변환기의 `--target_backend`/`--target_soc_model`로 융합을 맡기는 길은 QAIRT 2.41.0 설치본에서 `is_soc_model_supported()`가 모든 SoC 문자열에 `False`를 돌려줘 시험하지 못했습니다.

### 3.2 W16A16에서는 QAT를 하지 않는다

W16A16 QuantSim은 FP32와 1e-3 dB 안에서 같아 QAT가 메울 양자화 손실이 없습니다. 학습을 돌리면 모델만 움직입니다. 2026-09-10에 기본값이던 30에폭 QAT는 Phase-2 종점(LR 1e-7)에서 `--qat_lr 5e-5`로 다시 뜨겁게 시작했고, 손실 warm-up도 처음부터 다시 돌아 첫 에폭이 순수 MSE였습니다. 그 결과 합성 val 점수는 올랐지만 실기 IR 프레임에서는 더 흐리고 링잉이 늘었습니다(x3 선명도 앵커 대비 0.798 → 0.737, 링잉 1.86 % → 3.47 %). 그래서 기본값이 0이 됐고 출하본은 모두 PTQ입니다.

- `--qat_epochs 0`이면 Trainer를 만들지 않습니다. 그래서 EO config의 소프트닝 넷(`checkpoints/csuav_deploy/train_svfocusdenoise_..._260309/best.pth`)도 읽지 않고, 재사용한 폴더에 남은 `qat_checkpoints/best.pth`를 복원하는 일도 없습니다.
- 0보다 크게 주면 Trainer가 그 소프트닝 넷을 읽습니다(EO). 파일이 없으면 멈춥니다.
- `Post-QAT > FP32`가 나오면 양자화 이득이 아니라 추가 학습의 효과입니다. 합성 val 점수만 보고 판단하지 않습니다.

### 3.3 그 밖의 함정

| 함정 | 증상 | 대처 |
| --- | --- | --- |
| config 대 체크포인트 검사는 6단계에만 있음 | 1~5·7단계에서 IR 체크포인트를 EO config로 열면 무채색 접기 없는 그래프가 오류 없이 만들어짐 | 체크포인트를 학습한 config를 그대로 씁니다. `achromatic_output`은 가중치가 없어 strict 로드로도 걸리지 않습니다 |
| `--data_config`와 config의 `data_config_path`를 맞춰 보지 않음 | 다른 도메인 레시피로 보정해도 정상 종료 | EO는 `sr_finetune_eo_phase-2.yaml`, IR은 `sr_finetune_ir_phase-2.yaml`, Denoise는 학습 런의 `data_config.yaml` |
| `--encodings` 경로가 틀림 | 경고 없이 새로 보정 | 로그의 `>> Loading encodings from ...` 줄을 확인 |
| `--mmp_config`의 층 이름이 틀림 | `[Warning] Could not find module '<name>' in model.`만 찍고 그 층은 기본 비트폭 | 1단계의 모듈 표나 4단계의 `quant_layer_config.txt`에서 이름을 복사 |
| 7단계는 encodings를 `strict=False`로 읽음 | 그래프가 달라도 일부만 읽힌 채 진행 | 7단계에 6단계와 같은 `--quantsim_config`·비트폭·`--mmp_config`를 줍니다 |
| 출력 폴더 덮어쓰기 | 출하 export가 사라짐 | 항상 새 `--output_dir`. export.sh 두 개는 기존 폴더를 거부하지만 직접 명령은 거부하지 않습니다 |
| 컨테이너 안 GPU 번호 | `--device <호스트 번호>`가 없는 GPU를 가리킴 | `--gpu <N>` + `--device 0` |
| 컨테이너 밖 비-0 GPU export | 예전에 `sim.export`가 장치 불일치로 실패 | 스크립트가 `torch.cuda.set_device()`로 막아 둠. 지울 코드가 아닙니다 |

## 4. W8A8 도구 (1~5단계, 7단계)

1~5단계와 7단계는 W8A8(그리고 W8A4 혼합 정밀도) 모델을 분석·압축·QAT하는 도구로, 일부러 남겨 두었습니다. 출하 W16A16은 쓰지 않습니다.

**실행 상태**: 2026-10-16에 호스트(이미지 이전 환경)에서 CPU·작은 설정으로 다음이 끝까지 돌았습니다. SVFocusSRNet EO x2 체크포인트(`checkpoints/csuav_deploy/finetune_svfocussrnet_eo_sr_x2_*`)로 1→2→3→4→5→6(5단계 encodings, QAT 1에폭)→7, 2단계 압축 모델로 6, Bicubic++ IR x3 출하 체크포인트로 6(QAT 1에폭)→7. **작업 이미지 `usdp:20261019`에서는 6단계만 돌려 봤습니다**(인수 시험과 위 사슬). 1~5·7단계는 컨테이너에서 확인 안 됨이며, pytest는 5·6단계의 인자 검사만 봅니다. 아래 명령은 인자와 코드를 대조해 적은 것입니다.

W8A8 흐름: 1(기준) → 2(압축, 선택) → 3·4(민감도·층별 분석) → 5(혼합 정밀도 인코딩) → 6(`--encodings <5단계 출력>/mmp_model_torch.encodings --qat_epochs <N>`) → 7(QuantSim 추론). 6단계에는 5단계와 같은 `--param_bw`·`--output_bw`·`--mmp_config`·`--quantsim_config`를 줍니다(5단계 기본은 W8A4, 6단계 기본은 W8A8이라 그냥 두면 어긋납니다). 이 일치는 코드상 필요한 조건으로 보지만 어긋났을 때를 시험하지는 않았습니다. 7단계에는 6단계와 같은 값을 줍니다.

예시는 EO x4 출하 체크포인트(320x180)로 적었습니다. 다른 모델은 [1.2절](#12-입력-크기와-폴더-이름)의 크기를 씁니다.

### 4.1 단계 1 — FP32 기준 (`1_measure_fp32_model.py`)

학습된 FP32 모델의 val PSNR(휘도)을 `switch_to_deploy()` 뒤 그래프로 재고, 말단 모듈의 이름·타입 표를 찍습니다. 이 표는 2단계 `--ignore_layers`와 MMP JSON의 층 이름을 고를 때 씁니다. 파일은 쓰지 않습니다. config 대 체크포인트 검사가 없고, 2단계의 `compressed_model.pth`(모듈 통째)는 읽지 못합니다.

```bash
docker/run.sh --gpu <N> -- python3 examples/aimet_quantization/1_measure_fp32_model.py \
    --config configs/finetune/BicubicPlusPlus/bicubicplusplus_4x_eo.yaml \
    --checkpoint results/release_v3/checkpoints/EO_x4.pth --device 0
```

| 옵션 | 기본값 | 설명 |
| --- | --- | --- |
| `--config` | (필수) | val 경로(`data.val_hr_root`, `data.val_lr_root`)를 읽음. 없으면 PSNR을 건너뜀 |
| `--checkpoint` | (필수) | FP32 체크포인트 |
| `--device` | `cuda` | 숫자를 줬는데 CUDA가 없으면 경고 후 CPU |

### 4.2 단계 2 — 압축 (`2_compress_svd_pruning.py`)

Channel Pruning(또는 Spatial SVD)으로 채널과 MAC을 줄이고, 단계마다 복구 파인튜닝을 합니다. 결과 `<output_dir>/compressed_model.pth`는 모듈 통째로 저장돼 3~7단계(7단계는 `--base_model`)와 tools가 그대로 읽습니다.

```bash
docker/run.sh --gpu <N> --name compress_x4 -- python3 examples/aimet_quantization/2_compress_svd_pruning.py \
    --config configs/finetune/BicubicPlusPlus/bicubicplusplus_4x_eo.yaml \
    --data_config configs/data/bicubicpp/sr_finetune_eo_phase-2.yaml \
    --checkpoint results/release_v3/checkpoints/EO_x4.pth --output_dir <out2> --width 320 --height 180 \
    --use_pruning --target_ratio 0.5 --calib_batches 500 \
    --ft_epochs 30 --ft_lr 5e-5 --dataset_ratio 0.5 --device 0
```

- 출력: `<out2>/compressed_model.pth`, `<out2>/finetune_final/`(Pruning) 또는 `<out2>/finetune_svd/`(SVD) 안에 `train_config.yaml`, `logs/`, Trainer 체크포인트. AIMET greedy 탐색 캐시가 저장소 루트의 `./data/`에 남습니다(git-ignore 대상).
- Pruning은 HTP 레이아웃(32채널 정렬)에 맞춘 자체 탐색으로, 출력 채널이 32보다 큰 `Conv2d`만 32채널씩 줄입니다. dim 32 모델에서 대상이 되는 층은 x4 업샘플러뿐입니다(Bicubic++ x4 `conv_out` 64, SVFocusSRNet x4 `upsample.0`·`base_upsampler.net` 48). Bicubic++ x2·x3, Denoise, SVFocusSRNet x2는 줄일 층이 없어 비율 1.0으로 끝납니다(2026-10-16 확인). SVFocusSRNet x4의 `base_upsampler.net`은 학습하지 않는 고정 anchor라 `--ignore_layers base_upsampler`로 뺍니다.
- `--ft_epochs 0`이면 `compressed_model.pth`를 저장한 다음 마지막 요약 출력에서 `UnboundLocalError`(`psnr_cp_ft`/`psnr_svd_ft`, `NameError`의 하위 클래스)로 끝납니다. 압축 결과는 남지만 종료 코드는 실패입니다.
- `--ft_epochs` > 0이면 Trainer가 돌므로 EO config는 소프트닝 넷을 읽습니다.

| 옵션 | 기본값 | 설명 |
| --- | --- | --- |
| `--config`, `--data_config`, `--checkpoint` | (필수) | `--data_config` 파일이 없으면 멈춤 |
| `--output_dir` | `results/compression` | 출력 폴더 |
| `--width` / `--height` | 640 / 360 | 비용(MAC) 계산용 입력 크기 |
| `--use_pruning` | 꺼짐 | Channel Pruning |
| `--use_svd` | 꺼짐 | Spatial SVD. 추론이 느려져서 쓰지 않는다고 help에 적혀 있습니다 |
| `--ignore_layers` | 없음 | 제외할 층 이름의 부분 문자열(여러 개) |
| `--target_ratio` | 0.1 | **Pruning**: 대상 Conv(32채널 초과)의 채널 중 남길 비율. **SVD**: 코드가 `1 - target_ratio`를 AIMET `target_comp_ratio`로 넘깁니다. 같은 0.1이 Pruning에서는 10 % 남김, SVD에서는 90 % 남김이 됩니다 |
| `--num_comp_ratio_candidates` | 10 | SVD greedy 탐색 후보 수. Pruning은 자체 탐색 뒤 AIMET manual 모드라 쓰지 않습니다 |
| `--calib_batches` | 500 | Channel Pruning 재구성(reconstruction) 표본 수. greedy 점수는 val 전체로 매깁니다 |
| `--ft_epochs` | 20 | 단계마다 복구 파인튜닝 에폭 |
| `--ft_lr` | 1e-5 | 복구 파인튜닝 학습률 |
| `--dataset_ratio` | 0.4 | 복구 파인튜닝에 쓸 학습 데이터 비율 |
| `--device` | `cuda` | 장치 |

### 4.3 단계 3 — 비트폭·스킴 민감도 (`3_analyze_quant_sensitivity.py`)

스킴 3개(`post_training_tf`, `post_training_tf_enhanced`, `post_training_percentile`) × 정수 W{4,8,16}A{4,8,16} 9개 + float W16A16 1개 = QuantSim 30개를 만들어 각각 보정하고 val 전체로 채점합니다. 오래 걸리므로 예시는 `--calib_batches 10`입니다.

```bash
docker/run.sh --gpu <N> --name sens_x4 -- python3 examples/aimet_quantization/3_analyze_quant_sensitivity.py \
    --config configs/finetune/BicubicPlusPlus/bicubicplusplus_4x_eo.yaml \
    --data_config configs/data/bicubicpp/sr_finetune_eo_phase-2.yaml \
    --checkpoint results/release_v3/checkpoints/EO_x4.pth --width 320 --height 180 \
    --calib_batches 10 --device 0 --output_dir <out3>
```

- 출력: `<out3>/sensitivity_results.csv`, `<out3>/sensitivity_psnr.png`, 로그의 요약 표.
- 옵션: `--config`·`--data_config`·`--checkpoint`(필수), `--output_dir`(기본 `results/sensitivity`), `--calib_batches`(500), `--device`(`cuda`), `--quantsim_config`(저장소 config), `--width`/`--height`(640/360). BN fold와 CLE는 배포 그래프와 같게 항상 적용하므로 별도 플래그가 없습니다.

### 4.4 단계 4 — 층별 양자화 오차 (`4_apply_quant_analyzer.py`)

AIMET `QuantAnalyzer`로 층마다 양자화를 껐다 켜며 MSE를 재서, 정확도를 떨어뜨리는 층을 찾습니다.

```bash
docker/run.sh --gpu <N> --name qa_x4 -- python3 examples/aimet_quantization/4_apply_quant_analyzer.py \
    --config configs/finetune/BicubicPlusPlus/bicubicplusplus_4x_eo.yaml \
    --data_config configs/data/bicubicpp/sr_finetune_eo_phase-2.yaml \
    --checkpoint results/release_v3/checkpoints/EO_x4.pth --output_dir <out4> \
    --scheme post_training_tf_enhanced --param_bw 8 --output_bw 8 --width 320 --height 180 \
    --calib_batches 100 --device 0
```

- 출력: 먼저 `<out4>/quant_layer_config.json`과 `.txt`(기본 quantizer 배치, MMP 층 이름 출처), 이어서 QuantAnalyzer 결과(HTML/JSON)를 `<out4>`에 씁니다. config에 val 경로가 없으면 `quant_layer_config`까지만 쓰고 끝납니다.
- 옵션: `--config`·`--data_config`·`--checkpoint`(필수), `--output_dir`(`results/quant_analyzer`), `--scheme`(`post_training_tf_enhanced`; `post_training_tf`, `post_training_percentile` 가능), `--param_bw`(8), `--output_bw`(8), `--width`/`--height`(640/360), `--calib_batches`(500), `--quantsim_config`(저장소 config), `--device`(`cuda`).

### 4.5 단계 5 — 혼합 정밀도 인코딩 (`5_apply_manual_mixed_precision.py`)

4단계에서 찾은 취약 층만 높은 비트폭으로 올리고 나머지는 기본(W8A4)으로 둔 인코딩을 만듭니다.

```bash
docker/run.sh --gpu <N> -- python3 examples/aimet_quantization/5_apply_manual_mixed_precision.py \
    --config <cfg> --data_config <data_cfg> --checkpoint <ckpt> --mmp_config <my_mmp.json> \
    --output_dir <out5> --width 320 --height 180 --device 0
```

- 출력: `<out5>/mmp_model.onnx`, `mmp_model.encodings`, `mmp_model_torch.encodings`. `_torch` 파일을 6단계 `--encodings`로 넘깁니다.
- 옵션: `--config`·`--data_config`·`--checkpoint`·`--mmp_config`(모두 필수, `--mmp_config`가 없으면 종료 코드 2), `--output_dir`(`results/mmp`), `--param_bw`(8), `--output_bw`(**4**), `--scheme`(`post_training_tf_enhanced`), `--width`/`--height`(640/360), `--calib_batches`(500), `--quantsim_config`(저장소 config), `--device`(`cuda`).
- MMP JSON 형식(층 이름은 QuantSim 모델 안의 점 경로):

```json
{
    "layers": {
        "<module path, e.g. conv_out>": {"output_bw": 8, "input_bw": 8, "param_bw": 8}
    }
}
```

`configs/aimet/mmp_noprune_w16a16.json`과 `mmp_noprune_w8a8.json`은 `layers`가 빈 파일입니다.

### 4.6 단계 6 — W8A8 QAT

옵션은 [1.9절](#19-6_apply_qatpy-옵션-전체)과 같습니다. 5단계 인코딩에서 시작하는 형태:

```bash
docker/run.sh --gpu <N> --name qat_w8 -- python3 examples/aimet_quantization/6_apply_qat.py \
    --config <cfg> --data_config <data_cfg> --checkpoint <ckpt> --output_dir <out6> \
    --param_bw 8 --output_bw 4 --mmp_config <my_mmp.json> \
    --encodings <out5>/mmp_model_torch.encodings --qat_epochs <N> --qat_lr 5e-5 --train_data_ratio 0.3 \
    --width 320 --height 180 --device 0
```

- 출력: `<out6>/qat_model.*`와 `<out6>/qat_checkpoints/`(`best.pth`, `logs/`). 학습 뒤 `best.pth`를 복원해 export합니다. cosine 스케줄러면 `T_max`를 `--qat_epochs`로 맞춥니다.
- 손실 warm-up이 있는 config는 QAT 첫 에폭이 다시 순수 MSE가 됩니다([3.2절](#32-w16a16에서는-qat를-하지-않는다)). QAT 결과는 합성 val이 아니라 실기 프레임으로 판단합니다.

### 4.7 단계 7 — QuantSim 추론 (`7_inference_qat.py`)

6단계 QAT 체크포인트로 QuantSim을 다시 만들어, 장비 양자화를 흉내 낸 결과 이미지·영상을 얻습니다. QAT 0에폭 export에는 `qat_checkpoints/`가 없으므로 W8A8 QAT 전용입니다. IR 무채색 접기는 `switch_to_deploy()` 안에 있어 그래프가 6단계와 같습니다.

```bash
docker/run.sh --gpu <N> -- python3 examples/aimet_quantization/7_inference_qat.py \
    --config <6단계와 같은 cfg> --checkpoint <out6>/qat_checkpoints/best.pth \
    --encodings <out6>/qat_model_torch.encodings --mmp_config <6단계와 같은 json> \
    --param_bw <6단계와 같게> --output_bw <6단계와 같게> \
    --input <이미지 | 영상 | 폴더> --output_dir <out7> --width 320 --height 180 --device 0
```

- 이미지는 원래 크기 그대로 돌려 같은 파일 이름으로 `<out7>`에 저장합니다. 영상(`.mp4`, `.avi`, `.mkv`, `.mov`, `.ts`)은 가운데를 `--width` x `--height`로 잘라 처리하고 `<out7>/<이름>_output.mp4`로 씁니다. 영상 결과는 용량 때문에 `results/` 대신 `/mnt/data_server/etc/jshong/SuperResolution/results/` 아래에 둡니다(소유자 방침, 2026-10-13).
- 2단계로 줄인 모델은 `--base_model <out2>/compressed_model.pth`를 더합니다. 경로가 없으면 아무 말 없이 config로 모델을 만듭니다.
- 스킴은 `training_range_learning_with_tf_init`으로 고정입니다. `--encodings`가 없으면 경고만 하고 기본 통계로 돕니다. encodings는 `strict=False`로 읽고, 체크포인트 상태는 strict로 읽어 어긋나면 멈춥니다.
- 옵션: `--config`·`--checkpoint`·`--input`(필수), `--output_dir`(`results/qat_inference`), `--param_bw`(8), `--output_bw`(8), `--encodings`, `--mmp_config`, `--quantsim_config`(저장소 config), `--base_model`, `--width`/`--height`(640/360), `--device`(`cuda`).

## 5. 채점 스크립트 (`evaluation/`)

합성 val 점수(PSNR/LPIPS)는 실기 화질과 자주 엇갈렸습니다. 그래서 도메인별로 실기 입력을 쓰는 채점기를 만들었습니다. 최종 판정은 어느 경우든 보드 98 화면을 소유자가 보고 내립니다.

**입력 파일이 git에 없습니다.** 채점기의 기본 입력(프레임, 클립, sha256 목록, 핫픽셀 json, 차트 키트, 기록값 json)은 모두 `results/**`나 `checkpoints/` 아래라 git-ignore 대상이고 이 서버 디스크에만 있습니다. 새로 clone한 곳에서는 채점기가 파일을 못 찾고, 이를 고정하는 테스트(`TC_GRID_04_1`, `TC_CHART_06_1`)는 skip됩니다. 이 폴더들을 지우면 되살릴 수 없습니다.

| 입력 | 위치 |
| --- | --- |
| IR P2 프레임 600장 + 목록 | `results/release_v3/eval/ir92_raw_600/`, `results/release_v3/eval/ir92_p2_frames.sha256` (설명: `results/release_v3/eval/README_p2_frames.md`) |
| IR x4 출하 기록값 | `checkpoints/bicubicpp_deploy/finetune_bicubicplusplus_ir_sr_x4_dim32_v3/grid_best.json` |
| Denoise P1.1 클립 3개 + 목록 | `results/denoise_release_v1/eval/{z10_a9,z01_a9,z10_a9_nr3doff}.nv12`, `P1_clips.sha256` |
| 차트 키트(인쇄용 PDF, 보드 캡처 스크립트) | `results/denoise_release_v1/eval/kit/` |
| 핫픽셀 위치 | `results/denoise_release_v1/analysis/noise_census/hot_pixels_a0.json` |
| Denoise 출하 기록값 | `results/denoise_release_v1/analysis/p1_scores/P1.1/EXP-09-1.json` |

### 5.1 `score_grid_artifact.py` — SR 업샘플러 격자(바둑판) 점수, 프로토콜 P2

PixelShuffle이나 stride ConvTranspose는 출력 화소를 `scale` x `scale` 위상 중 하나에서 만듭니다. 위상끼리 어긋나면 `scale` 화소마다 반복되는 무늬가 생기고, 파워 스펙트럼의 k/scale cycles/px에 스파이크로 나타납니다. 이 스크립트는 체크포인트를 FP32로 IR 원본 프레임(640x512)에 돌려, 프레임 전체 스펙트럼을 더한 뒤 그 스파이크 대 주변 중앙값 비를 가로(h)·세로(v)·대각(d)별로 냅니다. 1 근처면 격자가 없는 것입니다. 격자를 흐림으로 없앤 모델을 가려내려고 선명도(같은 입력의 bicubic 확대 대비 평균 Sobel 크기 비)를 함께 냅니다. 설계와 시험: [MOD-GridScore](../docs/modules/20260928-MOD-GridScore.md).

```bash
docker/run.sh -- env PYTHONPATH=. python3 examples/evaluation/score_grid_artifact.py \
    --ckpt results/release_v3/checkpoints/IR_x4.pth --json <out>/grid.json
```

- 출력: `key : value` 줄(`1/4h`, `1/4v`, `1/4d`, `1/2h`, …, `sharpness`, `ckpt`, `epoch`, `n_frames`, …)과 `Grid Score Finish!`, `--json`을 주면 같은 내용의 JSON.
- 기준값: 출하 IR x4(ep89, 위 체크포인트와 `checkpoints/bicubicpp_deploy/finetune_bicubicplusplus_ir_sr_x4_dim32_v3/best.pth`는 같은 파일)는 `1/4h 5.41`, `1/4v 5.48`, `1/4d 14.30`, `sharpness 1.20`입니다(`grid_best.json`, `TC_GRID_04_1`이 비트 단위로 재현). 격자가 심했던 옛 v3(ep84)는 63.07 / 3.62 / 38.90 / 1.52였습니다. 모듈 문서에 합격 문턱 숫자는 없습니다. 기존 모델과 나란히 놓고 비교합니다.
- 모델은 체크포인트에 저장된 config로만 만듭니다(`--config` 인자가 없음). 저장된 config가 없으면 `ValueError`입니다.

| 옵션 | 기본값 | 설명 |
| --- | --- | --- |
| `--ckpt` | (필수) | config가 저장된 체크포인트(모든 Trainer 체크포인트) |
| `--frames` | `results/release_v3/eval/ir92_raw_600` | 원본 PNG 프레임 폴더 |
| `--crop` | `160,166,320,180` | 보드가 모델에 넣는 LR 크롭 `x,y,w,h`. **기본값은 x4 IR 크롭**입니다. x2는 `0,76,640,360`, x3은 `106,136,428,240`(`results/release_v3/eval/score_p4.py`의 보드 크롭 표) |
| `--index` | `0:600:12` | 정렬된 프레임 목록의 `start:stop:step`(50장) |
| `--min_frames` | 50 | 이보다 적게 고르면 체크포인트를 열기 전에 `ValueError` |
| `--manifest` | `results/release_v3/eval/ir92_p2_frames.sha256` | 고른 프레임의 이름·sha256이 순서까지 같아야 함. 프레임 폴더나 `--index`를 바꾸면 새 목록을 주거나 `''`로 끕니다 |
| `--device` | `cpu` | 프로토콜 기본값. `cuda`는 conv가 TF32로 돌 수 있어 기록값과 달라집니다 |
| `--json` | 없음 | 결과 JSON 경로 |

참고: P2 비율은 모델이 흐려져 주변 배경이 같이 줄면 크게 나올 수 있습니다. 선명도와 함께 봅니다. IR 출하 판정에 쓴 P4 게이트 묶음(`results/release_v3/eval/score_p4.py`)은 이제 기록으로만 남습니다. cam98 원본 프레임과 FLIR 등 홀드아웃 세트가 데이터 서버에서 지워져(2026-10-15 확인) 다시 채점할 수 없습니다([results/release_v3/MANIFEST.md](../results/release_v3/MANIFEST.md)).

### 5.2 `score_eo_chart.py` — Denoise 회의실 차트 점수, 프로토콜 P1.1

보드에서 캡처한 NV12 1920x1080 클립마다 5° 기울어진 검정 사각형을 찾아 차트 둘레 1280x720 창을 잘라(보드와 같은 크기), Denoise를 FP32로 돌리고 8비트로 반올림합니다. 같은 측정을 카메라 입력과 Denoise 출력에 똑같이 해서 비교합니다: 회색 패치 P4 평탄부 노이즈, P2·P3 윗변 넘침/꺼짐, 사각형 MTF50(가로·세로)과 넘침, P3 색 노이즈, 가장 작은 글자 줄의 가짜 색, 센서 핫픽셀 잔차. 차트 밖 위험은 보조 지표로 봅니다: 카메라 모델을 거친 점 보존, 깨끗한 영상의 질감 비. 설계와 시험: [MOD-ChartScore](../docs/modules/20261008-MOD-ChartScore.md).

```bash
docker/run.sh -- env PYTHONPATH=. python3 examples/evaluation/score_eo_chart.py \
    --checkpoint results/denoise_release_v1/checkpoint/best.pth --json <out>/p11.json
```

- 출력: 클립마다 한 줄(`noise <입력> -> <출력> | added over P2 .. P3 .. %p | MTF50 ratio h/v | chroma | false colour | square added | hot residual`), 이어서 `guards ...` 한 줄과 `Score Finish!`. JSON에는 `protocol: P1.1`, 체크포인트 md5와 에폭, 클립별 camera/denoise/derived 값이 들어갑니다.
- 출하 v1을 위 기본값과 `--camera_config results/denoise_release_v1/checkpoint/data_config.yaml`로 채점하면 기록값 161개가 비트 단위로 같습니다(`TC_CHART_06_1`). 지금은 두 파일의 `camera_model` 블록이 같아 기본값으로 채점해도 같습니다.
- 합격 기준 **초안**([RESEARCH-EODenoiseV3](../docs/20261008-RESEARCH-EODenoiseV3.md) §1, §6에 소유자 확정 대기로 남아 있음): 평탄부 노이즈 ≤ BL1 × 1.10, 더한 넘침 P2·P3 각각 ≤ +1.0 %p, MTF50 비 ≥ 0.95(가로·세로), 합성 val PSNR ≥ BL1 − 0.1 dB. 가짜 색과 글자 가독은 보고만 합니다. 최종 판정은 보드 98 화면(소유자)입니다.

왜 P1.1인가: 처음 프로토콜(P1)은 엣지 직선을 노이즈 낀 프레임마다 맞추고 각 열을 자기 노이즈 낀 중심으로 정렬했습니다. 그러면 입력 쪽 엣지가 번져 보여서, 노이즈만 완벽히 지운 Denoise도 MTF 비 1.31~2.19, 더한 넘침 +7~+29 %p로 불합격이 나왔습니다(2026-10-08 발견). "출력이 1.34배 선명하다 = v2 샤프닝이 남았다"던 판단도 이 착시였습니다. P1.1은 엣지 직선을 카메라 입력에서 한 번 맞춰 출력에도 쓰고, 넘침을 프레임 평균 엣지 프로파일에서 재며, 노이즈 낀 입력에 완벽한 Denoise를 넣는 시험(`MREQ-CHART-03`)을 통과해야 합니다. 평탄 노이즈와 색 지표는 원래 영향을 받지 않았습니다. 엣지 지표를 새로 만들 때도 같은 오라클 시험을 먼저 통과시킵니다.

| 옵션 | 기본값 | 설명 |
| --- | --- | --- |
| `--checkpoint` | (필수) | config가 저장된 Denoise 체크포인트 |
| `--clips` | `z10_a9=…/z10_a9.nv12 z01_a9=…/z01_a9.nv12 z10_a9_nr3doff=…/z10_a9_nr3doff.nv12` (`results/denoise_release_v1/eval/`) | `이름=경로` 여러 개. 이름은 차트 id(`z10` 또는 `z01`)로 시작해야 합니다 |
| `--frames` | `20::10` | `start::step` 프레임 선택 |
| `--n_frames` | 10 | 클립마다 쓸 프레임 수 |
| `--min_frames` | 10 | 이보다 적게 고른 클립은 거부 |
| `--manifest` | `results/denoise_release_v1/eval/P1_clips.sha256` | 클립 sha256 목록(`''`이면 생략) |
| `--hot_pixels` | `results/denoise_release_v1/analysis/noise_census/hot_pixels_a0.json` | 핫픽셀 위치(`''`이면 생략) |
| `--camera_config` | `configs/data/bicubicpp/denoise.yaml` | 점 보존 보조 지표를 만들 `camera_model` 블록(`''`이면 생략) |
| `--texture_root` | `/mnt/data2/jshong/SuperResolution/Pretrained_Dataset/val_denoise/HR` | 질감 비용 깨끗한 영상(`''`이면 생략) |
| `--device` | `cpu` | 장치 |
| `--json` | `''` | 결과 JSON 경로 |

새 클립을 찍을 때는 차트 키트의 `capture_eo_raw.sh`(보드에서 `./capture_eo_raw.sh <tag> [seconds]`, 기본 출력 `/data/eo_cap`)를 씁니다. 보드 작업은 소유자 몫입니다.

### 5.3 `score_sr_tensors.py` — 보드 텐서 선명도 비교 (legacy)

보드에서 인코더 앞 RGBA를 뽑은 폴더(`meta.txt`, `lr`/`anchor`/`model` `.rgba`)를 읽어 선명도(Sobel), 앵커(qtivtransform) 대비 비, 고주파 비중, 오버슈트 %를 프레임 중앙값으로 냅니다. 화면 녹화는 H.264 인코딩과 VLC 확대를 거쳐 모델이 아니라 표시 경로를 재게 되므로(2026-09-08: 녹화 0.813, 텐서 1.379) 이 방식이 생겼습니다.

```bash
scp -r root@192.168.100.98:/data/l0/tensors <dir>          # host
docker/run.sh -- python3 examples/evaluation/score_sr_tensors.py <dir> --cv2 --frames 12
```

- 옵션: `dir`(위치 인자), `--frames`(파일당 표본 프레임 수, 기본 12), `--cv2`(cv2 bilinear·bicubic 기준 행 추가). 출력은 화면 표뿐입니다.
- 캡처 스크립트 `capture_sr_tensors.sh`는 2026-10-06에 지워졌습니다. 필요하면 `git show 3fe100d^:examples/deployment/capture_sr_tensors.sh`로 꺼냅니다. 캡처들은 서로 다른 gst 실행이라 프레임이 짝지어지지 않으므로 정지 장면에서만 비교가 성립합니다.

## 6. 검증 세트 만들기 (`dataset_preparation/`)

두 생성기 모두 시드를 고정해(이미지 i는 `seed + i`) 같은 코드로 다시 만들면 같은 결과가 나옵니다. 둘 다 출력 폴더를 묻지 않고 덮어쓰므로 **반드시 새 폴더**에 만듭니다. 쓰고 있는 val 폴더(아래 표)를 출력으로 주면 모든 게이트 세트가 바뀝니다.

| 용도 | 지금 쓰는 val 폴더 |
| --- | --- |
| SR Phase-1 | `/mnt/data_server/etc/jshong/SuperResolution/Pretrained_Dataset/val/LRx<s>_v3` |
| SR EO Phase-2 | `/mnt/data_server/etc/jshong/SuperResolution/Pretrained_Dataset/val/LRx<s>_weaknoise_eo_v3` |
| SR IR Phase-2 | `/mnt/data_server/etc/jshong/SuperResolution/Finetunning_Dataset/val/LRx<s>_weaknoise_ir` |
| Denoise | `/mnt/data2/jshong/SuperResolution/Pretrained_Dataset/val_denoise_cam_a9nr5` (`LR/`만 쓰고 정답은 `val_denoise/HR`) |

### 6.1 `make_val_lr.py` — SR val LR

val HR에 학습과 같은 열화 레시피를 입혀 LR을 만듭니다. 예전 `val/LRx{2,3,4}`는 기록되지 않은 훨씬 센 레시피로 만들어져, 약한 노이즈 분포에 맞춰 가는 파인튜닝을 엉뚱한 기준으로 재고 있었습니다. 그래서 Phase-1과 Phase-2는 각자 학습 레시피(와 소프트닝 비율)로 만든 val로 검증합니다.

`--denoise_checkpoint`를 주면 배포 때 SR 앞에 Denoise가 있는 카메라와 없는 카메라를 둘 다 반영하려고, 학습(`Trainer._soften`)처럼 일부 이미지를 Denoise로 연화합니다. 학습은 무작위로 나누지만 여기서는 재현성을 위해 고정 간격(기본 0.5 → 짝수 인덱스)으로 나눕니다.

```bash
# EO Phase-2, x3
docker/run.sh -- python3 examples/dataset_preparation/make_val_lr.py \
    --data_config configs/data/bicubicpp/sr_finetune_eo_phase-2.yaml --scale 3 \
    --hr_root /mnt/data_server/etc/jshong/SuperResolution/Pretrained_Dataset/val/HR \
    --out_dir <새 폴더> \
    --denoise_checkpoint checkpoints/csuav_deploy/train_svfocusdenoise_adv_x1_dim32_block3_epoch300_lr1e-4_charbonnier_edge_ssim_hotpixel_260309/best.pth

# IR Phase-2, x4 (no softening)
docker/run.sh -- python3 examples/dataset_preparation/make_val_lr.py \
    --data_config configs/data/bicubicpp/sr_finetune_ir_phase-2.yaml --scale 4 \
    --hr_root /mnt/data_server/etc/jshong/SuperResolution/Finetunning_Dataset/val/HR \
    --out_dir <새 폴더>
```

Phase-1은 `--data_config configs/data/bicubicpp/sr_train_phase-1.yaml`에 EO와 같은 `--denoise_checkpoint`를 줍니다.

| 옵션 | 기본값 | 설명 |
| --- | --- | --- |
| `--data_config` | (필수) | 열화 레시피 |
| `--scale` | (필수) | 배율. HR을 배율의 배수로 잘라 씁니다 |
| `--hr_root` | (필수) | HR 폴더(`.png`, `.jpg`, `.jpeg`) |
| `--out_dir` | (필수) | 출력 폴더. `<stem>.png`로 씁니다 |
| `--seed` | 2026 | 기본 시드 |
| `--denoise_checkpoint` | 없음 | 연화에 쓸 Denoise 체크포인트(config 저장 필요) |
| `--denoise_fraction` | 0.5 | 연화 비율(학습의 `denoise_prob`에 맞춤) |
| `--denoise_noise_sigma` | `2,15` | Denoise 앞에 더할 센서 노이즈 범위(`Trainer._soften`과 같음). `''`이면 끔 |

마지막 줄: `<out_dir>: N images (seed 2026, recipe ..., softened 50% via best.pth)`.

**디스크에 있는 val 세트는 지금 코드로 비트 단위 재현이 되지 않습니다.** 그 세트들은 2026-10-16의 레시피 수정(노이즈 반올림, [docs/CONFIG_RATIONALE.md](../docs/CONFIG_RATIONALE.md) 4.4절) 전에 만들어졌고, 커밋 `de502f6`에서만 같은 파일이 나옵니다. 새 레시피로 val을 바꾸는 것은 게이트 기준을 바꾸는 일이므로 소유자와 정한 뒤 새 폴더 이름으로 만듭니다.

### 6.2 `make_val_camera.py` — Denoise val

깨끗한 원본에 카메라 모델(data config의 `camera_model` 블록)을 입혀 `<out_dir>/HR/`(정답, `camera_model.target`을 따름)과 `<out_dir>/LR/`(노이즈 입력)을 만듭니다. LR은 정답 방식과 무관하게 같은 시드면 같습니다. 재생성은 비트 단위로 같습니다(`TC_CAM_05_2`).

```bash
docker/run.sh -- env PYTHONPATH=. python3 examples/dataset_preparation/make_val_camera.py \
    --data_config configs/data/bicubicpp/denoise.yaml \
    --hr_root /mnt/data2/jshong/SuperResolution/Pretrained_Dataset/val_denoise/HR \
    --out_dir <새 폴더>
```

- 옵션: `--data_config`(필수, `camera_model.enabled`가 꺼져 있거나 없으면 `ValueError`), `--hr_root`(필수), `--out_dir`(필수), `--seed`(2026).
- 출력 끝: `Images : N -> <out_dir> (seed 2026)`, `Make Val Finish!`.
- 출하 세트 `val_denoise_cam_a9nr5`는 덮어쓰지 않습니다. Denoise config는 이 세트의 `LR/`과 `val_denoise/HR`(원본)을 짝으로 씁니다(`target: source`).

## 7. 이 영역의 테스트

```bash
docker/run.sh --offline -- env PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q -rs -p no:cacheprovider \
    tests/test_release_golden.py tests/test_config_guards.py tests/test_score_grid_artifact.py \
    tests/test_score_eo_chart.py tests/test_make_val_lr.py tests/test_camera_model.py
```

요약 줄이 `N passed`로 시작하고(뒤에 `, M warnings in …`이 붙고, 그 아래에 `sys:1: DeprecationWarning: builtin type swigvarlink …` 한 줄이 더 찍힐 수 있습니다) `-rs` 보고에 `SKIPPED` 줄이 없으면 통과입니다. git에 없는 릴리스 데이터가 없으면(새 clone 등) `TC_REL_02_1`, `TC_GRID_04_1`, `TC_CHART_06_1` 등이 skip되므로, skip이 있으면 통과로 보지 않습니다. 환경 전체 인수 시험(`docker/accept.sh`, 2026-10-19 pytest 3499 passed, skip 0)은 [docker/README.md](../docker/README.md)에 있습니다.

## 8. 관련 문서

| 문서 | 내용 |
| --- | --- |
| [README.md](../README.md) | 플랫폼 개요, 새 모델을 보드까지 올리는 전체 흐름 |
| [docker/README.md](../docker/README.md) | `run.sh` 옵션·거부 메시지, GPU 고르기, `qairt-env`, 인수 시험 |
| [tools/README.md](../tools/README.md) | 학습·평가·추론 CLI, 체크포인트 종류, 재개 규칙 |
| [configs/README.md](../configs/README.md) | config 키, Phase-1/Phase-2, 데이터 레시피, AIMET JSON |
| [src/README.md](../src/README.md) | 코드 지도: 모델·손실·데이터셋·열화 추가 |
| [deployment/README.md](deployment/README.md) | 변환, DLC 이름, 폴백 목록, 보드 스크립트 |
| [results/release_v3/MANIFEST.md](../results/release_v3/MANIFEST.md) | SR v3 출하 기록 |
| [results/denoise_release_v1/MANIFEST.md](../results/denoise_release_v1/MANIFEST.md) | Denoise v1 출하 기록 |
| [docs/modules/20260928-MOD-GridScore.md](../docs/modules/20260928-MOD-GridScore.md), [docs/modules/20261008-MOD-ChartScore.md](../docs/modules/20261008-MOD-ChartScore.md) | 두 채점기의 요구·설계·시험 |
