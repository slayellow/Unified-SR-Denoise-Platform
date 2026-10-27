# examples/deployment

AIMET export가 끝난 모델을 QCS8550(Hexagon V73) 보드용 DLC로 바꾸고, 보드에 올려 속도와 화질을 확인하는 도구들이다.
일이 세 곳에서 나뉘어 일어나므로 어느 명령이 어디서 도는지부터 적는다.

| 곳 | 들어가는 법 | 여기서 하는 일 |
|---|---|---|
| 서버 호스트 (사용자 jshong) | 셸 | `docker/run.sh` 시작, 보드 `ssh` / `scp` |
| 작업 컨테이너 (이미지 usdp, QAIRT 2.41 포함) | `docker/run.sh [옵션] -- <명령>` | `gen_convert_all.py`, `convert_all.sh`(`qairt-env` 안에서), `make_bench_raw.py` |
| 보드 98 (`root@192.168.100.98`, QCS8550 테스트 보드) | 호스트에서 `ssh` | `bench_dlc.sh`, `gst_compare_*.sh` |

- 호스트의 `python3`(3.8, onnx·aimet 없음)는 작업 환경이 아니다. 생성기와 변환은 모두 컨테이너에서 돈다.
- `docker/run.sh`는 저장소 루트에서 실행한다. `--log` 경로와 셸 리다이렉트(`> log`)는 호스트의 현재 폴더 기준이고,
  명령 안의 경로는 컨테이너 안의 저장소 루트 기준이다. 저장소 루트에서 시작해야 둘이 같은 곳을 가리킨다.
- `docker/run.sh`의 옵션, 거부 메시지(GPU 사용 중, `/mnt/data_server` 마운트 없음 등)와 대처는 [docker/README.md](../../docker/README.md)에 있다.
  CPU만 쓰는 변환도 `/mnt/data_server`가 읽히는 sshfs 마운트가 아니면 시작을 거부한다.
- 192.168.100.87은 운용 SW 장비다. 이 폴더의 스크립트는 98번 테스트 보드 기준이다.

| 파일 | 도는 곳 | 하는 일 |
|---|---|---|
| `gen_convert_all.py` | 컨테이너 | export 폴더 → `convert_all.sh` 생성 |
| `convert_all.sh` (생성물) | 컨테이너 `qairt-env` | ONNX + encodings → HTP prepare까지 끝난 DLC |
| `make_bench_raw.py` | 컨테이너 | 속도 측정용 RAW 입력 생성 |
| `bench_dlc.sh` | 보드 | DLC 추론 지연 측정 |
| `gst_compare_eo.sh` | 보드 | EO 카메라: Denoise → 업스케일러 비교, 2x2 모자이크 RTSP (포트 20000) |
| `gst_compare_eo_denoise.sh` | 보드 | EO 카메라: Denoise 여러 개(`ROWS`) 각각 → Bicubic++ x2·x3·x4, 줄 × 배율 모자이크 RTSP |
| `gst_compare_eo4k.sh` | 보드 | 4K USB EO 카메라: 업스케일러 비교(Denoise 없음), 2x2 모자이크 RTSP (포트 30000) |
| `gst_compare_eo_dual.sh` | 보드 | EO 센서 두 개 동시: 위 MC-G105 Denoise → Bicubic++, 아래 4K USB Bicubic++만. `RECORD=1`이면 두 센서 원본 저장 |
| `gst_compare_ir.sh` | 보드 | IR 카메라(720 / 1024 프로파일): 업스케일러 비교, 2x2 모자이크 RTSP |

gst 파이프라인 실행과 화면 확인은 소유자가 직접 한다. 자동화나 다른 사람이 대신 돌리지 않는다.

## 흐름 한눈에

학습·평가·export까지의 앞부분은 최상위 [README.md](../../README.md)의 "새 모델을 보드까지"가 다룬다. 이 문서는 export 폴더부터다.

```
<export_root>/x<s>_<eo|ir>[_1024]/qat/qat_model.{onnx,encodings}     <- 6_apply_qat.py (1.1)
   |  docker/run.sh --offline -- python3 examples/deployment/gen_convert_all.py <export_root> --version <ver>
   v
<export_root>/convert_all.sh
   |  docker/run.sh --offline -- qairt-env bash <export_root>/convert_all.sh > <export_root>/convert.log 2>&1
   v
<export_root>/BicubicPP_<EO|IR>_<W>x<H>_x<s>_<ver>.dlc               <- 로그 확인 (1.5)
   |  (호스트) scp <dlc> root@192.168.100.98:/data/dlc/              <- 출하 이름과 겹치지 않게 (2.3)
   v
보드: bench_dlc.sh (지연, 3절)  ->  gst_compare_*.sh (화면, 소유자, 4절)
```

2026-10-19에 이 서버에서 EO x2 Phase-2 1에폭 런으로 위 사슬을 끝까지 돌렸다(`results/smoke_20261019/`, 이미지 usdp:20261019).
이 문서의 출력 예시는 그 실행에서 가져왔다. 보드 단계는 아직 돌리지 않았다.

## 지켜야 할 규칙 네 가지

1. **출하 폴더(`results/release_v3`, `results/denoise_release_v1`) 안에서 `convert_all.sh`를 돌리지 않는다.**
   런마다 마지막 단계의 `mv -f`가 같은 이름의 DLC를 묻지 않고 덮어쓴다. Denoise 폴더에서는 출하 DLC 자체가 바뀌고,
   v3 폴더에서는 HTP prepare가 없는 DLC 9개가 출하 이름으로 `results/release_v3/`에 생긴다. 항상 복사본 폴더에서 돌린다.
2. **새로 변환한 DLC를 보드의 출하 이름 위로 올리지 않는다.** 그러면 다음에 기본값으로 돌린 gst 스크립트가 모르는 사이에
   시험 DLC를 읽고, 출하본과 나란히 비교할 수도 없게 된다. 새 버전 이름(`--version`)이나 다른 파일 이름으로 올린다.
3. **출하 DLC를 보드에 다시 올릴 때는 저장소의 출하 사본을 쓴다**(`results/release_v3/dlc/*.dlc`,
   `results/denoise_release_v1/BicubicPPDenoise_EO_1280x720_v1.dlc`). 다시 변환한 파일은 md5가 매번 달라 출하본임을 증명할 수 없다(1.7).
4. **DLC의 출력 레이어는 `final_act`이다.** 미리 컴파일된 HTP 그래프는 런타임이 같은 출력 레이어를 요청할 때만 쓰인다(1.4).

---

## 1. 변환 — export 폴더 + `convert_all.sh`

### 1.1 export 폴더 모양

`gen_convert_all.py`는 `<export_root>` 바로 아래에서 이름이 `x<배율>_<eo|ir>` 또는 `x<배율>_ir_1024`인 폴더만 런으로 본다
(정규식 `x(\d)_(eo|ir)(_1024)?$`). 런 폴더에는 `qat/qat_model.onnx`와 `qat/qat_model.encodings`가 둘 다 있어야 한다.
같은 `qat/`에 생기는 `qat_model.pth`, `qat_model_torch.encodings`, `temp_onnx_model_with_all_markers.onnx`는 변환에 쓰이지 않는다.

| 런 폴더 | 만드는 명령 | 입력 크기 (`--width --height`) |
|---|---|---|
| `x<s>_eo` | `6_apply_qat.py --output_dir <export_root>/x<s>_eo/qat` (배율마다 한 번) | x2 640 360, x3 428 240, x4 320 180 |
| `x<s>_ir`, `x<s>_ir_1024` | `results/release_v3/eval/export.sh` (배율마다 한 번, 두 프로파일을 함께) | 720: 640x360 / 428x240 / 320x180, 1024: 640x512 / 428x342 / 320x256 |

입력 크기는 보드 스크립트가 자르는 크기와 같아야 한다. 보드 스크립트는 이 크기로 DLC 이름을 만들어 찾으므로,
다른 크기로 export하면 보드에서 `model not found: ...`로 멈춘다.

EO 예 (2026-10-19 실행, W16A16, QAT 0에폭 = PTQ, 출하 레시피):

```bash
docker/run.sh --gpu 2 --name smoke_x2_eo_export --log results/smoke_20261019/x2_eo_export.log -- \
    python3 examples/aimet_quantization/6_apply_qat.py \
    --config configs/finetune/BicubicPlusPlus/bicubicplusplus_2x_eo.yaml \
    --data_config configs/data/bicubicpp/sr_finetune_eo_phase-2.yaml \
    --checkpoint results/smoke_20261019/x2_eo_p2/last.pth \
    --output_dir results/smoke_20261019/export/x2_eo/qat \
    --mmp_config configs/aimet/mmp_noprune_w16a16.json --param_bw 16 --output_bw 16 --qat_epochs 0 \
    --width 640 --height 360 --device 0
```

- `--gpu 2`로 호스트의 GPU 2번 하나만 컨테이너에 보이므로, 컨테이너 안에서는 `--device 0`이다.
- 로그 끝에 `[run.sh] exit 0`, 그 앞에 FP32 / Pre-QAT / Post-QAT PSNR이 찍힌다(이 실행: 31.0827 / 31.0832 / 31.0832 dB).
- 옵션과 AIMET 함정(Conv+LeakyRelu supergroup 등)은 [examples/README.md](../README.md) 1절(옵션 전체는 1.9)과 3절이 다룬다.

IR 예 (CPU, 한 번에 두 프로파일):

```bash
docker/run.sh --offline -- env SCALE=<2|3|4> results/release_v3/eval/export.sh <checkpoint.pth> <export_root> [config]
```

- `<export_root>/x<s>_ir/qat`, `x<s>_ir_1024/qat`와 로그 `x<s>_ir.log`, `x<s>_ir_1024.log`를 쓴다. 경로는 저장소 루트 기준이다.
- 프로파일마다 `x<s>_ir exit <코드>`를 찍고, 스크립트 자체는 export가 실패해도 종료 코드 0으로 끝난다(아래처럼 멈출 때만 1). 그래서 찍힌 두 줄을 직접 확인한다.
- 같은 이름의 프로파일 폴더가 이미 있으면 `... exists -- move it away first`로 멈춘다.
- config를 주지 않으면 `configs/finetune/BicubicPlusPlus/bicubicplusplus_<s>x_ir.yaml`을 쓴다.

Denoise의 출하 export 스크립트(`results/denoise_release_v1/export.sh`)는 `<out>/qat`에 바로 쓰고 런 폴더가 없어서 생성기가 찾지 못한다. Denoise 변환은 1.8을 본다.

### 1.2 `gen_convert_all.py` — `convert_all.sh` 만들기

```bash
docker/run.sh --offline -- python3 examples/deployment/gen_convert_all.py <export_root> --version <ver>
```

| 인자 | 기본값 | 뜻 |
|---|---|---|
| `root` (필수) | | 런 폴더들이 있는 폴더 |
| `--version` | `v3` | DLC 이름 끝의 버전. 영문·숫자·`_`만 쓴다(1.6) |
| `--scales` | 전부 | 이 배율만 (`--scales 3 4`) |
| `--domains` | `eo ir` | 이 도메인만 (`--domains ir`) |
| `--htp_socs` | `qcs8550` | `snpe-dlc-graph-prepare` 대상 SoC |
| `--output_layer` | `final_act` | HTP 캐시의 출력 레이어. 보드 gst의 `qtimlsnpe layers=`와 같아야 한다 |

`--version`의 기본값 `v3`는 출하 v3의 이름이다. 출하 v3 자체가 아닌 것을 변환할 때는 반드시 새 버전 이름을 준다.
그러지 않으면 규칙 2를 어기는 이름(`..._v3.dlc`)이 만들어진다.

생성기는 이 런 폴더 안의 ONNX에서 텐서 이름과 입력 크기를, encodings의 `quantizer_args.param_bitwidth`에서 가중치 비트폭을 읽는다.
그래서 손으로 적을 것이 없다. 텐서 이름은 배율마다 다르다(v3: 입력 `t.1`, 출력 x2 `37`, x3 `34`, x4 `47`).
비트폭이 16 이상인 런의 줄 끝에는 `16`이 붙고, 스크립트가 그 런에 `--restrict_quantization_steps "-0x8000 0x7F7F"`를 bash 배열로 한 인자에 담아 넘긴다.

2026-10-19 실행 출력:

```
results/smoke_20261019/export/convert_all.sh: 1 runs
  x2_eo        in=t.1    out=37   -> BicubicPP_EO_640x360_x2_smoke.dlc
```

| 메시지 | 뜻 | 할 일 |
|---|---|---|
| `no such results root: <root>` (종료 1) | 폴더가 없다 | 경로가 저장소 루트 기준인지 확인 |
| `no runs to convert under <root>` (종료 1) | 변환할 런이 없다. 아래 `!` 줄이 이유 | 폴더 이름과 qat 파일 확인 |
| `! <run>: qat_model.onnx 와 qat_model.encodings 가 다 있지 않음 -- 빠짐` | export가 덜 끝났다 | 그 런을 다시 export |
| `! <run>: qat_model.onnx 를 읽을 수 없음 (...) -- 빠짐` | ONNX가 깨졌다 | 그 런을 다시 export |
| `! 이름 충돌: <run> / <run> 가 모두 <dlc> 로 갑니다 ...` | 같은 도메인·배율·입력 크기의 런이 둘 | 하나를 다른 root로 옮긴다. 스크립트는 두 번째부터 거부한다 |
| `<root>/convert_all.sh already exists -- move it away first ...` (종료 1) | 이미 있는 스크립트는 덮어쓰지 않는다 | 기존 파일을 `mv`로 옮긴 뒤 다시 |
| `ValueError: ...: quantizer_args.param_bitwidth 없음 -- export 가 깨졌다` | encodings에 비트폭이 없다 | 다시 export. 이때 `convert_all.sh`는 만들어지지 않는다 |

생성되는 스크립트 본문(`script_lines()`)을 고칠 때 알아 둘 것:

- `tests/test_gen_convert_all.py`의 `test_denoise_release_script_unchanged`가 Denoise 출하 스크립트와 같은 본문을 sha256으로 고정한다. 본문을 바꾸면 같은 커밋에서 해시를 고친다.
- `docker/accept.sh`가 로그의 `생성 1개 / 건너뜀 0개` 문구를 grep한다. 끝의 echo 문구를 바꾸면 인수 시험이 깨진다.

### 1.3 `convert_all.sh` 돌리기

```bash
docker/run.sh --offline -- qairt-env bash <export_root>/convert_all.sh > <export_root>/convert.log 2>&1
```

- `qairt-env`는 QAIRT 도구와 그 전용 Python(NumPy 1 venv)을 켜고 뒤의 명령을 실행한다. 이것 없이는 `qairt-converter` 등이 PATH에 없다.
  반대로 `gen_convert_all.py`는 `onnx`가 있는 기본 환경에서 돌리므로 `qairt-env` 없이 실행한다.
- GPU는 필요 없다. 로그 첫 줄은 `run.sh: GPU: none (CPU run)`이다. `--offline`은 컨테이너를 네트워크 없이 띄운다.
- 스크립트는 자기 폴더로 `cd`한 뒤 시작하므로 DLC는 `convert_all.sh` 옆(`<export_root>/`)에 생긴다. 중간 파일은 컨테이너의 `/tmp`에 만들고 끝나면 지운다.
- 2026-10-19 실행에서 x2 EO 한 런은 1분 안에 끝났다.
- 같은 폴더에서 다시 돌리면 이전 실행이 남긴 같은 이름의 DLC를 경고 없이 덮어쓴다. 이 때문에 출하 폴더에서 돌리면 안 된다(규칙 1).
- 오래 걸리는 묶음은 분리 실행도 된다: `docker/run.sh --offline --name <이름> --log <export_root>/convert.log -- qairt-env bash <export_root>/convert_all.sh`.
  로그는 `[run.sh] exit <코드>`로 끝나고 `docker stop usdp-<이름>`으로 멈춘다. 이 형식은 학습에서 확인했고 변환에는 돌려 보지 않았다.
- QAIRT SDK가 있는 다른 PC(회사 PC의 WSL)에서도 된다. `convert_all.sh`와 `x*_*/qat/`만 옮겨 그 PC에서 `./convert_all.sh > convert.log 2>&1`로 돌린다. 이제는 선택 사항이다.

로그 끝(2026-10-19 실행):

```
생성 1개 / 건너뜀 0개
-rw-r----- 1 jshong jshong 281715 Oct 19 23:43 ./BicubicPP_EO_640x360_x2_smoke.dlc
```

| 스크립트 메시지 | 뜻 |
|---|---|
| `=== <run> -> <dlc> ===` | 그 런의 변환 시작 |
| `=== <run>: qat_model.onnx/encodings 없음 -> 건너뜀` | export 파일이 없어 건너뜀(`건너뜀` 개수에 들어감) |
| `!!! <run> 변환 실패` | 단계 하나가 실패. 배포 이름의 파일은 쓰지 않았다 |
| `(<dlc> 는 이전 실행에서 남은 파일입니다)` | 실패했는데 같은 이름의 옛 DLC가 폴더에 있다. 그 파일은 이번 결과가 아니다 |
| `!!! <run>: <dlc> 는 이번 실행에서 이미 만들어졌습니다 -> 덮어쓰지 않고 중단` | 한 스크립트 안에서 두 런이 같은 이름으로 간다 |
| `실패: <run> ...` | 끝에 모은 실패 목록. 이때 종료 코드는 0이 아니다 |

변환이 실패해도 묶음은 멈추지 않고 나머지 런을 계속 변환한다.

### 1.4 런 하나의 단계와 HTP prepare

| 순서 | 도구 | 하는 일 |
|---|---|---|
| 1 | `qairt-converter` | ONNX를 DLC로 바꾼다. AIMET encodings를 `--quantization_overrides`로 넘기고(로그에 `Processed <N> quantization encodings`), 입출력 배치를 NCHW → NHWC로 바꾼다 |
| 2 | `qairt-quantizer --float_fallback` | W16 런에는 `--restrict_quantization_steps "-0x8000 0x7F7F"`가 더 붙는다 |
| 3 | `snpe-dlc-graph-prepare --htp_socs qcs8550 --set_output_layers final_act` | HTP용 그래프를 미리 컴파일해 DLC 안에 `backend.metadata0`로 넣는다 |
| 4 | `mv -f` | 임시 폴더의 결과를 배포 이름으로 옮긴다. 앞 단계가 모두 성공했을 때만 실행된다 |

3단계가 있으면 보드가 DLC를 열 때 그래프를 다시 컴파일하지 않는다. 소유자가 2026-10-07 보드에서 CMA 사용량이 줄어드는 것을 확인했다.
이 캐시는 런타임이 prepare 때와 같은 출력 레이어를 요청할 때만 쓰인다. 보드 gst는 `qtimlsnpe layers=<final_act>`로 요청한다(`BPP_LAYER`, `DENOISE_LAYER` 기본값).
둘이 다르면 오류 없이 보드가 시작할 때 온라인 prepare로 돌아가고, CMA 절약만 조용히 사라진다.
그래서 새 모델은 마지막 모듈 이름을 `final_act`로 두거나([src/README.md](../../src/README.md)), 생성기에 `--output_layer`를 주고 보드에서도 같은 `BPP_LAYER` / `DENOISE_LAYER`를 준다.
prepare가 실패한 런은 실패로 처리되므로 prepare 안 된 DLC가 배포 이름으로 나가는 일은 없다.

DLC가 prepare됐는지는 호스트에서 바로 볼 수 있다(2026-10-19 smoke DLC, 출하 v3, 출하 Denoise로 확인):

```bash
unzip -l <dlc>        # 목록에 backend.metadata0 이 있어야 prepare된 DLC
grep -a -o '"\(set_output_layers\|htp_socs\)": "[^"]*"' <dlc>
# "set_output_layers": "[final_act]"
# "htp_socs": "[qcs8550]"
```

`results/release_v3/convert_all.sh`에는 3단계가 없다. prepare 단계가 생성기에 들어간 커밋(`1ee647f`)보다 먼저 만든 스크립트이기 때문이다.
보드에 올라간 v3 DLC 9개는 변환 뒤 2026-10-07에 회사 PC에서 손으로 prepare한 것이고, 그 사본이 `results/release_v3/dlc/`에 있다.

### 1.5 변환 로그 읽기

변환은 op가 float로 떨어져도 성공으로 끝나고 DLC를 만든다. 그래서 로그를 직접 읽어야 한다.
QAIRT 2.41은 런마다 한 번, `qairt-converter` 부분의 `INFO_CONVERSION_SUCCESS` 바로 앞에 폴백 줄을 찍는다.

| 경우 | 로그 |
|---|---|
| 폴백 없음 | `[ WARNING ] Following OPs fallback to float.` (마침표로 끝나고, 다음 줄은 `INFO_CONVERSION_SUCCESS`) |
| 폴백 있음 | `[ WARNING ] Following OPs fallback to float:` (쌍점), 다음 줄에 op 목록 |

폴백이 있을 때의 예(AIMET supergroup 함정, Bicubic++ 8개 연산 중 6개가 float):

```
[ WARNING ] Following OPs fallback to float:
conv0, module_act_2, conv2, conv1, module_act_1, act.
```

확인 명령(2026-10-19 로그에서 차례로 `1`, `1`, 출력 없음, `1`, `생성 1개 / 건너뜀 0개`):

```bash
L=<export_root>/convert.log
grep -c '=== .* -> .* ===' $L                         # 시도한 런 수
grep -c 'Following OPs fallback to float\.$' $L       # 위 숫자와 같아야 한다
grep -A1 'Following OPs fallback to float:' $L        # 아무것도 나오지 않아야 한다
grep -c 'Successfully created cache for QCS8550' $L   # 런 수와 같아야 한다 (prepare 성공)
grep '생성 .*개 / 건너뜀' $L                           # 기대한 개수, 건너뜀 0개
```

예전 문서의 `grep -A1 "fallback to float"`는 깨끗한 경우에도 다음 로그 줄을 함께 보여 주므로 판정에 쓰지 않는다.

폴백의 가장 흔한 원인은 AIMET의 Conv+LeakyRelu supergroup이다. AIMET이 Conv 출력 양자화기를 꺼 두는데 `qairt-converter` 2.41은 그 둘을 합치지 않아
Conv 출력에 encodings가 없고, 그 op들이 float로 남는다. 저장소의 기본 quantsim config(`configs/aimet/htp_v73_no_leakyrelu_supergroup.json`)가 이 supergroup을 뺀 것이다.
`6_apply_qat.py --quantsim_config`로 내장 `htp_v73` 같은 다른 config를 주면 다시 생긴다. 자세한 내용은 [examples/README.md](../README.md).

아래 줄들은 정상적으로 쓰인 DLC를 만든 변환에도 찍혔다. 2026-10-19 서버의 Denoise 변환(결과가 출하 DLC와 내용이 같았다, 1.7)과 SR smoke 변환에 모두 있었다.

| 줄 (일부) | 단계 |
|---|---|
| `Unable to register converter supported Operation [Inverse:Version 1] ...` | converter |
| `--desired_input_shape and -d are deprecated. ...` | converter |
| `Can't simplify the model when custom ops, quantization overrides or dynamic input shapes are specified, ...` | converter |
| `WARNING_OP_VERSION_NOT_SUPPORTED: Operation LeakyRelu Not Supported. Expected operator version: [1, 6], instead got version: [16]` (Clip은 `[13]`) | converter |
| `Unused Input nodes found: []` | converter |
| `IrQuantizer: Param Quantizer should be set to symmetric for 32 bit biases. Will ignore param quantizer option: tf for biases` | converter |
| `--float_fallback flag is deprecated, use --enable_float_fallback.` | quantizer |
| `Quantization is disabled as --enable_float_fallback flag is provided Some Ops may fallback to float datatype` | quantizer |
| `Restrict_quantization_steps is only supported for --param_quantizer = symmetric or per channel/row quantization. Value will be ignored.` | quantizer |
| `Skipping Float Fallback. Float Fallback is already applied to the provided DLC.` | quantizer |
| `Performance Estimates unsupported on soc QCS8550`, `Could not find ncvt for conv cost ...`, `Selecting disabled op for ... q::crouton_to_vtcm`, `No schematic bin found to add` | prepare |

`Quantization is disabled ...`와 `Restrict_quantization_steps ... Value will be ignored.` 두 줄은 출하 레시피(`qairt-quantizer --float_fallback`에 W16 플래그)로 한 모든 변환에 찍힌다.
2026-10-19 서버 변환들과 2026-09-30 회사 PC 변환 로그에 모두 있었고, 출하 DLC도 이 레시피로 만들었다.
이 플래그가 결과 DLC를 바꾸는지는 따로 확인하지 않았다.

### 1.6 DLC 이름과 버전

| 모델 | 파일 이름 | 만드는 곳 | 보드에서 고르는 법 |
|---|---|---|---|
| Bicubic++ SR | `BicubicPP_<EO\|IR>_<W>x<H>_x<s>_<ver>.dlc` | `gen_convert_all.py` (`<W>x<H>`는 ONNX 입력에서) | `BICUBICPP_VER=<ver>` (기본 `v3`) |
| BicubicPPDenoise | `BicubicPPDenoise_EO_1280x720_<ver>.dlc` | 손으로 적은 convert 줄(1.8) | `DENOISE=<경로>`(`gst_compare_eo.sh`, `_dual`), `DENOISE_V1=<경로>`(`_denoise`) |
| SVFocusSRNet (이전 세대, 비교 타일) | `SVFocusSRNet_<EO\|IR>_<W>x<H>_x<s>_<SVFOCUS_VER>.dlc`, IR 1024만 버전 없는 `SVFocusSRNet_IR_<W>x<H>_x<s>.dlc` | 생성기가 만들지 않는다 | `SVFOCUS_VER` (기본 `v2`) |
| 이전 Denoise | `SVFocusDenoiseNet_1280x720_v2.dlc`(출력 레이어 `n4_5`), `SVFocusDenoiseNet_EO_1280x720_x1_v3c_w8a8.dlc` | | `DENOISE_V2`, `DENOISE_V3` (`_denoise`) |

- 입력 크기: EO와 IR 720은 x2 640x360, x3 428x240, x4 320x180. IR 1024는 640x512, 428x342, 320x256. Denoise는 1280x720.
  IR 1024 프로파일은 이름의 크기로만 구별된다(`x2_ir` → `640x360`, `x2_ir_1024` → `640x512`).
- 버전은 파일 이름 끝에만 있다. 보드 스크립트는 고치지 않고 `BICUBICPP_VER` / `SVFOCUS_VER`로 고른다.
- 버전에는 영문·숫자·`_`만 쓴다(`v4`, `v3_1`은 되고 `v3.1`은 안 된다). 라벨이 gst-launch 인자로 쪼개질 때 공백 등이 값을 끊기 때문에,
  보드 스크립트가 다른 글자를 `bad version name: [...]`(`_denoise`, `_dual`은 `bad version: [...]`)으로 거부한다.
- `bench_dlc.sh`는 이름에서 `_EO_` / `_IR_`와 `_<W>x<H>_`를 읽어 입력 폴더를 고른다. 이 두 부분이 없는 이름은 `입력없음`이 된다.
- `BicubicPP_` 접두어는 생성기와 모든 gst 스크립트에 박혀 있다. 다른 구조의 모델도 이 이름 규칙과 출력 레이어 `final_act`를 지키면 스크립트를 고치지 않고 돌린다. 지키지 않으면 스크립트를 고쳐야 한다.
- 보드의 출하 이름(덮어쓰지 않는다): `BicubicPP_*_v3.dlc` 9개, `BicubicPPDenoise_EO_1280x720_v1.dlc`, 그리고 위 표의 이전 세대 DLC들.

### 1.7 DLC의 md5는 재현되지 않는다

같은 입력을 같은 서버에서 두 번 변환해도 md5가 다르다(2026-10-19 측정). 원인은 두 가지다.

- QAIRT 2.41이 `dlc.metadata`의 `startOpIndex` / `endOpIndex`에 초기화하지 않은 값을 쓴다(출하 Denoise 96/0, 서버 변환 5506000/22026).
- prepare 명령의 인자가 metadata에 기록된다. 생성기 흐름에서는 그 안에 `mktemp -d` 임시 폴더 경로(`/tmp/tmp.XXXXXXXXXX/...`)가 들어가 매번 달라진다.

그래서 md5는 "같은 파일의 복사본인가"를 확인할 때만 쓴다. 예를 들어 scp 뒤 보드 파일과 로컬 파일, 또는 출하 MANIFEST의 md5 표와 비교할 때다.
변환이 재현됐는지는 md5로 판단하지 않는다.

- 내용 비교: DLC는 압축하지 않은 zip이다. `docker/accept.sh`의 `same_dlc`는 각 항목의 원래 바이트를 읽고 두 값만 가린 뒤 비교한다.
  `docker/accept.sh --only TC_DOCK_11_1`이 출하 Denoise 입력을 복사본에서 다시 변환해(임시 경로는 출하 때 값으로 고정) 출하 DLC와 비교한다.
  2026-10-19에 통과했다. 직접 만든 임의의 DLC 파일을 검사하는 명령은 아니다. accept.sh 사용법은 [docker/README.md](../../docker/README.md).
- zip의 CRC 칸이 0이라 Python `zipfile`의 `read()`는 `Bad CRC-32`로 실패한다. 목록은 `unzip -l`로 볼 수 있다.
- 다시 변환한 v3 DLC는 두 값을 가려도 `results/release_v3/dlc/`의 사본과 같을 수 없다. 사본은 손으로 prepare해서
  metadata의 `input_dlc` / `output_dlc`가 배포 이름 기준(`[BicubicPP_EO_640x360_x2_v3.dlc]` / `[BicubicPP_EO_640x360_x2_v3_htp.dlc]`)이고, 생성기 흐름은 임시 경로이기 때문이다. v3 재변환은 보드에서 출하본과 비교해 확인한다(2.3).

### 1.8 Denoise DLC 변환

출하 Denoise 스크립트(`results/denoise_release_v1/convert_all.sh`)는 생성기 본문에 convert 줄 하나를 손으로 적은 것이다:
`convert dn_bpp_half4_w16a16 t.259 55 BicubicPPDenoise_EO_1280x720_v1.dlc 16`. 복사본 폴더에서 돌린다.

```bash
W=results/convert_dn_<날짜>
mkdir -p $W/dn_bpp_half4_w16a16
cp results/denoise_release_v1/convert_all.sh $W/
cp -r results/denoise_release_v1/dn_bpp_half4_w16a16/qat $W/dn_bpp_half4_w16a16/
docker/run.sh --offline -- qairt-env bash $W/convert_all.sh > $W/convert.log 2>&1
```

- `accept.sh`의 TC_DOCK_11이 같은 복사와 실행을 한다. 결과 `$W/BicubicPPDenoise_EO_1280x720_v1.dlc`는 **출하본과 파일 이름이 같으므로** 보드에는 다른 이름으로 올린다(2.3).
- 새 Denoise 모델이면 복사한 스크립트의 convert 줄을 고친다: 런 폴더, 입력·출력 텐서 이름, DLC 이름, 비트폭(16이면 `16`, 8이면 생략).
  텐서 이름은 ONNX에서 읽는다. 생성기의 `read_graph`와 같은 읽기다(2026-10-20 출하 Denoise ONNX에서 `t.259 55`를 찍었다):
  `docker/run.sh --offline -- python3 -c "import onnx, sys; g = onnx.load(sys.argv[1]).graph; print(g.input[0].name, g.output[0].name)" <run>/qat/qat_model.onnx`
- 다른 길: `6_apply_qat.py`로 `<root>/x1_eo/qat`에 export하면 정규식이 `x1_eo`도 받아 생성기가 `BicubicPP_EO_1280x720_x1_<ver>.dlc`를 만든다.
  이 이름은 gst 스크립트의 Denoise 기본 이름과 다르므로 scp할 때 `BicubicPPDenoise_EO_1280x720_<ver>.dlc`로 바꾸거나 `DENOISE=<경로>`로 고른다. 코드로만 확인했고 돌려 보지 않았다.

### 1.9 출하 DLC 다시 올리기와 v3 다시 변환하기

보드의 출하 DLC를 되돌릴 때(호스트, 저장소 루트):

```bash
scp results/release_v3/dlc/*.dlc root@192.168.100.98:/data/dlc/
scp results/denoise_release_v1/BicubicPPDenoise_EO_1280x720_v1.dlc root@192.168.100.98:/data/dlc/
ssh root@192.168.100.98 'md5sum /data/dlc/BicubicPP_*_v3.dlc /data/dlc/BicubicPPDenoise_EO_1280x720_v1.dlc'
```

md5는 [results/release_v3/MANIFEST.md](../../results/release_v3/MANIFEST.md)의 "보드 DLC 사본" 표와
[results/denoise_release_v1/MANIFEST.md](../../results/denoise_release_v1/MANIFEST.md)의 `62b56d8f6afad86df29ee27ce4572b31`과 같아야 한다.
보드에서 gst나 bench가 그 DLC를 쓰는 중에는 덮어쓰지 않는다.

`results/release_v3/convert_all.sh`를 그대로 돌리지 않는다. prepare 단계가 없어 보드 DLC와 다른 것이 나오고, 그 폴더에서 돌리면 출하 이름의 파일이 생긴다.
prepare까지 넣어 v3를 다시 변환해 보고 싶으면(예: 서버 변환 도구를 SR로 점검할 때) 입력만 새 폴더로 복사하고 생성기로 스크립트를 새로 만든다.
출하 이름과 겹치지 않게 다른 버전 이름을 준다.

```bash
W=results/convert_v3_<날짜>
mkdir -p $W && (cd results/release_v3 && tar cf - x*_*/qat) | tar xf - -C $W
docker/run.sh --offline -- python3 examples/deployment/gen_convert_all.py $W --version v3srv
docker/run.sh --offline -- qairt-env bash $W/convert_all.sh > $W/convert.log 2>&1
# -> $W/BicubicPP_<EO|IR>_<W>x<H>_x<s>_v3srv.dlc 9개, 보드에서는 BICUBICPP_VER=v3srv
```

이 묶음은 돌려 보지 않았다. 2026-10-19에 생성기가 만드는 본문을 출하 스크립트와 비교해, 9개 convert 줄(비트폭 16 포함)이 같고 prepare 단계만 더해진다는 것을 확인했다.
작업 폴더 이름에 `release`를 넣지 않는다. `.gitignore`가 `results/*release*/`의 `*.md`, `*.sh`를 추적하므로 작업 폴더의 `convert_all.sh`가 git에 잡힌다.

---

## 2. 보드에 올리기

### 2.1 보드 배치 (98번, 2026-10-17 기록)

```
/data/gst_compare_eo.sh  gst_compare_eo_denoise.sh  gst_compare_eo_dual.sh
/data/gst_compare_eo4k.sh  gst_compare_ir.sh  bench_dlc.sh      <- 10-17에 다시 복사, md5가 저장소와 같았다
/data/dlc/     모든 DLC를 한 폴더에 평평하게 (이름에 모델·도메인·크기·배율·버전)
/data/bench/   make_bench_raw.py 산출물
/data/old_scripts_20261013/   그 전에 쓰던 스크립트 백업, capture_eo_raw.sh, cma_watch.py
```

2026-10-19에는 다시 확인하지 못했다. 저장소 쪽 스크립트는 10-17 이후 바뀌지 않았다. 보드와 저장소가 같은지는 이렇게 본다:

```bash
md5sum examples/deployment/*.sh
ssh root@192.168.100.98 'md5sum /data/*.sh'
```

보드 시계는 배터리가 없어 켤 때 2000년으로 시작한다. 녹화 폴더 이름에 시각이 들어가므로 먼저 맞춘다:
`ssh root@192.168.100.98 "date -s '$(date '+%Y-%m-%d %H:%M:%S')'"` (또는 녹화 때 `STAMP=`를 준다).

### 2.2 스크립트를 고쳤을 때

```bash
scp examples/deployment/gst_compare_*.sh examples/deployment/bench_dlc.sh root@192.168.100.98:/data/
```

- 실행 중인 스크립트 파일을 덮어쓰지 않는다. bash는 실행하면서 파일을 이어 읽으므로, 끝날 때 엉뚱한 줄이 실행될 수 있다.
- `gst_compare_eo_denoise.sh`는 git에 실행 권한 없이(100644) 들어 있다. 새로 복사했으면 보드에서 `chmod +x`하거나 `bash gst_compare_eo_denoise.sh`로 시작한다. 다른 다섯 개는 100755다.
- 복사한 뒤 2.1의 md5 비교를 다시 한다.

### 2.3 새로 변환한 DLC 확인 체크리스트

1. **이름을 정한다.** 출하 이름(1.6 끝)과 겹치지 않게 한다. SR은 새 `--version`(예: `v4`, `smoke`)으로 이미 겹치지 않는다.
   출하본과 같은 이름이 나오는 경우(Denoise 재변환, `--version` 없이 만든 v3)는 이름을 바꿔 올린다(예: `BicubicPPDenoise_EO_1280x720_v1_server.dlc`).
2. **보드에 그 이름이 없는지 본다.** `ssh root@192.168.100.98 'ls -la /data/dlc/<이름>'`가 `No such file`이어야 한다.
3. **올리고 md5를 맞춘다.**
   ```bash
   scp <로컬.dlc> root@192.168.100.98:/data/dlc/<이름>
   md5sum <로컬.dlc>; ssh root@192.168.100.98 'md5sum /data/dlc/<이름>'
   ```
4. **지연을 출하본과 같은 세션에서 잰다(3절).** 필터를 새 DLC와 같은 크기의 출하 DLC가 함께 걸리게 준다.
   예: `ssh root@192.168.100.98 'cd /data && ./bench_dlc.sh EO_640x360_x2'`는 `..._x2_v3`와 `..._x2_smoke`를 한 번에 잰다. `입력없음`이 나오면 3.1로 입력을 만든다.
5. **화면 비교(소유자).** gst 스크립트는 한 번에 Bicubic++ 버전 하나만 보여 준다. 그래서 같은 인자로 새 버전과 출하 v3를 차례로 돌리고, 둘 다 `RECORD=1`로 남겨 `mosaic.mp4`와 `models.txt`(DLC md5 포함)를 비교한다.
   ```bash
   cd /data && RECORD=1 BICUBICPP_VER=<ver> ./gst_compare_eo.sh 2
   cd /data && RECORD=1 ./gst_compare_eo.sh 2                          # 출하 v3
   cd /data && RECORD=1 BICUBICPP_VER=<ver> ./gst_compare_ir.sh 4 720   # IR이면
   ```
   Denoise는 한 모자이크에 두 줄로 나란히 본다(행 이름은 영문·숫자·`_`·`[]`, 두 줄의 DLC 경로는 달라야 한다):
   ```bash
   cd /data && ROWS="v1 v3" \
       DENOISE_V3=/data/dlc/BicubicPPDenoise_EO_1280x720_v1.dlc DENOISE_V3_NAME=shipped_v1 \
       DENOISE_V1=/data/dlc/<새 Denoise>.dlc DENOISE_V1_NAME=new_v1 \
       bash gst_compare_eo_denoise.sh 2
   ```
6. **기록한다.** 지연(프로파일 이름과 함께), 화면 판정, 보드에 올린 파일의 md5를 실험 기록이나 릴리스 MANIFEST에 남긴다. 형식은 [docs/experiments/README.md](../../docs/experiments/README.md).

예: 2026-10-20 서버에서 다시 변환한 Denoise DLC `results/docker_accept_20261020_final/dn_convert1/BicubicPPDenoise_EO_1280x720_v1.dlc`
(md5 `83241334f5342158a8a5ba77a97650be`, 두 값을 가리면 출하본과 같음)는 아직 보드에서 확인하지 않았다. 1번 예시 이름으로 올려 5번 Denoise 명령으로 비교하면 된다.

---

## 3. 속도 측정 — `make_bench_raw.py` + `bench_dlc.sh`

### 3.1 입력 만들기 (컨테이너)

`snpe-net-run`은 헤더 없는 RAW 텐서를 읽으므로 입력 크기마다 폴더가 하나씩 필요하다.

```bash
docker/run.sh --offline -- python3 examples/deployment/make_bench_raw.py --from-results <export_root> --out tmp/bench
scp -r tmp/bench root@192.168.100.98:/data/          # 호스트 -> /data/bench
```

| 옵션 | 기본값 | 뜻 |
|---|---|---|
| `--out` | `bench` | 출력 폴더(저장소 루트 기준). 기본값 `bench/`는 git이 무시하지 않으므로 `tmp/bench`(무시됨)를 준다 |
| `--frames` | `8` | 크기마다 RAW 개수 |
| `--domains` | `EO IR` | 만들 도메인 |
| `--from-results ROOT` | 없음 | 크기를 `ROOT/x*_*/qat/qat_model.onnx`에서 읽는다. 없으면 내장 표(출하 SR 크기 전부 + Denoise `EO_1280x720`) |
| `--eo_src` | `/mnt/data_server/etc/jshong/SuperResolution/Pretrained_Dataset/val/HR` | EO 원본 이미지 |
| `--ir_src` | `/mnt/data_server/etc/jshong/SuperResolution/Finetunning_Dataset/val/HR` | IR 원본 이미지 |

- 출력: `<out>/<DOM>_<W>x<H>/0.raw ...`(float32 NHWC RGB, [0,1]), `input_list.txt`(bench 루트 기준 상대 경로), `<out>/manifest.txt`.
- 크기: `--from-results results/release_v3`이면 약 128 MB, 내장 표면 약 217 MB. 명령 한 줄로 언제든 같게 다시 만들어지므로 저장소에 두지 않는다.
- `--from-results`는 SR 런 폴더(`x<s>_<eo|ir>[_1024]`)만 읽는다. Denoise(1280x720) 입력은 `--from-results` 없이 만든다.
- 원본 이미지를 sshfs 마운트에서 읽어 느리다.

### 3.2 재기 (보드)

```bash
ssh root@192.168.100.98 'cd /data && ./bench_dlc.sh <필터>'
```

인자는 DLC 파일 이름의 부분 일치 필터다(`v4`, `IR`, `x3`, `640x512`, `EO_640x360_x2` ...). 출력 한 줄은 이렇게 생겼다:

```
BicubicPP_IR_640x360_x2_v4                    2.048 ms      488.3 fps   (2.041~2.056)
```

| 변수 | 기본값 | 뜻 |
|---|---|---|
| `PERF` | `burst` | 전력 프로파일 하나, 또는 `all`(11개 전부) |
| `REPEATS` | `5` | 프로세스를 새로 띄워 평균내는 횟수 |
| `DURATION` | `5` | 한 번 실행에서 입력을 반복하는 초 |
| `RUNTIME` | `dsp` | `dsp` \| `cpu` \| `gpu` |
| `MODEL_DIR` | `/data/dlc` | 잴 DLC 폴더 |
| `BENCH` | `/data/bench` | RAW 입력 폴더. 출력 텐서는 그 아래 `out/`에 쓰고 끝나면 지운다(따로 정할 수 없다) |

| 출력 | 뜻 |
|---|---|
| `입력없음` | 이름에 `_EO_`/`_IR_`나 `_<W>x<H>_`가 없거나, 그 크기의 입력 폴더가 `BENCH`에 없다 |
| `실패` | `snpe-net-run`이 실패했다. 마지막 실행 로그는 보드의 `/tmp/bench_run.log` |
| `입력 폴더 없음: <BENCH> ...` | `BENCH` 폴더가 없다. 3.1로 만들어 올린다 |

평균이 두 단인 이유: 한 번의 실행은 그 안에서 이미 여러 추론을 평균낸다. 그것만으로는 프로세스 시작, DSP 초기화, 발열 상태에 따른 실행 간 차이가 보이지 않아 실행 자체를 `REPEATS`번 반복하고 min~max를 옆에 찍는다. 그 폭이 넓으면 평균을 소수점 셋째 자리까지 믿지 않는다.

### 3.3 전력 프로파일과 숫자 읽는 법

`PERF=all`이 도는 11개: `burst high_performance sustained_high_performance balanced low_balanced default system_settings power_saver low_power_saver high_power_saver extreme_power_saver`.
DLC 하나에 프로파일마다 `REPEATS × DURATION`(기본 25초)씩 걸리므로 `all`은 DLC 하나에 적어도 약 5분이다.

릴리스 기록의 두 숫자는 이렇게 잰다.

| 기록 이름 | 명령 |
|---|---|
| burst | `./bench_dlc.sh <필터>` (기본, 5 × 5초) |
| 1분 | `PERF=sustained_high_performance DURATION=60 ./bench_dlc.sh <필터>` (5 × 60초) |

- 2026-10-12 보드 98에서 1분 값이 burst보다 23~46 % 높았다. 두 측정 모두 NSP가 약 90 °C였으므로 차이는 주로 프로파일 탓이고 열 탓이 아니다. 그래서 지연을 적을 때는 항상 burst인지 1분인지 함께 적는다. 30 fps 카메라 경로는 프레임마다 짧게 몰아 도는 부하라서 기본값이 burst다.
- 같은 그래프도 세션이 바뀌면 약 2 ms(약 7 %) 흔들렸다. 7 % 아래 차이는 잡음으로 본다.
- 발열은 숫자를 크게 바꾼다. 10-06 보드가 87.6 °C일 때 모든 모델이 출시 기록보다 약 46 % 느렸고, 같은 DLC가 식은 보드에서 7.5 ms, 뜨거운 보드에서 17.3 ms였다.
- 그래서 버전 비교는 절대값이 아니라 같은 세션에서 함께 잰 기준 DLC(같은 크기의 출하 DLC)에 대한 비율로 한다. 보드를 식힌 뒤(NSP 75 °C 이하) 같은 필터로 여러 번 돌려 라운드마다 비율을 보고, 한 묶음에서 마지막에 도는 DLC가 가장 느리게 나온다는 점을 감안한다.
- gst 비교 스크립트와 같이 돌리지 않는다. DSP를 나눠 쓰고, gst 스크립트는 `snpe-net-run`이 보이면 시작을 거부한다.
- Ctrl-C는 전체를 멈추고(종료 코드 130) `out/`을 지운다.
- 측정 결과는 해당 릴리스 `MANIFEST.md`의 성능 절에 남긴다.

---

## 4. 화질 비교 — `gst_compare_{eo,eo4k,ir,eo_denoise,eo_dual}.sh` (소유자)

보드에서 카메라를 열어 같은 크롭을 여러 업스케일러로 동시에 돌리고, 결과를 모자이크 한 화면으로 합쳐 RTSP 하나(포트 20000, `gst_compare_eo4k.sh`만 30000)로 내보낸다.
모든 타일이 같은 카메라 프레임이라 회의실에서 같은 순간을 나란히 본다. 타일 왼쪽 위에 이름이 노란 글자로 찍힌다.

```bash
# 보드
cd /data && ./gst_compare_ir.sh 4 720         # 인자: scale [2|3|4], profile [720|1024]
cd /data && ./gst_compare_eo.sh 2             # 인자: scale
cd /data && ./gst_compare_eo4k.sh 3
cd /data && BICUBICPP_VER=v4 ./gst_compare_ir.sh 4 1024

# 맥: 배너가 찍어 주는 명령 그대로 (gst_compare_eo4k.sh는 :30000)
open -n -a VLC --args --rtsp-tcp rtsp://192.168.100.98:20000
```

종료는 Ctrl-C. 한 번에 하나만 돌린다. 다른 `gst-launch-1.0`, 같은 카메라를 잡은 파이프라인, `snpe-net-run`(보드 평가, `bench_dlc.sh`)이 떠 있으면
그 `ps` 줄을 보여 주고 시작을 거부한다(카메라 충돌, DSP 나눠 쓰기). `gst_compare_ir.sh`만 `FORCE=1`이 있다(아래).

### 타일 배치

타일은 1280x720 (IR 1024 프로파일은 1280x1024).

```
x2, x4  2560x1440 (IR 1024: 2560x2048)        x3  1280x1440 (IR 1024: 1280x2048)
+---------------------+-----------------+     +---------------------+
| 앵커 (단순 업스케일) | camera          |     | 앵커 (단순 업스케일) |
+---------------------+-----------------+     +---------------------+
| SVFocusSRNet_<ver>  | Bicubic++_<ver> |     | Bicubic++_<ver>     |
+---------------------+-----------------+     +---------------------+
```

x3에는 SVFocusSRNet이 없으므로 앵커 위, Bicubic++ 아래 두 칸만 쓴다.

| 스크립트 | 앵커 (라벨) | camera 타일 | 크롭 |
|---|---|---|---|
| `gst_compare_eo.sh` | `qtivtransform`: Denoise된 크롭을 qtivtransform으로 확대 | 1920x1080 전체를 **Denoise 없이** 1280x720으로 | 1920x1080 → 가운데 1280x720 → Denoise → 가운데 SR 입력 |
| `gst_compare_eo4k.sh` | `videoscale`: CPU videocrop → videoscale 4-tap | 3840x2160 전체를 1280x720으로 | 4K 가운데를 원래 화소 그대로 (640x360이면 4K 면적의 1/36, 화각 좁음) |
| `gst_compare_ir.sh` | `qtivtransform`: 크롭 + bilinear 확대 (SR 없는 하드웨어 줌) | 640x512 전체: 720은 900x720 가운데 + 좌우 여백, 1024는 1280x1024 꽉 | 640x512 가운데 |

- 앵커는 보드가 SR 없이 줌할 때 쓰는 하드웨어 스케일러(qtivtransform, bilinear)다. IR은 2026-10-06에 CPU videoscale 4-tap에서 qtivtransform으로 바꿨고
  (모델 칸과 같은 크롭 + GBM 형식 지정, 보드에서 x2·x3·x4 × 720·1024 확인) 시작 때 GBM 경고도 줄었다.
  EO4K는 4K 카메라 규칙(크롭 뒤 형식 강제 금지) 때문에 아직 CPU `videoscale`이다. 라벨은 실제 요소 이름을 쓴다.
- 앵커는 모델과 같은 크롭을 같은 배율로 키우고 같은 트림을 받는다. x3는 1280이 3으로 안 나눠져 모델이 1284폭을 내고 양옆 2화소씩 잘라 맞춘다(IR 1024 x3는 아래 2줄도).
- SR 입력 크기: x2 640x360, x3 428x240, x4 320x180. IR 1024 프로파일은 640x512, 428x342, 320x256.
- 한 번에 Bicubic++ 버전 하나만 보인다. 버전 비교는 차례로 두 번 돌리고 `RECORD=1`로 남긴다(2.3).

### 필요한 DLC (`MODEL_DIR`, 기본 `/data/dlc`)

| 스크립트 | 없으면 시작 안 함 | x2/x4에서 찾아보고 없으면 빈 타일 |
|---|---|---|
| `gst_compare_eo.sh` | `BicubicPPDenoise_EO_1280x720_v1.dlc`(`DENOISE`), `BicubicPP_EO_<W>x<H>_x<s>_<BICUBICPP_VER>.dlc` | `SVFocusSRNet_EO_<W>x<H>_x<s>_<SVFOCUS_VER>.dlc` |
| `gst_compare_eo4k.sh` | `BicubicPP_EO_<W>x<H>_x<s>_<BICUBICPP_VER>.dlc` | `SVFocusSRNet_EO_<W>x<H>_x<s>_<SVFOCUS_VER>.dlc` |
| `gst_compare_ir.sh` 720 | `BicubicPP_IR_<W>x<H>_x<s>_<BICUBICPP_VER>.dlc` | `SVFocusSRNet_IR_<W>x<H>_x<s>_<SVFOCUS_VER>.dlc` |
| `gst_compare_ir.sh` 1024 | 위와 같음 (1024 입력 크기) | `SVFocusSRNet_IR_<W>x<H>_x<s>.dlc` (버전 없음) |

- Bicubic++(EO는 Denoise도)가 없으면 아무것도 만들기 전에 `model not found: <경로> (...)`를 찍고 종료 1.
- SVFocusSRNet이 없으면 그 타일은 비워 두고 라벨에 `SVFocusSRNet_v2_DLC_missing`처럼 적는다. 빈 타일과 IR 720 camera 타일의 여백은 composer 배경색이다.
- 새 버전: `BicubicPP_<EO|IR>_<W>x<H>_x<s>_<ver>.dlc`를 `/data/dlc`에 올리고 `BICUBICPP_VER=<ver>`로 실행하면 끝이다. 라벨도 `Bicubic++_<ver>`가 된다.

### 손잡이 (환경 변수)

| 변수 | 기본값 | 뜻 |
|---|---|---|
| `BICUBICPP_VER` | `v3` | Bicubic++ DLC 버전 |
| `SVFOCUS_VER` | `v2` | SVFocusSRNet DLC 버전 (IR 1024는 버전 없는 이름이라 무시) |
| `RECORD` | `0` | `1`이면 모자이크를 파일로도 저장 (아래) |
| `STAMP` | 보드 날짜-시각 | 녹화 폴더 이름 앞부분 |
| `OUT_ROOT` | `/data` | 녹화 폴더 위치 |
| `PORT_OFFSET` | `0` | 포트를 민다 (`98` → 20098, `gst_compare_eo4k.sh`는 30098). 두 보드를 한 뷰어로 볼 때 |
| `TRANSPORT` | `rtsp` | `rtsp` = gst-rtsp-server로 TCP 송출, `udp` = `HOST`로 RTP 직접 |
| `HOST` | `192.168.100.112` | `TRANSPORT=udp`일 때 받는 쪽 |
| `NET_QUEUE` | leaky 30 버퍼 | 송신 앞 큐. 네트워크가 막히면 여기서 버려 인코더·녹화·타일이 멈추지 않는다. `NET_QUEUE=""`면 막힘 |
| `MOSAIC_BITRATE` | `24000000` | qtic2venc target-bitrate |
| `ENC_OPTS` | (없음) | qtic2venc 추가 속성 |
| `MOSAIC_LATENCY` | `33` (ms) | composer가 늦은 타일을 기다리는 시간. 모델 타일이 뒤처지면 올린다 |
| `LABEL` | `1` | `0`이면 글자 없이 |
| `CAMERA_TILE` | `1` | `0`이면 camera 타일을 비운다. `crop`(EO만)이면 전체 프레임 대신 SR 입력 영역의 카메라 픽셀을 확대해 보여 준다(Denoise·모델 없음, 조리개·3D NR 판단용, 보드 실행 기록 없음) |
| `MODEL_DIR` | `/data/dlc` | DLC 폴더 |
| `BOARD_IP` | 첫 global IPv4 | 배너의 접속 주소 (인터페이스가 여럿이면 지정) |
| `SVFOCUS_LAYER`, `BPP_LAYER` | `final_act` | 출력 레이어 이름. DLC를 prepare한 레이어와 같아야 캐시가 쓰인다 |

스크립트별:

- `gst_compare_eo.sh`: `DENOISE`(Denoise DLC 경로, 기본 BicubicPPDenoise[v1]), `DENOISE_LAYER`(`final_act`). 이전 배포본 v2는 `DENOISE=/data/dlc/SVFocusDenoiseNet_1280x720_v2.dlc DENOISE_LAYER=n4_5`.
- `gst_compare_eo4k.sh`: `DEVICE`(`/dev/video4`), `DELEGATE`(`dsp`; `gpu`는 Adreno float).
- `gst_compare_ir.sh`: `DEVICE`(`/dev/video2`), `FORCE`(`0`; `1`이면 다른 gst 작업이나 `snpe-net-run`이 돌고 있어도 시작한다. DSP를 나눠 써서 타일 fps가 떨어질 수 있고, IR 카메라를 잡고 있는 파이프라인이 있으면 `FORCE`와 상관없이 멈춘다).

### 녹화 (`RECORD=1`)

```bash
cd /data && RECORD=1 ./gst_compare_ir.sh 4 720
# -> /data/<STAMP>_gst_compare_ir_x4_720/  mosaic.mp4  models.txt  rtsp_mosaic.log
```

- 폴더: `<STAMP>_gst_compare_eo_x<s>`, `<STAMP>_gst_compare_eo4k_x<s>`, `<STAMP>_gst_compare_ir_x<s>_<profile>`. 이미 있으면 멈춘다.
- `models.txt`: 타일 위치, 모델, DLC md5, 경로 (EO는 Denoise 줄 포함). 어떤 DLC로 찍은 영상인지 이 파일로 확인한다.
- 반드시 Ctrl-C로 끝낸다. `gst-launch -e`가 Ctrl-C를 EOS로 바꿔 mp4 인덱스를 쓴다. kill로 죽이면 인덱스 없는 mp4가 남는다. `mp4mux`가 없는 보드면 `mosaic.h264`.
- 시계가 2000년이면 경고를 찍는다. 먼저 맞추거나(2.1) `STAMP`를 준다: `cd /data && RECORD=1 STAMP=20261006_140000 ./gst_compare_eo.sh 4`.

녹화하지 않을 때 RTSP 중계 로그는 `/tmp/rtsp_mosaic.log`. 이전 실행의 `gst-rtsp-server`가 포트를 잡고 있으면 시작을 거부한다
(`ps -ef | grep gst-rtsp-server`로 찾아 kill).

### 보드에서 아직 안 돌려 본 것

모자이크 출력부(qtivcomposer → qtivoverlay → qtic2venc → RTSP 중계)와 camera 타일은 2026-10-06 98번 보드에서 EO x2·x3·x4, IR x2·x3·x4 × 720·1024(2560x2048 모자이크 포함)로 돌렸다. 아래는 보드 실행 기록이 없다.

- `gst_compare_eo4k.sh` 전체(4K 카메라의 camera 타일과 앵커 출력을 composer가 받는 것 포함). 10-13에 4K 카메라로 돌린 것은 `gst_compare_eo_dual.sh`의 4K → Bicubic++ 줄뿐이다.
- `CAMERA_TILE=crop`(10-08 추가, 가짜 gst-launch로만 확인).
- EO는 모델 타일이 DSP 두 단(Denoise → SR)을 거치고 camera 타일은 거치지 않는다. 모델 타일이 눈에 띄게 늦으면 `MOSAIC_LATENCY`를 올린다.

### 디노이즈 v2 / v3 / BicubicPPDenoise[v1] 비교 — `gst_compare_eo_denoise.sh`

같은 카메라 창(1920x1080 가운데 1280x720)을 Denoise 여러 개에 넣고, 각 출력을 고른 배율의 Bicubic++로 키워 한 모자이크로 보낸다.
줄은 `ROWS`에 적은 순서(위에서 아래), 열은 배율이다. 줄 후보는 v2(이전 배포본, SVFocusDenoiseNet W8A8), v3(SVFocusDenoiseNet W8A8 v3 C, 후보였음),
v1(BicubicPPDenoise[v1], W16A16, 10-13 배포 확정, 릴리스 `results/denoise_release_v1/`)이고 기본은 `ROWS="v2 v1"`(10-13 소유자 요청)이다.
크롭·트림은 `gst_compare_eo.sh`와 같다. 스크립트 머리말의 "[v2], the deployed one", "v1 ... the W16A16 candidate"는 10-13 배포 결정 전의 표현이다.

```bash
cd /data && bash gst_compare_eo_denoise.sh                  # v2 / v1 두 줄 × x2·x3·x4 -> 3840x1440
cd /data && bash gst_compare_eo_denoise.sh 2                # v2 / v1 × x2 -> 1280x1440
cd /data && ROWS="v2 v3 v1" bash gst_compare_eo_denoise.sh  # 세 줄 -> 3840x2160
cd /data && ROWS="v1 v3" bash gst_compare_eo_denoise.sh 2   # v1 / v3 × x2 (일렁임 비교용)
```

```
+-----------------------------------------+-----------------------+-----------------------+
| DN_v2+Bicubic++_v3_x2                   | DN_v2+Bicubic++_v3_x3 | DN_v2+Bicubic++_v3_x4 |
+-----------------------------------------+-----------------------+-----------------------+
| DN_BicubicPPDenoise[v1]+Bicubic++_v3_x2 | ... x3                | ... x4                |
+-----------------------------------------+-----------------------+-----------------------+
```

| 변수 | 기본값 | 뜻 |
|---|---|---|
| `ROWS` | `v2 v1` | 보일 줄과 순서. v2, v3, v1 중에서 |
| `DENOISE_V2`, `DENOISE_V2_LAYER`, `DENOISE_V2_NAME` | `SVFocusDenoiseNet_1280x720_v2.dlc`, `n4_5`, `v2` | v2 줄 |
| `DENOISE_V3`, `DENOISE_V3_LAYER`, `DENOISE_V3_NAME` | `SVFocusDenoiseNet_EO_1280x720_x1_v3c_w8a8.dlc`, `final_act`, `v3` | v3 줄 |
| `DENOISE_V1`, `DENOISE_V1_LAYER`, `DENOISE_V1_NAME` | `BicubicPPDenoise_EO_1280x720_v1.dlc`, `final_act`, `BicubicPPDenoise[v1]` | v1 줄 |
| `MOSAIC_BITRATE` | 타일당 6 Mbit/s (2줄 × 2 3 4면 36000000) | 2x2 스크립트의 타일당 비트레이트와 같게 |
| `PROBE` | `0` | `1`이면 모자이크 앞에 identity(`probe_<번호>`)를 끼워 버퍼마다 한 줄씩 찍는다. fps 측정용 |

나머지(`BICUBICPP_VER`, `RECORD`, `STAMP`, `TRANSPORT`, `PORT_OFFSET`, `MOSAIC_LATENCY`, `LABEL` ...)는 위 표와 같다.
`ROWS`에 든 줄의 DLC가 모두 있어야 시작한다. 이름(`_NAME`)은 라벨에 들어가며 영문·숫자·`_`·`[]`만 된다.
같은 이름이나 같은 DLC를 두 줄에 쓰면 멈춘다. 다른 DLC는 `DENOISE_V1=/data/dlc/<파일>.dlc`처럼 바꾼다(2.3의 Denoise 비교).
녹화 폴더는 `<STAMP>_gst_compare_eo_denoise_x234`(배율을 이어 붙인 이름).

Denoise마다 배율 수만큼 Bicubic++가 붙어 한 HTP를 나눠 쓰므로 배율·줄이 많으면 30 fps를 못 지킨다. `PROBE=1`로 98번 보드에서 잰 값이다(같은 카메라 프레임은 한 번만 셈):

| 줄 | 인자 | 타일별 fps | 모든 타일이 받은 카메라 프레임 | 측정 |
|---|---|---|---|---|
| v2, v1 | `2` | 19.7 ~ 20.2 | 11.5 /s | 10-13 오전, 학습된 v1 DLC, 1시간 모자이크 직후 뜨거운 보드(NSP 약 85 °C) |
| v2, v1 | `2 3 4` | 10.9 ~ 11.1 | 1.3 /s | 같은 조건 |
| v2, v3, v1 | `2 3 4` | 11.7 ~ 12.0 | 0.5 /s | 10-12 밤, v1 자리에 같은 그래프의 무작위 가중치 DLC |
| v2, v3, v1 | `2` | 22.5 ~ 22.9 | 10.4 /s | 같은 조건 |
| v1, v3 | `2` | 26.5 ~ 26.9 | 23.7 /s | 같은 조건 |
| v2, v3 | `2 3 4` | 18.8 ~ 19.0 | 8.9 /s | 10-12 오후 |
| v2, v3 | `2` | 30.0 | 30 /s | 10-12 오후 |

- 보드 온도에 따라 DSP가 크게 느려지므로 위 값은 측정 때 온도에서만 맞다(3.3).
- 움직이는 장면(일렁임)을 같은 순간으로 비교하려면 배율 하나, 줄 둘로 줄인다.
- 배율이 하나(가로 1280)이면 qtivoverlay가 짧은 라벨의 상자를 짧게 잡아 끝 글자가 잘린다(`..._x`, 10-12부터 그랬음).
- 3줄 × `2 3 4`(3840x2160)는 10-12 밤 한 번 돌린 시험에서 Ctrl-C 뒤 EOS까지 끝난(mp4는 정상) 다음 gst-launch가 SIGSEGV로 멈춰 'Spinning'에 머물렀다.
  그러면 Ctrl-C를 한 번 더 누르거나 찍힌 PID를 kill한다. 남아 있으면 다음 실행이 카메라 사용 중으로 거부된다.

### EO 센서 두 개 동시 — `gst_compare_eo_dual.sh`

MC-G105 EO 카메라와 4K USB 카메라를 한 파이프라인에 열어 한 모자이크로 보낸다(포트 20000).
위 줄은 MC-G105 → Denoise → Bicubic++, 아래 줄은 4K → Bicubic++만(4K 센서는 Denoise 없음, 소유자 10-13), 열은 배율이다.
기하는 위 줄이 `gst_compare_eo_denoise.sh`, 아래 줄이 `gst_compare_eo4k.sh`와 같다(둘 다 원래 화소, 렌즈가 달라 두 줄의 화각은 다르다).

```bash
cd /data && ./gst_compare_eo_dual.sh            # x2·x3·x4 -> 3840x1440
cd /data && ./gst_compare_eo_dual.sh 2          # x2만 -> 1280x1440
cd /data && RECORD=1 STAMP=20261013_140000 ./gst_compare_eo_dual.sh   # 보면서 두 센서 원본 저장
```

```
+----------------------------------------------+---------------------+---------------------+
| EO_DN_BicubicPPDenoise[v1]+Bicubic++_v3_x2   | ..._x3              | ..._x4              |
+----------------------------------------------+---------------------+---------------------+
| EO4K_Bicubic++_v3_x2                         | EO4K_..._x3         | EO4K_..._x4         |
+----------------------------------------------+---------------------+---------------------+
```

| 변수 | 기본값 | 뜻 |
|---|---|---|
| `DENOISE`, `DENOISE_LAYER`, `DENOISE_NAME` | `BicubicPPDenoise_EO_1280x720_v1.dlc`, `final_act`, `BicubicPPDenoise[v1]` | 위 줄 Denoise. 이전 배포본 v2로 보려면 `DENOISE=/data/dlc/SVFocusDenoiseNet_1280x720_v2.dlc DENOISE_LAYER=n4_5 DENOISE_NAME=v2` |
| `DEVICE` | `/dev/video4` | 4K USB 카메라 |
| `RECORD`, `STAMP`, `OUT_ROOT` | `0`, 보드 날짜-시각, `/data` | 원본 저장 (아래) |
| `BICUBICPP_VER`, `MOSAIC_BITRATE`, `PROBE`, `TRANSPORT`, `PORT_OFFSET` ... | 위 표와 같음 | `MOSAIC_BITRATE`는 타일당 6 Mbit/s |

`RECORD=1`이면 모자이크는 저장하지 않고 두 센서가 내보낸 그대로만 `<OUT_ROOT>/<STAMP>_gst_compare_eo_dual_x<배율>/`에 쓴다.

| 파일 | 내용 | 읽는 법 |
|---|---|---|
| `eo_1920x1080.nv12` | MC-G105 프레임, NV12 1920x1080, 한 프레임 3,110,400바이트를 이어 붙임 | P1 도구와 같은 형식(`score_eo_chart.read_frame`), 또는 `ffmpeg -f rawvideo -pix_fmt nv12 -s 1920x1080 -i ...` |
| `eo4k_3840x2160.mjpeg` | 4K 카메라의 JPEG 프레임을 받은 그대로 이어 붙임 | `ffmpeg -f mjpeg -i eo4k_3840x2160.mjpeg ...` |
| `info.txt` | 시작 시각, 카메라 설정, 쓴 모델과 md5 | |

- 저장 갈래는 프레임을 버리지 않는다. 저장이 못 따라가면 그 카메라를 붙잡는다. NV12만 초당 약 93 MB이고 MJPEG가 더해진다
  (10-13 시험: 25초에 NV12 2.21 GB, MJPEG 327 MB). 배너가 남은 용량을 찍는다. 끝낼 때는 Ctrl-C(EOS로 두 파일 모두 프레임 단위로 끝남).
- 두 카메라는 각자 시계로 돌아서, 위아래 줄과 두 파일은 프레임 단위로 같은 순간이 아니다.

10-13 보드 98 실측(`PROBE=1`, `2 3 4`, 식은 보드, 저장 켠 상태): 위 줄 타일 29.5~29.9 fps, 아래 줄 타일 24.0~24.8 fps
(4K MJPEG 디코드와 변환이 CPU에서 돈다). 저장 파일은 NV12 712프레임(30 fps 전부), MJPEG도 30 fps로 읽혔다.

---

## 5. 새 버전 한 바퀴 (v4 예)

학습한 체크포인트에서 보드까지. 학습과 평가는 [tools/README.md](../../tools/README.md), export 옵션은 [examples/README.md](../README.md), 릴리스 기록은 [docs/experiments/README.md](../../docs/experiments/README.md)가 다룬다.
모두 저장소 루트에서 실행한다.

```bash
R=results/release_v4     # 이름에 release가 들어가면 MANIFEST.md와 *.sh가 git에 잡힌다

# 1) export. EO는 배율마다 한 번(GPU), 크기는 1.1 표
docker/run.sh --gpu <N> --name v4_x2_eo_export --log $R/x2_eo_export.log -- \
    python3 examples/aimet_quantization/6_apply_qat.py \
    --config <EO x2 config> --data_config configs/data/bicubicpp/sr_finetune_eo_phase-2.yaml \
    --checkpoint <EO x2 체크포인트> --output_dir $R/x2_eo/qat \
    --mmp_config configs/aimet/mmp_noprune_w16a16.json --param_bw 16 --output_bw 16 --qat_epochs 0 \
    --width 640 --height 360 --device 0
#    --name은 분리 실행이라 바로 돌아온다. 2)로 가기 전에 $R/x2_eo_export.log 끝의 [run.sh] exit 0을 확인한다
#    IR은 배율마다 한 번, 두 프로파일 함께(CPU)
docker/run.sh --offline -- env SCALE=<s> results/release_v3/eval/export.sh <IR x<s> 체크포인트> $R [config]

# 2) 변환 스크립트 생성, 3) 변환
docker/run.sh --offline -- python3 examples/deployment/gen_convert_all.py $R --version v4
docker/run.sh --offline -- qairt-env bash $R/convert_all.sh > $R/convert.log 2>&1

# 4) 로그 확인 (1.5의 grep 다섯 줄)

# 5) 호스트: 보드에 올리기 (_v4는 출하 이름과 겹치지 않는다)
scp $R/BicubicPP_*_v4.dlc root@192.168.100.98:/data/dlc/

# 6) 보드: 지연은 같은 크기의 v3와 같은 세션에서
ssh root@192.168.100.98 'cd /data && ./bench_dlc.sh EO_640x360_x2'
#    화면(소유자)
#    cd /data && RECORD=1 BICUBICPP_VER=v4 ./gst_compare_ir.sh 4 720
```

- Phase-2 런이 기준선을 넘지 못해 `best.pth`가 없으면 `best_attempt.pth`나 `last.pth`를 export한다. 2026-10-19 smoke는 `last.pth`를 썼다.
- 일부 배율만 다시 학습한 버전은 export한 런만 넣거나 생성기에 `--scales` / `--domains`를 준다.
- 기록용으로 체크포인트를 `$R/checkpoints/`에 복사하고, 보드에 올린 DLC를 `$R/dlc/`에 둔다(v3 선례).

---

## 6. PNG 폴더를 보드 SR 경로로 돌리기 (영상 작업)

카메라 대신 PNG 프레임 폴더를 배포 SR 경로에 넣어 출력 프레임을 받는 스크립트는 저장소 밖, 데이터 서버에 있다.

| 폴더 | 스크립트 |
|---|---|
| `/mnt/data_server/etc/jshong/SuperResolution/results/261006_ir_sr_video/` | `board_sr_frames.sh`(보드), `pull_frames.sh`(호스트, 기본 보드 폴더 `/data/jshong/srvideo`), `make_videos.sh`, `compose_videos.sh` |
| `/mnt/data_server/etc/jshong/SuperResolution/results/261013_eo_sr_video/` | `board_seg_frames.sh`(보드, Denoise는 BicubicPPDenoise[v1]) 등 |

- 영상과 프레임 산출물은 저장소의 `results/`에 만들지 않는다. 한 작업이 58 GB까지 커져 서버 디스크를 채운 적이 있다. 위 데이터 서버 경로나 로컬 디스크 `/mnt/data2`를 쓰고, 새 작업은 쓸 위치부터 정한다.
- `261006`의 `board_sr_frames.sh`는 머리말이 지금은 없는 옛 gst 스크립트를 가리키고, EO `dn` 작업의 기본 `DENOISE`가 이전 배포본 v2(`n4_5`)다. v1으로 돌리려면 `DENOISE`와 `DENOISE_LAYER=final_act`를 준다.
- 두 `qtivtransform`을 시스템 메모리로 이으면 한 가지 색으로 칠해진 프레임이 나온다. 중간 caps는 `video/x-raw(memory:GBM)`이어야 한다.
- DSP 런타임이 마지막 프레임 뒤 정리 단계에서 멈출 때가 있다(SVFocusSRNet x4, IR 1024의 Bicubic++). 스크립트는 프레임이 다 생기면 `EXIT_GRACE`초(기본 30) 뒤 프로세스를 끝내고, 성공 여부는 프레임 개수로 판단한다.
- 보드 NIC는 100 Mbps(약 11 MB/s)라서 끝난 작업을 다음 작업이 도는 동안 받아 온다(`pull_frames.sh`).

옛 보드 스크립트(260914_*, 260918_*, `bench_parallel.sh`, `capture_sr_tensors.sh`, `make_snpe_raw_inputs.py` 등)는 `3fe100d`에서, `rtsp_serve_eo4k.sh`는 `a4475e8`에서 지웠다.
필요하면 git에서 꺼낸다. 예: `git show 3fe100d^:examples/deployment/capture_sr_tensors.sh`(`examples/evaluation/score_sr_tensors.py`의 입력을 보드에서 받는 스크립트).

---

## 7. 이 폴더의 테스트

```bash
docker/run.sh --offline -- env PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q -rs -p no:cacheprovider \
    tests/test_gen_convert_all.py tests/test_board_scripts.py
```

- 통과하면 `14 passed`이고 SKIPPED 줄이 없다(생성기 9개, 보드 스크립트 5개). 이 두 파일만 따로 돌려 보지는 않았다. 2026-10-19 `docker/accept.sh`의 전체 pytest(3499 passed, 0 skipped)에 들어 있었다.
- `test_denoise_release_script_unchanged`는 `results/denoise_release_v1`의 export가 없는 기계에서는 실패하지 않고 건너뛴다.
- 보드 스크립트 테스트는 가짜 `gst-launch-1.0` / `snpe-net-run`으로 시작 거부(`gst_compare_eo.sh`, `_eo4k`, `_ir`), IR `FORCE=1`, `bench_dlc.sh`의 `out/` 정리만 본다. `gst_compare_eo_denoise.sh`와 `_dual`은 테스트가 없다.
