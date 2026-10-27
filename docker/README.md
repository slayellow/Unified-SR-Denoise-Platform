# docker/ — 작업 환경

학습, 평가, inference, AIMET export, DLC 변환, pytest, 영상 작업은 모두 Docker 이미지 `usdp` 안에서 돈다.
편집(VSCode), git, 보드 ssh/scp, TensorBoard 보기는 호스트에서 한다.
호스트의 `python3`(3.8.10)는 이 저장소의 작업 환경이 아니다. 다른 프로젝트용 torch 2.4.1만 있고 aimet은 없다.

이 문서는 사용법만 다룬다. 설계 근거와 시험 기록은 모듈 문서에 있다.

| 문서 | 내용 |
|---|---|
| [DockerEnv](../docs/modules/20261019-MOD-DockerEnv.md) | 이미지, `qairt-env`, `accept.sh`, 패키지 목록, 옛 컨테이너 이관 |
| [DockerRun](../docs/modules/20261019-MOD-DockerRun.md) | `run.sh`, `entry.sh`, 차단 조건과 그 시험 |
| [README](../README.md) | 새 모델을 보드까지 보내는 전체 흐름 |
| [tools/README.md](../tools/README.md) | train·evaluate·inference·export 옵션, 재개 규칙 |
| [examples/deployment/README.md](../examples/deployment/README.md) | 변환(`convert_all.sh`), DLC, 보드 |
| [configs/README.md](../configs/README.md) | config와 데이터 경로 |

## 1 파일

| 파일 | 도는 곳 | 하는 일 | 고치면 이미지를 다시 빌드하나 |
|---|---|---|---|
| `run.sh` | 호스트 | 작업 컨테이너 하나를 띄운다 | 아니오 |
| `entry.sh` | 컨테이너 안 | GPU 확인, 백그라운드 로그, 명령 실행. 저장소 마운트에서 바로 실행된다 | 아니오 |
| `build.sh` | 호스트 | 이미지 `usdp:<YYYYMMDD>`와 `usdp:latest`를 만든다 | — |
| `Dockerfile` | 빌드 | 이미지 정의 | 예 |
| `qairt-env` | 컨테이너 안 `/usr/local/bin/qairt-env` | QAIRT 변환 환경에서 명령 하나를 실행한다 | 예(빌드 때 복사) |
| `qairt-requirements.txt` | 빌드 | QAIRT 전용 venv의 고정 목록(`pip freeze --all`, 7줄) | 예 |
| `apt-versions.txt` | 시험 | 받아들인 이미지의 apt 패키지 목록(319개). `TC-DOCK-01-6`이 실제 이미지와 비교한다 | 아니오(기록) |
| `accept.sh` | 호스트 | 환경 합격 확인 15건 | 아니오 |
| `../requirements.txt` | 빌드 | 시스템 python3의 패키지 178개(`==` 고정) | 예 |
| `../.dockerignore` | 빌드 | 빌드 컨텍스트를 `requirements.txt`와 `docker/`로 제한한다 | — |

저장소의 코드(`src/`, `tools/`, `examples/`, `configs/`)는 이미지에 들어가지 않는다. 컨테이너는 저장소를 마운트해서 쓰므로, 코드를 고친 뒤 다시 빌드할 필요가 없다.

## 2 무엇이 어디서 도는가

| 호스트(jshong) | 작업 컨테이너(`docker/run.sh`) |
|---|---|
| VSCode 편집, git, 보드 ssh/scp, TensorBoard 보기, `docker/build.sh`, `docker/accept.sh` | 학습, 평가, inference, AIMET export, `gen_convert_all.py`, DLC 변환(`qairt-env`), pytest, ffmpeg 영상 작업 |

- **TensorBoard**는 호스트의 `~/.local/bin/tensorboard`(2.14.0)로 `<work_dir>/logs`를 직접 연다(예: `~/.local/bin/tensorboard --logdir <work_dir>/logs`). event 파일은 저장소 마운트를 통해 호스트에 있다. 컨테이너는 포트를 열지 않으므로 컨테이너 안에서 띄운 TensorBoard에는 접속할 수 없다.
- **영상.** ffmpeg 7.0.2 정적 바이너리가 이미지의 `/usr/local/bin/ffmpeg`에 있다. 영상 스크립트(`make_videos.sh`, `compose_videos.sh` 등)는 `/mnt/data_server/etc/jshong/SuperResolution/results/<폴더>/`에 있고 컨테이너에서 돈다. 보드에서 프레임을 받는 `pull_frames.sh`는 ssh를 쓰므로 호스트에서 돈다. 영상 출력은 용량 때문에 저장소 `results/`가 아니라 그 `/mnt/data_server/.../results/` 아래에 둔다.
- **옛 컨테이너 `sr`**(이미지 `sr/unified-platform:v2.0`, root 실행)는 이 흐름에서 쓰지 않는다. 지우는 시점은 소유자가 정한다([DockerEnv §2.7](../docs/modules/20261019-MOD-DockerEnv.md)).

## 3 `docker/run.sh`

### 3.1 형식과 옵션

```text
docker/run.sh [--gpu N] [--name NAME [--log FILE]] [--mount SRC[:DST[:ro]]]... [--offline] [--force] -- 명령...
```

`--` 앞은 `run.sh`의 옵션이고, `--` 뒤는 컨테이너에서 실행할 명령이다. 명령의 인자(공백, 따옴표, 한글, `--force` 같은 문자열 포함)는 그대로 전달된다.

| 옵션 | 뜻 | 규칙 |
|---|---|---|
| `--gpu N` | 호스트 GPU N 한 장을 붙인다. 컨테이너 안에서는 그 GPU가 `cuda:0`이다 | 0 이상의 숫자 하나(`2`). `02`, `2,3`, `all`, `GPU-<uuid>`, `-1`은 거부. 한 작업에 한 번만 |
| `--name NAME` | 백그라운드로 띄운다. 컨테이너 이름은 `usdp-NAME`, 로그 기본값은 `<저장소>/logs/NAME.log` | `[A-Za-z0-9][A-Za-z0-9_.-]*` |
| `--log FILE` | 백그라운드 로그 위치를 바꾼다 | `--name`과 함께만. 컨테이너가 같은 경로에 쓸 수 있는 곳(저장소, `/mnt`, 같은 경로·쓰기로 붙인 `--mount`) 아래 |
| `--mount SRC[:DST[:ro]]` | bind 마운트를 더한다(여러 번 줄 수 있음) | SRC가 있어야 한다. DST를 생략하면 적은 경로 그대로 붙는다(symlink 경로 유지). 모드는 `ro`만. 경로에 쉼표와 `:`는 쓸 수 없다 |
| `--offline` | `--network none`. 네트워크 없이 돈다 | 받을 것이 없는 작업(pytest, export, 변환, 가중치가 캐시에 있는 학습)에 쓴다. 숨은 다운로드를 잡아낸다 |
| `--force` | "GPU 사용 중" 거부만 넘긴다 | 반드시 `--` 앞에. 다른 거부(마운트, 이미지 등)는 넘기지 못한다 |

종료 코드는 다음과 같다.

| 상황 | 종료 코드 |
|---|---|
| 포그라운드(`--name` 없음) | 명령의 종료 코드 |
| 백그라운드(`--name`) | 0 = 시작됨. 작업 결과는 로그 마지막 줄 `[run.sh] exit <코드>`를 본다 |
| 사용법 오류(`run.sh: usage:`) | 2 |
| 사전 검사 거부(`run.sh: refused:`) | 1 |
| 컨테이너 안 CUDA 거부(`entry.sh: refused:`) | 3 |

사전 검사를 통과한 뒤 docker 자체가 실패하면 docker의 오류 메시지와 종료 코드가 그대로 나온다.

### 3.2 컨테이너 안의 모습과 그 이유

| 항목 | 값 | 이유 |
|---|---|---|
| 사용자 | 호출자의 uid:gid. 이미지 라벨 `usdp.uid`·`usdp.gid`와 같아야 한다 | 2026-10-19 root가 만든 파일이 소유자의 git checkout을 막았다. 결과 파일은 항상 호출자 소유로 남는다 |
| 작업 폴더 | 저장소 루트(호스트와 같은 절대 경로) | config와 도구 기본값의 상대 경로(`checkpoints/…`, `configs/…`, `results/…`)가 호스트 저장소에 그대로 떨어진다 |
| 마운트 | 저장소(쓰기), `/mnt`(쓰기, `rslave`), 호스트 `~/.cache/torch` → 컨테이너 `~/.cache/torch` 세 곳뿐 | 홈 전체는 붙이지 않는다. 보드 ssh 키, HF 토큰, root 소유 HF 캐시가 있기 때문이다 |
| `/mnt` 전파 `rslave` | 호스트에서 sshfs를 다시 마운트하면 실행 중인 컨테이너에도 보인다 | 옛 컨테이너는 `rprivate`라서 재부팅 뒤 다시 붙인 sshfs를 보지 못했다 |
| IPC | `--ipc host` | DataLoader 워커가 공유 메모리를 최악 약 10 GB 쓴다. docker 기본 `/dev/shm`(64 MB)으로는 모자란다 |
| 프로세스 | `--init` + `TINI_KILL_PROCESS_GROUP=1` | `docker stop`과 Ctrl-C가 명령의 프로세스 그룹 전체(DataLoader 워커 포함)에 간다 |
| 네트워크 | 기본 bridge, `--offline`이면 none. 포트는 열지 않는다 | 작업이 포트를 쓰지 않는다 |
| GPU | `--gpu`가 없으면 없음(`NVIDIA_VISIBLE_DEVICES=void`) | `--gpu` 없이 띄운 작업이 남의 GPU를 잡지 않는다 |
| `NUMPY_MADVISE_HUGEPAGE=0` | 이미지 ENV(그리고 `tools/train.py`도 기본값으로 둔다) | 2026-09-10 NumPy의 hugepage 요청과 호스트 메모리 파편화가 겹쳐 학습이 6분/epoch에서 3시간/epoch로 느려졌다. 증상은 워커 CPU의 97 %가 커널 시간이고 GPU는 0 %인 상태였다 |
| `PYTHONUNBUFFERED=1` | 이미지 ENV | 백그라운드 로그에 출력이 바로 쌓인다 |
| `TZ=Asia/Seoul`, `LANG=C.UTF-8` | 이미지 ENV | 로그 시각이 KST이고 한글 출력(`convert_all.sh`의 `생성 N개 / 건너뜀 M개` 등)이 깨지지 않는다 |
| 수명 | `--rm` | 끝나면 컨테이너와 그 안의 HOME(Triton 캐시, VSCode 서버)이 사라진다. 결과는 마운트한 경로에만 남는다. Triton 커널은 GPU export마다 다시 컴파일된다(몇 초) |
| 자원 제한 | `--cpus`, `--memory` 없음 | — |

### 3.3 경로와 환경 변수 규칙

- **`run.sh`는 저장소 루트에서 부른다.** `--log`와 `--mount`의 상대 경로는 호스트 셸의 현재 폴더 기준으로 풀린다. 반대로 `--` 뒤 명령 안의 상대 경로는 컨테이너의 작업 폴더(저장소 루트) 기준이다. 다른 폴더에서 부르면 두 기준이 어긋난다.
- **호스트 `/tmp`와 Claude 스크래치패드는 컨테이너에서 보이지 않는다.** 출력(`--work_dir`, `--output_dir`, `--save_dir`, `--log`)은 저장소(`results/`, `checkpoints/`) 또는 `/mnt` 아래에 둔다. 저장소 밖 데이터는 `--mount`로 붙인다.
- **호스트의 환경 변수는 넘어가지 않는다.** `-- env SCALE=4 bash results/release_v3/eval/export.sh …`처럼 명령 앞에 `env`로 준다.
- **`CUDA_VISIBLE_DEVICES`를 쓰지 않는다.** GPU는 `--gpu`로 고르고, 컨테이너 안에서는 `cuda:0`이므로 스크립트에는 `--device 0`(또는 `cuda`)을 준다. 호스트 GPU 번호를 넘기면 안 된다.
- **표준 입력.** 입출력이 모두 터미널이면 `-it`, 표준 입력이 파일·파이프·heredoc이면 `-i`로 붙는다. `docker/run.sh -- python3 - <<'EOF' … EOF`가 된다.
- **`run.sh: GPU: none (CPU run)`이 보이면 GPU가 없는 작업이다.** 학습에서 `--gpu`를 빠뜨리면 거부되지 않고 CPU로 돈다. 이 줄은 stderr에 나오고, 학습 로그에는 `Warning: CUDA not available, falling back to CPU.`와 `Using device: cpu`가 남는다.

### 3.4 `run.sh`가 읽는 호스트 환경 변수

| 변수 | 기본값 | 용도 |
|---|---|---|
| `USDP_IMAGE` | `usdp:latest` | 다른 태그로 실행할 때(예: `USDP_IMAGE=usdp:20261019 docker/run.sh …`). 새 빌드가 문제일 때 이전 날짜 태그로 되돌리는 데 쓴다 |
| `USDP_DATA_MOUNT` | `/mnt/data_server` | 시작 전에 확인하는 sshfs 마운트 위치. fstype이 `fuse.sshfs`여야 하므로 로컬 디스크(`/mnt/data2`)로 돌려 검사를 피할 수는 없다 |
| `USDP_PROBE_TIMEOUT` | `10`(초) | sshfs 읽기 탐침과 `nvidia-smi` 조회의 제한 시간. 0보다 큰 초 단위 숫자(`0.5`처럼 소수 가능)여야 하고, 아니면 사용법 오류(종료 2) |

### 3.5 자주 쓰는 형태

```bash
# Detached GPU training; the log is logs/<run>.log
docker/run.sh --gpu <N> --name <run> -- python3 tools/train.py --config <config> --work_dir <work_dir>

# Same, with the log next to the outputs
docker/run.sh --gpu <N> --name <run> --log results/<폴더>/<run>.log -- python3 tools/train.py --config <config> --work_dir results/<폴더>/<run>

# Foreground evaluation (output on the terminal)
docker/run.sh --gpu <N> -- python3 tools/evaluate.py --config <config> --checkpoint <ckpt> --save_dir <save_dir>

# W16A16 export on the GPU (inside the container the GPU is cuda:0)
docker/run.sh --gpu <N> -- python3 examples/aimet_quantization/6_apply_qat.py ... --device 0

# Generate the conversion script (main Python, not qairt-env), then convert
docker/run.sh --offline -- python3 examples/deployment/gen_convert_all.py <export_root> --version <ver>
docker/run.sh --offline -- qairt-env bash <export_root>/convert_all.sh > <export_root>/convert.log 2>&1

# Full test suite, no GPU, no network (what accept.sh runs)
docker/run.sh --offline -- env PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q -rs -p no:cacheprovider

# Interactive shells
docker/run.sh -- bash
docker/run.sh --gpu <N> -- bash
docker/run.sh --offline -- qairt-env bash
```

각 도구의 옵션은 [tools/README.md](../tools/README.md), export 옵션은 [examples/README.md](../examples/README.md), 변환은 [examples/deployment/README.md](../examples/deployment/README.md)를 본다.

## 4 GPU 고르기와 나눠 쓰기

이 서버의 RTX A6000 5장(0–4번)은 다른 사용자와 k3s 파드(이 호스트는 k3s 노드다)가 같이 쓴다. `run.sh`는 고른 GPU가 다음 중 하나라도 해당하면 "사용 중"으로 보고 거부한다.

1. `nvidia-smi --query-compute-apps`에 그 GPU의 UUID로 잡힌 프로세스가 있다(다른 사용자의 학습, k3s 파드 포함).
2. 라벨 `usdp.gpu=<N>`이 붙은 실행 중 컨테이너가 있다. `--gpu`로 띄운 모든 `run.sh` 작업(포그라운드 포함)이 이 라벨을 단다. 아직 CUDA를 열지 않은 작업도 이것으로 보인다.

비어 있는 GPU는 다음으로 찾는다(호스트, 읽기만 함). `run.sh`의 1번 판정과 같은 방법이다.

```bash
APPS=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader); nvidia-smi --query-gpu=index,uuid --format=csv,noheader | while IFS=', ' read -r I U; do grep -qxF "$U" <<< "$APPS" || echo "GPU $I idle"; done
```

`GPU 2 idle`처럼 한 줄씩 나온다. 2번 판정(usdp 작업)은 다음으로 본다.

```bash
docker ps --filter label=usdp.gpu --format '{{.Names}} gpu={{.Label "usdp.gpu"}}'
```

- `--force`는 그 GPU를 쓰는 것이 자기 작업이고 메모리가 남는 것을 알 때만 쓴다(`--` 앞에). 넘기면 stderr에 `run.sh: --force: starting on GPU <N> although it is in use by <…>`가 남는다.
- 한 작업은 GPU 한 장만 쓴다. `accelerate launch` 다중 GPU는 `run.sh`의 범위 밖이다.
- 막지 못하는 경우: 검사와 시작 사이에 다른 프로세스가 GPU를 잡는 경쟁, 아직 GPU 메모리를 잡지 않은 다른 사용자의 프로세스, 시작 뒤 k3s가 같은 GPU에 파드를 배치하는 경우(k3s device plugin은 docker 작업을 모른다). 오래 도는 학습은 시작 뒤 `nvidia-smi`로 한 번 더 본다.

## 5 백그라운드 작업과 로그

`--name`을 주면 `run.sh`는 컨테이너를 `-d`로 띄우고 바로 끝난다.

```text
run.sh: started usdp-<NAME>, log <로그 절대 경로> (stop: docker stop usdp-<NAME>)
```

- 작업은 호출한 셸과 상관없이 돈다. ssh 연결이 끊기거나 로그아웃해도 계속된다.
- 명령의 표준 출력과 오류는 모두 로그 파일에 쌓이고, 끝나면 마지막 줄에 `[run.sh] exit <코드>`가 붙는다. `run.sh`가 0으로 끝났어도 작업이 성공했다는 뜻은 아니다. 결과는 이 줄로 판단한다.

| 로그 마지막 줄 | 뜻 |
|---|---|
| `[run.sh] exit 0` | 명령이 성공했다 |
| `[run.sh] exit 143` | `docker stop`(SIGTERM)으로 멈췄다 |
| `[run.sh] exit 130` | SIGINT로 멈췄다 |
| `[run.sh] exit 3` | 컨테이너 안 CUDA 확인에서 거부됐다. 바로 위 줄에 `entry.sh: refused: …` |
| 그 밖의 숫자 | 명령의 종료 코드. 위쪽 Traceback을 본다 |
| 종료 줄이 없음 | 아직 도는 중이거나, 호스트 재부팅·docker 재시작·SIGKILL로 끊겼다 |

로그 다루는 법:

```bash
tail -f logs/<run>.log                                   # follow progress
tail -n 1 logs/<run>.log                                 # "[run.sh] exit <rc>" once finished
tr '\r' '\n' < logs/<run>.log | grep -a '^\[' | tail -n 20   # tagged lines only
docker ps --filter name=usdp- --format '{{.Names}}\t{{.Status}}\tgpu={{.Label "usdp.gpu"}}'   # running jobs
```

- tqdm 진행 막대는 캐리지 리턴(`\r`)으로 줄을 덮어쓴다. `tail -f`는 문제없지만 grep은 긴 한 줄로 본다. `grep -a`와 `tr '\r' '\n'`를 쓴다.
- 같은 `--name`을 다시 쓰면 같은 `logs/<NAME>.log`에 **이어 붙는다**. 옛 `[run.sh] exit` 줄이 중간에 남으므로, 작업마다 새 이름을 쓰거나 `--log`로 새 파일을 준다.
- `docker logs usdp-<NAME>`에는 거의 아무것도 없다. 명령 출력은 로그 파일로 가기 때문이다. 컨테이너가 끝나면(`--rm`) `docker logs`도 사라진다.
- `logs/`는 git이 무시한다. 첫 `--name` 실행 때 생긴다.
- 끝나기를 기다릴 때는 `docker wait usdp-<NAME>`을 쓴다. 명령의 종료 코드를 출력한다. 이미 끝나서 컨테이너가 지워졌다면 `No such container` 오류가 나므로, 그때도 로그 마지막 줄로 판단한다.

작업을 이어 붙이는 스크립트는 없다. 앞 작업이 성공했을 때만 다음 작업을 띄우려면, 로그아웃해도 살아 있도록 호스트의 `tmux` 안에서 다음처럼 한다(이 형태는 실행 기록이 없다).

```bash
docker wait usdp-<run1>; tail -n 1 logs/<run1>.log | grep -qx '\[run.sh\] exit 0' && docker/run.sh --gpu <N> --name <run2> -- python3 tools/train.py --config <다음 config>
```

## 6 멈추기, 이어서 하기, 재부팅 뒤

- **멈추기.** `docker stop usdp-<NAME>`. SIGTERM이 프로세스 그룹 전체에 가고, 로그 끝에 `[run.sh] exit 143`이 남는다(합격 확인 `TC-DRUN-04-4`). docker는 10초 안에 끝나지 않으면 SIGKILL을 보내며, 그때는 종료 줄이 남지 않는다.
- **멈추면 잃는 것.** Trainer에는 신호 처리기가 없어서 멈출 때 따로 저장하지 않는다. 진행 중이던 epoch는 버려진다. `last.pth`는 epoch가 끝날 때마다 저장되므로 거기서 이어 간다. `last.pth`를 쓰는 도중에 멈췄다면 그 파일이 잘렸을 수 있으니 `epoch_<N>.pth`나 `best_attempt.pth`에서 이어 간다.
- **이어서 학습.** `docker/run.sh --gpu <N> --name <새 이름> -- python3 tools/train.py --resume <work_dir>/last.pth`. config는 run 폴더의 `train_config.yaml`을 읽으므로 `--config`, `--data_config`, `--work_dir`, `--lr`는 함께 줄 수 없다(`tools/train.py`가 거부). 자세한 규칙은 [tools/README.md](../tools/README.md).
- **포그라운드 작업.** 터미널에서 띄운 작업(`-it`)은 Ctrl-C로 멈춘다. 출력을 파이프(`| tee`)로 넘긴 포그라운드 작업은 Ctrl-C가 작업까지 닿지 않을 수 있다(확인 필요). 오래 걸리는 작업은 `--name`으로 띄우거나, 파이프 대신 `> 파일 2>&1`로 받는다.
- **호스트 재부팅·docker 재시작.** 작업은 죽고 다시 시작되지 않는다(`--rm`, 재시작 정책 없음). 재부팅 뒤에는 sshfs 마운트가 없으므로(fstab에 없음), root가 다시 마운트하기 전까지 `run.sh`는 CPU 작업까지 모두 거부한다(7.2의 sshfs 항목). 다시 붙인 뒤 `--resume`으로 이어 간다.

## 7 거부 메시지와 대처

### 7.1 사용법 오류(종료 2)

stderr에 사용법 한 줄과 사유 한 줄이 나온다. 컨테이너는 만들지 않는다.

```text
run.sh: usage: docker/run.sh [--gpu N] [--name NAME [--log FILE]] [--mount SRC[:DST[:ro]]]... [--offline] [--force] -- CMD...
run.sh: <사유>
```

| 사유(그대로) | 원인 | 대처 |
|---|---|---|
| `missing -- before the command` | `--`가 없다 | 명령 앞에 `--`를 넣는다 |
| `missing command after --` | `--` 뒤가 비었다 | 명령을 적는다 |
| `unknown option '<인자>' (the command goes after --)` | 모르는 옵션(오타 `--forc`, `--gpus` 등) 또는 `--` 없이 명령을 적었다 | 옵션 이름을 고치거나 명령 앞에 `--`를 넣는다 |
| `<옵션> needs a value` | `--gpu`·`--name`·`--log`·`--mount` 값이 없거나, 값이 `--`로 시작한다(`--name --force`), 또는 `--log ''` | 값을 준다 |
| `--gpu given twice (one GPU per job)` | `--gpu`를 두 번 줬다 | 한 장만 준다 |
| `--name must match [A-Za-z0-9][A-Za-z0-9_.-]*` | 이름에 공백·`/` 등이 있거나 `-`로 시작한다 | 영문·숫자·`_.-`만 쓴다 |
| `USDP_PROBE_TIMEOUT must be a positive number of seconds, got '<값>'` | `USDP_PROBE_TIMEOUT`가 0, 음수, 숫자가 아님 | `10`이나 `0.5`처럼 0보다 큰 초 |
| `--log needs --name` | 포그라운드에 `--log`를 줬다 | `--name`을 함께 주거나, 포그라운드는 `> 파일 2>&1`로 받는다 |
| `--mount needs SRC[:DST[:ro]]` | `--mount` 원본이 비었다 | 원본 경로를 준다 |
| `--mount mode must be ro, got '<모드>'` | 세 번째 칸이 `ro`가 아니다(`rw` 포함) | 쓰기 마운트는 모드를 생략한다 |

### 7.2 사전 검사 거부(종료 1)

stderr에 `run.sh: refused: <사유>`가 나온다. 컨테이너는 만들지 않는다. 검사 순서는 인자 → 도구 → 이미지 → sshfs → 경로 → GPU이다. 멈춘 sshfs 위의 경로를 건드리면 run.sh도 멈추므로, sshfs를 먼저 확인한 뒤에 `--mount`·`--log` 경로를 푼다.

| 사유(그대로, `<>`는 바뀌는 값) | 원인 | 대처 |
|---|---|---|
| `<도구> missing` | 호스트에 `docker`, `findmnt`, `timeout`, `realpath`, `awk`가 없다. `--gpu`면 `nvidia-smi`도 | 그 도구를 설치한다 |
| `image <이미지> not found; build it first with docker/build.sh` | 이미지가 없다 | `docker/build.sh`(10절). 다른 태그면 `USDP_IMAGE`를 확인한다 |
| `docker image inspect <이미지> failed: <docker 오류 첫 줄>` | docker 데몬이 안 돌거나 소켓 권한이 없다(docker 그룹이 아님). 성공했을 때 docker가 stderr에 찍는 경고(`~/.docker/config.json`을 못 읽음 등)는 무시한다 | docker 데몬 상태와 `id`의 docker 그룹을 확인한다 |
| `image <이미지> has no usdp.uid/usdp.gid/usdp.home labels; build it with docker/build.sh` | `build.sh`로 만들지 않은 이미지다(`USDP_IMAGE=sr/unified-platform:v2.0` 등) | `build.sh`로 만든 `usdp` 이미지를 쓴다 |
| `run as your own user, not root (the image runs as uid:gid <UID>:<GID>)` | root(또는 sudo)로 불렀다 | 자기 계정으로 부른다. root로 만든 파일은 소유자의 git을 막는다 |
| `image <이미지> was built for uid:gid <UID>:<GID> but the caller is <uid>:<gid>; rebuild with docker/build.sh` | 다른 계정이 만든 이미지다 | 자기 계정으로 `docker/build.sh`를 돌린다. 한 호스트의 두 계정은 12절 참고 |
| `the repository path <경로> contains a comma, which docker --mount cannot take` | 저장소 경로에 쉼표가 있다 | 쉼표 없는 경로로 옮긴다 |
| `HOME <경로> contains a comma, which docker --mount cannot take` | HOME 경로에 쉼표가 있다 | 같음 |
| `--mount source <경로> contains a comma, which docker --mount cannot take` (`--mount target …`도 같음) | `--mount` 경로에 쉼표가 있다 | 쉼표 없는 경로나 symlink를 쓴다 |
| `--mount source <경로> does not exist` | `--mount` 원본이 없다(상대 경로는 호스트 현재 폴더 기준) | 경로를 고치거나 저장소 루트에서 부른다 |
| `--log <경로> is outside the mounted paths, so the container cannot write it` | 로그가 저장소·`/mnt`·`~/.cache/torch`·`--mount` 밖(예: `/tmp`)이다. 적은 경로와 symlink를 푼 경로를 둘 다 본다 | 저장소나 `/mnt` 아래로 옮긴다 |
| `--log <경로> is under --mount <SRC>:<DST>[:ro], so the container cannot write it at that path` | 로그를 덮는 가장 깊은 마운트가 읽기 전용이거나, 컨테이너에서 그 경로에 쓰면 다른 호스트 파일로 간다(다른 경로로 붙인 `--mount`, 로그 경로에 정확히 붙인 파일 마운트) | 로그를 저장소 아래에 두거나, 그 `--mount`를 같은 경로·쓰기로 붙인다 |
| `container name usdp-<NAME> is in use` | 같은 이름의 컨테이너가 아직 있다(돌고 있음) | 새 이름을 쓰거나, 그 작업이 끝나기를 기다리거나 `docker stop usdp-<NAME>` |
| `<마운트> is not a readable sshfs mount (fstype '<fstype>'); root must remount with sshfs -o allow_other` | `/mnt/data_server`가 마운트되지 않았거나(`fstype 'none'`), sshfs가 아니거나, `allow_other` 없이 붙어 jshong이 읽지 못하거나, 연결이 멈춰 `USDP_PROBE_TIMEOUT`(10초) 안에 읽히지 않는다 | 아래 "sshfs 다시 붙이기" |
| `bad GPU index '<값>'; give one number, e.g. --gpu 2` | GPU 번호 형식이 틀렸다 | 숫자 하나(`--gpu 2`) |
| `nvidia-smi --query-gpu failed` | `nvidia-smi`가 실패했거나 10초 안에 답하지 않았다 | 호스트에서 `nvidia-smi`를 직접 돌려 드라이버 상태를 본다 |
| `GPU <N> not found by nvidia-smi` | 없는 번호다(이 서버는 0–4) | 번호를 고친다 |
| `nvidia-smi --query-compute-apps failed` | 사용 중 조회가 실패했다 | 위와 같다 |
| `docker ps failed, so run.sh cannot tell whether GPU <N> is in use` | 같은 GPU를 쓰는 `usdp` 컨테이너를 찾는 `docker ps`가 실패했다(이때 비어 있다고 보지 않는다) | docker 데몬 상태를 본다 |
| `GPU <N> is in use by a running process; pick an idle GPU or pass --force before --` | 그 GPU에 CUDA 프로세스가 있다(다른 사용자, k3s 파드 포함) | 4절로 빈 GPU를 고른다 |
| `GPU <N> is in use by usdp container <ID>; pick an idle GPU or pass --force before --` | 같은 GPU에 다른 `run.sh` 작업이 있다(VSCode용 `dev` 컨테이너 포함) | 빈 GPU를 고르거나 그 작업을 멈춘다. 자기 작업과 나눠 쓸 때만 `--force` |

**sshfs 다시 붙이기.** sshfs 마운트는 fstab에 없어서 호스트 재부팅 뒤 사라진다. 2026-10-19에는 root가 다음으로 다시 붙였다(지금 마운트 원본과 같다).

```bash
# as root on the host
sshfs -o allow_other solvit@192.168.1.165:/mnt/data /mnt/data_server
sshfs -o allow_other solvit@192.168.1.165:/mnt/data2 /mnt/data_server2
```

- `allow_other`가 빠지면 root만 읽을 수 있어 uid 1017 컨테이너가 읽지 못한다. 그래서 `run.sh`는 마운트가 있어도 읽기 탐침이 실패하면 거부한다.
- 저장소는 `/mnt/data_server`만 쓴다. `/mnt/data_server2`는 저장소가 쓰지 않지만 10-19에 함께 붙였다.
- `run.sh`가 "폴더가 비어 있지 않음" 대신 fstype을 보는 이유: 10-19 재부팅 직후 마운트 전의 빈 마운트 지점에 root 소유 폴더가 생겼다. 내용이 있어 보여도 데이터가 아닐 수 있다.
- 연결이 멈춘 sshfs는 먼저 풀어야 다시 붙는다. 이때 `umount -l /mnt/data_server` 같은 해제가 필요한지는 확인 필요. 멈춘 마운트를 읽던 `run.sh`의 탐침(`ls`)은 신호로 끝나지 않아 거부 뒤에도 남고, 연결이 풀리면 스스로 끝난다.

### 7.3 컨테이너 안 거부(`entry.sh`)

| 메시지·종료 | 원인 | 대처 |
|---|---|---|
| `entry.sh: refused: CUDA does not see exactly one GPU; check --gpu and the NVIDIA runtime`, 종료 3 | `--gpu`를 줬는데 컨테이너 안 torch가 CUDA 장치를 정확히 1장 보지 못했다(최대 60초 기다림). 명령은 실행되지 않는다 | 호스트 `nvidia-smi`, NVIDIA container toolkit을 확인한다. 백그라운드면 이 줄이 로그에도 남고 로그 끝은 `[run.sh] exit 3` |
| 종료 143 / 130 (명령 출력 없음) | CUDA 확인 중에 `docker stop` / Ctrl-C가 왔다. 명령은 돌지 않았다 | 다시 띄운다 |

### 7.4 거부가 아닌 알림

| 줄 | 뜻 |
|---|---|
| `run.sh: GPU: none (CPU run)` | `--gpu`가 없다. GPU 작업이면 `--gpu`를 빠뜨린 것이다 |
| `run.sh: --force: starting on GPU <N> although it is in use by <…>` | `--force`로 사용 중 거부를 넘겼다 |
| `run.sh: started usdp-<NAME>, log <경로> (stop: docker stop usdp-<NAME>)` | 백그라운드 작업이 시작됐다 |

## 8 VSCode 붙이기

편집은 호스트에서 한다. 호스트에는 torch 2.9.0과 aimet이 없으므로, IntelliSense나 디버거가 필요할 때만 컨테이너를 띄워 붙인다(설계 결정 Q8. 붙이는 단계의 실행 기록은 없어 확인 필요).

1. `docker/run.sh --name dev -- sleep infinity` (GPU 코드를 디버깅할 때만 `--gpu <N>`를 더한다).
2. VSCode 명령 팔레트에서 "Dev Containers: Attach to Running Container..." → `usdp-dev`. 폴더는 저장소 경로, 인터프리터는 `/usr/bin/python3`.
3. 다 쓰면 `docker stop usdp-dev`.

- VSCode 서버는 컨테이너 HOME에 설치되고 컨테이너와 함께 사라지므로, 붙일 때마다 다시 설치된다(1–2분 예상).
- `--gpu`로 띄운 `dev`는 그동안 그 GPU의 `usdp.gpu` 라벨을 잡고 있어, 같은 GPU의 다른 `run.sh` 작업이 거부된다. 쓰지 않을 때는 멈춘다.
- `dev`도 일반 작업과 같은 검사(이미지, sshfs, 이름)를 거친다. 로그는 `logs/dev.log`에 쌓인다.

## 9 `qairt-env` — DLC 변환 환경

QAIRT 2.41 SDK는 이미지의 `/opt/qairt`(root 소유, 읽기 전용)에, 그 Python 환경은 `/opt/qairt-venv`에 있다. `qairt-env 명령…`은 그 명령 하나에만 다음을 설정하고 `exec`한다.

| 변수 | 값 |
|---|---|
| `QAIRT_SDK_ROOT`, `QNN_SDK_ROOT`, `SNPE_ROOT` | `/opt/qairt` |
| `VIRTUAL_ENV` | `/opt/qairt-venv` |
| `PATH` | `/opt/qairt-venv/bin:/opt/qairt/bin/x86_64-linux-clang:` + 기존 |
| `LD_LIBRARY_PATH` | `/opt/qairt/lib/x86_64-linux-clang` + 기존 |
| `PYTHONPATH` | `/opt/qairt/lib/python` |

- **왜 따로 두나.** QAIRT의 Python 도구는 NumPy 1(1.26.4)과 onnx 1.16.1 ABI로 빌드돼 있고, 저장소 환경은 NumPy 2.2.6과 onnx 1.17.0이다. 도구 스크립트가 `#!/usr/bin/env python3`로 시작하므로 venv를 PATH 맨 앞에 둔다. 이 명령 밖에서는 `qairt-converter` 등이 PATH에 없고 `import qti`도 안 된다(`TC-DOCK-02-2`).
- **왜 컨테이너에서만 되나.** QAIRT 바이너리는 glibc 2.35(Ubuntu 22.04)를 요구하고 호스트는 2.31이다.
- 안에 있는 도구(시험으로 확인): `qairt-converter`, `qairt-quantizer`, `snpe-dlc-graph-prepare`, `snpe-dlc-info`, `qairt-dlc-diff`.
- `gen_convert_all.py`는 저장소 Python(`docker/run.sh -- python3 …`)에서 돌린다. `qairt-env` 안에서 저장소 도구(`tools/*.py`)를 돌리면 venv의 python3가 잡혀 torch가 없다.
- 확인: `docker/run.sh --offline -- qairt-env qairt-converter --help`가 종료 0이면 변환 환경이 산다. 인자 없이 `qairt-env`만 부르면 `qairt-env: usage: qairt-env CMD...`와 종료 2.
- 변환 절차, 로그 읽기(float 대체 목록), DLC 이름, md5가 매번 다른 이유는 [examples/deployment/README.md](../examples/deployment/README.md)에 있다. 이미 출하된 릴리스 폴더에서 `convert_all.sh`를 바로 돌리면 `results/denoise_release_v1`에서는 출하 DLC를 덮어쓰고, `results/release_v3`에서는 HTP prepare가 없는 DLC 9개가 출하 이름으로 그 폴더에 생긴다. 복사본에서 돌린다.

## 10 이미지 다시 빌드

### 10.1 언제 빌드하나

| 바꾼 것 | 다시 빌드 |
|---|---|
| `docker/Dockerfile`, `requirements.txt`, `docker/qairt-requirements.txt`, `docker/qairt-env` | 필요 |
| `docker/run.sh`, `docker/entry.sh`, `docker/accept.sh`, `docker/apt-versions.txt` | 불필요(호스트나 저장소 마운트에서 바로 쓰인다) |
| 저장소 코드, config | 불필요 |
| 계정(uid:gid)이 바뀌거나 새 호스트 | 필요(라벨이 빌드한 계정에 묶인다) |

빌드한 뒤에는 항상 `docker/accept.sh --gpu <N>`(11절)로 합격 확인을 다시 한다. PyPI 파일은 버전으로만 고정하고 해시는 고정하지 않으며, apt는 버전을 고정하지 않기 때문에 캐시 없는 재빌드는 다른 파일을 받을 수 있다.

### 10.2 `docker/build.sh`

```bash
docker/build.sh                                          # defaults
QAIRT_TARBALL=<tarball 경로> USDP_TAG=<태그> docker/build.sh
```

| 환경 변수 | 기본값 | 뜻 |
|---|---|---|
| `QAIRT_TARBALL` | `$HOME/Code/qualcomm-vlm/env/QAIRT_2.41.tar.gz` | QAIRT 2.41 tarball. sha256 `4e73e38f5e60e083b7c1e535ceaabb8a18e7523d912ad3b7dbe99d9217e3520d`과 같아야 한다. symlink면 실제 파일을 쓴다 |
| `USDP_TAG` | 오늘 날짜(`YYYYMMDD`) | 태그 `usdp:<USDP_TAG>`. `usdp:latest`도 항상 같이 옮긴다 |

하는 일:
1. `------ Build Start! ------`를 찍고 도구(`docker`, `sha256sum`, `id`, `date`, `mktemp`, `realpath`, `docker buildx`)를 확인한다.
2. tarball의 sha256을 확인한다.
3. `mktemp -d` 임시 폴더에 tarball 하드링크를 만든다(다른 파일시스템이면 복사). `qualcomm-vlm/env`를 직접 넘기지 않는 이유는 그 폴더의 `.dockerignore`가 tarball을 가리기 때문이다. 임시 폴더는 끝나면 지운다.
4. `docker buildx build --builder default`로 빌드한다. 빌드 인자 `UID`·`GID`·`USERNAME`은 부른 사람의 `id -u`·`id -g`·`id -un`이다. 이미지 안에 그 사용자를 만들고 라벨 `usdp.uid`·`usdp.gid`·`usdp.home`을 단다. Dockerfile 기본값(1017, 1017, jshong)은 `build.sh` 없이 빌드할 때만 쓰인다.
5. 이미지 ID를 찍고 `Build Finish!!`.

| 메시지 | 대처 |
|---|---|
| `build.sh: <도구> missing`, `build.sh: docker buildx missing` | 도구를 설치한다 |
| `build.sh: QAIRT tarball not found: <경로> (set QAIRT_TARBALL)` | `QAIRT_TARBALL`로 위치를 준다 |
| `build.sh: <경로> does not match sha256 <sha256>` | 다른 파일이다. 원본 tarball을 다시 받는다 |

- 첫 빌드는 27분 걸렸다(2026-10-19). 바뀐 것이 없으면 모든 층이 캐시에 맞아 몇 초에 끝나고 이미지 ID도 그대로다. 이미지는 약 19 GB다.
- 빌드는 네트워크를 쓴다: Docker Hub(`docker/dockerfile:1` frontend, `nvidia/cuda:12.6.0-cudnn-devel-ubuntu22.04`, 둘 다 digest 고정), Ubuntu apt, PyPI(ffmpeg는 `imageio_ffmpeg-0.6.0` 휠에서 꺼내고 sha256을 확인한다).
- 무거운 pip 층이 QAIRT·ffmpeg 층보다 앞에 있어, QAIRT나 ffmpeg만 바꾸면 pip 층은 다시 받지 않는다.
- 자기 계정으로 돌린다. root로 빌드하면 라벨이 uid 0이 되고, `run.sh`는 root 호출을 거부하므로 쓸 수 없다.
- **QAIRT 버전을 바꾸는 일은 코드 수정이다.** `build.sh`의 `QAIRT_SHA256`과 파일 이름 `QAIRT_2.41.tar.gz`, `Dockerfile`의 `QAIRT_2.41/examples` 경로가 2.41에 고정돼 있다.

### 10.3 태그, 되돌리기, 옛 이미지 정리

- 다른 날 빌드하면 옛 `usdp:<옛 날짜>` 태그가 남는다. 새 이미지에 문제가 있으면 `USDP_IMAGE=usdp:<옛 날짜> docker/run.sh …`로 되돌린다.
- 같은 날 다시 빌드하면 두 태그가 새 이미지로 옮겨 가고 옛 이미지는 태그 없이 남는다. `docker image ls --filter dangling=true --filter label=usdp.uid`로 보고(`docker image ls usdp`에는 태그 없는 이미지가 나오지 않는다) `docker image rm <옛 이미지 ID>`로 지운다. 이 호스트에는 다른 사용자의 이미지가 있으므로 `docker image prune` 같은 일괄 정리는 쓰지 않는다.
- 2026-10-20 기준 `usdp:20261019`와 `usdp:latest`는 같은 이미지 `7fd2f0ce9454`(10-19 23:49에 다시 빌드)다. 이 이미지로 `docker/accept.sh --gpu 2`가 15건 모두 통과했다(2026-10-20 01:57–02:15, `results/docker_accept_20261020_final`, pytest 4394 passed). 그 전 이미지 `112b82a375dd`는 지웠다.
- `usdp` 이미지는 레지스트리에 올리지 않는다(Qualcomm 비공개 SDK 포함). QAIRT tarball과 SDK는 git에 넣지 않는다.

### 10.4 Python 패키지를 더할 때

1. `requirements.txt`에 그 패키지와 **의존성 전부**를 `==`로 넣는다. 이미지는 `pip install --no-deps`로 설치한 뒤 `pip check`를 하므로, 의존성이 하나라도 빠지면 빌드가 실패한다.
2. `tests/test_docker_image.py`의 `assert len(pins) == 178`을 새 개수로 고친다(`TC-DOCK-01-1`).
3. 다시 빌드하고 `docker/accept.sh --gpu <N>`.

새 의존성은 이미지 크기와 재현성에 영향을 주므로, 커밋 제목이나 모듈 문서에 한 줄 이유를 남긴다.

### 10.5 `docker/apt-versions.txt`

apt 패키지는 버전을 고정하지 않으므로, 다시 빌드한 날의 Ubuntu 저장소 상태를 따른다. 대신 받아들인 이미지의 목록(`패키지=버전` 319줄)을 이 파일에 두고, `TC-DOCK-01-6`이 이미지 안의 `dpkg-query -W` 결과와 비교한다. 목록이 다르면 pytest가 실패하고 합격 확인 `TC-DOCK-09-1`도 FAIL이 된다. 바뀐 이미지를 모르고 쓰는 일을 막기 위해서다.

갱신 절차(재빌드 뒤 apt 목록만 달라졌을 때):
1. `docker/accept.sh --gpu <N> --out results/docker_accept_<날짜>_<이름>`을 돌린다.
2. FAIL이 `TC-DOCK-09-1` 하나이고, 그 `pytest.log`의 실패가 `test_TC_DOCK_01_6_AptPackagesMatchRecord`뿐인지 본다. export·DLC·GPU·학습 건은 모두 PASS여야 한다.
3. 바뀐 패키지를 본다.
   ```bash
   docker/run.sh --offline -- dpkg-query -W -f='${Package}=${Version}\n' | LC_ALL=C sort | diff docker/apt-versions.txt -
   ```
4. 받아들이면 파일을 다시 쓰고, `docker/accept.sh --only TC_DOCK_09_1,TC_DOCK_09_2 --out <새 폴더>`로 확인한다.
   ```bash
   docker/run.sh --offline -- dpkg-query -W -f='${Package}=${Version}\n' | LC_ALL=C sort > docker/apt-versions.txt
   ```
5. 파일을 커밋한다. 예: `[YYYYMMDD] DockerEnv Update apt-versions.txt for usdp:<태그>`.

시험은 공백으로 나눈 뒤 정렬해서 비교하므로 줄 순서는 상관없다. 기록 파일은 `LC_ALL=C` 순서로 정렬돼 있고 `dpkg-query` 출력은 정렬돼 있지 않아서, 3에서 `sort`를 빼면 같은 목록도 diff가 수십 줄 나온다. 3의 명령은 2026-10-20 `7fd2f0ce9454`에서 돌려 diff가 비는 것을 확인했다. 4는 파일을 쓰므로 실행하지 않았다.

### 10.6 `docker/qairt-requirements.txt`

QAIRT venv의 `pip freeze --all` 결과(pip과 setuptools 포함 7줄)다. numpy 1.26.4, onnx 1.16.1, packaging 24.0, pip 22.0.2, protobuf 3.20.3, PyYAML 6.0.2, setuptools 59.6.0. 첫 변환이 성공한 venv를 그대로 고정했다(R4). 빌드는 이 파일로 venv를 만들고, `TC-DOCK-02-1`이 `qairt-env pip freeze --all`과 비교한다.

평소에는 바꾸지 않는다. venv 구성을 일부러 바꿀 때(QAIRT 버전 변경 등)만 다음 순서로 갱신한다.
1. 파일과 Dockerfile을 고치고 다시 빌드한다.
2. 변환이 되는지 본다: `docker/accept.sh --only TC_DOCK_11_1,TC_DOCK_11_2 --out <새 폴더>`(출하 Denoise DLC와 내용 비교).
3. 통과하면 실제 venv를 기록한다(시험 코드의 비교 형식, 이 문서에서 실행하지 않음).
   ```bash
   docker/run.sh --offline -- qairt-env pip freeze --all > docker/qairt-requirements.txt
   ```
4. 전체 `docker/accept.sh --gpu <N>`.

## 11 합격 확인 `docker/accept.sh`

환경 자체가 출하 결과를 다시 만드는지 15건으로 확인한다. 호스트에서 자기 계정으로 돌린다.

```bash
docker/accept.sh --gpu <N>                                        # all 15 cases
docker/accept.sh --gpu <N> --out results/docker_accept_<날짜>_<이름>   # second run on the same day
docker/accept.sh --only TC_DOCK_11_1,TC_DOCK_11_2 --out <새 폴더>    # selected cases
USDP_IMAGE=usdp:<태그> docker/accept.sh --gpu <N> --out <새 폴더>      # check another tag
```

| 옵션 | 뜻 |
|---|---|
| `--gpu N` | GPU 건(5건)에 쓸 GPU. 없으면 그 5건은 `NOT RUN … needs --gpu N`이 되고 전체는 종료 1 |
| `--only TC_…,TC_…` | 고른 건만. 이름은 아래 표의 밑줄 형식(`TC_DOCK_09_1`). 모르는 이름이면 `accept.sh: unknown test case <ID>`, 종료 2 |
| `--out DIR` | 출력 폴더. 기본 `results/docker_accept_<YYYYMMDD>`(git 무시). 저장소나 `/mnt` 아래여야 하고(아니면 종료 2), sshfs(`/mnt/data_server*`) 위는 피한다(파일 소유자가 원격 계정이라 소유자 확인 건이 실패한다). 이미 있으면 `accept.sh: <폴더> exists; pass --out <new folder>`, 종료 1 |

**언제 돌리나.** 이미지를 다시 빌드한 뒤, `run.sh`·`entry.sh`·`Dockerfile`·두 requirements 파일을 고친 뒤, 새 호스트에서, 테스트를 더한 뒤.

**얼마나 걸리나(2026-10-19, GPU 2).**

| 범위 | 걸린 시간 |
|---|---|
| 전체 15건 | 21:06–21:20, 약 14분. 그중 pytest 380초(3499 passed) |
| pytest·export·`TC_DRUN_01_3` 6건 | 18:00–18:09 |
| `--gpu 2`로 돌린 7건(12·13·DRUN 03-3·04-4·05-5) | 19:24–19:28 |

그동안 고른 GPU는 비어 있어야 한다. `accept.sh`는 `--force`를 쓰지 않으므로, GPU가 사용 중이면 GPU 건이 FAIL로 찍힌다.

**출력.** 건마다 한 줄, 끝에 요약 한 줄이다. 모두 PASS일 때만 종료 0이다.

```text
PASS TC-DOCK-09-1
FAIL TC-DOCK-12-1: <사유와 볼 파일>
NOT RUN TC-DOCK-13-1: needs --gpu N
accept.sh: 14 passed, 1 not passed (outputs in <폴더>)
```

작업이 거부되거나 도구가 실패해도 그 건만 FAIL로 찍고 다음 건으로 간다. 요약 전에 스크립트가 끝나면 `accept.sh: aborted before the summary (exit <코드>); only the cases above ran`이 나온다. Ctrl-C면 130, `kill`(SIGTERM)이면 143이다.

### 11.1 15건이 보는 것

| TC | GPU | 확인하는 것 | 실패할 때 볼 파일 |
|---|---|---|---|
| `TC-DOCK-09-1` | | 컨테이너 안 전체 pytest(GPU·네트워크 없음)가 정확히 "`<기대 개수>` passed"이고 skip·fail이 없다. 기대 개수 = `BASE_TESTS`(280) + `test_docker_run.py`·`test_docker_image.py` 수집 수. skip을 실패로 치는 이유: 출하 파일이 없으면 golden 시험이 조용히 skip돼 통과처럼 보이기 때문이다 | `pytest.log`, `pytest.expected` |
| `TC-DOCK-09-2` | | pytest가 저장소에 파일을 남기지 않는다(`git status --porcelain --ignored` 전후 같음, 새 파일 0) | `git_before.txt`, `git_after.txt`, `pytest_new_files.txt` |
| `TC-DOCK-10-1` | | 출하 Denoise v1을 CPU로 다시 export하면 ONNX md5가 `81ac40c78450489a6fac04c05569a2ec`이고 `param_encodings`가 출하 파일과 같다 | `dn/export.log`, `dn_export.out` |
| `TC-DOCK-10-2` | | 출하 IR x4 두 프로파일(`x4_ir` 320×180, `x4_ir_1024` 320×256)을 다시 export하면 md5와 `param_encodings`가 같다 | `ir/x4_ir.log`, `ir/x4_ir_1024.log`, `ir_export.out` |
| `TC-DOCK-10-3` | | export 출력이 모두 호출자 소유이고 로그에 Traceback이 없다 | 같음 |
| `TC-DOCK-11-1` | | 출하 Denoise 입력의 복사본을 변환하면 종료 0, `생성 1개 / 건너뜀 0개`, float 대체 op 없음(`Following OPs fallback to float.`), DLC가 출하 DLC와 op 번호 두 칸(`startOpIndex`·`endOpIndex`)을 빼고 바이트 단위로 같다. 출하 DLC의 md5가 여전히 `62b56d8f6afad86df29ee27ce4572b31`인지(덮어쓰지 않았는지)도 본다 | `dn_convert1.log`, `dn_convert1/` |
| `TC-DOCK-11-2` | | 두 번째 변환이 첫 번째와 같은 기준으로 같다(변환이 결정적이다) | `dn_convert2.log`, `dn_convert2/` |
| `TC-DOCK-12-1` | 예 | EO x2 W16A16 export를 GPU로 하면 ONNX md5가 GPU 출하본 `f4a757f3551f194f1e0dd35944fcefa5`이다 | `x2_eo_gpu.log` |
| `TC-DOCK-12-2` | 예 | 그 export가 정말 GPU에서 돌았다: md5가 CPU export `7e2fe59ffbeb1660b7f83091e63a87e5`와 다르고 Triton `.so`가 2개 이상 컴파일됐다 | `x2_eo_gpu.log`의 `triton so files: N` |
| `TC-DOCK-13-1` | 예 | Denoise config 1 epoch 학습을 백그라운드로 돌리면 로그 끝이 `[run.sh] exit 0`이고 `best.pth`, `last.pth`, `training_log.csv`가 생긴다 | `train_dn_1ep.log`, `train_start.out` |
| `TC-DOCK-13-2` | 예 | 학습 출력이 모두 호출자 소유다(`--offline`) | `train_dn_1ep/` |
| `TC-DRUN-01-3` | | 컨테이너가 만든 파일과 컨테이너 안 `~/.cache`가 호출자 소유다 | `own_probe` |
| `TC-DRUN-03-3` | 예 | `--gpu N`이면 컨테이너 안 `nvidia-smi -L`이 1줄이고 UUID가 호스트 GPU N과 같다. `--gpu`가 없으면 torch가 장치 0개를 보고 stderr에 `GPU: none`이 있다 | `drun_03_3_gpu.err`, `drun_03_3_nogpu.err` |
| `TC-DRUN-04-4` | | 백그라운드 작업을 `docker stop`하면 로그 끝이 `[run.sh] exit 143`이다 | `stop_probe.log`, `stop_probe.out` |
| `TC-DRUN-05-5` | | `run.sh`를 거치지 않고 GPU 없이 `USDP_GPU_CHECK=1` 컨테이너를 띄우면 종료 3, `entry.sh: refused`, 명령이 돌지 않는다 | `drun_05_5.err` |

DLC를 md5로 비교하지 않는 이유: QAIRT 2.41은 `dlc.metadata`의 두 칸에 초기화하지 않은 메모리를 써서, 같은 서버에서 두 번 변환해도 md5가 다르다. 그래서 11-1·11-2는 그 두 값만 가리고 비교한다(`same_dlc`). 변환이 기록하는 `mktemp -d` 경로는 대체 스크립트(`shim/mktemp`)가 출하본과 같은 `/tmp/tmp.nIupqkU2Au`로 고정한다. 서버에서 만든 DLC의 보드 확인은 소유자 몫이다.

### 11.2 출력 폴더

| 경로 | 내용 |
|---|---|
| `pytest.log`, `pytest.expected`, `pytest.stamp` | pytest 출력, 기대 개수, 새 파일 기준 시각 |
| `git_before.txt`, `git_after.txt`, `pytest_new_files.txt` | 09-2의 전후 비교 |
| `dn/`, `dn_export.out` | Denoise 재-export(`dn/qat/`, `dn/export.log`) |
| `ir/`, `ir_export.out` | IR x4 재-export(`ir/x4_ir/qat/`, `ir/x4_ir_1024/qat/`, 로그) |
| `dn_convert1/`, `dn_convert2/`, `dn_convert<k>.log` | 출하 입력 복사본, 서버에서 만든 DLC, 변환 로그 |
| `shim/mktemp` | 임시 경로 고정 스크립트 |
| `x2_eo_gpu/`, `x2_eo_gpu.log` | EO x2 GPU export |
| `train_dn_1ep/`, `train_dn_1ep.log`, `train_start.out` | 1 epoch 학습 run 폴더, 로그, `run.sh` 시작 메시지 |
| `stop_probe.log`, `stop_probe.out`, `drun_*`, `own_probe` | DRUN 통합 시험 |

이 폴더는 코드나 테스트가 읽지 않는다. 모듈 문서 §5가 기록으로 경로를 든다. 최신 전체 통과 기록은 `results/docker_accept_20261020_final`(2026-10-20, 이미지 `7fd2f0ce9454`)이고, 그 전 출력 폴더는 정리했다(§5에 남은 경로는 그때의 기록이다).

### 11.3 자주 나는 FAIL

- **`TC-DOCK-09-1` "want '<N> passed' and no skip".**
  - 테스트를 더하거나 뺐는데 `docker/accept.sh`의 `BASE_TESTS=280`을 고치지 않았다. Docker 시험 파일 두 개 밖의 시험 수가 바뀌면 같은 커밋에서 `BASE_TESTS`를 고친다.
  - git이 추적하지 않는 출하 파일(12절 4번)이 없어 golden 시험이 skip됐다. `pytest.log`에서 `SKIPPED` 줄을 본다.
  - `~/.cache/torch/hub`에 가중치가 없어 `--offline` 시험이 실패했다(12절 3번).
  - `/mnt/data_server`가 없으면 `run.sh`가 pytest를 시작하지 않는다.
- **GPU 건 FAIL "export exited 1" 등.** 고른 GPU가 사용 중이라 `run.sh`가 거부했다. 출력 파일(`x2_eo_gpu.log`, `train_start.out`)에 `run.sh: refused:` 사유가 있다.
- **같은 날 두 번째 실행.** 기본 출력 폴더가 이미 있어 종료 1. `--out`으로 새 폴더를 준다.

## 12 다른 호스트로 옮기기

1. **호스트 준비.** Docker와 buildx, NVIDIA container toolkit, 드라이버. 이 서버는 Docker 20.10.11, buildx 0.23.0, toolkit 1.13.5, 드라이버 550.54.14이고, 이미지의 torch 2.9.0+cu128이 이 드라이버에서 돈다. 다른 드라이버에서 되는지는 확인 필요. 작업 계정을 docker 그룹에 넣는다. 호스트 도구 `findmnt`, `timeout`, `realpath`, `nvidia-smi`, `sha256sum`, `mktemp`가 필요하다.
2. **저장소와 이미지.** 저장소를 쉼표 없는 경로에 둔다(어느 경로든 같은 경로로 마운트된다). QAIRT tarball을 저장소 밖에 복사하고 `QAIRT_TARBALL=<경로> docker/build.sh`를 **작업할 계정으로** 돌린다. 라벨이 그 계정의 uid:gid에 묶이므로 계정마다 따로 빌드한다. 한 호스트에서 두 계정이 빌드하면 `usdp:latest`를 서로 덮어쓴다. 그때는 계정마다 `USDP_TAG`를 다르게 주고 `USDP_IMAGE`로 자기 태그를 고른다. tarball과 `mktemp -d`(`$TMPDIR`, 기본 `/tmp`)가 다른 파일시스템이면 tarball(약 500 MB)이 복사된다. 같은 파일시스템의 폴더를 `TMPDIR`로 주면 하드링크로 끝난다.
3. **가중치 캐시.** `--offline` 작업과 pytest는 호스트 `~/.cache/torch/hub`의 가중치 4개를 쓴다. 같은 상대 위치로 복사하고 md5를 대조한다. 없으면 `--offline` pytest의 가중치 시험이 이름 해석 실패로 떨어진다(10-19 이관 전 관측). 네트워크가 있을 때 첫 실행이 받아 오는지는 확인 필요.

   | 파일(`~/.cache/torch/hub/` 아래) | md5 | 쓰는 곳 |
   |---|---|---|
   | `checkpoints/alexnet-owt-7be5be79.pth` | `cc6360b220eb7cfb64d83dc81b1963b1` | pyiqa LPIPS의 alex 백본(저장소 코드는 직접 부르지 않음) |
   | `checkpoints/vgg19-dcbb9e9d.pth` | `92881fe292bd7d2408ecff58a101fd03` | EO Phase-2 perceptual loss |
   | `pyiqa/LPIPS_v0.1_alex-df73285e.pth` | `9642209e2b57a85d20f86d812320f9e6` | LPIPS 지표 |
   | `pyiqa/niqe_modelparameters.mat` | `8601a29b769c9c836bb67a50a574e9c5` | NIQE 지표 |

4. **git이 추적하지 않는 출하 파일.** golden 시험과 `accept.sh`가 읽는다. 없으면 시험이 skip되고 `TC-DOCK-09-1`과 export·DLC·GPU 건이 FAIL이다. 다음 폴더를 통째로 복사한다(2026-10-19 기준 약 2.6 GB).
   - `results/release_v3/`(checkpoints, `x*_*/qat`, `dlc/`, `eval/` 프레임 등)
   - `results/denoise_release_v1/`(checkpoint, `dn_bpp_half4_w16a16/qat`, 출하 DLC, `eval/*.nv12`, `analysis/`)
   - `checkpoints/`(`bicubicpp_deploy/`, `csuav_deploy/`)
5. **데이터.** `run.sh`는 `/mnt/data_server`가 읽히는 `fuse.sshfs` 마운트여야 시작한다(7.2). root가 `sshfs -o allow_other`로 붙인다. 로컬 사본 `/mnt/data2/jshong/SuperResolution`은 Denoise config와 일부 시험이 읽는다. 컨테이너는 `/mnt` 전체를 같은 경로로 보므로, 새 호스트에서도 같은 경로에 두면 config를 고치지 않아도 된다. 경로가 다르면 config를 고친다([configs/README.md](../configs/README.md)). `/mnt` 밖의 데이터는 `--mount`로 붙인다.
6. **합격 확인.** `docker/accept.sh --gpu <N>`. EO x2 GPU export md5(`TC-DOCK-12-1`)는 RTX A6000에서 맞춘 값이다. 다른 GPU 모델에서도 같은지는 확인 필요. 다르면 DockerEnv §2.7의 R2 규칙(옛 환경이 같은 커밋에서 낸 값과 같고 CPU 값과 다를 것)을 따른다.
7. **보드.** 새 호스트 계정의 ssh 공개 키를 보드 `root@192.168.100.98`의 `authorized_keys`에 넣는다(소유자).

## 13 알려진 한계

- GPU 사용 중 판정은 시작하는 순간의 상태만 본다(4절 끝).
- 멈춘 sshfs를 읽던 탐침은 거부 뒤에도 남는다. 연결이 풀리면 스스로 끝난다.
- `--gpu` 없이 띄운 GPU 작업은 CPU로 돈다. `run.sh: GPU: none (CPU run)`만 나온다.
- 명령이 검사 뒤에 GPU를 가리는 경우(`CUDA_VISIBLE_DEVICES=`)는 막지 않는다.
- `--mount`는 신뢰 입력이다. 민감한 경로(홈, `~/.ssh`)를 붙이지 않는다.
- 작업은 재부팅을 넘기지 못하고, CPU·메모리 제한이 없다.
