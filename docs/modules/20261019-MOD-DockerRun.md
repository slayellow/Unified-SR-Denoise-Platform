# DockerRun 모듈 문서

| 문서번호 | 버전 | 작성일 | 작성자 | 시스템 | 상위 문서 |
|---|---|---|---|---|---|
| USRD-MOD-DRUN-001 | 0.4 | 2026-10-20 | JSHONG | unified-sr-denoise-platform | USRD-MOD-DOCK-001(DockerEnv) §6 결정 Q4·Q5, "실패로 보는 것" |

개정 이력
| 버전 | 일자 | 변경 내용 | 작성자 |
|---|---|---|---|
| 0.1 | 2026-10-19 | 최초 작성. DockerEnv v0.1의 MREQ-DOCK-04~08을 옮김(→ DRUN-01~05). 상세 설계, 시험 케이스. 차단 조건에 "이미지 없음", "이미 쓰는 이름"을 더함 | JSHONG |
| 0.2 | 2026-10-19 | 브랜치 전체 리뷰 반영. `--log`는 같은 경로·쓰기로 붙은 마운트 아래만 허용(읽기 전용·다른 경로 `--mount` 아래면 거부). 포그라운드 표준 입력 전달(`-i`). §4.1 G에 사용 중 조회만 실패·멈춤, R에 `--mount` 꼴 3가지와 `--log` 위치 1가지를 더함(2,214 → 3,190건). `--offline`·표준 입력 TC 추가 | JSHONG |
| 0.3 | 2026-10-19 | 리뷰에서 미룬 Minor 처리. 사용 중 판정을 `usdp.gpu` 라벨로(포그라운드 GPU 작업 포함). 멈춘 sshfs는 백그라운드 읽기 탐침으로 `USDP_PROBE_TIMEOUT` 안에 거부. 모든 실행에 `TINI_KILL_PROCESS_GROUP=1`, CUDA 확인 중 중지·Ctrl-C도 종료 줄을 남기고 명령을 돌리지 않음. 저장소는 실제 경로(`pwd -P`), symlink `--mount`는 적은 경로에 붙음, 경로의 쉼표 거부, 옵션처럼 생긴 값 거부, docker 오류·root 호출 사유를 정확히. 스텁을 실제 도구처럼(필터, 라벨 출력, findmnt 경로), CUDA 확인은 가짜 torch로 실제 코드 실행. 곱집합 3,190 → 4,050건 | JSHONG |
| 0.4 | 2026-10-20 | 2차 수정 리뷰 반영(0.3에서 생긴 회귀 포함). 이미지 라벨은 stdout에서만 읽음(stderr 경고 무시). `--log` 판정을 "그 경로를 덮는 가장 깊은 마운트가 쓰기이고 같은 호스트 파일로 이어짐"으로 일반화(적은 경로→푼 경로 순, 정확히 그 경로의 마운트 포함): symlink `--mount` 아래 로그 허용, 로그를 가리는 파일 마운트 거부. 사전 확인 순서를 이미지 → sshfs → 경로 → GPU로 바꾸고 `--mount` 해석을 sshfs 확인 뒤로. `docker ps` 실패는 사용 중으로 보고 거부. `USDP_PROBE_TIMEOUT`은 0보다 큰 초만. entry.sh는 신호를 trap에서 기록하고(탐침이 살아도 130/143), 전체를 `main`에 넣어 실행 중 수정에 안전. 곱집합은 그대로 4,050건 | JSHONG |

작업 컨테이너를 띄우는 `docker/run.sh`와 컨테이너 안 진입점 `docker/entry.sh`를 다룬다. 이미지는 DockerEnv 문서에 있다. 모듈 약어는 `DRUN`이다.

## 1 요구
| ID | 유형 | 요구사항 | 수용 기준 | 상위 | 검증 |
|---|---|---|---|---|---|
| MREQ-DRUN-01 | 기능 | 컨테이너는 `run.sh`를 부른 사용자의 uid:gid로 실행되어 결과 파일과 HOME 캐시를 그 사용자 소유로 남겨야 한다 | 스텁 `docker` argv에 `--user <id -u>:<id -g>`. 실제 실행 뒤 만든 파일과 컨테이너 안 `~/.cache`의 소유자가 jshong | Q4, Q5 | T |
| MREQ-DRUN-02 | 기능 | `--mount`를 주지 않으면 `run.sh`는 저장소(같은 경로, 쓰기), `/mnt`(같은 경로, 쓰기, rslave), 호스트 `~/.cache/torch` 세 곳만 `--mount type=bind`로 붙이고, 저장소 루트에서 명령을 받은 그대로 실행하여야 한다 | 스텁 argv에 bind 3개가 정확히 있고, `-v`가 없고, `-w <저장소>`(실제 경로)가 있다. `--` 뒤 인자는 공백·따옴표·한글·`--force`까지 docker argv와 entry.sh `exec` 양쪽에서 그대로 전달되고, 포그라운드 종료 코드는 명령의 코드다. symlink `--mount`는 적은 경로에 붙고 `USDP_IMAGE`가 지정되면 그 이미지를 쓴다. `--offline`이면 `--network none`이 한 번 있고, 없으면 `--network`가 없다. 표준 입력이 터미널이 아니면 `-i`(입출력이 모두 터미널이면 `-it`)로 붙여 파이프·heredoc이 명령에 닿는다 | Q4 | T |
| MREQ-DRUN-03 | 기능 | `--gpu N`을 주면 컨테이너는 호스트 GPU N 한 장만 보아야 하고, 주지 않으면 GPU를 보지 않아야 한다 | 실제 컨테이너 안 `nvidia-smi -L`이 1줄이고 UUID가 호스트 GPU N과 같다. `--gpu`가 없으면 torch GPU 0장, stderr에 `GPU: none` | Q5 | T |
| MREQ-DRUN-04 | 기능 | `--name`으로 띄운 작업은 호출 셸과 상관없이 계속 돌고, 로그 끝에 종료 코드를 남겨야 한다. 중지 신호(`docker stop`, Ctrl-C)는 포그라운드·백그라운드 모두 명령의 프로세스 그룹 전체에 가야 한다 | 정상 종료면 로그 마지막 줄 `[run.sh] exit <코드>`. `docker stop`이면 `[run.sh] exit 143`. 모든 실행 argv에 `TINI_KILL_PROCESS_GROUP=1`. CUDA 확인 중 SIGTERM이면 종료 143, SIGINT면 130으로 끝나고(탐침이 신호를 받지 않고 끝나도 같다) 명령은 돌지 않으며 로그에 그 종료 줄이 남는다 | Q5 | T |
| MREQ-DRUN-05 | 차단 | `run.sh`는 다음 중 하나라도 해당하면 컨테이너를 만들지 않아야 한다: 사용법 오류, 이미지 없음, 호출자 uid·gid ≠ 이미지 라벨, 컨테이너가 같은 경로에 쓸 수 없는 `--log`(마운트 밖, 또는 읽기 전용이거나 다른 경로로 붙인 `--mount` 아래), 없는 `--mount` 원본, 이미 쓰는 `--name`, `/mnt/data_server`를 읽을 수 있는 fuse.sshfs로 확인하지 못함, 잘못되거나 없는 GPU 번호, GPU 조회 실패, 지정 GPU 사용 중(`--` 앞 `--force`일 때만 예외). 또 GPU 작업에서 컨테이너 안 CUDA 장치가 정확히 1장이 아니면 명령을 실행하지 않아야 한다 | 사용법 오류는 종료 2와 `run.sh: usage:`, 그 밖의 거부는 종료 1과 `run.sh: refused:` 사유, 컨테이너 안 CUDA 거부는 종료 3과 `entry.sh: refused:`. §4.1 곱집합 4,050건 전수에서 잘못 통과 0건. 경로의 쉼표, 옵션처럼 생긴 값(`--name --force`), 빈 `--log`, root 호출, 0 이하나 숫자가 아닌 `USDP_PROBE_TIMEOUT`(종료 2), 사용 중 확인용 `docker ps` 실패도 거부 | Q5, 실패로 보는 것 | T |

## 2 설계
### 2.1 구성 요소
| 항목 | 내용 |
|---|---|
| 파일 | `docker/run.sh`(호스트), `docker/entry.sh`(컨테이너 안. 저장소 마운트로 실행하므로 고쳐도 다시 빌드하지 않음), `tests/test_docker_run.py` |
| 의존 | 호스트: docker CLI, nvidia-smi, findmnt, timeout, id. 이미지 `usdp:latest`와 라벨 `usdp.uid`·`usdp.gid`·`usdp.home`(DockerEnv) |

### 2.2 실행 개념
- 흐름: 인자 해석 → 사전 확인(§3.2) → `docker run --rm --init --user <uid>:<gid> --ipc host` + bind 3개 + `-w <저장소>` → `<저장소>/docker/entry.sh 명령…`.
- 포그라운드: 표준 입력과 출력이 모두 터미널이면 `-it`, 표준 입력이 터미널이 아니면 `-i`(파이프·heredoc 전달). entry.sh가 GPU 확인 뒤 `exec`한다.
- 백그라운드(`--name N`): `-d --name usdp-N -e USDP_LOG=<로그>`. `TINI_KILL_PROCESS_GROUP=1`은 모든 실행에 붙어 중지 신호가 명령 전체에 간다. 로그 기본값은 `logs/N.log`(git 무시)이고, 호스트가 먼저 만든다. `docker stop usdp-N`이면 학습이 바로 끊기고, `last.pth`에서 재개한다.
- GPU: `--gpu 2` → `--gpus '"device=2"'` + `--label usdp.gpu=2` + `-e USDP_GPU_CHECK=1`. 컨테이너 안에서는 cuda:0이다. `CUDA_VISIBLE_DEVICES`는 쓰지 않는다. `--gpu`가 없으면 이미지 기본값 `NVIDIA_VISIBLE_DEVICES=void`로 GPU 없이 돈다.
- 환경 변수: 호스트 변수는 넘어가지 않는다. `-- env SCALE=4 스크립트`처럼 준다.
- VSCode: `run.sh --name dev [--gpu N] -- sleep infinity`로 띄운 뒤 "Attach to Running Container".
- 호스트 /tmp와 Claude 스크래치패드는 컨테이너에서 보이지 않는다. `--mount`로 붙이거나 출력을 저장소 `results/` 아래에 둔다.

### 2.3 인터페이스
| 대상 | 형태 |
|---|---|
| `docker/run.sh [옵션] -- 명령…` | `--gpu N`(한 번만), `--name NAME`(`[A-Za-z0-9][A-Za-z0-9_.-]*`), `--log FILE`(`--name` 필요), `--mount SRC[:DST[:ro]]`(반복, 신뢰 입력), `--offline`(`--network none`), `--force`(GPU 사용 중만 넘김). 종료 코드: 포그라운드는 명령의 코드, 백그라운드는 0(시작됨), 거부 1, 사용법 오류 2 |
| 환경 변수 | `USDP_IMAGE`(기본 `usdp:latest`), `USDP_DATA_MOUNT`(기본 `/mnt/data_server`), `USDP_PROBE_TIMEOUT`(초, 기본 10) |
| `docker/entry.sh 명령…` | `USDP_GPU_CHECK=1`이면 CUDA 장치 1장 확인(`USDP_CUDA_TIMEOUT` 기본 60초). 실패하면 사유를 남기고 종료 3. `USDP_LOG`가 있으면 trap 래퍼, 없으면 `exec` |

## 3 상세 설계
### 3.1 소프트웨어 단위
| 단위 | 파일 | 역할 | MREQ |
|---|---|---|---|
| `parse_args` | `docker/run.sh` | `--` 앞 옵션만 해석. 모르는 옵션·줄임 표기·`--` 없음·빈 명령은 종료 2 | 05 |
| `check_image`, `check_paths`, `check_data_mount`, `check_gpu` | `docker/run.sh` | §3.2의 사전 확인. 거부는 `refuse "<사유>"`(stderr `run.sh: refused: …`, 종료 1) | 01, 02, 05 |
| `docker_args` | `docker/run.sh` | docker 인자 배열 조립 | 01–04 |
| `docker/entry.sh` | `docker/entry.sh` | CUDA 확인, 백그라운드 래퍼, `exec` | 03–05 |

### 3.2 `docker/run.sh` (`set -euo pipefail`)
0. 순서: 인자 → 도구 → `check_image` → `check_data_mount` → `check_paths` → `check_gpu`. 멈춘 sshfs 위 경로를 건드리면 run.sh도 멈추므로 경로 해석은 sshfs 확인 뒤에 한다.
1. `check_image`: 라벨은 stdout에서만 읽는다(docker가 stderr에 경고를 찍을 수 있음). `docker image inspect "$USDP_IMAGE"`가 "No such image"로 실패하면 거부("docker/build.sh 먼저"), 다른 오류(데몬 없음, 권한)면 docker의 첫 줄을 사유로 거부. 라벨이 없으면 거부. 호출자가 root(uid 0)면 "자기 사용자로" 거부. 라벨 `usdp.uid`·`usdp.gid`가 `id -u`·`id -g`와 다르면 거부.
2. `check_paths`
   - 저장소 경로, `HOME`, `--mount` 원본·대상에 쉼표가 있으면 거부(docker `--mount`는 쉼표로 필드를 나눔).
   - `--mount` 원본이 없으면 거부. 원본은 `realpath -m`으로 풀고, DST를 안 주면 적은 경로(`realpath -ms`, symlink 유지)에 붙인다.
   - `--log`: 호스트 파일은 `realpath -m`으로 푼 경로다. 컨테이너가 쓸 경로 후보는 적은 경로(`realpath -ms`, symlink 유지)와 푼 경로 순이다. 후보를 덮는(같거나 그 아래인) 마운트(저장소, `/mnt`, `~/.cache/torch`, `--mount`) 가운데 가장 깊은 것이 쓰기이고, 그 마운트를 따라가면 같은 호스트 파일이 나오는 첫 후보를 `USDP_LOG`로 쓴다. 덮는 마운트가 없으면 "outside", 있는데 조건을 못 맞추면 "under --mount" 사유로 거부.
   - `docker ps -a --filter name=^usdp-NAME$ -q`가 비어 있지 않으면 거부.
3. `check_data_mount`: `findmnt -C -rn -M "$USDP_DATA_MOUNT" -o FSTYPE | tail -n 1`(마운트 표만 읽고, 겹친 마운트면 맨 위)이 `fuse.sshfs`이고, 백그라운드 `ls -A` 탐침이 `USDP_PROBE_TIMEOUT` 안에 성공해야 한다. 멈춘 sshfs의 읽기는 신호로도 끝나지 않으므로 기다리지 않고 탐침을 남겨 둔 채 거부한다. 아니면 거부하고 "root가 `sshfs -o allow_other`로 다시 마운트" 안내를 출력한다. `--force`로 넘길 수 없다.
4. `check_gpu`(`--gpu`가 있을 때)
   - 번호는 `^(0|[1-9][0-9]*)$`여야 한다.
   - `timeout … nvidia-smi --query-gpu=index,uuid --format=csv,noheader`가 실패하거나 해석되지 않거나 번호가 없으면 거부.
   - 사용 중 판정: `nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader`에 그 UUID가 있거나, `docker ps --filter label=usdp.gpu=N`에 실행 중인 컨테이너가 있으면(포그라운드 GPU 작업도 라벨을 단다) 사용 중이다. 이 조회도 `timeout`이 걸리고 실패하면 거부. 라벨 조회(`docker ps`)가 실패해도 거부한다(비어 있다고 보지 않음).
   - 사용 중이면 `--force`가 없을 때 거부하고, 있을 때는 넘긴 사실을 stderr에 남긴다.
5. `mkdir -p "$HOME/.cache/torch"`(호출자 소유). 백그라운드면 로그 폴더를 만들고 로그 파일을 만든다(`: >> "$LOG"`).
6. `docker_args`
   - bind 3개: `type=bind,source=$REPO,target=$REPO`, `type=bind,source=/mnt,target=/mnt,bind-propagation=rslave`, `type=bind,source=$HOME/.cache/torch,target=<라벨 usdp.home>/.cache/torch`.
   - `--mount` 추가분, `--network none`, GPU·백그라운드 인자를 더한다.
   - `--gpu`가 없으면 stderr에 `GPU: none (CPU run)`을 출력한다.
7. 포그라운드는 `exec docker "${ARGS[@]}"`. 백그라운드는 컨테이너 이름과 로그 경로를 출력한다.

### 3.3 `docker/entry.sh` (`set -u`)
- 전체가 `main`에 있어 bash가 파일을 다 읽은 뒤 실행한다(실행 중인 작업이 고친 entry.sh를 이어 읽지 않음).
- 처음에 `trap 'SIG=143' TERM`, `trap 'SIG=130' INT`를 건다. 무시(`trap ''`)가 아니라 기록이므로 자식에 상속되지 않고, 탐침이 신호를 받지 않고 끝나도 신호를 놓치지 않는다.
- `USDP_GPU_CHECK=1`: `timeout --foreground "$USDP_CUDA_TIMEOUT" python3 -c '…torch.cuda.is_available() and torch.cuda.device_count() == 1…'`. `--foreground`라 탐침이 같은 프로세스 그룹에 있어 중지 신호를 받는다. 확인 중 신호가 기록됐거나 종료 130·143이면 그 코드로 끝내고(백그라운드면 로그에 `[run.sh] exit <코드>`), 그 밖의 실패면 `entry.sh: refused: CUDA …`를 출력하고(백그라운드면 로그에도) 종료 3.
- `USDP_LOG`가 있으면 `"$@" >>"$USDP_LOG" 2>&1` 뒤 `[run.sh] exit <코드>`를 붙이고 그 코드로 끝낸다.
- 없으면 `exec "$@"`.

## 4 단위 시험 케이스
### 4.1 회피 차원 (MREQ-DRUN-05)
검사는 서로 다른 입력을 보는 네 묶음이다. 묶음마다 `tests/test_docker_run.py`가 곱집합을 `itertools.product`로 만들어 전수 실행한다. 스텁은 `docker`(필터·이미지 오류·경고·`ps` 실패·종료 코드), `nvidia-smi`, `findmnt`(`-C`와 `-M <경로>`가 있어야 답함), `id`, `ls`이고, CUDA 확인은 가짜 `torch` 패키지(PYTHONPATH)를 실제 python3와 실제 확인 코드가 import한다. 시험에서는 `USDP_DATA_MOUNT`를 tmp_path로 돌리고, `USDP_PROBE_TIMEOUT`과 `USDP_CUDA_TIMEOUT`을 줄인다.

| 묶음 | 차원 = 값 | 건수 | 통과 규칙 |
|---|---|---|---|
| G GPU | 지정 = 없음 / `2` / `7`(없음) / `02` / `2,3` / `all` / `-1` / 빈 값 / `GPU-<uuid>` / `--gpu` 두 번. GPU 2 = 비어 있음 / 프로세스 / `usdp` 컨테이너(라벨 `usdp.gpu=2`) / GPU 3만 프로세스 / GPU 3만 `usdp` 컨테이너. 조회 = 정상 / 종료≠0 / 빈 출력 / 깨진 출력 / 목록 조회 멈춤 / 사용 중 조회만 종료≠0 / 사용 중 조회만 멈춤. `--force` = 없음 / `--` 앞 / `--` 뒤 / `--` 없음 / `--forc`. 모드 = 포그라운드 / 백그라운드 | 10×5×7×5×2 = 3,500 | `--` 없음·`--forc`·잘못된 지정은 거부. 지정 없음이면 통과. `2`는 조회가 정상이고 (비어 있음 또는 GPU 3만 사용 중, 또는 `--` 앞 `--force`)일 때만 통과 |
| M 마운트 | 상태 = 읽히는 fuse.sshfs / 빈 폴더 / 폴더 없음 / 마운트 아님 + 파일 있음(10-19 관측) / allow_other 없음(EACCES) / 끊김(신호를 무시하고 30초 멈춤). `--force` 5값. 모드 2값 | 6×5×2 = 60 | 읽히는 fuse.sshfs이고 사용법이 맞을 때만 통과. 거부는 5초 안 |
| R 실행 조건 | uid = 같음 / 다름. 이미지 = 있음 / 없음 / docker 데몬 오류. `--log` = 없음 / 저장소 안 / 마운트 밖(/tmp) / `--mount` 원본 아래. `--mount` = 없음 / 있음(/x로) / 있음(같은 경로) / 있음(같은 경로, ro) / 없는 경로. 이름 = 새 이름 / 사용 중. 모드 2값 | 2×3×4×5×2×2 = 480 | uid 같음, 이미지 있음, 원본 있음이어야 한다. 백그라운드는 로그가 기본값·저장소 안이거나 같은 경로·쓰기로 붙인 `--mount` 원본 아래이고 이름이 새것일 때, 포그라운드는 `--log`가 없을 때 통과(`--log`는 `--name` 필요) |
| C 컨테이너 안 CUDA | 가짜 torch(실제 확인 코드가 import) = 장치 1장 / 0장 / ImportError / 2장 / 멈춤(5초 뒤 1장). 모드 2값 | 5×2 = 10 | 1장일 때만 명령 실행 |

합계 4,050건, 잘못 통과 0건이어야 한다. 건수는 `test_TC_DRUN_05_9_SweepSizesMatchSpec`이 고정한다.

막을 수 없는 것(문서화된 한계):
- 검사와 시작 사이에 다른 프로세스가 GPU를 잡는 경쟁, 그리고 아직 GPU 메모리를 잡지 않은 다른 사용자의 프로세스(run.sh가 띄운 컨테이너는 라벨로 보임).
- 멈춘 sshfs 탐침은 거부 뒤에도 남는다(신호로 끝나지 않음). 연결이 풀리면 스스로 끝난다.
- 시작 뒤 k3s가 같은 GPU에 파드를 배치하는 경우. device plugin은 docker 작업을 모른다.
- `--gpu` 없이 띄운 GPU 작업은 CPU로 돈다. `GPU: none`만 출력한다.
- 명령이 검사 뒤에 GPU를 가리는 경우(`CUDA_VISIBLE_DEVICES=`).
- `--mount`로 민감한 경로를 붙이는 경우. 신뢰 입력으로 본다.

### 4.2 시험 케이스
| TC | MREQ | 구분 | 테스트 이름 | 기대 결과 |
|---|---|---|---|---|
| TC-DRUN-01-1 | 01 | 정상 | `test_TC_DRUN_01_1_RunsAsCallerUid` | argv에 `--user 1017:1017` |
| TC-DRUN-01-2 | 01 | 오류 | `test_TC_DRUN_01_2_RefusesImageWithoutUidLabel` | 라벨이 빈 문자열(실제 docker 출력)이면 종료 1, `usdp.uid` 사유 |
| TC-DRUN-01-3 | 01 | 정상(통합) | `docker/accept.sh` `TC_DRUN_01_3_RealOutputsOwnedByCaller` | 컨테이너가 만든 파일과 `~/.cache` 소유자 jshong |
| TC-DRUN-01-4 | 01 | 오류 | `test_TC_DRUN_01_4_RefusesGidOnlyMismatch` | 라벨 `1017 1018`이면 gid만 달라도 거부 |
| TC-DRUN-01-5 | 01 | 오류 | `test_TC_DRUN_01_5_RootCallerToldToUseOwnUser` | uid 0이면 "not root" 사유, "rebuild" 안내 없음 |
| TC-DRUN-01-6 | 01 | 경계 | `test_TC_DRUN_01_6_InspectWarningOnStderrIgnored` | 성공한 `docker image inspect`가 stderr에 경고를 찍어도 시작 |
| TC-DRUN-02-1 | 02 | 정상 | `test_TC_DRUN_02_1_MountsExactlyThreeBinds` | bind 3개, `-v` 없음, `-w` 저장소 |
| TC-DRUN-02-2 | 02 | 경계 | `test_TC_DRUN_02_2_ExtraMountAndKoreanSpacePaths` | 같은 경로 `--mount "a b/한글"`과 `--mount "참고 자료:/x:ro"`가 4·5번째 bind(readonly)로 들어가고, 앞 마운트 안 `로그 파일.log`가 만들어진다 |
| TC-DRUN-02-3 | 02 | 경계 | `test_TC_DRUN_02_3_CreatesMissingTorchCache` | HOME에 `.cache/torch`가 없으면 만든 뒤 진행 |
| TC-DRUN-02-4 | 02 | 경계 | `test_TC_DRUN_02_4_CommandArgvPassedVerbatim` | `python3 -c … 'a b' '"q"' --force 한글`이 docker argv 끝에 그대로, entry.sh `exec` 뒤 `sys.argv`도 그대로 |
| TC-DRUN-02-5 | 02 | 정상 | `test_TC_DRUN_02_5_OfflineAddsNetworkNone` | `--offline`이면 `--network none` 한 번, 없으면 `--network` 없음 |
| TC-DRUN-02-6 | 02 | 경계 | `test_TC_DRUN_02_6_ForegroundStdinAttached` (3) | 표준 입력 파일 → `-i`, 입출력 터미널 → `-it`, 입력만 터미널 → 둘 다 없음 |
| TC-DRUN-02-7 | 02 | 경계 | `test_TC_DRUN_02_7_SymlinkedRepoPathUsesRealPath` | symlink로 부른 run.sh도 실제 경로로 bind·`-w`·기본 로그 |
| TC-DRUN-02-8 | 02 | 경계 | `test_TC_DRUN_02_8_SymlinkMountKeepsTypedPath` | symlink `--mount`는 source=실제 경로, target=적은 경로 |
| TC-DRUN-02-9 | 02 | 정상 | `test_TC_DRUN_02_9_UsdpImageOverride` | `USDP_IMAGE=usdp:20261019`면 그 이미지로 실행 |
| TC-DRUN-02-10 | 02 | 정상 | `test_TC_DRUN_02_10_ForegroundExitCodePassedThrough` | docker가 7로 끝나면 run.sh도 7 |
| TC-DRUN-02-11 | 02 | 경계 | `test_TC_DRUN_02_11_LogUnderSymlinkMountUsesTypedPath` | symlink `--mount` 아래 로그: `USDP_LOG`=적은 경로, 호스트 파일은 실제 경로 |
| TC-DRUN-03-1 | 03 | 정상 | `test_TC_DRUN_03_1_SingleGpuQuotedDeviceRequest` | argv에 `--gpus`, `"device=2"`(따옴표 포함 한 원소), `--label usdp.gpu=2`, `USDP_GPU_CHECK=1` |
| TC-DRUN-03-2 | 03 | 경계 | `test_TC_DRUN_03_2_NoGpuPrintsNotice` | `--gpus` 없음, stderr `GPU: none` |
| TC-DRUN-03-3 | 03 | 정상(통합) | `docker/accept.sh` `TC_DRUN_03_3_RealContainerSeesOneGpu` | `nvidia-smi -L` 1줄, UUID 일치. `--gpu` 없으면 장치 0 |
| TC-DRUN-04-1 | 04 | 정상 | `test_TC_DRUN_04_1_DetachedArgvAndLogCreated` | `-d --name usdp-t`, `TINI_KILL_PROCESS_GROUP=1`, `USDP_LOG=<절대경로>`, 로그 파일 생성 |
| TC-DRUN-04-2 | 04 | 정상 | `test_TC_DRUN_04_2_EntryAppendsExitCode` (rc 0, 3) | 로그 마지막 줄 `[run.sh] exit 0` / `exit 3` |
| TC-DRUN-04-3 | 04 | 오류 | `test_TC_DRUN_04_3_EntryRecordsSigterm143` | 프로세스 그룹에 SIGTERM → `[run.sh] exit 143`, 자식 종료 |
| TC-DRUN-04-4 | 04 | 오류(통합) | `docker/accept.sh` `TC_DRUN_04_4_RealDockerStopLogs143` | `docker stop` 뒤 로그 끝 `exit 143` |
| TC-DRUN-04-5 | 04 | 정상 | `test_TC_DRUN_04_5_ForegroundSignalsWholeGroup` | 포그라운드 argv에도 `TINI_KILL_PROCESS_GROUP=1` |
| TC-DRUN-04-6 | 04 | 오류 | `test_TC_DRUN_04_6_StopDuringCudaCheckEndsJob` (SIGTERM, SIGINT) | CUDA 확인 중 신호 → 종료 143/130, 명령 미실행, 로그 끝 종료 줄 |
| TC-DRUN-04-7 | 04 | 오류 | `test_TC_DRUN_04_7_SignalToEntryDuringCheckHonored` | CUDA 확인 중 entry.sh에만 SIGINT(탐침은 살아 끝남) → 종료 130, 명령 미실행, 로그 끝 `exit 130` |
| TC-DRUN-05-1 | 05 | 오용 | `test_TC_DRUN_05_1_GpuGuardSweep` (3,500) | §4.1 G 규칙과 전건 일치 |
| TC-DRUN-05-2 | 05 | 오용 | `test_TC_DRUN_05_2_DataMountGuardSweep` (60) | M 규칙과 전건 일치, 거부는 5초 안 |
| TC-DRUN-05-3 | 05 | 오용 | `test_TC_DRUN_05_3_RunConditionGuardSweep` (480) | R 규칙과 전건 일치, 이미지 없음·데몬 오류 사유 구분 |
| TC-DRUN-05-4 | 05 | 오용 | `test_TC_DRUN_05_4_InContainerCudaGuardSweep` (10) | C 규칙과 전건 일치 |
| TC-DRUN-05-5 | 05 | 오류(통합) | `docker/accept.sh` `TC_DRUN_05_5_RealNoGpuInjectedRefused` | `--gpus` 없이 `USDP_GPU_CHECK=1` → 종료 3, 명령 미실행 |
| TC-DRUN-05-7 | 05 | 오용 | `test_TC_DRUN_05_7_CommaPathsRefused` (4) | `--mount` 원본·대상, 저장소, HOME 경로의 쉼표 → 종료 1 |
| TC-DRUN-05-8 | 05 | 오용 | `test_TC_DRUN_05_8_OptionLookingValuesRejected` (6) | `--name --force`, `--name -x`, `--name 'a b'`, `--log ''`, `--gpu --force`, `--mount --offline` → 종료 2 |
| TC-DRUN-05-9 | 05 | 경계 | `test_TC_DRUN_05_9_SweepSizesMatchSpec` | G·M·R·C 건수 = 3,500·60·480·10 |
| TC-DRUN-05-10 | 05 | 정상 | `test_TC_DRUN_05_10_ForcePrintsNotice` | `--force`로 넘기면 stderr에 넘긴 사실 |
| TC-DRUN-05-11 | 05 | 오용 | `test_TC_DRUN_05_11_LogShadowedByFileMountRefused` | 로그 경로에 정확히 붙인 파일 `--mount` → 종료 1 |
| TC-DRUN-05-12 | 05 | 오용 | `test_TC_DRUN_05_12_ProbeTimeoutValidated` (4) | `USDP_PROBE_TIMEOUT` `abc`·`0`·`-1` → 종료 2, `0.01` → 시작 |
| TC-DRUN-05-13 | 05 | 오용 | `test_TC_DRUN_05_13_BusyCheckFailsClosed` | 라벨 조회 `docker ps` 실패 → 종료 1 |

## 5 단위 시험 결과
| 일자 | 환경 | 명령 | 결과 |
|---|---|---|---|
| 2026-10-19 | 옛 이미지 `sr/unified-platform:v2.0`로 띄운 uid 1017 임시 컨테이너(새 이미지 빌드 전 부트스트랩), Python 3.10.12, pytest 9.0.3 | `docker run --rm --user 1017:1017 -e HOME=/tmp -e PYTHONDONTWRITEBYTECODE=1 --mount type=bind,source=<R>,target=<R> -w <R> sr/unified-platform:v2.0 python3 -m pytest -q -p no:cacheprovider tests/test_docker_run.py` | 구현 전 2226 failed(run.sh·entry.sh 없음). 구현 뒤 2226 passed, 101.95 s |
```text
2226 passed in 101.95s (0:01:41)
```

| 일자 | 환경 | 명령 | 결과 |
|---|---|---|---|
| 2026-10-19 | `usdp:20261019`(`112b82a375dd`), 실제 docker | `docker/accept.sh --only …,TC_DRUN_01_3` | TC-DRUN-01-3 PASS: 컨테이너가 만든 `own_probe`와 컨테이너 안 `~/.cache`의 소유자가 jshong |
| 2026-10-19 | 같은 이미지 안 | `docker/run.sh --offline -- … pytest -q -rs -p no:cacheprovider`(전체) | `tests/test_docker_run.py` 2226건 포함 2517 passed, SKIPPED 0 |
| 2026-10-19 | 같은 이미지, 실제 docker, GPU 2 | `docker/accept.sh --gpu 2 --only …,TC_DRUN_03_3,TC_DRUN_04_4,TC_DRUN_05_5` | TC-DRUN-03-3 PASS: `--gpu 2`면 `nvidia-smi -L` 1줄이고 UUID `GPU-57f64a18-4151-1d8f-b1e1-d5311d7630d2`와 같다. `--gpu`가 없으면 torch 장치 0개, stderr `GPU: none`. TC-DRUN-04-4 PASS: `docker stop` 뒤 로그 끝 `[run.sh] exit 143`. TC-DRUN-05-5 PASS: run.sh 없이 `--gpus` 없는 컨테이너에 `USDP_GPU_CHECK=1`을 주면 종료 3, `entry.sh: refused`, 명령(`touch`)은 돌지 않는다. 같은 명령에서 `USDP_GPU_CHECK`를 빼면 파일이 생기므로 판정이 실행을 잡아낸다 |

| 일자 | 환경 | 명령 | 결과 |
|---|---|---|---|
| 2026-10-19 | `usdp:20261019`(`112b82a375dd`) 안, 스텁 | `docker/run.sh --offline -- env PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q -p no:cacheprovider tests/test_docker_run.py` (+ DOCK-15 2건) | 브랜치 리뷰 반영. 반영 전 5 failed: TC-DRUN-02-6 파일 입력(`-i` 없음), TC-DRUN-05-3 2건(다른 경로·ro `--mount` 아래 로그로 시작됨), DOCK-15 2건. 반영 뒤 3208 passed(이 파일 3206). 뮤턴트: GPU 목록 조회 타임아웃 제거 16 failed(스텁 수정 전에는 400건 모두 통과), 사용 중 조회 실패를 빈 값으로 20 failed, 사용 중 조회 타임아웃 제거 16 failed, `--network none` 제거 1 failed |
| 2026-10-19 | 같은 이미지, 실제 docker, GPU 2, 21:06–21:20 | `docker/accept.sh --gpu 2 --out results/docker_accept_20261019_fix` | 15건 모두 PASS(DRUN 01-3·03-3·04-4·05-5 포함). pytest 3499 passed, SKIPPED 0 |
```text
3208 passed in 163.45s (0:02:43)
```

| 일자 | 환경 | 명령 | 결과 |
|---|---|---|---|
| 2026-10-19 | `usdp:20261019`(`112b82a375dd`) 안, 스텁 | `docker/run.sh --offline -- env PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q -p no:cacheprovider tests/test_docker_run.py` | 미룬 Minor 처리(v0.3). 반영 전 124 failed / 3962 passed: 01-5, 02-7, 02-8, 03-1(라벨), 04-5, 04-6 2건, 05-1 4건(다른 GPU의 usdp 컨테이너), 05-2 6건(멈춘 sshfs, 30초 대기), 05-3 100건(데몬 오류 사유·이름 필터), 05-7 3건, 05-8 4건. 반영 뒤 4086 passed, 346 s |
| 2026-10-20 | `usdp:20261019`(`7fd2f0ce9454`, 다시 빌드), 실제 docker, GPU 2, 00:57–01:15 | `docker/accept.sh --gpu 2 --out results/docker_accept_20261020` | 15건 모두 PASS. pytest 4382 passed(이 파일 4086), SKIPPED 0 |
```text
4086 passed in 346.19s (0:05:46)
```

| 일자 | 환경 | 명령 | 결과 |
|---|---|---|---|
| 2026-10-20 | `usdp:20261019`(`7fd2f0ce9454`) 안, 스텁 | `docker/run.sh --offline -- … pytest -q -p no:cacheprovider tests/test_docker_run.py tests/test_docker_image.py` | 2차 수정(v0.4). 반영 전 10 failed: 01-6, 02-11, 04-7, 05-11, 05-12 4건, 05-13, DOCK-15-6. 반영 뒤 4114 passed(이 파일 4096) |
| 2026-10-20 | 같은 이미지, 실제 docker, GPU 2, 01:57–02:15 | `docker/accept.sh --gpu 2 --out results/docker_accept_20261020_final` | 15건 모두 PASS. pytest 4394 passed, SKIPPED 0 |
```text
4114 passed, 4 warnings in 447.95s (0:07:27)
```

## 6 미해결 · 결정
- 결정은 DockerEnv §6(Q4·Q5)을 따른다.
- "이미지 없음"과 "이미 쓰는 이름" 거부는 계획 단계에서 Review Focus로 더한 작성자 판단이다.
