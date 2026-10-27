# Docker 작업 환경 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 학습·AIMET export·QAIRT DLC 변환·pytest·영상 작업을 다시 만들 수 있는 Docker 이미지에서 jshong(uid 1017) 소유 결과로 돌리고, 검증이 끝나면 옛 root 컨테이너 `sr`를 지울 수 있게 한다.

**Architecture:** 이미지 하나(`usdp`)에 다음을 담는다: 고정한 파이썬 환경(178개), QAIRT 2.41 SDK와 전용 venv(`qairt-env` 안에서만 보임), ffmpeg 7.0.2. 작업은 호스트의 `docker/run.sh`가 사전 확인 뒤 `docker run --rm`으로 띄운다. 컨테이너 안 진입점은 저장소 마운트의 `docker/entry.sh`다. 합격 확인은 호스트의 `docker/accept.sh`가 TC별 함수로 돌린다.

**Tech Stack:** Docker 20.10.11 + buildx 0.23.0(기본 docker 드라이버), nvidia-container-toolkit 1.13.5, `nvidia/cuda:12.6.0-cudnn-devel-ubuntu22.04`, Python 3.10.12, bash, pytest 9.0.3.

**Spec:** `docs/modules/20261019-MOD-DockerEnv.md`(v0.2, 이미지·합격 확인·이관), `docs/modules/20261019-MOD-DockerRun.md`(v0.1, run.sh·entry.sh). 실행자는 두 문서와 이 계획을 함께 읽는다.

## Global Constraints
- 파이썬 패키지 버전은 하나도 바꾸지 않는다. torch 2.9.0, aimet-torch 2.29.0, onnx 1.17.0, protobuf 6.32.1, onnxscript 0.2.5, numpy 2.2.6, opencv-python·opencv-python-headless 4.12.0.88.
- 베이스: `nvidia/cuda:12.6.0-cudnn-devel-ubuntu22.04@sha256:60cfa1fb926e4830ec1dbee1b7511eeba0fee2333b16cbec79c356e2afc2f359`. Dockerfile 첫 줄은 `# syntax=docker/dockerfile:1`이고, 빌드는 `docker buildx build --builder default`로만 한다.
- QAIRT tarball sha256 `4e73e38f5e60e083b7c1e535ceaabb8a18e7523d912ad3b7dbe99d9217e3520d`. QAIRT 파일은 `git add`하지 않는다. 이미지는 `docker push`하지 않는다.
- ffmpeg: `https://files.pythonhosted.org/packages/a0/2d/43c8522a2038e9d0e7dbdf3a61195ecc31ca576fb1527a528c877e87d973/imageio_ffmpeg-0.6.0-py3-none-manylinux2014_x86_64.whl`, sha256 `c7e46fcec401dd990405049d2e2f475e2b397779df2519b544b8aab515195282`. 꺼낸 바이너리 md5는 `d5c698e0e98e5cf7de03fe14bd1cda6a`.
- UID·GID 빌드 인자 기본값은 1017이다. 스크립트에 `/home/<user>` 리터럴을 쓰지 않는다(`$HOME`, `SCRIPT_DIR`).
- 셸 스크립트 규칙
  - 첫 줄 `#!/bin/bash`, 다음 줄 `set -euo pipefail`. `entry.sh`만 `set -u`.
  - 변수는 UPPER_SNAKE, 함수는 snake_case이고 중괄호는 다음 줄에 둔다.
  - 모든 확장을 따옴표로 감싼다. 주석과 출력은 영어.
- 시험
  - pytest와 `@pytest.mark.parametrize`를 쓰고, 이름은 `test_TC_<MOD>_<nn>_<k>_<CamelCase>`로 짓는다.
  - 테스트 함수와 `accept.sh` TC 함수의 이름은 두 모듈 문서 §4.2 표에 적힌 이름을 글자 그대로 쓴다.
  - 스텁은 `tests/test_board_scripts.py`의 `_stubs` 방식을 따른다.
  - 시험은 컨테이너 안에서 `PYTHONDONTWRITEBYTECODE=1`과 `-p no:cacheprovider`로 돌린다. 저장소에 root 파일이나 캐시를 남기지 않는다.
- 출하 폴더 보호
  - `results/denoise_release_v1`와 `results/release_v3`에서 `convert_all.sh`를 돌리지 않는다.
  - 합격 확인 산출물은 `results/docker_accept_<YYYYMMDD>/`에만, 로그는 `logs/`에만 둔다.
- GPU는 `docker/run.sh --gpu N`으로만 고른다. 컨테이너 안에서 `CUDA_VISIBLE_DEVICES`를 쓰지 않는다. GPU 작업 전에 빈 GPU를 확인한다(10-19 기준 0·1·4번은 다른 사용자가 썼다).
- 옛 컨테이너와 이미지
  - 옛 컨테이너 `298b322c6c92`와 이미지 `sr/unified-platform:v2.0`은 소유자 승인 없이 정지·재시작·삭제하지 않는다.
  - 읽기 전용 `docker exec`와 `docker cp`는 된다.
  - 옛 이미지로 새 컨테이너를 `--user "$(id -u):$(id -g)"`로 띄우는 것도 된다.
- `rm -rf`는 쓰지 않는다(훅이 막는다). 파일은 하나씩 지운다.
- 커밋
  - 제목만 쓰고, 꼴은 `[20261019] <Component> <Verb> <what>`이다. attribution은 넣지 않는다.
  - 문서와 코드는 같은 커밋에 넣는다.
  - 커밋은 소유자가 승인한 뒤에만 한다.
- 보드 gst 실행과 화면 확인은 소유자 몫이다.

## Review Focus
1. `--` 뒤 인자에 공백·따옴표·한글·`--force`가 있어도 그대로 전달되어야 한다 → Task 1 TC-DRUN-02-4.
2. `usdp:latest`가 아직 없을 때(빌드 전) `run.sh`는 "docker/build.sh 먼저"라는 사유로 거부해야 한다 → Task 1 TC-DRUN-05-3(이미지 없음 값) + 메시지 단언.
3. 같은 `--name`으로 두 번 띄우면 거부하고 기존 로그를 건드리지 않아야 한다 → Task 1 TC-DRUN-05-3(이름 사용 중 값) + 로그 내용 불변 단언.
4. 한글·공백이 든 `--log`·`--mount` 경로도 동작해야 한다 → Task 1 TC-DRUN-02-2.
5. 컨테이너 안 한글 출력과 KST 시각(런 폴더 이름)이 맞아야 한다 → Task 2 TC-DOCK-01-5.

## 파일 구조
| 파일 | 새로/수정 | 책임 | Task |
|---|---|---|---|
| `docker/run.sh` | 새로 | 사전 확인과 `docker run` 조립(DockerRun §3.2) | 1 |
| `docker/entry.sh` | 새로 | 컨테이너 안 CUDA 확인, 백그라운드 래퍼, `exec`(§3.3) | 1 |
| `tests/test_docker_run.py` | 새로 | 스텁 시험과 곱집합 2,214건 | 1 |
| `requirements.txt` | 수정 | 178개 고정 | 2 |
| `docker/Dockerfile`, `.dockerignore`, `docker/build.sh` | 새로 | 이미지(DockerEnv §2.4, §3.2) | 2–4 |
| `tests/test_docker_image.py` | 새로 | 이미지 안 환경 시험 | 2–4 |
| `docker/qairt-env`, `docker/qairt-requirements.txt` | 새로 | QAIRT 범위(§3.3) | 4 |
| `docker/accept.sh` | 새로 | 합격 확인 TC 함수(§3.4) | 6–8 |
| `README.md`, `tools/README.md`, `examples/README.md`, `examples/deployment/README.md`, `examples/deployment/gen_convert_all.py`(docstring), `tests/test_gen_convert_all.py`(docstring), `results/denoise_release_v1/MANIFEST.md` | 수정 | 컨테이너 사용법과 서버 변환 | 1, 2, 4 |
| 두 모듈 문서 §5 | 수정 | 시험 결과 원문 | 각 Task |

시험을 돌리는 명령(`<R>`은 저장소 절대 경로, 다음 둘에서 공통으로 쓴다):
- **Task 1만, 새 이미지가 없을 때:** 옛 이미지로 새 컨테이너를 띄운다.
  `docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp -e PYTHONDONTWRITEBYTECODE=1 --mount type=bind,source=<R>,target=<R> -w <R> sr/unified-platform:v2.0 python3 -m pytest -q -p no:cacheprovider <시험 파일>`
- **Task 2 이후:** `docker/run.sh --offline -- env PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q -rs -p no:cacheprovider <시험 파일>`

---

### Task 1: 실행 스크립트 `run.sh` + `entry.sh` (DockerRun MREQ-DRUN-01~05 단위 시험)

**Files:**
- Create: `docker/run.sh`, `docker/entry.sh`, `tests/test_docker_run.py`
- Modify: `tools/README.md:21`(GPU 선택 문장), `docs/modules/20261019-MOD-DockerRun.md` §5

**Interfaces:**
- Produces
  - `docker/run.sh [--gpu N] [--name NAME] [--log FILE] [--mount SRC[:DST[:ro]]]... [--offline] [--force] -- CMD...`
  - 종료 코드: 포그라운드는 CMD의 코드, 백그라운드 시작은 0, 거부는 1, 사용법 오류는 2. 거부 사유는 stderr에 `run.sh: refused: <사유>`로 낸다.
  - 환경 변수: `USDP_IMAGE`(기본 `usdp:latest`), `USDP_DATA_MOUNT`(기본 `/mnt/data_server`), `USDP_PROBE_TIMEOUT`(기본 10).
  - `REPO`는 `$(cd "$SCRIPT_DIR/.." && pwd)`로 정한다. 시험은 run.sh와 entry.sh를 tmp의 가짜 저장소 `docker/`로 복사해 돌린다(실제 저장소에 로그가 생기지 않게).
- run.sh가 부르는 외부 명령의 정확한 꼴(스텁이 이 꼴에 답한다):
  - `docker image inspect -f '{{index .Config.Labels "usdp.uid"}} {{index .Config.Labels "usdp.gid"}} {{index .Config.Labels "usdp.home"}}' "$USDP_IMAGE"`
  - `docker ps -a -q --filter "name=^usdp-$NAME\$"`
  - `docker ps -q --filter "name=^usdp-"`와 `docker inspect -f '{{range .HostConfig.DeviceRequests}}{{range .DeviceIDs}}{{.}} {{end}}{{end}}' <id>`(GPU 번호를 공백으로 나열. JSON 해석 없음)
  - `nvidia-smi --query-gpu=index,uuid --format=csv,noheader`
  - `nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader`
  - `findmnt -rn -M "$USDP_DATA_MOUNT" -o FSTYPE`
  - `timeout "$USDP_PROBE_TIMEOUT" ls -A "$USDP_DATA_MOUNT"`
  - `id -u`, `id -g`
- docker run 인자 배열(시험이 단언하는 원소)
  - 공통: `run --rm --init --user U:G --ipc host`
  - bind 3개: `--mount type=bind,source=$REPO,target=$REPO`, `--mount type=bind,source=/mnt,target=/mnt,bind-propagation=rslave`, `--mount type=bind,source=$HOME/.cache/torch,target=<usdp.home>/.cache/torch`
  - 작업 폴더: `-w $REPO`
  - 추가 `--mount`: `type=bind,source=SRC,target=DST[,readonly]`
  - `--offline`이면 `--network none`
  - GPU: `--gpus` 다음 원소가 따옴표까지 포함한 `"device=N"`, 그리고 `-e USDP_GPU_CHECK=1`
  - 백그라운드: `-d --name usdp-NAME -e TINI_KILL_PROCESS_GROUP=1 -e USDP_LOG=<절대경로>`
  - 포그라운드에서 stdin·stdout이 TTY면 `-it`
  - 끝: `$USDP_IMAGE $REPO/docker/entry.sh CMD...`
- `entry.sh` 계약
  - 환경 변수 `USDP_GPU_CHECK`, `USDP_LOG`, `USDP_CUDA_TIMEOUT`(기본 60).
  - CUDA 확인 명령: `python3 -c 'import sys, torch; sys.exit(0 if torch.cuda.is_available() and torch.cuda.device_count() == 1 else 1)'`.
  - 거부하면 종료 3.
  - 로그 끝 줄은 `[run.sh] exit <rc>` 꼴이다.

- [ ] **Step 1: 시험 작성 — `tests/test_docker_run.py`**
  - **도우미**
    - `_repo(tmp_path)`: run.sh와 entry.sh를 `tmp_path/repo/docker/`로 복사한다.
    - `_env(tmp_path, **state)`: 스텁 `docker`·`nvidia-smi`·`findmnt`·`id`·`ls`를 쓰고 HOME=`tmp_path/home`으로 둔다. 스텁 `docker run`은 `printf '%s\0' "$@" > argv.bin`으로 인자를 남긴다.
    - `_run(args, env) -> (rc, stdout, stderr, argv|None)`.
    - GPU 스텁: 0~4번, UUID는 `GPU-0000000N`.
  - **기능 시험(DockerRun §4.2):** 01-1, 01-2, 02-1~4, 03-1, 03-2, 04-1, 04-2(rc 0·3 parametrize), 04-3. 단언 예:
```python
def test_TC_DRUN_03_1_SingleGpuQuotedDeviceRequest(tmp_path):
    rc, _, _, argv = _run(["--gpu", "2", "--", "true"], _env(tmp_path))
    assert rc == 0 and argv[argv.index("--gpus") + 1] == '"device=2"' and "USDP_GPU_CHECK=1" in argv

def test_TC_DRUN_02_4_CommandArgvPassedVerbatim(tmp_path):
    cmd = ["python3", "-c", 'print("a b")', "--force", "한글"]
    rc, _, _, argv = _run(["--", *cmd], _env(tmp_path))
    assert rc == 0 and argv[-len(cmd):] == cmd
```
  - **TC-DRUN-04-3:** `subprocess.Popen(..., start_new_session=True)`로 entry.sh가 `sleep 30`을 돌리게 한다. `os.killpg(pgid, SIGTERM)`을 보낸 뒤 로그 마지막 줄 `[run.sh] exit 143`을 단언하고, 그룹에 남은 프로세스가 없음을 단언한다.
  - **곱집합 4개:** DockerRun §4.1의 표 그대로다. `itertools.product`로 만들고 `ids`를 붙인다. 각 경우는 기대 함수 `_expect_g/_m/_r/_c(case) -> bool`과 비교한다.
    - 통과: `rc == 0`이고 `argv`가 있다.
    - 거부: `rc in (1, 2)`이고 argv가 없고, stderr에 `run.sh: refused:` 또는 `usage`가 있다.
    - G의 `--gpu` 두 번, `--` 없음, `--forc`는 rc 2다.
    - M의 상태는 tmp 폴더와 스텁(`findmnt` 출력, `ls` 권한 거부·멈춤)으로 만든다.
    - R의 "이미지 없음"은 거부 메시지에 `docker/build.sh`가 있는지 단언한다(RF2). "이름 사용 중"은 기존 로그 내용이 바뀌지 않았는지 단언한다(RF3).
    - C는 entry.sh를 직접 돌리고 스텁 `python3`을 쓴다. 명령이 실행됐는지는 `touch` 파일로 판정한다.
  - 시간 제한은 `USDP_PROBE_TIMEOUT=0.5`, `USDP_CUDA_TIMEOUT=0.5`로 줄인다.

- [ ] **Step 2: 실패 확인**
  - 실행: 위 "Task 1만" 명령, 시험 파일 `tests/test_docker_run.py`
  - 기대: 전부 실패(run.sh 없음)

- [ ] **Step 3: `docker/run.sh` 구현.** DockerRun §3.2의 1~7을 함수 `usage`, `refuse`, `parse_args`, `check_image`, `check_paths`, `check_data_mount`, `check_gpu`, `docker_args`, `main`으로 나눠 구현한다.
  - `parse_args`는 `--` 앞만 해석한다. `--gpu`는 한 번만 받는다. `--log`는 `--name`이 있을 때만 받는다.
  - 데이터 마운트 거부 메시지에는 "root must remount with sshfs -o allow_other"를 넣는다.

- [ ] **Step 4: `docker/entry.sh` 구현.** DockerRun §3.3대로 한다. 백그라운드 래퍼는 `trap : TERM INT`이다(`trap ''` 금지).

- [ ] **Step 5: 통과 확인**
  - 실행: Step 2와 같은 명령
  - 기대: `2226 passed`(기능 12 + 곱집합 2,214). 실행 시간이 5분을 넘으면 보고한다.

- [ ] **Step 6: 문서**
  - `tools/README.md:21`의 GPU 문장을 "GPU는 `docker/run.sh --gpu N`으로 고릅니다(컨테이너 안에서는 cuda:0, `CUDA_VISIBLE_DEVICES`는 쓰지 않음)"로 바꾼다.
  - DockerRun §5에 Step 5의 출력 원문을 넣는다(일자, 환경 = 옛 이미지 부트스트랩).

- [ ] **Step 7: 커밋(소유자 승인 뒤)**
```bash
git add docker/run.sh docker/entry.sh tests/test_docker_run.py tools/README.md docs/modules/20261019-MOD-DockerRun.md
git commit -m "[20261019] DockerRun Add run.sh and entry.sh with preflight guards"
```

### Task 2: 이미지 기본 — requirements, Dockerfile, build.sh (MREQ-DOCK-01)

**Files:**
- Create: `docker/Dockerfile`, `.dockerignore`, `docker/build.sh`, `tests/test_docker_image.py`
- Modify: `requirements.txt`, `README.md:7,32-49`(환경 절, 학습 명령, 다중 GPU 문장), `docs/modules/20261019-MOD-DockerEnv.md`(§1에 DockerRun 문서 경로, §5)

**Interfaces:**
- Consumes: Task 1의 `docker/run.sh`.
- Produces
  - 이미지 `usdp:<YYYYMMDD>`와 `usdp:latest`. 라벨 `usdp.uid=1017`, `usdp.gid=1017`, `usdp.home=/home/jshong`. 기본 사용자는 jshong.
  - `docker/build.sh`의 환경 변수 `QAIRT_TARBALL`(기본 `$HOME/Code/qualcomm-vlm/env/QAIRT_2.41.tar.gz`), `USDP_TAG`(기본 `date +%Y%m%d`).
  - 이 Task의 Dockerfile에는 QAIRT·ffmpeg 단계가 아직 없다. build.sh는 처음부터 최종 꼴이고, named context를 넘겨도 쓰지 않으면 무해하다.

- [ ] **Step 1: `requirements.txt` 만들기**
  - 옛 컨테이너 `docker exec 298b322c6c92 pip freeze`(읽기 전용)에서 DockerEnv §2.5 표의 39개 이름을 빼고 이름순(`sort -f`)으로 쓴다.
  - 확인: `wc -l requirements.txt` → `178`. `grep -c -E '^(aimet-torch==2.29.0|thop==0.1.1.post2209072238|scikit-learn==1.7.2)$' requirements.txt` → `3`. `grep -c -i -E '^(streamlit|basicsr|tb-nightly|lpips|mamba)' requirements.txt` → `0`.

- [ ] **Step 2: 시험 작성 — `tests/test_docker_image.py`의 TC-DOCK-01-1~5(DockerEnv §4.2).**
  - 01-1: `importlib.metadata.distributions()` 중 `sys.prefix` 쪽(`/usr/local`, `/usr`) 배포판에서 pip·setuptools·wheel을 뺀 `{정규화 이름: 버전}`이 `requirements.txt`를 해석한 dict와 같아야 한다.
  - 01-3: `cv2/cv2.abi3.so`의 sha256(urlsafe base64)이 `distribution("opencv-python-headless").files` RECORD 값과 같아야 한다.
  - 01-5: `subprocess.run(["python3", "-c", "print('한글 출력')"])`의 rc가 0이고, `"KST" in time.tzname`이고, 다음이 모두 맞아야 한다: `os.environ["NUMPY_MADVISE_HUGEPAGE"] == "0"`, `PYTHONUNBUFFERED == "1"`, `NVIDIA_VISIBLE_DEVICES == "void"`(pytest는 `--gpu` 없이 돈다), `LANG == "C.UTF-8"`, `os.stat(os.path.expanduser("~/.cache")).st_uid == os.getuid()`.
  - 이 시험들은 skip하지 않는다. 이미지 밖에서는 실패해야 한다.

- [ ] **Step 3: 실패 확인(차이를 잡는지)**
  - 실행: "Task 1만" 명령(옛 이미지), 시험 파일 `tests/test_docker_image.py`
  - 기대: TC-DOCK-01-1 FAIL(streamlit 등 초과, aimet-torch 없음). 01-5도 FAIL(ENV 없음).

- [ ] **Step 4: `.dockerignore`, `docker/Dockerfile`(베이스·apt·pip·사용자·ENV), `docker/build.sh` 작성**
  - `.dockerignore`는 `*`, `!requirements.txt`, `!docker/`이다.
  - Dockerfile은 DockerEnv §2.4의 베이스, apt, pip 1~5, 사용자, ENV를 그대로 따른다.
  - build.sh는 §3.2의 1~5이고, 시작·끝에 `echo "------ Build Start! ------"`, `echo "Build Finish!!"`를 넣는다.

- [ ] **Step 5: 빌드**
  - 실행: `docker/build.sh`
  - 기대: 종료 0, 마지막에 이미지 ID, `docker image inspect usdp:latest -f '{{.Config.User}}'` → `1017:1017`.
  - 빌드 로그에 `pip check` 통과(`No broken requirements found.`)가 있어야 한다.
  - 첫 빌드의 `docker/dockerfile:1` frontend digest를 빌드 로그에서 찾아 syntax 줄에 고정한다.

- [ ] **Step 6: 통과 확인**
  - 실행: "Task 2 이후" 명령, 시험 파일 `tests/test_docker_image.py tests/test_docker_run.py`
  - 기대: `2231 passed`(이미지 5 + 실행 2226), SKIPPED 0줄.

- [ ] **Step 7: 문서**
  - `README.md`의 환경 절(32-35행)을 Docker 사용법으로 바꾼다. 담을 내용:
    - `docker/build.sh` 한 번 실행
    - 작업은 `docker/run.sh [--gpu N] -- 명령`
    - 호스트 python3(3.8)은 작업 환경이 아님
    - QAIRT tarball 위치 변수
  - 학습 명령(41-44행)은 `docker/run.sh --gpu <N> -- python3 tools/train.py --config …`로 바꾼다. `NUMPY_MADVISE_HUGEPAGE`는 이미지 기본값이라는 문장으로 47행을 고친다.
  - 7·49행의 다중 GPU 문장은 "코드상 `accelerate launch`로 가능하지만 `docker/run.sh`는 GPU 한 장만 붙인다"로 고친다.
  - DockerEnv §5에 Step 6의 출력을 넣는다.

- [ ] **Step 8: 커밋(승인 뒤)**
```bash
git add .dockerignore docker/Dockerfile docker/build.sh requirements.txt tests/test_docker_image.py README.md docs/modules/20261019-MOD-DockerEnv.md
git commit -m "[20261019] DockerEnv Add the image Dockerfile, build script and pinned requirements"
```

### Task 3: ffmpeg 7.0.2 (MREQ-DOCK-03)

**Files:**
- Modify: `docker/Dockerfile`(pip 층 다음에 ffmpeg 층), `tests/test_docker_image.py`, DockerEnv §5

- [ ] **Step 1: 시험 작성.** TC-DOCK-03-1과 03-2를 쓴다.
  - 03-1: `/usr/local/bin/ffmpeg`의 md5가 `d5c698e0e98e5cf7de03fe14bd1cda6a`이고, `ffmpeg -version` 첫 줄에 `7.0.2`가 있어야 한다.
  - 03-2: `os.access(…, os.X_OK)`가 참이고, `ffmpeg -hide_banner -f lavfi -i testsrc=d=0.1 -c:v libx264 -f null -`가 종료 0이어야 한다.
- [ ] **Step 2: 실패 확인.** 실행: "Task 2 이후" 명령, `-k DOCK_03` → 2 failed.
- [ ] **Step 3: ffmpeg 층 구현.** 다음 순서로 한다.
  1. `curl -fsSL "$URL" -o /tmp/ff.whl`
  2. `echo "$SHA  /tmp/ff.whl" | sha256sum -c -`
  3. `python3 -c` zipfile로 `imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2`를 꺼낸다.
  4. `install -m 0755 … /usr/local/bin/ffmpeg`, 휠은 지운다.
- [ ] **Step 4: 다시 빌드하고 통과 확인.** `docker/build.sh`를 돌린 뒤 `tests/test_docker_image.py`를 실행한다. 기대: `7 passed`.
- [ ] **Step 5: 커밋(승인 뒤)** `[20261019] DockerEnv Add the static ffmpeg 7.0.2 to the image`

### Task 4: QAIRT 2.41 + `qairt-env` + 서버 변환 문서 (MREQ-DOCK-02)

**Files:**
- Create: `docker/qairt-env`, `docker/qairt-requirements.txt`
- Modify: `docker/Dockerfile`(ffmpeg 다음 QAIRT 층), `tests/test_docker_image.py`, `README.md:35,90`, `examples/README.md:34,42,157`, `examples/deployment/README.md:4-10,43-61,332-345,377-392`, `examples/deployment/gen_convert_all.py:3-4`(docstring), `tests/test_gen_convert_all.py:1-3`(docstring), `results/denoise_release_v1/MANIFEST.md:54`(경고 한 줄), DockerEnv §5

**Interfaces:**
- Produces
  - `/usr/local/bin/qairt-env CMD...`: 인자가 없으면 종료 2. 변수는 DockerEnv §3.3대로 설정한다.
  - SDK 위치 `/opt/qairt`(root 소유, `examples/` 없음). venv는 `/opt/qairt-venv`.
- `docker/qairt-requirements.txt` 처음 내용: `numpy==1.26.4`, `onnx==1.16.1`, `protobuf==3.20.3`, `PyYAML==6.0.2`, `packaging==24.0`. 나중에 Task 7에서 venv 전체 freeze로 바꾼다.

- [ ] **Step 1: 시험 작성.** TC-DOCK-02-1~4(DockerEnv §4.2)를 쓴다.
  - 02-1: `qairt-env bash -c 'command -v qairt-converter qairt-quantizer snpe-dlc-graph-prepare snpe-dlc-info qairt-dlc-diff'`의 5줄이 모두 `/opt/qairt/`로 시작하고, `qairt-env python3 -c 'import numpy; print(numpy.__version__)'`이 `1.26.4`이고, `qairt-env qairt-converter --help`가 종료 0이어야 한다.
  - 02-2(누수):
    - `shutil.which()`로 다섯 도구가 모두 None이고, `bash -lc 'command -v qairt-converter'` 출력이 비어 있다.
    - `python3 -c 'import qti'` rc가 0이 아니다.
    - PATH·LD_LIBRARY_PATH·PYTHONPATH에 `/opt/qairt`가 없고, `ldconfig -p`에도 없다.
    - `numpy.__version__ == "2.2.6"`.
  - 02-3: 변환 경로 ELF(`/opt/qairt/bin/x86_64-linux-clang/*`, `/opt/qairt/lib/x86_64-linux-clang/*.so`, `/opt/qairt/lib/python/qti/aisw/**/*.so`. `qti/tvm`은 제외)에 `LD_LIBRARY_PATH=/opt/qairt/lib/x86_64-linux-clang ldd`를 돌려 `not found` 줄이 0개여야 한다.
  - 02-4: `qairt-env`를 인자 없이 부르면 rc 2.
- [ ] **Step 2: 실패 확인.** 실행: `-k DOCK_02` → 4 failed.
- [ ] **Step 3: QAIRT 층, `qairt-env`, `qairt-requirements.txt` 구현.**
  - 층: `RUN --mount=type=bind,from=qairt,target=/ctx` 아래 `tar --no-same-owner --strip-components=1 --exclude='QAIRT_2.41/examples' -xzf /ctx/QAIRT_2.41.tar.gz -C /opt/qairt`.
  - venv: `python3 -m venv /opt/qairt-venv && /opt/qairt-venv/bin/pip install --no-cache-dir -r qairt-requirements.txt && /opt/qairt-venv/bin/pip check`.
  - ldd 검사를 빌드 단계에 넣는다(02-3과 같은 대상). `not found`면 `exit 1`.
- [ ] **Step 4: 다시 빌드하고 통과 확인.** `tests/test_docker_image.py tests/test_gen_convert_all.py`를 실행한다.
  - 기대: 모두 passed, SKIPPED 0줄.
  - 빌드나 02-1이 protobuf·PyYAML 때문에 실패하면, venv 안의 그 핀만 바꾸고 바꾼 이유를 DockerEnv §6에 적는다.
- [ ] **Step 5: 문서**
  - 서버 변환을 기본 흐름으로 쓴다: 복사본 폴더에서 `docker/run.sh -- qairt-env bash <폴더>/convert_all.sh`. 회사 PC tgz 흐름은 "다른 PC에서 할 때"로 남긴다.
  - `examples/deployment/README.md`의 '도는 곳' 표
    - `gen_convert_all.py`, `make_bench_raw.py` → "작업 컨테이너"
    - `convert_all.sh` → "작업 컨테이너(`qairt-env`) 또는 변환 PC"
  - §3 명령은 `docker/run.sh -- python3 …`로 바꾸고, 보드 scp는 호스트에서 한다.
  - docstring
    - `gen_convert_all.py:3-4`를 "the job container (`qairt-env`) or any machine with the QAIRT SDK"로 바꾼다. 생성 스크립트 머리말 문자열(:114)은 바꾸지 않는다(해시 고정).
    - `tests/test_gen_convert_all.py` docstring도 같은 뜻으로 고친다.
  - `MANIFEST.md:54` 위에 경고 한 줄을 넣는다: "> 이 서버의 작업 컨테이너에서는 이 폴더에서 바로 돌리지 말고 복사본에서 돌린다(출하 DLC를 덮어씀). 절차: `docs/modules/20261019-MOD-DockerEnv.md` §3.4."
- [ ] **Step 6: 커밋(승인 뒤)** `[20261019] DockerEnv Add QAIRT 2.41 behind qairt-env and document server conversion`

### Task 5: 가중치 4개 이관 (TC-DOCK-14-1)

**Files:** 저장소 파일은 없다(호스트 `~/.cache/torch/hub/`). 결과는 Task 6 커밋의 §5에 넣는다.

- [ ] **Step 1: 원본 md5 기록.** `docker exec 298b322c6c92 md5sum`으로 다음 4개의 md5를 기록한다:
  - `/root/.cache/torch/hub/checkpoints/alexnet-owt-7be5be79.pth`
  - `/root/.cache/torch/hub/checkpoints/vgg19-dcbb9e9d.pth`
  - `/root/.cache/torch/hub/pyiqa/LPIPS_v0.1_alex-df73285e.pth`
  - `/root/.cache/torch/hub/pyiqa/niqe_modelparameters.mat`
- [ ] **Step 2: 복사.** `mkdir -p ~/.cache/torch/hub/checkpoints ~/.cache/torch/hub/pyiqa`를 한 뒤, 파일마다 `docker cp 298b322c6c92:<원본> ~/.cache/torch/hub/<같은 하위 경로>`를 한다(호스트 CLI가 쓰므로 jshong 소유).
- [ ] **Step 3: 확인**
  - 호스트 `md5sum` 4개가 Step 1과 같고, `stat -c %U`가 jshong이어야 한다.
  - 이어서 `docker/run.sh --offline -- env PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q -p no:cacheprovider tests/test_losses.py tests/test_tools_evaluate.py`를 돌린다. 기대: 모두 passed(네트워크 없이 캐시로 동작).

### Task 6: `accept.sh` — pytest·export 합격 확인 (MREQ-DOCK-09, 10, DRUN-01 통합)

**Files:**
- Create: `docker/accept.sh`
- Modify: DockerEnv §5, DockerRun §5

**Interfaces:**
- Produces
  - `docker/accept.sh [--gpu N] [--only TC_A,TC_B] [--out DIR]`. `--out` 기본값은 `$REPO/results/docker_accept_$(date +%Y%m%d)`이고, 이미 있으면 종료 1.
  - TC 함수마다 한 줄을 낸다: `PASS TC-…`, `FAIL TC-…: <이유>`, `NOT RUN TC-…: <이유>`. 마지막에 개수를 요약하고, PASS가 아닌 것이 있으면 종료 1.
  - 상수 `BASE_TESTS=280`.
- 이 Task의 함수: `TC_DOCK_09_1_PytestAllPassNoSkip`, `TC_DOCK_09_2_PytestLeavesNoFiles`, `TC_DOCK_10_1_DenoiseExportMatchesShipped`, `TC_DOCK_10_2_IrX4ExportMatchesShipped`, `TC_DOCK_10_3_ExportOutputsOwnedNoTraceback`, `TC_DRUN_01_3_RealOutputsOwnedByCaller`. 명령과 판정은 DockerEnv §3.4를 그대로 따른다.
- 판정 세부
  - 09-1: 기대 수는 `BASE_TESTS` + `pytest --collect-only -q tests/test_docker_run.py tests/test_docker_image.py`의 항목 수다. 요약 줄은 `^<기대 수> passed`여야 하고, `skipped|failed|error`가 없어야 한다.
  - 10-*: `param_encodings`는 호스트 `python3 -c 'import json,sys; …'`로 두 JSON의 `param_encodings` 키 값이 같은지 본다.

- [ ] **Step 1:** `accept.sh`의 골격(인자, 결과 줄, 요약)과 위 6개 함수를 작성한다.
- [ ] **Step 2: 실행.** `docker/accept.sh --only TC_DOCK_09_1,TC_DOCK_09_2,TC_DOCK_10_1,TC_DOCK_10_2,TC_DOCK_10_3,TC_DRUN_01_3`
  - 기대: 6줄 모두 PASS, 종료 0.
  - md5 기준값: DN `81ac40c78450489a6fac04c05569a2ec`, `x4_ir` `5de6458e5cfce8c4d0666b54727b4294`, `x4_ir_1024` `37d8f6ff051a97d1df9dbc2f8fa5434d`.
- [ ] **Step 3:** 출력 원문을 DockerEnv §5(TC-DOCK-09·10·14-1)와 DockerRun §5(TC-DRUN-01-3)에 넣는다.
- [ ] **Step 4: 커밋(승인 뒤)** `[20261019] DockerEnv Add accept.sh with the pytest and export gates`

### Task 7: DLC 합격 확인 + QAIRT venv 고정 (MREQ-DOCK-11, R1·R4)

**Files:**
- Modify: `docker/accept.sh`, `docker/qairt-requirements.txt`, DockerEnv §5

- [ ] **Step 1: 함수 작성.** `TC_DOCK_11_1_DenoiseDlcMatchesShipped`와 `TC_DOCK_11_2_DlcConversionDeterministic`를 쓴다.
  - 준비: k = 1, 2마다 `$OUT/dn_convert<k>/`에 `results/denoise_release_v1/convert_all.sh`와 `dn_bpp_half4_w16a16/qat/`만 복사한다(.dlc는 넣지 않는다).
  - `$OUT/shim/mktemp`를 만든다. `-d`면 `/tmp/tmp.nIupqkU2Au`를 만들어 그 경로를 출력하고, 아니면 `exec /usr/bin/mktemp "$@"`.
  - 실행: `docker/run.sh -- env PATH="$OUT/shim:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" qairt-env bash "$OUT/dn_convert<k>/convert_all.sh" > "$OUT/dn_convert<k>.log" 2>&1`
  - 판정
    - 종료 0.
    - 로그에 `생성 1개 / 건너뜀 0개`가 있다.
    - `grep -A1 "fallback to float"`의 둘째 줄이 비어 있다. 폴백 줄 자체가 없으면 FAIL(사유: 폴백 목록 미확인).
    - md5 비교.
- [ ] **Step 2: 실행.** `docker/accept.sh --only TC_DOCK_11_1,TC_DOCK_11_2` → 둘 다 PASS. 두 md5가 모두 `62b56d8f6afad86df29ee27ce4572b31`이어야 한다.
  - md5가 다르면 여기서 멈추고 소유자에게 보고한다. B안(구성 파일 바이트 비교, metadata 경로 가림)은 그 뒤 별도 단계로 만든다.
- [ ] **Step 3: R4 고정.**
  - `docker/run.sh -- /opt/qairt-venv/bin/pip freeze`의 출력으로 `docker/qairt-requirements.txt`를 바꾼다.
  - `docker/build.sh`로 다시 빌드한 뒤 `-k DOCK_02` 시험과 Step 2를 다시 돌려 모두 PASS인지 본다.
- [ ] **Step 4:** §5에 원문을 넣는다.
- [ ] **Step 5: 커밋(승인 뒤)** `[20261019] DockerEnv Add the DLC gate and pin the QAIRT venv`

### Task 8: GPU 합격 확인 (MREQ-DOCK-12, 13, DRUN-03·04·05 통합, R2)

**Files:**
- Modify: `docker/accept.sh`, 두 문서 §5

- [ ] **Step 1: 함수 작성**
  - `TC_DOCK_12_1_GpuExportMatchesShipped`, `TC_DOCK_12_2_GpuExportReallyOnGpu`
  - `TC_DOCK_13_1_OneEpochTrainingCompletes`, `TC_DOCK_13_2_TrainingOutputsOwnedOffline`. 이 둘은 `--name smoke_train_<날짜> --log $OUT/train_dn_1ep.log`로 띄우고 `docker wait`한다.
  - `TC_DRUN_03_3_RealContainerSeesOneGpu`
    - `run.sh --gpu N -- nvidia-smi -L`이 1줄이고, 호스트 GPU N의 UUID를 담아야 한다.
    - `run.sh -- python3 -c 'import torch; print(torch.cuda.device_count())'`가 `0`이어야 한다.
  - `TC_DRUN_04_4_RealDockerStopLogs143`: `run.sh --name stop_probe --log $OUT/stop_probe.log -- sleep 300`을 띄우고, 3초 뒤 `docker stop usdp-stop_probe`. 로그 마지막 줄이 `[run.sh] exit 143`이어야 한다.
  - `TC_DRUN_05_5_RealNoGpuInjectedRefused`: run.sh를 거치지 않고 `--gpus` 없이 `docker run --rm --user U:G -e USDP_GPU_CHECK=1 --mount type=bind,source=$REPO,target=$REPO usdp:latest $REPO/docker/entry.sh true`를 돌린다. 종료 3이고 stderr에 `entry.sh: refused`가 있어야 한다.
  - `--gpu`가 없으면 GPU가 필요한 TC는 `NOT RUN`이다.
- [ ] **Step 2: 빈 GPU 고르기.** `nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader`와 `--query-gpu=index,uuid`로 프로세스가 없는 번호 N을 고른다.
- [ ] **Step 3: 실행.** `docker/accept.sh --gpu N --only TC_DOCK_12_1,TC_DOCK_12_2,TC_DOCK_13_1,TC_DOCK_13_2,TC_DRUN_03_3,TC_DRUN_04_4,TC_DRUN_05_5`
  - 기대: 7줄 모두 PASS. 12-1의 md5는 `f4a757f3551f194f1e0dd35944fcefa5`이다.
- [ ] **Step 4: R2(12-1만 실패하고 md5 X ≠ `7e2fe59ffbeb1660b7f83091e63a87e5`일 때)**
  1. 멈추고, 옛 컨테이너 재시작 승인을 받는다. 붙어 있는 VSCode 창이 끊긴다고 함께 알린다.
  2. 승인 뒤 `docker restart 298b322c6c92`.
  3. 저장소에 root 파일이 생기지 않게, `-w /tmp`와 절대 경로로 같은 명령을 돌린다: `docker exec -w /tmp -e PYTHONDONTWRITEBYTECODE=1 -e NUMPY_MADVISE_HUGEPAGE=0 -e CUDA_VISIBLE_DEVICES=N 298b322c6c92 python3 <R>/examples/aimet_quantization/6_apply_qat.py <같은 인자, 경로는 <R>/… 절대 경로> --output_dir /tmp/g4_old && md5sum /tmp/g4_old/qat_model.onnx`
  4. 새 이미지의 값과 같고 CPU 값과 다르면 통과로 기록한다(R2).
  5. 실행 뒤 `find <R> ! -user jshong`이 0건인지 확인한다.
- [ ] **Step 5:** 두 문서 §5에 원문을 넣는다.
- [ ] **Step 6: 커밋(승인 뒤)** `[20261019] DockerEnv Add the GPU export and training gates`

### Task 9: Claude 기록 이관과 마무리 (TC-DOCK-14-2)

**Files:**
- Modify: DockerEnv §5
- 메모리(`~/.claude/projects/-home-jshong-Code-unfied-sr-denoise-platform/memory/`)의 `devices-and-workflow.md`, `docker-env-design.md`

- [ ] **Step 1: 백업과 기준 체크섬.** 세션 스크래치패드에 다음 두 가지를 만든다.
  - `tar -C ~/.claude/projects/-home-jshong-Code-unfied-sr-denoise-platform -cf <scratch>/memory_backup.tar memory`
  - 같은 폴더에서 `find memory -type f -exec md5sum {} + | sort > <scratch>/memory_before.md5`
- [ ] **Step 2: 스테이징.** 옛 컨테이너 `/root/.claude/projects/`의 5개 폴더(이 프로젝트와 `…-results-*` 4개)마다 `docker cp 298b322c6c92:/root/.claude/projects/<폴더> - | tar -x -C <scratch>/claude_staging --exclude='*/memory'`.
- [ ] **Step 3: 합치기.** `cp -rn <scratch>/claude_staging/. ~/.claude/projects/`(덮어쓰기 금지).
- [ ] **Step 4: 확인**
  - 스테이징의 파일마다 대상에 같은 크기의 파일이 있어야 한다(파일 수·총 크기 출력).
  - `memory/` md5 목록이 Step 1과 같아야 한다.
  - 기대: 세션 jsonl 9개 + results 폴더 4개.
- [ ] **Step 5: 메모리 갱신.**
  - devices-and-workflow: 변환은 이제 서버 작업 컨테이너에서 하고, 회사 PC는 선택이다.
  - docker-env-design: 완료 상태.
- [ ] **Step 6: 보고.**
  - DockerEnv §5에 TC-DOCK-14-2 원문을 넣는다(커밋은 승인 뒤 `[20261019] DockerEnv Record the migration and acceptance results`).
  - 소유자에게 쉬운 말로 결과를 보고한다: 게이트 표, 남은 일.
  - 삭제를 요청한다: 소유자가 직접 `docker rm -f sr`와 `docker rmi sr/unified-platform:v2.0`을 하거나, 지시하면 Claude가 한다.
