# DockerEnv 모듈 문서

| 문서번호 | 버전 | 작성일 | 작성자 | 시스템 | 상위 문서 |
|---|---|---|---|---|---|
| USRD-MOD-DOCK-001 | 0.6 | 2026-10-20 | JSHONG | unified-sr-denoise-platform | Docker 환경 설계 결정 2026-10-19 (소유자 결정 Q1–Q8, 검토 답변 R1–R4, §6) |

개정 이력
| 버전 | 일자 | 변경 내용 | 작성자 |
|---|---|---|---|
| 0.1 | 2026-10-19 | 최초 작성. 요구, 설계, 회피 차원. 검토 4건(빌더, 기술, 결정 충실도, 저장소 영향)과 소유자 답변 반영. 소유자 승인 | JSHONG |
| 0.2 | 2026-10-19 | 실행 스크립트 부분(MREQ-DOCK-04~08과 회피 차원)을 DockerRun(USRD-MOD-DRUN-001, MREQ-DRUN-01~05)으로 옮김. 상세 설계(§3)와 시험 케이스(§4.2) 작성. 합격 확인을 `docker/accept.sh`로 묶음 | JSHONG |
| 0.3 | 2026-10-19 | MREQ-DOCK-11 합격 기준을 A안(md5)에서 B안(op 번호 두 칸만 가린 구성 파일 바이트 비교)과 소유자 보드 확인으로 바꿈. QAIRT 2.41이 그 두 칸에 초기화하지 않은 메모리를 써서 md5는 같은 서버에서도 매번 다르다(소유자 결정). float 대체 판정을 실제 로그 형식에 맞춤 | JSHONG |
| 0.4 | 2026-10-19 | 브랜치 전체 리뷰 반영. `accept.sh` 보고 규약을 MREQ-DOCK-15로 올림: 작업이 거부·실패해도 고른 TC마다 한 줄과 요약을 찍고, 요약 전에 끝나면 그렇다고 알린다. TC-DOCK-09-2는 pytest가 돌지 않았으면 FAIL | JSHONG |
| 0.5 | 2026-10-19 | 리뷰에서 미룬 Minor 처리. 빌드 초기 pip 단계에 packaging 25.0도 `--no-deps`로 고정, QAIRT venv를 `pip freeze --all`(pip·setuptools 포함 7줄)로 고정, apt 목록을 `docker/apt-versions.txt`에 기록하고 시험으로 비교(TC-DOCK-01-6). build.sh는 tarball symlink를 실제 파일로 푼다. accept.sh는 `--out`이 저장소·`/mnt` 밖이면 거부하고(TC-DOCK-15-4), 09-1 실패 사유에 pytest 요약 줄을 쓴다(TC-DOCK-15-5). §2.4·§2.5·§3.2·§3.4·§4.2를 구현과 맞춤 | JSHONG |
| 0.6 | 2026-10-20 | 2차 수정 리뷰 반영. accept.sh는 TERM·INT를 받으면 143·130으로 끝나 중단 줄에 그 코드가 남는다(TC-DOCK-15-6). symlink로 부른 저장소에서도 `--out` 판정이 맞음을 시험으로 고정(TC-DOCK-15-7). sshfs 위 `--out`은 소유자 확인 건이 실패하므로 피하라고 docker/README에 적음 | JSHONG |

이 저장소에는 SRS가 없다. 상위 근거는 2026-10-19 소유자 결정(§6)이다. 모듈 약어는 `DOCK`이고, 작업 실행은 DockerRun(`DRUN`) 문서에 있다. **삭제 조건**은 소유자가 정한 옛 컨테이너 삭제 조건(MREQ-DOCK-09~13)을 뜻한다.

**배경.** 지금까지 모든 작업은 장기 실행 컨테이너 `sr`에서 했다(이미지 `sr/unified-platform:v2.0`, root 실행, VSCode 부착). 문제가 셋이다.
- root가 만든 파일이 저장소 소유자(jshong, uid 1017)의 git을 막았다(10-19).
- 환경이 `docker commit` 레이어로만 있어 다시 만들 수 없다. `requirements.txt`(165줄)는 aimet-torch를 한 줄로만 고정하고, 그 의존성 27개는 고정하지 않는다.
- DLC 변환은 회사 PC(WSL)에서만 된다.

**목표.** VSCode, Claude, git, 보드 ssh/scp는 호스트에서 jshong으로 쓴다. 학습, AIMET export, DLC 변환, pytest, 영상 작업은 작업할 때만 띄우는 컨테이너에서 jshong으로 돈다. 이미지는 저장소의 Dockerfile로 다시 만들 수 있고, QAIRT 2.41을 품어 변환까지 서버에서 끝낸다.

**비목표.** 다른 PC용 이미지, 컨테이너 안 GUI, k3s 연동, 패키지 버전 올리기, Docker 엔진 업그레이드(20.10.11 유지), 여러 GPU를 쓰는 작업.

**실패로 보는 것.**
- 결과 파일이 root 소유로 생긴다.
- 테스트가 실패 대신 skip으로 끝나 통과처럼 보인다.
- 출하 ONNX·DLC를 덮어쓴다.
- 다른 사람(k3s 파드, 다른 사용자)이 쓰는 GPU에 작업을 띄운다.

**금지.** QAIRT tarball과 SDK는 git에 넣지 않는다. `usdp` 이미지는 레지스트리에 올리지 않는다(Qualcomm 비공개 SDK 포함). `build.sh`의 임시 빌드 컨텍스트는 저장소 밖(`mktemp -d`, /tmp)에 만든다.

## 1 요구
MREQ-DOCK-04~08은 DockerRun(`docs/modules/20261019-MOD-DockerRun.md`)의 MREQ-DRUN-01~05로 옮겼다.

| ID | 유형 | 요구사항 | 수용 기준 | 상위 | 검증 |
|---|---|---|---|---|---|
| MREQ-DOCK-01 | 기능 | 이미지의 시스템 python3은 `requirements.txt`의 178개 배포판을 그 버전 그대로 가져야 한다 | 다음을 모두 만족: 배포판 집합과 버전이 `requirements.txt`와 정확히 같다(pip·setuptools·wheel 제외). pip 26.0.1, setuptools 82.0.0, wheel 0.46.3. `pip check` 통과. `torch.__version__` 2.9.0+cu128, numpy 2.2.6, cv2 4.12.0. `aimet_torch`, `aimet_common.defs`, `aimet_torch.common.defs`, `pyiqa`, `torch.utils.tensorboard` import 성공. `cv2/cv2.abi3.so` sha256이 opencv-python-headless RECORD 값과 같다. apt 패키지 목록(`dpkg-query -W`)이 `docker/apt-versions.txt`(319개)와 같다 | Q2 | T |
| MREQ-DOCK-02 | 기능 | QAIRT 2.41 변환 도구와 그 파이썬 환경은 `qairt-env` 명령 안에서만 보여야 한다 | 안: `qairt-converter`, `qairt-quantizer`, `snpe-dlc-graph-prepare`, `snpe-dlc-info`, `qairt-dlc-diff`가 `/opt/qairt`로 잡히고 numpy 1.26.4. venv의 `pip freeze --all`이 `docker/qairt-requirements.txt`(7줄)와 같다. 변환 경로 바이너리·확장 모듈(`qti/tvm` 제외)의 `ldd`에 `not found` 0건. 밖(`bash -lc` 포함): 다섯 도구가 PATH에 없고, `import qti` 실패, env와 `ldconfig -p`에 QAIRT 경로 없음, numpy 2.2.6 | Q1, Q3 | T |
| MREQ-DOCK-03 | 기능 | 이미지는 ffmpeg 7.0.2 정적 바이너리를 실행 가능한 상태로 PATH에 가져야 한다 | `/usr/local/bin/ffmpeg` md5 `d5c698e0e98e5cf7de03fe14bd1cda6a`, `ffmpeg -version` 종료 0, 첫 줄에 7.0.2, libx264 인코딩 성공 | Q6 | T |
| MREQ-DOCK-09 | 기능 | 컨테이너 안 pytest는 GPU와 네트워크 없이 모두 통과하여야 한다 | passed = 수집 항목 수 = 280 + 새 시험 파일(`test_docker_run.py`, `test_docker_image.py`) 항목 수, SKIPPED 0줄. 실행 전후 `git status --porcelain --ignored` 같음, `find . -newer <stamp>` 0건 | 삭제 조건 | T |
| MREQ-DOCK-10 | 기능 | 컨테이너 안 출하 export 스크립트는 출하 ONNX를 다시 만들어야 한다 | DN ONNX md5 `81ac40c78450489a6fac04c05569a2ec`, IR `x4_ir`(320×180) `5de6458e5cfce8c4d0666b54727b4294`, `x4_ir_1024`(320×256) `37d8f6ff051a97d1df9dbc2f8fa5434d`. 셋 모두 `param_encodings` JSON이 출하 파일과 같다 | 삭제 조건 | T |
| MREQ-DOCK-11 | 기능 | 컨테이너 안 DLC 변환은 출하 Denoise DLC와 같은 내용의 파일을 만들어야 한다 | R1 = B안(소유자 결정, §6). 종료 0, 로그에 `생성 1개 / 건너뜀 0개`와 `Following OPs fallback to float.`(대체 op 0개). DLC 구성 파일 5개가 출하 DLC(md5 `62b56d8f6afad86df29ee27ce4572b31`)와 바이트 단위로 같다. 단 `dlc.metadata`의 `startOpIndex`·`endOpIndex` 두 값은 가린다. 두 번 실행한 결과도 같은 기준으로 같다. 그다음 소유자 보드 확인 | 삭제 조건 | T |
| MREQ-DOCK-12 | 기능 | GPU export는 컨테이너 안에서 실제 GPU로 돌아야 한다 | ONNX md5 `f4a757f3551f194f1e0dd35944fcefa5`(09-10 GPU 출하본). 다르면 R2 규칙: 옛 컨테이너가 같은 커밋에서 낸 값과 같고 CPU 값 `7e2fe59ffbeb1660b7f83091e63a87e5`와 달라야 한다. 같은 실행 안에서 Triton `.so` 2개 이상 | 삭제 조건(R2) | T |
| MREQ-DOCK-13 | 기능 | 컨테이너 안 GPU 학습은 끝까지 돌고 결과를 jshong 소유로 남겨야 한다 | 로그 마지막 줄 `[run.sh] exit 0`. work_dir에 `best.pth`, `last.pth`, `training_log.csv`. work_dir 안 `find ! -user jshong` 0건, 가중치 다운로드 없음(`--offline`) | 삭제 조건(R2) | T |
| MREQ-DOCK-14 | 기능 | 옛 컨테이너는 이관을 마치고 삭제 조건을 통과한 뒤에만 지워야 한다 | 가중치 4개 md5 일치. Claude 기록 파일 수·크기가 스테이징 사본과 같고, 호스트 `memory/` 체크섬이 이관 전후 같다. §2.7 실행 설정 요약이 이 문서에 있다. MREQ-DOCK-09~13 통과가 §5에 기록돼 있다 | Q7 | I |
| MREQ-DOCK-15 | 기능 | `docker/accept.sh`는 고른 TC마다 `PASS`·`FAIL`·`NOT RUN` 한 줄과 요약 줄을 찍어야 하고, 하나라도 통과하지 않으면 종료 1이어야 한다. 시작한 작업이 거부되거나 실패해도 같다 | `run.sh`가 모든 작업을 거부하고 `nvidia-smi`가 실패해도 고른 6건이 한 줄씩, 요약 `0 passed, 6 not passed`, 종료 1, `aborted` 없음. `--gpu`가 없으면 GPU 2건은 `NOT RUN`. pytest가 돌지 않았으면 TC-DOCK-09-2도 FAIL | §3.4 "조용한 생략 없음" | T |

## 2 설계
### 2.1 구성 요소
| 항목 | 내용 |
|---|---|
| 책임 | 작업(학습, export, 변환, pytest, 영상)을 다시 만들 수 있는 이미지와 그 합격 확인, 옛 컨테이너 이관 |
| 새 파일 | `docker/Dockerfile`, `docker/build.sh`, `docker/qairt-env`, `docker/qairt-requirements.txt`, `docker/accept.sh`, `.dockerignore`, `tests/test_docker_image.py`. 실행 스크립트는 DockerRun |
| 바뀌는 파일 | `requirements.txt`(165 → 178), `README.md`, `examples/README.md`, `examples/deployment/README.md`, `tools/README.md`, `gen_convert_all.py`·`tests/test_gen_convert_all.py`의 docstring, `results/denoise_release_v1/MANIFEST.md` 변환 절에 경고 한 줄. `src/`, `tools/`, `examples/`의 코드는 바꾸지 않는다 |
| 자원 | 이미지 약 20 GB(베이스 8, pip 약 10, QAIRT 약 1.2, ffmpeg 0.08) |
| 의존 | 호스트: Docker 20.10.11, buildx 0.23.0, nvidia-container-toolkit 1.13.5, 드라이버 550.54.14. 빌드: QAIRT tarball, PyPI, Docker Hub(`docker/dockerfile:1` frontend) |

### 2.2 실행 개념
- **빌드.** `docker/build.sh`가 tarball의 sha256을 확인한다. 그다음 `mktemp -d`에 tarball 하드링크를 만든다(다른 파일시스템이면 복사). 이 폴더를 `--build-context qairt=`로 넘겨 `docker buildx build --builder default`로 `usdp:<YYYYMMDD>`와 `usdp:latest`를 만든다. 끝나면 임시 폴더를 지운다. `qualcomm-vlm/env`를 직접 넘기지 않는 이유는 그 `.dockerignore`가 OE tarball만 통과시키기 때문이다. 저장소 컨텍스트는 `.dockerignore` 허용 목록으로 `requirements.txt`와 `docker/`만 보낸다.
- **작업 실행.** `docker/run.sh`로 한다(DockerRun §2.2).
- **변환.** 생성은 저장소 환경에서 `python3 examples/deployment/gen_convert_all.py <root>`로 한다. 실행은 `qairt-env bash <폴더>/convert_all.sh`로 한다. 이미 출하된 릴리스 폴더(`results/denoise_release_v1`, `results/release_v3`)에서는 돌리지 않는다. 출하 DLC를 덮어쓰기 때문이다.
- **영상.** `make_videos.sh`와 `compose_videos.sh`는 저장소 밖 `/mnt/data_server/.../results/` 아래에 있고, 컨테이너에서 돈다. ffmpeg는 PATH로 부른다(10-19 확인: `make_videos.sh:33`, `compose_videos.sh:96`). 보드에서 프레임을 받는 `pull_frames.sh`는 호스트에서 돈다.
- **호스트에 남는 일.** 보드 ssh/scp, git push, tensorboard 보기.

### 2.3 인터페이스
| 대상 | 형태 | MREQ |
|---|---|---|
| `docker/build.sh` | 인자 없음. `QAIRT_TARBALL`(기본 `$HOME/Code/qualcomm-vlm/env/QAIRT_2.41.tar.gz`, sha256 `4e73e38f5e60e083b7c1e535ceaabb8a18e7523d912ad3b7dbe99d9217e3520d` 고정), `USDP_TAG`(기본 오늘 날짜). UID·GID는 실행한 사용자의 것 | 01–03 |
| `qairt-env 명령…` | 이미지 안 `/usr/local/bin/qairt-env`. QAIRT venv와 SDK의 PATH·LD_LIBRARY_PATH·PYTHONPATH를 그 명령에만 적용. 인자가 없으면 종료 2 | 02 |
| `docker/accept.sh [--gpu N] [--only TC_…] [--out DIR]` | 호스트에서 합격 확인(§3.4). 결과 줄 `PASS\|FAIL\|NOT RUN TC-…`. PASS가 아닌 것이 하나라도 있으면 종료 1. `--out` 기본 `results/docker_accept_<YYYYMMDD>`(git 무시) | 09–13, DRUN |

### 2.4 이미지 내용
- **베이스.** `FROM nvidia/cuda:12.6.0-cudnn-devel-ubuntu22.04@sha256:60cfa1fb926e4830ec1dbee1b7511eeba0fee2333b16cbec79c356e2afc2f359`(로컬 사본, 옛 이미지와 같음). 드라이버 550 통과 절이 있고, GPU QAT의 Triton 컴파일에 필요한 gcc·`Python.h`·cuda stub·cuda-compat이 갖춰져 slim으로 바꾸지 않는다. 첫 줄은 `# syntax=docker/dockerfile:1`이고, 첫 빌드 뒤 digest로 고정한다. `ENTRYPOINT []`로 NVIDIA 배너를 끈다.
- **apt.** 버전을 고정하지 않으므로 다시 빌드한 날의 Ubuntu 저장소 상태를 따른다. 대신 받아들인 이미지의 목록을 `docker/apt-versions.txt`에 두고 TC-DOCK-01-6이 비교한다(다르면 실패하므로, 바뀐 이미지를 쓰려면 합격 확인 뒤 그 파일을 갱신한다). `ARG DEBIAN_FRONTEND=noninteractive` 아래 다음을 설치한다: python3(3.10.12), python3-pip, python3-dev, python3.10-venv, gcc, libc6-dev, git, libgl1, libglib2.0-0, libsm6, libxext6, libxrender1, libc++1-14, libc++abi1-14, libunwind-14, tzdata, curl, ca-certificates.
- **pip.** 다음 순서로 설치한다.
  1. `PIP_NO_CACHE_DIR=1`로 둔다.
  2. pip 26.0.1, setuptools 82.0.0, wheel 0.46.3, packaging 25.0을 `--no-deps`로 고정한다(의존성 해석을 켜면 wheel이 그날 최신 packaging을 끌어온다).
  3. `pip install --no-deps --no-build-isolation -r requirements.txt`. 고정한 setuptools·packaging으로 sdist(`filterpy`, `openai-clip`)를 빌드하기 위해서다.
  4. `pip install --no-deps --force-reinstall opencv-python-headless==4.12.0.88`. 옛 환경에서 실제로 쓰이던 cv2 파일이 headless 쪽이기 때문이다.
  5. `pip check`.
- **QAIRT.** `tar --no-same-owner --strip-components=1 --exclude='QAIRT_2.41/examples'`로 `/opt/qairt`(root 소유)에 푼다. venv `/opt/qairt-venv`는 `docker/qairt-requirements.txt`를 쓴다(numpy 1.26.4, onnx 1.16.1, packaging 24.0, pip 22.0.2, protobuf 3.20.3, PyYAML 6.0.2, setuptools 59.6.0). 첫 변환이 성공한 venv 전체를 `pip freeze --all`로 고정했다(R4). pip·setuptools는 Ubuntu의 ensurepip 휠에서 오므로 이 파일로 다시 맞춘다. 빌드 끝에 `ldd` 검사를 하고 `not found`면 빌드를 실패시킨다. 대상은 변환 경로이고, `libLLVM-14`가 필요한 `qti/tvm`은 뺀다.
- **ffmpeg.** 휠 `imageio_ffmpeg-0.6.0-py3-none-manylinux2014_x86_64.whl`(sha256 `c7e46fcec401dd990405049d2e2f475e2b397779df2519b544b8aab515195282`)을 PyPI에서 받아, `ffmpeg-linux-x86_64-v7.0.2`만 꺼내 `install -m 0755`로 `/usr/local/bin/ffmpeg`에 둔다.
- **사용자.** 빌드 인자 UID·GID·USERNAME(build.sh가 `id -u`·`id -g`·`id -un`을 넘김, 이 서버에서는 jshong 1017)으로 사용자를 만든다. `/home/<USERNAME>/.cache`, `.cache/torch`, `.config`를 그 사용자 소유로 미리 만들고, `LABEL usdp.uid`·`usdp.gid`·`usdp.home`을 단다. 마지막 줄은 `USER ${UID}:${GID}`라서 `run.sh` 없이 띄워도 root가 아니다.
- **ENV.** `TZ=Asia/Seoul`, `LANG=C.UTF-8`, `NUMPY_MADVISE_HUGEPAGE=0`, `PYTHONUNBUFFERED=1`, `NVIDIA_VISIBLE_DEVICES=void`. QAIRT 변수는 전역에 두지 않는다.

### 2.5 패키지 목록 (Q2 = B안)
옛 컨테이너 220개(pip freeze 217개 + pip·setuptools·wheel)에서 출발하고, 버전은 하나도 바꾸지 않는다. 출하 md5가 torch 2.9.0, aimet-torch 2.29.0, onnx 1.17.0, protobuf 6.32.1, onnxscript 0.2.5에 묶여 있기 때문이다.
- **넣는 것(161개).** 저장소가 import하는 17개 모듈의 의존 전체다. 지금 `requirements.txt`에 없던 aimet 의존성 27개가 여기서 고정된다.
- **남기는 것(17개).** 쓰는 라이브러리가 선택적으로 import한다. thop은 `tools/export.py`가 선택적으로 import한다(GFLOPs 줄).
  - onnx-ir(torch.onnx), onnxruntime(aimet, onnx, onnxscript), h5py, PyWavelets, cryptography
  - google-auth 계열 7개(google-auth, google-auth-oauthlib, oauthlib, requests-oauthlib, rsa, pyasn1, pyasn1-modules)
  - importlib_resources, coloredlogs, humanfriendly, flatbuffers
  - thop(`tools/export.py`의 GFLOPs 출력. 옛 환경에서는 root user-site에만 있었다)
- **빼는 것(39개).** 아래 표다. `requirements.txt`(178) + 이 표 = 옛 pip freeze 217.

| 묶음 | 개수 | 패키지 |
|---|---|---|
| 문서·압축·git 도구 | 11 | `pymupdf==1.28.2`, `python-pptx==1.0.2`, `lxml==6.1.2`, `xlsxwriter==3.2.9`, `pypdf==6.14.2`, `rarfile==4.5`, `unrar-cffi==0.2.2`, `libarchive-c==5.3`, `git-filter-repo==2.47.0`, `pexpect==4.9.0`, `ptyprocess==0.7.0` |
| 퇴역 모델·빌드 도구 | 5 | `mamba_ssm==2.3.0`, `causal_conv1d==1.6.0`, `ninja==1.13.0`, `cmake==4.2.3`, `lit==18.1.8` |
| streamlit 계열 | 11 | `streamlit==1.56.0`, `altair==6.0.0`, `pydeck==0.9.1`, `blinker==1.9.0`, `GitPython==3.1.46`, `gitdb==4.0.12`, `smmap==5.0.3`, `tenacity==9.1.4`, `toml==0.10.2`, `watchdog==6.0.0`, `cachetools==7.0.2` |
| 안 쓰는 복원 라이브러리 | 11 | `basicsr==1.4.2`, `gfpgan==1.3.8`, `tb-nightly==2.21.0a20251023`, `albumentations==2.0.8`, `albucore==0.0.24`, `pydantic==2.12.5`, `pydantic_core==2.41.5`, `annotated-types==0.7.0`, `typing-inspection==0.4.2`, `simsimd==6.5.16`, `stringzilla==4.6.0` |
| 기타 | 1 | `lpips==0.1.4` (저장소는 pyiqa의 LPIPS를 쓴다) |

### 2.6 마운트 · HOME · 캐시 (Q4 = B안)
- 붙이는 것은 저장소, `/mnt`, 호스트 `~/.cache/torch` 셋뿐이다(DockerRun MREQ-DRUN-02). 홈 전체는 붙이지 않는다. 보드 키, HF 토큰, root 소유 `~/.cache/huggingface/hub`가 있기 때문이다.
- 코드가 쓰는 절대 경로는 저장소와 `/mnt`뿐이다(10-19 조사).
- `/mnt/data2/jshong`은 jshong 소유다(R3, 10-19 소유자가 변경하고 확인함). 그래서 검증 세트 생성도 컨테이너에서 된다.
- Triton 캐시는 컨테이너 HOME 안에 있어 작업마다 다시 컴파일한다(몇 초).
- **캐시 이관.** 옛 컨테이너 `/root/.cache/torch/hub`에서 4개만 호스트 `~/.cache/torch/hub/`의 같은 위치로 복사하고 md5를 대조한다: `checkpoints/alexnet-owt-7be5be79.pth`, `checkpoints/vgg19-dcbb9e9d.pth`, `pyiqa/LPIPS_v0.1_alex-df73285e.pth`, `pyiqa/niqe_modelparameters.mat`.

### 2.7 옛 컨테이너 기록 · 이관 · 정리 (Q7 = A안)
옛 실행 설정(10-19 `docker inspect`)과 바꾼 이유:

| 항목 | 옛 `sr` | 새 작업 컨테이너 | 이유 |
|---|---|---|---|
| 사용자 · 작업 폴더 | root · `/home/jshong` | 호출자 uid:gid · 저장소 루트 | 소유권 사고(10-19) |
| GPU · IPC | `--gpus all` · host | `--gpu N` 한 장 · host(유지) | GPU 공유. DataLoader 공유 메모리 최악 약 10 GB |
| 네트워크 | host | bridge(합격 확인은 none) | 작업이 포트를 쓰지 않음 |
| 마운트 | `/home/jshong`, `/mnt`(rprivate), localtime, timezone, X11 익명 볼륨(깨짐) | 저장소, `/mnt`(rslave), `~/.cache/torch`, `TZ` env | Q4. sshfs 재마운트를 실행 중에도 받음 |
| 권한 | SYS_PTRACE, seccomp=unconfined, label=disable, core 무제한 | 기본값 | 디버거 attach가 필요해지면 그때 추가 |
| 이미지 | nvidia/cuda 12.6 + `docker commit` 레이어 4개(Dockerfile 없음) | `docker/Dockerfile` | 재현성 |

**순서.**
1. 가중치 4개를 이관한다.
2. 이미지를 빌드한다.
3. MREQ-DOCK-01~03·09~13과 DRUN 통합 시험을 확인한다.
4. Claude 기록을 이관한다.
5. 결과를 보고한다.
6. 소유자가 지우거나, 소유자가 지시하면 Claude가 컨테이너 `sr`와 이미지 `sr/unified-platform:v2.0`(약 65 GB)을 지운다.

**Claude 기록.** 이 프로젝트 세션 9개와, `results/` 아래에서 연 세션의 Claude 프로젝트 폴더 4개(약 0.7 GB)를 옮긴다.
1. 호스트 `memory/`를 백업한다.
2. `docker cp sr:<폴더> - | tar -x --exclude='*/memory'`로 스테이징 폴더에 받는다.
3. `cp -rn`(덮어쓰기 금지)으로 `~/.claude/projects/`에 합친다.

**버릴 것.** `/tmp/claude-0` 15 GB, pip 캐시 9.4 GB, `.vscode-server` 12 GB, 안 쓰는 가중치, root 보드 키(호스트 jshong 키로 대체, 10-19 등록).

**GPU export가 어긋날 때(R2).** MREQ-DOCK-12의 md5가 `f4a757f3…`이 아니면, 소유자 승인을 받아 옛 컨테이너를 재시작하고 같은 커밋에서 같은 명령을 돌린다. 재시작하면 붙어 있는 VSCode 창이 끊긴다.

## 3 상세 설계
### 3.1 소프트웨어 단위
| 단위 | 파일 | 역할 | MREQ |
|---|---|---|---|
| 이미지 · 빌드 | `docker/Dockerfile`, `.dockerignore`, `docker/build.sh` | §3.2 | 01–03 |
| QAIRT 환경 | `docker/qairt-env`, `docker/qairt-requirements.txt` | §3.3 | 02 |
| 합격 확인 | `docker/accept.sh` | §3.4. TC 함수 하나가 TC 하나 | 09–13 |
| 이미지 시험 | `tests/test_docker_image.py` | 이미지 안에서만 통과하는 환경 시험. 다른 환경이면 실패한다(skip 아님) | 01–03 |

### 3.2 `docker/Dockerfile`과 `docker/build.sh`
- 레이어는 §2.4 순서로 쌓는다: 베이스 → apt(`--no-install-recommends`) → pip → ffmpeg(`curl -fsSL`, `sha256sum -c`, zipfile 추출) → QAIRT(`RUN --mount=type=bind,from=qairt,target=/ctx tar … -C /opt/qairt`, venv, `COPY docker/qairt-env`, `ldd` 검사) → 사용자·LABEL·ENV·`USER`. 무거운 pip 층이 앞에 있어서, QAIRT나 ffmpeg를 바꿔도 pip 층은 다시 받지 않는다.
- `build.sh`(`set -euo pipefail`)의 순서:
  1. 필요한 도구가 있는지 확인한다.
  2. tarball sha256을 확인한다. 다르면 종료 1. symlink면 `realpath -e`로 실제 파일을 쓴다(빌드 컨텍스트에 링크로 들어가면 빌드가 못 연다).
  3. `CTX=$(mktemp -d)`에 `ln`(실패하면 `cp`)하고, `trap`으로 정리한다.
  4. `docker buildx build --builder default --progress plain --build-context qairt="$CTX" --build-arg UID="$(id -u)" --build-arg GID="$(id -g)" --build-arg USERNAME="$(id -un)" -f docker/Dockerfile -t "usdp:$USDP_TAG" -t usdp:latest "$REPO"`.
  5. 이미지 ID를 출력한다.

### 3.3 `docker/qairt-env` (`set -euo pipefail`)
인자가 없으면 사용법을 출력하고 종료 2. 다음 변수를 설정한 뒤 `exec "$@"`한다.
- `QAIRT_SDK_ROOT`, `QNN_SDK_ROOT`, `SNPE_ROOT` = `/opt/qairt`
- `VIRTUAL_ENV=/opt/qairt-venv`
- `PATH=/opt/qairt-venv/bin:/opt/qairt/bin/x86_64-linux-clang:$PATH`
- `LD_LIBRARY_PATH=/opt/qairt/lib/x86_64-linux-clang[:기존]`
- `PYTHONPATH=/opt/qairt/lib/python`

### 3.4 `docker/accept.sh` (호스트, `set -euo pipefail`)
출력 폴더 `$OUT`(기본 `results/docker_accept_<YYYYMMDD>`)은 저장소나 `/mnt` 아래여야 하고(컨테이너가 써야 함), 이미 있으면 거부한다. `TC_DRUN_05_5`(run.sh를 일부러 거치지 않음)를 빼면 모든 명령은 `docker/run.sh`로 실행한다.
- `TC_DOCK_09_1/2`
  - 실행 전후 `git status --porcelain --ignored`를 비교하고, `find -newer` stamp로 새 파일을 찾는다.
  - 명령: `run.sh --offline -- env PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q -rs -p no:cacheprovider`
  - 판정: 요약 줄이 `<수집 수> passed`뿐이고 수집 수 = 280 + 새 파일 항목 수. 실패면 사유에 pytest 요약 줄을 쓴다.
- `TC_DOCK_10_1~3`
  - DN: `run.sh --offline -- bash results/denoise_release_v1/export.sh $OUT/dn`
  - IR: `run.sh --offline -- env SCALE=4 bash results/release_v3/eval/export.sh results/release_v3/checkpoints/IR_x4.pth $OUT/ir`
  - 판정: md5와 `param_encodings` 비교. IR 스크립트는 실패해도 종료 0이므로 `x4_ir exit 0` 줄도 본다.
- `TC_DOCK_11_1/2`
  - 준비: `$OUT/dn_convert<k>/`에 `convert_all.sh`와 `dn_bpp_half4_w16a16/qat/`만 복사한다.
  - `mktemp` 대체 스크립트는 `-d`면 `/tmp/tmp.nIupqkU2Au`를 만들어 돌려주고, 그 밖에는 진짜 mktemp를 부른다.
  - 실행(포그라운드): `run.sh --offline -- env PATH=<대체 폴더>:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin qairt-env bash $OUT/dn_convert<k>/convert_all.sh > $OUT/dn_convert<k>.log 2>&1`. 로그는 호스트가 쓴다. 이것을 k = 1, 2로 두 번 한다.
  - 로그에 `생성 1개 / 건너뜀 0개`가 있어야 한다. qairt-quantizer 2.41은 대체 op가 있으면 `Following OPs fallback to float:` 다음 줄에 op 목록을 찍고, 없으면 `Following OPs fallback to float.`만 찍는다. 앞의 줄이 있거나 뒤의 줄이 없으면 FAIL이다.
  - DLC 비교(B안, `same_dlc`): DLC는 무압축 zip이다. 항목 이름·순서·압축 방식·바이트가 모두 같아야 하고, `dlc.metadata*`의 `startOpIndex`·`endOpIndex` 값만 가린다. 11-1은 1회차를 출하 DLC와 비교하고, 출하 DLC의 md5가 `62b56d8f…`인지도 본다(기준 파일이 덮어써지지 않았는지). 11-2는 2회차를 1회차와 비교한다.
- `TC_DOCK_12_1/2`
  - 명령: `run.sh --gpu N --offline -- bash -c '<6_apply_qat 명령> && echo "triton so files: $(find ~/.triton/cache -name "*.so" | wc -l)"'`. 12-2는 이 라벨 줄을 읽는다.
  - `<6_apply_qat 명령>`은 다음과 같다: `python3 examples/aimet_quantization/6_apply_qat.py --config configs/finetune/BicubicPlusPlus/bicubicplusplus_2x_eo.yaml --data_config configs/data/bicubicpp/sr_finetune_eo_phase-2.yaml --checkpoint results/release_v3/checkpoints/EO_x2.pth --output_dir $OUT/x2_eo_gpu --mmp_config configs/aimet/mmp_noprune_w16a16.json --param_bw 16 --output_bw 16 --qat_epochs 0 --width 640 --height 360 --device 0`
- `TC_DOCK_13_1/2`
  - 명령: `run.sh --gpu N --offline --name smoke_train_<날짜> --log $OUT/train_dn_1ep.log -- python3 tools/train.py --config configs/train/Denoise/bicubicppdenoise.yaml --epochs 1 --work_dir $OUT/train_dn_1ep`
  - 판정: `docker wait`로 끝나기를 기다린 뒤 판정한다.
- DRUN 통합 시험: `TC_DRUN_01_3`, `03_3`, `04_4`, `05_5`(DockerRun §4.2).
- `--gpu`가 없으면 GPU가 필요한 TC는 `NOT RUN`으로 찍히고, 전체가 종료 1이 된다(조용한 생략 없음).
- 작업이 거부되거나 도구가 실패해도(GPU 사용 중, 마운트 끊김, `nvidia-smi` 오류, 출하 입력 없음) 그 TC를 `FAIL`로 찍고 다음 TC로 간다. 요약 전에 스크립트가 끝나면 EXIT trap이 `accept.sh: aborted before the summary (exit <코드>)`를 찍는다. TERM·INT는 143·130으로 끝낸다.
- `TC_DOCK_09_2`는 `pytest.log`에 pytest 요약 줄이 없으면(pytest가 돌지 않았으면) FAIL이다.

## 4 단위 시험 케이스
### 4.1 회피 차원
해당 없음(차단 요구는 DockerRun MREQ-DRUN-05).

### 4.2 시험 케이스
| TC | MREQ | 구분 | 테스트 이름 | 기대 결과 |
|---|---|---|---|---|
| TC-DOCK-01-1 | 01 | 정상 | `test_TC_DOCK_01_1_DistributionsMatchRequirements` | 배포판 집합·버전 = `requirements.txt` |
| TC-DOCK-01-2 | 01 | 정상 | `test_TC_DOCK_01_2_PinnedToolsVersionsAndImports` | pip·setuptools·wheel 고정값, torch·numpy·cv2 버전, import 5개 |
| TC-DOCK-01-3 | 01 | 경계 | `test_TC_DOCK_01_3_HeadlessCv2Wins` | `cv2.abi3.so` sha256 = headless RECORD |
| TC-DOCK-01-4 | 01 | 경계 | `test_TC_DOCK_01_4_PipCheckClean` | `pip check` 종료 0 |
| TC-DOCK-01-5 | 01 | 경계 | `test_TC_DOCK_01_5_KoreanOutputTimezoneAndEnv` | 한글 출력 오류 없음, `time.tzname`에 KST, ENV 5개(TZ 포함), `~/.cache` 소유자 = 현재 uid |
| TC-DOCK-01-6 | 01 | 경계 | `test_TC_DOCK_01_6_AptPackagesMatchRecord` | `dpkg-query -W` 목록 = `docker/apt-versions.txt` |
| TC-DOCK-02-1 | 02 | 정상 | `test_TC_DOCK_02_1_QairtToolsInsideEnv` | 도구 5개가 `/opt/qairt`, numpy 1.26.4, `pip freeze --all` = `docker/qairt-requirements.txt`, `qairt-converter --help` 종료 0 |
| TC-DOCK-02-2 | 02 | 오류 | `test_TC_DOCK_02_2_QairtDoesNotLeak` | 밖(`bash -lc` 포함)에서 도구 없음, `import qti` 실패, env·`ldconfig -p`에 QAIRT 없음 |
| TC-DOCK-02-3 | 02 | 경계 | `test_TC_DOCK_02_3_NoMissingSharedLibraries` | 변환 경로 ELF의 `ldd` not found 0건 |
| TC-DOCK-02-4 | 02 | 오류 | `test_TC_DOCK_02_4_QairtEnvWithoutCommand` | 종료 2 |
| TC-DOCK-03-1 | 03 | 정상 | `test_TC_DOCK_03_1_FfmpegBinaryMatches` | md5와 `-version` |
| TC-DOCK-03-2 | 03 | 경계 | `test_TC_DOCK_03_2_FfmpegEncodesX264` | 실행 비트, `testsrc` 0.1초 libx264 인코딩 종료 0 |
| TC-DOCK-09-1 | 09 | 정상 | `docker/accept.sh` `TC_DOCK_09_1_PytestAllPassNoSkip` | passed = 수집 수 = 280 + 새 항목, skip 0 |
| TC-DOCK-09-2 | 09 | 경계 | `TC_DOCK_09_2_PytestLeavesNoFiles` | git status·find 전후 같음 |
| TC-DOCK-10-1 | 10 | 정상 | `TC_DOCK_10_1_DenoiseExportMatchesShipped` | DN md5 + param encodings |
| TC-DOCK-10-2 | 10 | 정상 | `TC_DOCK_10_2_IrX4ExportMatchesShipped` | IR 두 프로파일 md5 + param encodings |
| TC-DOCK-10-3 | 10 | 경계 | `TC_DOCK_10_3_ExportOutputsOwnedNoTraceback` | 출력 jshong 소유, 로그에 Traceback 없음 |
| TC-DOCK-11-1 | 11 | 정상 | `TC_DOCK_11_1_DenoiseDlcMatchesShipped` | 1회차 종료 0, 생성 1/건너뜀 0, 대체 op 없음, 두 칸만 가리고 출하 DLC와 바이트 같음 |
| TC-DOCK-11-2 | 11 | 경계 | `TC_DOCK_11_2_DlcConversionDeterministic` | 2회차가 같은 기준으로 1회차와 같음 |
| TC-DOCK-12-1 | 12 | 정상 | `TC_DOCK_12_1_GpuExportMatchesShipped` | md5 `f4a757f3…`(R2 규칙) |
| TC-DOCK-12-2 | 12 | 오류 | `TC_DOCK_12_2_GpuExportReallyOnGpu` | md5 ≠ CPU `7e2fe59f…`, Triton `.so` ≥ 2 |
| TC-DOCK-13-1 | 13 | 정상 | `TC_DOCK_13_1_OneEpochTrainingCompletes` | 로그 끝 `exit 0`, 파일 3개 |
| TC-DOCK-13-2 | 13 | 경계 | `TC_DOCK_13_2_TrainingOutputsOwnedOffline` | work_dir 전부 jshong 소유, `--offline` |
| TC-DOCK-14-1 | 14 | 정상(I) | 계획 Task 5 확인 명령 | 가중치 4개 md5 일치 |
| TC-DOCK-14-2 | 14 | 경계(I) | 계획 Task 9 확인 명령 | 기록 수·크기 일치, 호스트 `memory/` 체크섬 전후 같음 |
| TC-DOCK-15-1 | 15 | 정상(통합) | `docker/accept.sh` 실제 실행(§5) | 고른 TC마다 한 줄과 요약, 모두 PASS면 종료 0 |
| TC-DOCK-15-2 | 15 | 오류 | `test_TC_DOCK_15_2_AcceptReportsEveryCaseWhenJobsFail` | 거부하는 `run.sh`·실패하는 `nvidia-smi`로 6건 모두 FAIL 한 줄씩, 요약, 종료 1, `aborted` 없음 |
| TC-DOCK-15-3 | 15 | 경계 | `test_TC_DOCK_15_3_AcceptWithoutGpuMarksGpuCasesNotRun` | `--gpu` 없이 FAIL 4건 + NOT RUN 2건, 요약, 종료 1 |
| TC-DOCK-15-4 | 15 | 오류 | `test_TC_DOCK_15_4_AcceptRefusesOutOutsideMounts` | 저장소·`/mnt` 밖 `--out`이면 종료 2, 폴더를 만들지 않음 |
| TC-DOCK-15-5 | 15 | 경계 | `test_TC_DOCK_15_5_PytestFailReasonQuotesSummary` | 09-1 FAIL 사유에 `3 failed, 10 passed` 요약 줄 |
| TC-DOCK-15-6 | 15 | 오류 | `test_TC_DOCK_15_6_AcceptAbortReportsSignal` | 실행 중 프로세스 그룹에 SIGTERM → 종료 143, `aborted before the summary (exit 143)` |
| TC-DOCK-15-7 | 15 | 경계 | `test_TC_DOCK_15_7_AcceptThroughSymlinkedRepoPath` | symlink로 부른 저장소에서 그 링크 아래 `--out`도 받아들임(종료 2가 아님) |

## 5 단위 시험 결과
| 일자 | 환경 | 명령 | 결과 |
|---|---|---|---|
| 2026-10-19 | `usdp:20261019`(`01777afea3f4`, 17.6 GB), uid 1017, `--offline`, GPU 없음 | `docker/run.sh --offline -- env PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q -rs -p no:cacheprovider tests/test_docker_image.py tests/test_docker_run.py` | 구현 전(옛 이미지): TC-DOCK-01-1·01-2·01-5 실패(초과 27개, `aimet_torch` 없음, UTC). 빌드 27분, `pip check` 통과, sdist 2개는 `--no-build-isolation`으로 빌드됨. 2231 passed, SKIPPED 0, 119 s |
```text
2231 passed, 4 warnings in 119.06s (0:01:59)
```

| 일자 | 환경 | 명령 | 결과 |
|---|---|---|---|
| 2026-10-19 | `usdp:20261019`(`d4c20b78c38d`, ffmpeg 층 추가), uid 1017, `--offline` | `docker/run.sh --offline -- env PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q -rs -p no:cacheprovider tests/test_docker_image.py` | 층 추가 전 TC-DOCK-03-1·03-2 실패(ffmpeg 없음). 휠 sha256 OK, pip 층은 캐시. 7 passed, SKIPPED 0. `ffmpeg -version` 첫 줄 `ffmpeg version 7.0.2-static` |
```text
7 passed, 4 warnings in 10.03s
```

| 일자 | 환경 | 명령 | 결과 |
|---|---|---|---|
| 2026-10-19 | `usdp:20261019`(`112b82a375dd`, QAIRT 층 추가), uid 1017, `--offline` | `docker/run.sh --offline -- env PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q -rs -p no:cacheprovider tests/test_docker_image.py tests/test_gen_convert_all.py` | 층 추가 전 TC-DOCK-02-1·02-3·02-4 실패(02-2 누수 없음은 그 전에도 참인 보호 시험). 빌드 안 venv `pip check` 통과, ldd 검사 누락 0건. venv는 numpy 1.26.4, onnx 1.16.1, packaging 24.0, protobuf 3.20.3, PyYAML 6.0.2뿐. 20 passed(이미지 11 + 생성기 9), SKIPPED 0 |
```text
20 passed, 4 warnings in 12.80s
```

| 일자 | 환경 | 명령 | 결과 |
|---|---|---|---|
| 2026-10-19 | 호스트 `~/.cache/torch/hub` | `docker cp 298b322c6c92:/root/.cache/torch/hub/<파일> ~/.cache/torch/hub/<파일>` 4개, md5 대조 | TC-DOCK-14-1: 4/4 일치(alexnet `cc6360b2…`, vgg19 `92881fe2…`, LPIPS_alex `9642209e…`, NIQE `8601a29b…`), 소유자 jshong. 이관 전 `--offline` pytest의 가중치 시험 3건 실패(이름 해석 실패), 이관 뒤 10 passed |
| 2026-10-19 | `usdp:20261019`(`112b82a375dd`), 18:00–18:09 | `docker/accept.sh --only TC_DOCK_09_1,TC_DOCK_09_2,TC_DOCK_10_1,TC_DOCK_10_2,TC_DOCK_10_3,TC_DRUN_01_3` | 6 PASS. pytest 2517 passed(기존 280 + 새 2237), SKIPPED 0, 357 s, `git status --ignored` 변화 없음, 새 파일 0. DN ONNX `81ac40c78450489a6fac04c05569a2ec`, IR `x4_ir` `5de6458e5cfce8c4d0666b54727b4294`, `x4_ir_1024` `37d8f6ff051a97d1df9dbc2f8fa5434d`, param encodings 같음. DN PSNR FP32 35.0118 / Pre-QAT 35.0125 dB(MANIFEST와 같음) |
```text
PASS TC-DOCK-09-1
PASS TC-DOCK-09-2
PASS TC-DOCK-10-1
PASS TC-DOCK-10-2
PASS TC-DOCK-10-3
PASS TC-DRUN-01-3
accept.sh: 6 passed, 0 not passed (outputs in /home/jshong/Code/unfied-sr-denoise-platform/results/docker_accept_20261019)
2517 passed, 318 warnings in 357.02s (0:05:57)
```

| 일자 | 환경 | 명령 | 결과 |
|---|---|---|---|
| 2026-10-19 | `usdp:20261019`(`112b82a375dd`), `--offline`, 18:31 | `docker/accept.sh --only TC_DOCK_11_1,TC_DOCK_11_2 --out results/docker_accept_20261019_t7` (A안 md5) | 2 FAIL. 대체 op 판정이 실제 형식과 달랐다(목록이 비면 `float.` 줄 바로 다음이 다음 로그 줄). md5는 1회 `e40d5b95…`, 2회 `90797124…`, 출하 `62b56d8f…`. 구성 파일 4개(model, model.params, model.params.bin, backend.metadata0)는 셋 다 바이트가 같다. `dlc.metadata2.1.0`(JSON)은 `startOpIndex`·`endOpIndex`만 다르다: 출하 96/0, 1회 426196512/21916(합치면 힙 주소 `0x0000559C19670B20`), 2회 3140584720/21950. 고정한 임시 경로 2곳은 같다. 소유자 결정: B안 + 보드 확인 |
| 2026-10-19 | 같은 이미지, 19:15 | `docker/accept.sh --only TC_DOCK_11_1,TC_DOCK_11_2 --out results/docker_accept_20261019_t7b` (B안) | 2 PASS. 두 로그 모두 `Following OPs fallback to float.`와 `생성 1개 / 건너뜀 0개`. md5 1회 `7f800df5…`, 2회 `6fba5d39…`(두 칸 값만 다름). 판정 확인: HTP 결과 1비트, 임시 경로, 공백 1개, 파일 없음은 모두 "다름", 두 칸 값만 바꾼 파일은 "같음" |
| 2026-10-19 | 같은 이미지, 19:16 | `docker/run.sh --offline -- /opt/qairt-venv/bin/pip freeze`, `docker/build.sh`, `pytest tests/test_docker_image.py -k DOCK_02` | R4: freeze 5줄이 `docker/qairt-requirements.txt`와 같다(diff 없음). 다시 빌드 5초, 11단계 모두 캐시, 이미지 ID `112b82a375dd` 그대로. DOCK_02 4 passed |
```text
PASS TC-DOCK-11-1
PASS TC-DOCK-11-2
accept.sh: 2 passed, 0 not passed (outputs in /home/jshong/Code/unfied-sr-denoise-platform/results/docker_accept_20261019_t7b)
4 passed, 7 deselected in 3.19s
```
보드 확인(소유자): 대기. 대상은 최신 합격 확인의 `results/docker_accept_20261020_final/dn_convert1/BicubicPPDenoise_EO_1280x720_v1.dlc`(이미지 `7fd2f0ce9454`). 이 절의 다른 출력 폴더는 정리해서 없다(경로는 실행 기록으로 남김).

| 일자 | 환경 | 명령 | 결과 |
|---|---|---|---|
| 2026-10-19 | `usdp:20261019`(`112b82a375dd`), GPU 2(`GPU-57f64a18…`, 프로세스 없음), `--offline`, 19:24–19:28 | `docker/accept.sh --gpu 2 --only TC_DOCK_12_1,TC_DOCK_12_2,TC_DOCK_13_1,TC_DOCK_13_2,TC_DRUN_03_3,TC_DRUN_04_4,TC_DRUN_05_5 --out results/docker_accept_20261019_t8` | 구현 전 `unknown test case`(종료 2). `--gpu` 없이 돌리면 GPU 5건 NOT RUN, 종료 1. 실행: 7 PASS. GPU export ONNX `f4a757f3551f194f1e0dd35944fcefa5`(09-10 GPU 출하본과 같아 R2 불필요), Triton `.so` 2개, PSNR FP32 31.3186 / Pre-QAT 31.3184 dB. 1 epoch 학습: 로그 끝 `[run.sh] exit 0`, `best.pth`·`last.pth`·`training_log.csv` 있음, work_dir 전부 jshong 소유. 참고로 epoch 1 PSNR 33.29 dB(10-13 배포 런의 epoch 1은 33.36 dB) |
```text
PASS TC-DOCK-12-1
PASS TC-DOCK-12-2
PASS TC-DOCK-13-1
PASS TC-DOCK-13-2
PASS TC-DRUN-03-3
PASS TC-DRUN-04-4
PASS TC-DRUN-05-5
accept.sh: 7 passed, 0 not passed (outputs in /home/jshong/Code/unfied-sr-denoise-platform/results/docker_accept_20261019_t8)
```

| 일자 | 환경 | 명령 | 결과 |
|---|---|---|---|
| 2026-10-19 | 호스트 jshong, 옛 컨테이너 `298b322c6c92`는 읽기만 | 백업 `tar -C <P> -cf <scratch>/memory_backup.tar memory`와 `find memory -type f -exec md5sum {} + \| sort`. 5개 폴더마다 `docker cp 298b322c6c92:/root/.claude/projects/<폴더> - \| tar -x -C <scratch>/claude_staging --exclude='*/memory'`, 그다음 `cp -rn <scratch>/claude_staging/. ~/.claude/projects/` | TC-DOCK-14-2 PASS. 옛 컨테이너(memory 제외) 908개, 736,179,551바이트가 스테이징·대상과 같다(크기 불일치 0, md5 908개 모두 같음). 합치기 전 이름 겹침 0. 세션 jsonl 9개(대상 jsonl 10개 = 9 + 이번 세션)와 results 폴더 4개가 새로 생겼다. 호스트 `memory/` 47개 md5는 전후 같다. 옛 컨테이너와 이미지는 소유자 보드 확인 뒤 지운다 |
```text
staged files=908 bytes=736179551 missing_or_size_differs=0
all 908 files byte-identical
memory md5 unchanged (47 files)
```

| 일자 | 환경 | 명령 | 결과 |
|---|---|---|---|
| 2026-10-19 | `usdp:20261019`(`112b82a375dd`), GPU 2, `--offline`, 21:06–21:20 | `docker/accept.sh --gpu 2 --out results/docker_accept_20261019_fix` | 브랜치 리뷰 반영 뒤 전체 15건 PASS(TC-DOCK-15-1). pytest 3499 passed = 기대값 3499(280 + 3206 + 13), SKIPPED 0. DN ONNX `81ac40c7…`, GPU export `f4a757f3…` 그대로. TC-DOCK-15-2·15-3은 반영 전 출력이 중간에 끊겨 FAIL, 반영 뒤 PASS |
```text
PASS TC-DOCK-09-1
PASS TC-DOCK-09-2
PASS TC-DOCK-10-1
PASS TC-DOCK-10-2
PASS TC-DOCK-10-3
PASS TC-DOCK-11-1
PASS TC-DOCK-11-2
PASS TC-DOCK-12-1
PASS TC-DOCK-12-2
PASS TC-DOCK-13-1
PASS TC-DOCK-13-2
PASS TC-DRUN-01-3
PASS TC-DRUN-03-3
PASS TC-DRUN-04-4
PASS TC-DRUN-05-5
accept.sh: 15 passed, 0 not passed (outputs in /home/jshong/Code/unfied-sr-denoise-platform/results/docker_accept_20261019_fix)
3499 passed, 318 warnings in 379.71s (0:06:19)
```

| 일자 | 환경 | 명령 | 결과 |
|---|---|---|---|
| 2026-10-19 | `usdp:20261019`(`112b82a375dd`) 안 | `docker/run.sh --offline -- … pytest -q -p no:cacheprovider tests/test_docker_image.py` | 미룬 Minor 처리(v0.5). 반영 전 TC-DOCK-01-6(기록 파일 없음)·02-1(venv 목록에 pip·setuptools 없음)·15-4·15-5 실패, 반영 뒤 16 passed |
| 2026-10-19 | 호스트 | `docker/build.sh` | 23:25–23:51, 다시 빌드. 빌드 초기 단계 `Successfully installed packaging-25.0 pip-26.0.1 setuptools-82.0.0 wheel-0.46.3`, 이미지 `7fd2f0ce9454`. 새 이미지에서 `tests/test_docker_image.py` 16 passed(apt 319개·venv 7줄 기록과 같음) |
| 2026-10-20 | `usdp:20261019`(`7fd2f0ce9454`), GPU 2, `--offline`, 00:57–01:15 | `docker/accept.sh --gpu 2 --out results/docker_accept_20261020` | 15건 모두 PASS. pytest 4382 passed = 기대값 4382(280 + 4086 + 16), SKIPPED 0. DN ONNX `81ac40c7…`, GPU export `f4a757f3…` 그대로 |
```text
PASS TC-DOCK-09-1
PASS TC-DOCK-09-2
PASS TC-DOCK-10-1
PASS TC-DOCK-10-2
PASS TC-DOCK-10-3
PASS TC-DOCK-11-1
PASS TC-DOCK-11-2
PASS TC-DOCK-12-1
PASS TC-DOCK-12-2
PASS TC-DOCK-13-1
PASS TC-DOCK-13-2
PASS TC-DRUN-01-3
PASS TC-DRUN-03-3
PASS TC-DRUN-04-4
PASS TC-DRUN-05-5
accept.sh: 15 passed, 0 not passed (outputs in /home/jshong/Code/unfied-sr-denoise-platform/results/docker_accept_20261020)
4382 passed, 318 warnings in 637.13s (0:10:37)
```

| 일자 | 환경 | 명령 | 결과 |
|---|---|---|---|
| 2026-10-20 | `usdp:20261019`(`7fd2f0ce9454`), GPU 2, `--offline`, 01:57–02:15 | `docker/accept.sh --gpu 2 --out results/docker_accept_20261020_final` | 2차 수정(v0.6) 뒤 15건 모두 PASS. pytest 4394 passed = 기대값 4394(280 + 4096 + 18), SKIPPED 0. DN ONNX `81ac40c7…`, GPU export `f4a757f3…` 그대로. TC-DOCK-15-6은 반영 전 FAIL(중단 줄이 `exit 0`), 반영 뒤 PASS |
```text
PASS TC-DOCK-09-1
PASS TC-DOCK-09-2
PASS TC-DOCK-10-1
PASS TC-DOCK-10-2
PASS TC-DOCK-10-3
PASS TC-DOCK-11-1
PASS TC-DOCK-11-2
PASS TC-DOCK-12-1
PASS TC-DOCK-12-2
PASS TC-DOCK-13-1
PASS TC-DOCK-13-2
PASS TC-DRUN-01-3
PASS TC-DRUN-03-3
PASS TC-DRUN-04-4
PASS TC-DRUN-05-5
accept.sh: 15 passed, 0 not passed (outputs in /home/jshong/Code/unfied-sr-denoise-platform/results/docker_accept_20261020_final)
4394 passed, 318 warnings in 651.43s (0:10:51)
```

## 6 미해결 · 결정
**소유자 결정(2026-10-19)**
- Q1. 이미지 하나 + QAIRT 전용 venv
- Q2. 옛 컨테이너 목록에서 출발해 안 쓰는 39개만 뺀다(§2.5)
- Q3. QAIRT는 빌드할 때 tarball에서 풀어 굽는다
- Q4. 필요한 것만 붙인다(§2.6). `/mnt`는 쓰기 가능
- Q5. 실행 스크립트 하나, `--gpu N`(한 장). GPU 사용 중이면 `--force` 없이는 거부. 마운트가 없으면 항상 멈춤
- Q6. ffmpeg 7.0.2를 이미지에 넣는다
- Q7. 필요한 것만 옮기고, 삭제 조건 통과 뒤 지운다(§2.7)
- Q8. VSCode는 필요할 때 컨테이너에 붙인다
- R1. DLC는 임시 폴더 이름을 고정한 md5 비교(A안). 실측해 보니 md5는 같은 서버에서 두 번 돌려도 달랐다. QAIRT 2.41이 `startOpIndex`·`endOpIndex`(`QnnSystemContext_GraphInfoV3_t`, 초기값 0)에 초기화하지 않은 메모리를 쓰기 때문이다. 그래서 B안(그 두 값만 가린 구성 파일 바이트 비교)과 소유자 보드 확인으로 바꿨다(소유자 결정 10-19)
- R2. 삭제 조건에 GPU export와 1 epoch 학습을 넣는다. md5가 어긋나면 옛 컨테이너와 비교한다
- R3. 소유자가 `/mnt/data2/jshong`을 jshong 소유로 바꿨다
- R4. QAIRT venv는 첫 변환이 성공한 버전으로 고정한다(회사 PC는 WSL)

**작성자 판단(0.1에서 승인됨)**
- `build.sh`, `entry.sh`, `accept.sh`를 둔다.
- `run.sh` 부가 옵션(`--offline`, `USDP_*`, `--mount`의 DST·ro)을 둔다.
- `usdp-*` 컨테이너가 잡은 GPU도 사용 중으로 친다.
- `--init` + `TINI_KILL_PROCESS_GROUP=1`, `--ipc host`를 쓴다.
- 태그는 `usdp:<날짜>`·`latest`이고, 네트워크는 bridge이다.
- 이미지에 curl을 넣고, `NVIDIA_VISIBLE_DEVICES=void`를 기본값으로 둔다.
- 학습 중지는 즉시 끊고 `last.pth`에서 재개한다.
- DN MANIFEST에 경고 두 줄을 넣는다(복사본에서 변환, 변환한 DLC의 md5는 매번 다름).
- 문서를 DockerRun으로 나눈다(0.2).

**미해결**
- 이 서버는 k3s 노드라 GPU와 디스크를 클러스터와 나눈다. 이미지는 약 20 GB 늘고, 옛 컨테이너를 지우면 약 65 GB가 준다.
- 변환한 DLC는 md5로 재현을 확인할 수 없다. `mktemp` 임시 경로가 들어가고, QAIRT 2.41이 metadata 두 칸에 초기화하지 않은 메모리를 쓴다. 같은 내용인지는 `accept.sh`의 `same_dlc`로 본다. 생성 스크립트의 임시 경로를 고칠지는 이 모듈 밖의 별도 작업으로 둔다.
- PyPI 파일은 버전으로만 고정하고 해시(`--require-hashes`)는 쓰지 않는다. 같은 버전에 새 휠이 올라오면 캐시 없는 재빌드에서 다른 파일을 받을 수 있다. 재빌드 뒤에는 `docker/accept.sh`로 합격 확인을 다시 한다.
- `tests/test_gen_convert_all.py`는 `script_lines()` 출력 전체를 해시로 고정한다. 그래서 `gen_convert_all.py`는 docstring만 고친다.
