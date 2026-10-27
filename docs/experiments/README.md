# 실험·릴리스 기록

이 폴더는 모델과 학습 레시피 연구의 기록을 둔다. 이 파일은 네 가지를 다룬다.

1. 실험을 기록하는 법: 가설과 실험 ID, 노이즈 하한, 산출물을 두는 곳 (2절)
2. 릴리스를 기록하는 법: 릴리스 폴더, MANIFEST, golden 테스트, 모듈 문서 (3절)
3. 영상과 큰 산출물을 두는 곳 (4절)
4. IR x4 격자 연구의 실험 레지스트리. 2026-10-06에 마감한 기록이다 (5절)

명령은 모두 저장소 루트에서 `docker/run.sh`로 이미지 `usdp` 안에서 돌린다. 호스트의 `python3`(3.8.10, 다른 프로젝트용 torch 2.4.1만 있고 aimet 없음)는 작업 환경이 아니다. `docker/run.sh`의 옵션, 거절 메시지, GPU 고르는 법은 [docker/README.md](../../docker/README.md)에 있다. 학습·평가·export CLI의 옵션은 [tools/README.md](../../tools/README.md), config 작성법은 [configs/README.md](../../configs/README.md), 변환과 보드는 [examples/deployment/README.md](../../examples/deployment/README.md)를 본다.

## 1 지금 있는 연구 기록

| 연구 | 계획서 (RESEARCH) | 레지스트리 | 실험 문서 | 상태 | 결과물 |
|---|---|---|---|---|---|
| IR x4 격자 | [20260928-RESEARCH-X4IRGrid.md](../20260928-RESEARCH-X4IRGrid.md) | 이 파일 5절 | 이 폴더의 `2026MMDD-EXP-nn-k.md` 17개 | 마감 2026-10-06, BL2 = EXP-13-1 | IR v3 수정판, [release_v3 MANIFEST](../../results/release_v3/MANIFEST.md) |
| EO Denoise v3 | [20261008-RESEARCH-EODenoiseV3.md](../20261008-RESEARCH-EODenoiseV3.md) | [eo_denoise/README.md](eo_denoise/README.md) | 없음 (레지스트리 행만) | 종료 2026-10-13, EXP-09-1 배포 | BicubicPPDenoise[v1], [denoise_release_v1 MANIFEST](../../results/denoise_release_v1/MANIFEST.md) |

실험 ID는 한 연구 안에서만 유일하다. EO Denoise의 EXP-01-1과 IR 격자의 EXP-01-1은 다른 실험이다. 다른 문서에서 실험을 가리킬 때는 연구 이름을 같이 적는다(예: "IR 격자 EXP-13-1").

두 연구의 실험 config(`configs/experiments/`)와 체크포인트(`checkpoints/experiments/`)는 연구가 끝난 뒤 소유자 결정으로 지웠다. 문서에 남은 그 경로와 학습·평가 명령은 실제로 돌린 기록이고, 지금 그대로 치면 돌지 않는다. 출하에 쓰인 가중치와 채점 입력은 릴리스 폴더로 옮겨 두었다.

## 2 실험을 기록하는 법

### 2.1 실험으로 다룰 일과 모듈로 다룰 일

결과가 벤치마크 숫자인 일은 실험으로 기록한다. 손실 가중치, 증강, 데이터 구성, 구조 변형, 양자화 설정, 학습 일정이 여기에 든다.

매번 똑같이 동작해야 하는 코드는 실험이 아니라 모듈 작업이다. 데이터 로더, Trainer, export, 채점기 코드가 그렇다. 이것들은 `docs/modules/`의 모듈 문서와 단위 테스트로 다룬다(3.5절).

채점기는 두 쪽에 걸친다. 먼저 모듈 작업으로 만들고 테스트한 뒤(예: [MOD-GridScore](../modules/20260928-MOD-GridScore.md), [MOD-ChartScore](../modules/20261008-MOD-ChartScore.md)), 연구 계획서 §2에서 그 버전을 프로토콜로 고정한다.

### 2.2 문서 세 단계

| 단계 | 파일 | 담는 것 |
|---|---|---|
| 계획서 (연구당 하나) | `docs/<YYYYMMDD>-RESEARCH-<Topic>.md` | §1 목표와 측정 가능한 목표치, §2 평가 프로토콜 `P<v>`와 노이즈 하한, §3 베이스라인 `BL<v>`, §4 가설 `HYP-nn` 표, §5 결론 |
| 레지스트리 (연구당 하나) | `docs/experiments/<topic>/README.md` | 실험 한 줄씩, 최신이 위. 머리에 현재 `BL`과 `P`, 끝에 프로토콜 변경 이력 |
| 실험 문서 (실험당 하나) | `docs/experiments/<topic>/<YYYYMMDD>-EXP-nn-k.md` | §1 가설과 예상, §2 바꾼 축 하나(config diff, 코드 commit), §3 실행(명령, 시드, 채점한 체크포인트, 실제 epoch, 환경, 유효 배치, 소요, 산출물), §4 BL 대비 결과와 노이즈 하한, §5 판정 |

새 연구는 레지스트리와 실험 문서를 `docs/experiments/<topic>/` 아래에 둔다(EO Denoise가 그렇게 했다). IR 격자 연구의 파일은 기록이라 이 폴더 바로 아래에 그대로 둔다.

빈 양식은 `~/.claude/templates/docs/`의 `RESEARCH.md`, `EXPERIMENTS.md`, `EXPERIMENT.md`에 있다. 채워진 예로는 [20260928-RESEARCH-X4IRGrid.md](../20260928-RESEARCH-X4IRGrid.md)와 [20261006-EXP-13-1.md](20261006-EXP-13-1.md)가 가장 완전하다. 문서 이름의 날짜는 만든 날이고 바꾸지 않는다. 고친 내용은 문서 안 개정 이력 표에 적는다.

### 2.3 ID

| ID | 어디에 | 뜻 |
|---|---|---|
| `HYP-nn` | 계획서 §4 | "X를 바꾸면 Z 때문에 지표 M이 약 Y만큼 움직인다" |
| `EXP-nn-k` | 실험 문서 | `HYP-nn`을 시험하는 k번째 실험 |
| `EXP-00-k` | 실험 문서 | 가설 없는 실험. `EXP-00-1`은 베이스라인 재현, `EXP-00-2`부터는 시드만 바꾼 노이즈 하한 측정 |
| `P<v>` | 계획서 §2 | 평가 프로토콜 버전. 일부를 줄인 싼 변형은 `P<v>-proxy`로 따로 선언하고, 그 결과로는 채택하지 않는다 |
| `BL<v>` | 계획서 §3 | 베이스라인 버전. 실험이 채택될 때만 바뀐다 |

### 2.4 규칙

- **실험 하나에 축 하나.** 축은 바꾸는 키 하나, 또는 함께 움직여야 하는 이유를 계획서에 적어 둔 키 묶음 하나다. 두 축을 같이 바꾸면 결과를 어느 쪽 탓으로도 돌릴 수 없다.
- **코드와 환경도 축이다.** 실험의 코드 commit, 이미지, GPU 종류, 유효 배치(배치 × GPU 수 × grad accum. 예: EXP-13-1은 8 × 1 × 2 = 16)는 베이스라인과 같아야 한다. `docker/run.sh`는 GPU를 한 장만 붙이므로 GPU 수는 늘 1이다.
- **새 연구는 지금 코드와 이미지에서 `EXP-00-1`, `EXP-00-2`부터 돈다.** 2026-10-16에 학습 분포를 바꾼 수정 세 가지가 들어갔다(출하 학습을 비트 단위로 다시 하려면 그 전 커밋 `de502f6`, [CONFIG_RATIONALE](../CONFIG_RATIONALE.md) 4.4). 이전 실험은 지금의 `usdp` 이미지가 아닌 옛 환경에서 쟀다. 옛 표의 숫자와 새 런을 바로 비교하면 코드와 환경의 차이가 효과처럼 보인다.
- **프로토콜은 실험 동안 고정한다.** 검증 세트, 전처리, 시드, 채점 코드와 버전, 명령, 채점할 체크포인트를 고르는 규칙이 프로토콜이다. 하나라도 바꾸면 `P<v>`를 올리고, 노이즈 하한을 다시 재고, BL과 채택된 실험을 다시 채점하고, 레지스트리의 프로토콜 변경 이력에 적는다. 다른 `P` 버전의 숫자끼리는 비교하지 않는다.
- **결과를 쓰기 전에 실행 기록부터 남긴다.** 정확한 명령, 시드, 코드 commit(`git rev-parse --short HEAD`, 추적 파일에 커밋 안 한 변경이 있으면 그 사실), config 파일, 조기 종료가 있으면 실제로 돈 epoch, 채점한 체크포인트 파일. 이것이 빠진 결과는 버린다. 체크포인트 안에도 `git_sha`가 저장된다.
- **판정은 채택 / 기각 / 재실험 셋 중 하나다.** "유망"은 판정이 아니다.
- **채택은 한 커밋에.** 실험 문서 §5, 레지스트리 행과 머리, 계획서 §3(새 BL)과 §5를 같은 커밋에서 고친다. 커밋 제목 예: `[20261006] Research Close the X4IRGrid study with EXP-13-1 as the shipped recipe`.
- **합성 val만으로 채택하지 않는다.** 이 저장소에서는 합성 val이 좋아져도 실기에서 나빠진 일이 여러 번 있었다. W16 QAT 30에폭은 합성 PSNR을 올렸지만 실기 IR 프레임을 더 흐리게 만들었다([release_v3 MANIFEST](../../results/release_v3/MANIFEST.md) 'QAT → PTQ'). 실기 프레임 게이트(IR은 P2 격자, Denoise는 P1.1 차트)를 프로토콜에 넣는다.

### 2.5 노이즈 하한

노이즈 하한은 "아무것도 바꾸지 않아도 이만큼은 움직인다"는 폭이다. 이 폭을 넘지 못한 차이는 효과로 인정하지 않는다.

재는 법은 이렇다. `EXP-00-1`로 베이스라인 config를 그대로(시드 42) 다시 학습해 BL이 재현되는지 본다. 이어서 `EXP-00-2`로 시드만 43으로 바꿔 학습한다. 두 런의 지표별 차이(절댓값) 가운데 가장 큰 값을 계획서 §2에 하한으로 적는다.

같은 코드와 환경에서는 학습이 결정적이다. IR 격자의 EXP-00-1은 출하 v3와 가중치가 비트 단위로 같았다. 그래서 시드를 바꾸는 것이 곧 노이즈를 재는 방법이 된다.

| 연구 | 측정 | 하한 |
|---|---|---|
| IR 격자, P2 | EXP-00-1 대 EXP-00-2 (best) | 1/4h 1.02 · 1/4d 1.13 · 1/4v 0.75 · val LPIPS 0.00078 |
| EO Denoise, P1.1, A 레시피 | EXP-01-1 대 EXP-01-2 | 10배 노이즈 0.40 · 더한 넘침 0.15 %p |
| EO Denoise, P1.1, C 레시피 | EXP-04-1 대 EXP-04-2 | 노이즈 0.02 · 넘침 0.1 %p |

IR 격자 연구 후반에는 하한의 3배를 넘는 차이만 효과로 인정했다(EXP-13-1 §1). 위 EO의 A와 C처럼 레시피가 크게 바뀌면 하한도 달라진다. 그때는 새 레시피에서 시드 쌍을 다시 돌린다.

**채점기도 먼저 의심한다.** EO Denoise의 P1 차트 채점기는 잡음 낀 프레임마다 에지를 따로 맞췄다. 그래서 잡음 없는 완벽한 출력을 넣어도 넘침 +5.6 %p, MTF 비 1.73으로 읽었고, 어떤 Denoise도 공정하게 줄 세울 수 없었다. P1.1에서 에지를 입력에서 한 번 맞추고 출력에 다시 쓰도록 고쳤다([eo_denoise/README.md](eo_denoise/README.md) 프로토콜 변경 이력). 새 지표는 정답(완벽한 출력)을 넣었을 때 중립값이 나오는지부터 확인한다.

**채점 입력은 사본과 sha256 목록으로 고정한다.** IR 격자의 P1은 데이터 서버의 채점 폴더가 바뀌어 P2로 바꿔야 했다. P4는 cam98, FLIR, LLVIP, M3FD 입력이 데이터 서버에서 지워져(2026-10-15 확인) 이제 다시 채점할 수 없다. 지금 살아 있는 P2와 P1.1은 입력 사본이 릴리스 폴더에 있다: `results/release_v3/eval/ir92_raw_600/` + `ir92_p2_frames.sha256`, `results/denoise_release_v1/eval/*.nv12` + `P1_clips.sha256`.

### 2.6 무엇을 어디에 두는가

| 무엇 | 위치 | git |
|---|---|---|
| 계획서, 레지스트리, 실험 문서 | `docs/`, `docs/experiments/<topic>/` | 커밋한다 |
| 실험 config | `configs/experiments/<topic>/<name>.yaml` | 커밋하지 않는다(소유자 원칙, 2026-09-28). 실험 문서 §2에 diff를 적는다 |
| 런 폴더 (체크포인트, 콘솔 로그, 채점 JSON) | `checkpoints/experiments/<topic>/<YYYYMMDD>-EXP-nn-k/` | `checkpoints/`는 git이 무시한다 |
| 분석 그림, 일회성 스크립트 | `results/<YYMMDD>_<name>/` | `results/`는 릴리스 폴더의 기록 텍스트만 추적한다(3.1절) |
| 영상, 프레임 묶음 | 저장소 밖 (4절) | – |

`configs/experiments/`는 `.gitignore`에 들어 있지 않다. `git add -A`나 `git add .`를 쓰면 실험 config가 같이 커밋되므로, 커밋할 파일을 골라서 `git add`한다.

판정에 쓰는 지표가 된 일회성 스크립트는 `examples/evaluation/`으로 옮기고 모듈 문서와 테스트를 만든다. IR 격자의 `score_grid_artifact.py`가 그렇게 들어왔다. `results/` 안에 남은 채점 스크립트는 git이 추적하지 않아서, 폴더를 정리하면 같이 사라진다.

### 2.7 실행과 채점 명령

모두 저장소 루트에서 친다. `--log`의 상대 경로는 호스트의 현재 폴더 기준으로 풀리고, 명령 안의 경로는 컨테이너의 저장소 루트 기준으로 풀리기 때문이다. GPU는 `nvidia-smi`로 빈 것을 고른다. `nvidia-smi`에 프로세스가 보이거나 다른 `usdp` 작업이 쓰는 GPU는 `docker/run.sh`가 거절한다.

학습은 이름을 붙여 백그라운드로 띄우고, 콘솔 로그를 런 폴더에 둔다.

```bash
E=checkpoints/experiments/<topic>/<YYYYMMDD>-EXP-nn-k
docker/run.sh --gpu <N> --name <topic>_EXP-nn-k --log $E/train.log -- \
    python3 tools/train.py --config configs/experiments/<topic>/<name>.yaml --work_dir $E --seed 42
tail -f $E/train.log    # a last line "[run.sh] exit 0" means the job ended cleanly
```

`--name`에는 영문, 숫자, `_`, `.`, `-`만 쓸 수 있고 첫 글자는 영문이나 숫자다. 컨테이너 이름은 `usdp-<name>`이 된다. 멈출 때는 `docker stop usdp-<name>`이고, 이어서 학습할 때는 `docker/run.sh --gpu <N> --name <새 이름> -- python3 tools/train.py --resume $E/last.pth`다. 재개 규칙과 주의점은 [tools/README.md](../../tools/README.md)에 있다.

채점할 체크포인트는 `best.pth`(EMA 가중치)다. Phase-2처럼 사전학습 가중치에서 출발하는 런은 학습 전에 그 가중치를 한 번 검증하고, 그 값을 넘을 때만 `best.pth`를 쓴다. 콘솔 로그에는 다음과 같은 줄이 남는다.

```
[Trainer] Baseline from pretrained weights: lpips=0.167224 (loss=0.033917). best.pth is only written if this is beaten.
```

이 기준값은 `training_log.csv`에 들어가지 않는다(CSV는 epoch 1부터). 로그에서 실험 문서 §4로 옮겨 적는다. 끝까지 기준을 넘지 못하면 `best.pth`가 없고 `best_attempt.pth`(그 런의 최선), `last.pth`, `epoch_<N>.pth`만 남는다. 이때는 "사전학습 대비 개선 없음"으로 기록하고 `best_attempt.pth`를 채점한다. 2026-10-19의 EO x2 1에폭 시험 런(`results/smoke_20261019/x2_eo_p2`)이 그런 경우였다(기준 LPIPS 0.1672, epoch 1은 0.1772).

합성 val 채점은 `tools/evaluate.py`로 한다.

```bash
docker/run.sh --gpu <N> -- python3 tools/evaluate.py --config configs/experiments/<topic>/<name>.yaml \
    --checkpoint $E/best.pth --save_dir $E/eval
```

`$E/eval/best/metrics.csv`에 영상별 값이 남는다(하위 폴더 이름은 체크포인트 파일 이름). 같은 체크포인트라면 학습 중 검증과 같은 값이 나온다(2026-10-19 시험 런에서 PSNR 31.0827, LPIPS 0.1772로 일치). Phase-1, EO Phase-2, IR Phase-2, Denoise config는 서로 다른 LR 검증 세트를 쓴다. 다른 config의 런끼리 비교할 때는 `--hr_dir`, `--lr_dir`로 같은 세트를 준다.

실기 프레임 게이트는 CPU로 돌린다(`--gpu` 없이).

```bash
# IR: P2 grid score. The default crop is the x4 board input (320x180); other scales need --crop x,y,w,h.
docker/run.sh -- env PYTHONPATH=. python3 examples/evaluation/score_grid_artifact.py --ckpt $E/best.pth --json $E/grid_best.json
# Denoise: P1.1 chart score
docker/run.sh -- env PYTHONPATH=. python3 examples/evaluation/score_eo_chart.py --checkpoint $E/best.pth --json $E/p11_best.json
```

`examples/evaluation/`의 채점기는 `sys.path`를 고치지 않으므로 `env PYTHONPATH=.`가 필요하다. 채점을 CPU로 하는 이유는 GPU 합성곱이 TF32로 돌 수 있어서다. 그러면 같은 체크포인트도 프로토콜 값이 조금씩 달라진다.

## 3 릴리스를 기록하는 법

릴리스는 소유자가 보드 화면을 보고 배포를 정한 모델 묶음이다. 지금 두 개가 있다.

| 릴리스 | 폴더 | 배포 결정 |
|---|---|---|
| Bicubic++ SR v3 (EO/IR x2·x3·x4, W16A16) | [results/release_v3/MANIFEST.md](../../results/release_v3/MANIFEST.md) | 2026-09-14, IR 수정판 2026-10-06 |
| BicubicPPDenoise[v1] (EO 1280x720, W16A16) | [results/denoise_release_v1/MANIFEST.md](../../results/denoise_release_v1/MANIFEST.md) | 2026-10-13 |

새 릴리스는 이 두 폴더를 본떠 만든다.

### 3.1 폴더 이름과 git

릴리스 폴더는 `results/` 아래에, 이름에 `release`가 들어가게 만든다(예: `results/release_v4`, `results/denoise_release_v2`). `.gitignore`는 `results/`를 통째로 무시하고, 이름에 `release`가 든 폴더의 맨 위 `*.md`와 `*.sh`, 그리고 `eval/export.sh`만 다시 추적한다. 이름에 `release`가 없으면 MANIFEST도 커밋되지 않는다. 체크포인트, ONNX, DLC, `.nv12` 같은 바이너리는 어느 폴더에서든 추적하지 않는다.

그래서 릴리스 폴더의 바이너리는 이 서버 디스크에만 있다. golden 테스트(3.4절)가 이 파일들을 기준으로 쓰므로 지우거나 옮기지 않는다.

### 3.2 폴더 구성

| 내용 | SR v3 (`release_v3`) | Denoise v1 (`denoise_release_v1`) |
|---|---|---|
| 체크포인트 (학습 상태 그대로: EMA, ECB 안 접힘) | `checkpoints/<EO\|IR>_x<s>.pth` | `checkpoint/best.pth` |
| 학습 당시 config 사본과 실행 기록 | 없음 | `checkpoint/train_config.yaml`, `data_config.yaml`, `run.txt`, `code_commit.txt`, `train.log`, `training_log.csv` |
| export 결과 (변환 입력) | `x<s>_<eo\|ir>[_1024]/qat/qat_model.{onnx,encodings}` | `dn_bpp_half4_w16a16/qat/` |
| 변환 스크립트 (기록) | `convert_all.sh` | `convert_all.sh` |
| 보드에 올린 DLC 그대로 | `dlc/*.dlc` | `BicubicPPDenoise_EO_1280x720_v1.dlc` |
| 재-export 레시피 | `eval/export.sh`(IR), MANIFEST의 EO 명령 | `export.sh` |
| 채점 입력과 점수 | `eval/` (P2 프레임, `scores/`) | `eval/` (P1.1 클립, `scores/`), `analysis/` |
| 비교 그림 | `compare/` | `compare/` |

다음 릴리스에서 지킬 것이 세 가지 있다.

- **학습 당시 config 사본을 넣는다.** Denoise의 `export.sh`는 처음에 저장소의 config를 읽었는데, 2026-10-14 config 이동(`1817870`)으로 경로가 바뀌자 멈췄다. 지금은 `checkpoint/`의 사본을 읽는다. 런 폴더의 `train_config.yaml`과 `data_config.yaml`을 같이 복사해 두면 저장소가 바뀌어도 재-export가 된다.
- **보드에 올린 DLC 파일을 그대로 보관하고, 다시 올릴 때도 그 파일을 쓴다.** QAIRT 2.41로 변환한 DLC는 같은 입력으로 다시 변환해도 md5가 매번 다르다. `dlc.metadata`의 `startOpIndex`와 `endOpIndex`에 초기화하지 않은 값이 들어가고, `convert_all.sh`가 쓴 `mktemp` 임시 폴더 경로도 들어가기 때문이다. 다시 변환한 파일을 올리면 보드의 md5가 MANIFEST와 달라져, 무엇이 올라가 있는지 확인할 수 없게 된다. 내용을 비교하는 방법은 [examples/deployment/README.md](../../examples/deployment/README.md)에 있다.
- **변환은 릴리스 폴더 밖의 복사본에서 한다.** `convert_all.sh`는 DLC를 스크립트 옆에 `mv -f`로 쓴다. Denoise 폴더에서 돌리면 출하 DLC를 덮어쓰고, SR v3 폴더에서 돌리면 HTP prepare가 없는 DLC 9개가 보드와 같은 이름으로 생긴다.

### 3.3 MANIFEST에 쓰는 것

두 MANIFEST가 쓴 절을 기준으로 한다.

| 절 | 내용 |
|---|---|
| 제목·날짜와 한 줄 요약 | 이전 릴리스와 무엇이 다른가 |
| 재학습 재현 | 비트 단위로 다시 학습할 수 있는 커밋 |
| 배포 구성 | 모델, 체크포인트와 md5, 학습·데이터 config, 양자화(W16A16 PTQ, QAT 0에폭, 보정 수), DLC 이름과 md5, 입출력 텐서 이름, 출력층 `final_act`, HTP prepare 여부 |
| md5 표 | 체크포인트, qat ONNX와 encodings, 보드에 올린 DLC (호스트에서 `md5sum`) |
| 실측 | 합성 val, 실기 프레임 게이트, 보드(DLC 출력과 FP32의 정합 PSNR, `bench_dlc.sh` 지연과 그때 보드 온도) |
| 알려진 약점 | 남은 문제와 그 장면 |
| 변환과 배포 | 복사본에서 변환하는 명령, 보드에 올리는 명령 |
| 다시 만들기 | 학습, export, 채점 명령 (`docker/run.sh` 형식) |
| 미확인 | 재지 못한 것, 소유자 판단을 기다리는 것 |
| 구성 | 폴더 트리와 각 항목 설명 |

지연에는 burst인지 1분인지 꼭 적는다. `bench_dlc.sh`의 기본은 `PERF=burst`(5초 × 5회)이고, 1분 값은 `PERF=sustained_high_performance DURATION=60`으로 잰다. 2026-10-12 보드 98에서 두 설정의 차이(23~46 %)는 온도보다 전력 설정 탓이었고, 같은 그래프도 세션마다 약 2 ms(7 %) 흔들렸다. 그래서 구조끼리의 비교는 같은 세션에서 잰 기준 DLC 대비 비율로 적는다.

### 3.4 golden 테스트

출하 파일이 바뀌지 않았다는 것을 테스트가 고정한다. 새 릴리스를 만들면 같은 커밋에서 기록값을 더한다.

| 파일 | 상수 | 고정하는 것 |
|---|---|---|
| `tests/test_release_golden.py` | `CHECKPOINTS` | 체크포인트 md5, 체크포인트 안 model config가 config 파일의 `model` 블록과 같음, strict 로드, 배포 변환 뒤 state_dict와 고정 입력 출력의 sha256 |
| | `PROFILES` | 보드 프로파일별 W16A16 export의 ONNX md5와 param encodings |
| | `SEEDED_INIT` | 시드 고정 초기값 (구조를 새로 낼 때만) |
| `tests/test_data_recipe_golden.py` | `DIGESTS` | 데이터 config가 만드는 학습 쌍의 sha256 (새 데이터 config를 출하할 때) |

`PROFILES`의 ONNX md5 칸이 `None`이면 테스트가 CPU에서 export한 결과를 출하 qat 파일과 직접 비교한다. 출하 파일을 GPU에서 export했다면 CPU 결과와 W16 scale이 1 ulp 달라 md5가 다르다. 그때는 CPU export의 md5를 적는다(EO v3가 그렇다). 체크포인트와 ONNX의 md5는 호스트에서 `md5sum`으로 잰다. 테스트가 계산하는 sha256은 테스트가 평범한 `assert a == b`라서, 항목을 넣고 돌리면 실패 메시지에 계산값이 나온다.

같이 고칠 곳:

- [docs/modules/20261015-MOD-ReleaseGolden.md](../modules/20261015-MOD-ReleaseGolden.md): MREQ-REL의 개수, §4 TC 표의 parametrize 수, §5 결과 행. 기록값을 일부러 바꿀 때는 같은 커밋에서 다시 재고, 이유를 커밋과 그 문서 §5에 적는다.
- `docker/accept.sh`의 `BASE_TESTS`(2026-10-19 기준 280): Docker 테스트 두 파일 밖의 테스트 수다. 테스트를 더한 만큼 올린다. `accept.sh`는 이 수와 Docker 테스트 수의 합이 정확히 `N passed`로 나오기를 요구해서(TC-DOCK-09-1), 그대로 두면 인수 시험이 실패한다.
- 새 릴리스가 `accept.sh`의 기준(SR v3의 IR x4와 EO x2 export, Denoise v1의 export와 DLC)을 대신하게 되면 `accept.sh` 상수(`DN_ONNX_MD5`, `IR_X4_MD5`, `IR_X4_1024_MD5`, `DN_DLC`, `DN_DLC_MD5`, `DN_DLC_TMP`, `X2_EO_GPU_MD5`, `X2_EO_CPU_MD5`)와 [docs/modules/20261019-MOD-DockerEnv.md](../modules/20261019-MOD-DockerEnv.md)의 MREQ-DOCK-10~12를 고친다.

golden 테스트는 기준 파일이 없으면 실패하지 않고 skip한다. 평범한 pytest는 skip이 있어도 종료 코드가 0이라 놓치기 쉽다. 그래서 늘 `-rs`로 돌려 skip이 0인지 본다. 옛 릴리스 파일을 지울 때는 그 golden 항목도 같은 커밋에서 지운다. 그러지 않으면 skip이 생기고 `accept.sh`가 실패한다.

```bash
docker/run.sh --offline -- env PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q -rs -p no:cacheprovider \
    tests/test_release_golden.py tests/test_data_recipe_golden.py
```

2026-10-19 기준으로 통과는 `56 passed`이고 `skipped`가 없어야 한다. `PYTHONDONTWRITEBYTECODE=1`과 `-p no:cacheprovider`는 저장소에 `__pycache__`와 `.pytest_cache`를 남기지 않게 한다. GPU는 붙이지 않는다.

### 3.5 모듈 문서

릴리스가 기대는 코드 변경(새 구조, 새 손실, 새 열화 연산, 새 채점기)은 `docs/modules/<YYYYMMDD>-MOD-<Module>.md`에 적는다. 순서는 요구 → 설계 → 상세 설계 → 단위 시험 케이스 → 결과다. 테스트 함수 이름에 TC ID를 넣는다(예: `test_TC_REL_02_1_AimetExportMatchesShipped`). 그러면 어떤 요구가 테스트됐는지 grep으로 찾을 수 있다. 문서와 그 코드는 같은 커밋에서 바꾼다.

Denoise v1의 예: 모델 구조는 [MOD-BicubicPPDenoise](../modules/20261012-MOD-BicubicPPDenoise.md), 학습 입력을 만드는 카메라 모델은 [MOD-CameraModel](../modules/20261008-MOD-CameraModel.md), 게이트 채점기는 [MOD-ChartScore](../modules/20261008-MOD-ChartScore.md)에 있다. 연구의 숫자(지표, 판정)는 모듈 문서가 아니라 연구 문서에 남긴다.

### 3.6 같이 고칠 곳과 커밋

- [README.md](../../README.md)의 모델 목록.
- 보드 스크립트 기본값. SR은 `gst_compare_*.sh`의 `BICUBICPP_VER`(지금 `v3`), Denoise는 `gst_compare_eo.sh`와 `gst_compare_eo_dual.sh`의 `DENOISE`, `gst_compare_eo_denoise.sh`의 `DENOISE_V1`이다. 고친 스크립트를 보드에 복사하고 화면을 확인하는 일은 소유자가 한다([examples/deployment/README.md](../../examples/deployment/README.md)).
- 연구 레지스트리의 머리와 계획서 §5 (채택 커밋, 2.4절).
- 커밋 제목은 `[YYYYMMDD] <Component> <Verb> <what>` 형식이다. 지금까지의 예: `[20261016] Release Add the SR v3 and Denoise v1 release records`, `[20261016] Release Update both release records with the retraining reference commit`.

## 4 영상과 큰 산출물

비교 영상, 보드에서 뽑은 프레임 묶음, 녹화 파일은 저장소의 `results/`에 만들지 않는다(소유자 결정, 2026-10-13). 저장소가 있는 디스크는 2026-10-19에 86 % 찼다(남은 공간 124 GB). 영상 작업 하나가 58 GB까지 커져 작업 도중에 공간을 비워야 했던 일도 있었다.

| 위치 | 용도 |
|---|---|
| `/mnt/data_server/etc/jshong/SuperResolution/results/<YYMMDD>_<name>/` | 보관용. 지금 `261006_ir_sr_video`, `261013_eo_sr_video`가 있고, 그 폴더에 만들 때 쓴 스크립트도 같이 있다. sshfs라 느리다 |
| `/mnt/data2/jshong/...` | 작업 중인 큰 파일. 로컬 ext4 디스크라 빠르다(2026-10-19 남은 공간 785 GB) |

작업마다 둘 중 어디에 쓸지 먼저 정한다. 컨테이너는 `/mnt`를 그대로 보므로 작업이 그 경로에 바로 쓸 수 있다. `*.mp4`와 `*.nv12`는 어느 폴더에서든 git이 무시한다.

보드에서 `RECORD=1`로 gst 스크립트를 돌리면 보드의 `/data/<STAMP>_<스크립트 이름>_x<배율>.../`에 녹화 파일이 생긴다(`OUT_ROOT` 기본 `/data`). 예를 들어 `gst_compare_ir.sh`는 `mosaic.mp4`와, 타일별 DLC md5를 적은 `models.txt`를 쓴다. 이것도 받아 올 때는 위 두 곳 중 하나로 받는다.

## 5 IR x4 격자 실험 레지스트리 (마감, 기록)

- 현재 베이스라인: **BL2** = EXP-13-1 (2026-10-06 채택, IR x4 v3 출시) · 상태: **마감 (2026-10-06)** — 결론은 `docs/20260928-RESEARCH-X4IRGrid.md` §5. 이전 BL1 = EXP-00-1(출하 원판)
- 현재 프로토콜: **P4** (2026-10-04~, 이전 런은 P2·P3) (`docs/20260928-RESEARCH-X4IRGrid.md` §2) · 노이즈 하한 (best 기준): 1/4h 1.02 · 1/4d 1.13 · 1/4v 0.75 · val LPIPS 0.00078 (EXP-00-1 vs 00-2)
- 주 지표: 실기 FP32 격자 점수 1/4h · 1/4d (낮을수록 좋음, best.pth로 판정), 제약: val LPIPS, 선명도비. 보고 전용: 1/2h·1/2v·1/2d, x2·x3 격자, epoch_20~100 점수

| 일자 | EXP | HYP | 축 · 변경 (한 줄) | P | BL 기준 | 1/4h · 1/4d | delta | 판정 | 문서 |
|---|---|---|---|---|---|---|---|---|---|
| 2026-10-06 | EXP-14-1 | HYP-14 | 부호 edge 가중치 0.02 → 0.01 (12-1 기준) | P4 | EXP-12-1 | 6.72 · 14.44 (cam92) | +0.6 · −0.7 (vs 12-1) | HYP-14 지지 — P4 전 게이트 통과, 에지 3.73 px로 가장 가파름, 1/4v 9.6 경계 | `20261006-EXP-14-1.md` |
| 2026-10-06 | EXP-13-1 | HYP-13 | SSIM 가중치 0.02 → 0.05 (12-1 기준) | P4 | EXP-12-1 | 5.41 · 14.30 (cam92) | −0.7 · −0.8 (vs 12-1) | **채택** (출시 레시피, BL2) — HYP-13 부분 지지, P4 전 게이트 통과, 에지 단차 1.013·질감 0.843 | `20261006-EXP-13-1.md` |
| 2026-10-05 | EXP-12-2 | HYP-12 (재현) | EXP-12-1을 seed 43으로 (노이즈 하한) | P4 | EXP-12-1 | 5.93 · 11.79 (cam92) | −0.2 · −3.3 (vs 12-1) | 재현 확인 — P4 전 게이트 통과, 사전 한계 안 | `20261005-EXP-12-2.md` |
| 2026-10-04 | EXP-12-1 | HYP-12 | 학습 열 노이즈 σ 1~3 → 0.5~1.5 DN (09-1 기준) | P4 | EXP-09-1 | 6.12 · 15.12 (cam92) | +3.6 · −9.5 (vs 09-1) | HYP-12 지지 — P4 전 게이트 첫 통과, 저대비 질감 0.716 → 0.814, 보드 A/B 후보 | `20261004-EXP-12-1.md` |
| 2026-10-04 | EXP-11-1 | HYP-11 | 부호 edge 가중치 0.02 → 0.01 (09-1 기준) | P4 | EXP-09-1 | 3.63 · 25.10 (cam92) | +1.1 · +0.5 (vs 09-1) | HYP-11 지지 — 사전 GO 통과, P4 G7(0.745)만 미달, 에지 3.74 px로 가장 가파른 후보 | `20261004-EXP-11-1.md` |
| 2026-10-04 | EXP-10-1 | HYP-10 | SSIM 가중치 0.02 → 0.05 (09-1 기준) | P4 | EXP-09-1 | 2.55 · 6.89 (cam92) | +0.05 · −17.7 (vs 09-1) | HYP-10 부분 지지 — G7(저대비 0.730)만 미달, 대각 격자 해소·평탄부 G2 통과, 현재 최선 후보 | `20261004-EXP-10-1.md` |
| 2026-10-03 | EXP-09-1 | HYP-09 | replay + edge 부호 비교 (조합) | P3 | EXP-06-1 | 2.50 · 24.62 (cam92) | 0 · +20.8 (vs 06-1) | HYP-09 부분 지지 — 오버슈트 5.4 %·1/4v 5.4로 개선, 1/4d↑·선명도 1.17 | `20261003-EXP-09-1.md` |
| 2026-10-03 | EXP-08-1 | HYP-08 | IR에 Phase-1 레시피 (bicubic 축소만 + MSE만) | P3 | EXP-05-1 | ep40 46.53 · 11.53 (best=ep1, early stop) | ep40 −2.5 vs 05-1 | HYP-08 기각 — 열화가 아니라 IR 데이터 내용이 원인 | `20261003-EXP-08-1.md` |
| 2026-10-03 | EXP-07-1 | HYP-07 | edge 부호 비교 (`signed: true`) | P3 | BL1 | 22.80 · 7.83 (cam92) | −40.3 · −31.1 | HYP-07 부분 지지 — 격자·노이즈·링잉 감소, 1/4h·선명도 미달 | `20261003-EXP-07-1.md` |
| 2026-10-03 | EXP-06-1 | HYP-06 | 회색 LSDIR 922장 replay + edge 끔 | P3 | EXP-03-1 | 2.50 · 3.78 (cam92) | −29.1 · −9.8 (vs 03-1) | HYP-06 지지 — 격자 해소, 오버슈트 7.4 %(G3 미달) | `20261003-EXP-06-1.md` |
| 2026-09-28 | EXP-05-1 | HYP-05 | 보조 손실 3개 끄기 (MSE만, 폐쇄 검정) | P2 | BL1 | 60.62 · 15.95 | −2.45 · −22.95 | 기각 — 1/4d는 손실 기인(사라짐), 1/4h는 손실 외(남음), 흐려짐 | `20260928-EXP-05-1.md` |
| 2026-09-28 | EXP-04-1 | HYP-04 | 컬럼 노이즈 증강 끄기 | P2 | BL1 | 63.05 · 33.20 | −0.02 · −5.70 | 기각 — 가로 성분 그대로, 대각만 소폭 감소 | `20260928-EXP-04-1.md` |
| 2026-09-28 | EXP-03-1 | HYP-03 | edge 손실 끄기 | P2 | BL1 | 31.58 · 13.53 | −31.49 · −25.36 | 채택 안 함(목표 미달·LPIPS +0.022) — HYP-03 지지: edge가 최대 단일 원인 | `20260928-EXP-03-1.md` |
| 2026-09-28 | EXP-02-1 | HYP-02 | SSIM 손실 끄기 | P2 | BL1 | 538.18 · 59.31 (ep7, early stop ep57) | +475 · +20 | 기각 — 반대: SSIM 끄면 격자 폭증, SSIM이 억제 역할 | `20260928-EXP-02-1.md` |
| 2026-09-28 | EXP-01-1 | HYP-01 | GV 손실 끄기 | P2 | BL1 | 106.25 · 31.67 | +43.18 · −7.23 | 기각 — GV 끄면 격자 증가 (1/4v 3.6 → 78, 모눈 무늬) | `20260928-EXP-01-1.md` |
| 2026-09-28 | EXP-00-2 | – | 시드만 변경 (노이즈 하한) | P2 | BL1 | 64.08 · 37.77 | +1.02 · −1.13 | 노이즈 하한 확정 | `20260928-EXP-00-2.md` |
| 2026-09-28 | EXP-00-1 | – | 베이스라인 재현 (BL1) | P2 | – | 63.07 · 38.90 (출하와 비트 동일) | – | BL1 확정 | `20260928-EXP-00-1.md` |

`문서` 열의 파일은 이 폴더에 있다. 각 실험 문서 §3의 학습·평가 명령은 그때 호스트에서 돌린 그대로의 기록이다. 같은 일을 지금 하려면 2.7절의 `docker/run.sh` 형식을 쓴다. P2는 지금도 채점할 수 있고, P4는 입력이 지워져 다시 채점할 수 없다(2.5절).

### 산출물 정리 기록 (2026-10-06, 소유자 결정, 시간순)

1. 산출물 정리 (2026-10-06, 소유자 결정): 후보에서 제외된 런 01-1~05-1, 07-1, 08-1, 10-1, 11-1의 `checkpoints/experiments/` 폴더를 삭제했다. 남긴 런: 00-1(출하 v3와 비트 동일, 기준), 00-2(시드 하한), 06-1, 09-1, 12-1, 12-2, 13-1, 14-1. 삭제한 런의 수치는 각 문서와 `results/` json에 있고, 학습이 결정적이라 config + seed 42로 다시 만들 수 있다. (이 문장의 '다시 만들 수 있다'는 지금은 성립하지 않는다. 아래 3번 정리로 실험 config도 지워졌고, 2026-10-16에 학습 분포를 바꾼 수정이 들어갔다. [CONFIG_RATIONALE](../CONFIG_RATIONALE.md) 4.4)
2. results 정리 (2026-10-06, 소유자 결정): `results/`에는 `261006_ir_v3fix/`(이 연구의 도구·데이터·배포 산출물을 모은 곳, README에 경로 대응표), `release_v3/`, 영상 폴더 두 개만 남겼다. 09-29~10-04 분석 폴더는 지웠고, 문서에 남은 그 경로들은 기록이다. (이후 대체됨: `261006_ir_v3fix/`는 아래 4번 정리로 지웠다. 2026-10-19 지금 `results/`에는 이 영상 폴더들도 없고, 남은 영상은 데이터 서버 사본이다. 4절)
3. 실험 산출물 최종 정리 (2026-10-06 15:30, 소유자 결정): `configs/experiments/`와 `checkpoints/experiments/`의 남은 런(00-1, 00-2, 06-1, 09-1, 12-1, 12-2, 14-1)을 지웠다. EXP-13-1은 IR x4 출시본으로 `checkpoints/bicubicpp_deploy/finetune_bicubicplusplus_ir_sr_x4_dim32_v3/`로 옮겼다. 레시피는 `configs/finetune/BicubicPlusPlus/bicubicplusplus_{2,3,4}x_ir.yaml`(커밋 대상)에 남는다. 문서의 학습·평가 명령은 기록으로 남기며, 지운 config·체크포인트·검증 세트(`_v3`)를 가리키므로 그대로는 다시 실행되지 않는다.
4. results 최종 정리 (2026-10-06 16:00): `results/261006_ir_v3fix/`를 지웠다. 계속 쓰는 것만 `results/release_v3/`로 옮겼다: 변경 내역 `V3_CHANGES_20261006.md`, 채점 프레임·sha256·P4 채점기·점수 json·export 도구(`eval/`). 문서에 남은 `results/261006_ir_v3fix/...` 경로 중 위 대응이 없는 것(그림, 보드 비교 패키지, 번들)은 기록이다.

2026-10-19 지금 `checkpoints/experiments/`와 `configs/experiments/` 폴더는 없다.

### 프로토콜 변경 이력

| 버전 | 일자 | 이유 | 재평가한 행 | 새 노이즈 하한 |
|---|---|---|---|---|
| P1 | 2026-09-28 | 최초 | – | 측정 전 (P2로 대체) |
| P2 | 2026-09-29 | P1 채점 폴더 `RAW/92/20260922_112645/raw` 소실(소유자가 동영상용 600장으로 재구성) → 새 600장 사본 `[0:600:12]` 50장과 sha256 목록 대조, CPU 채점 고정, epoch_20~100 채점 보고 추가, 50장 미만 거부, best.pth 규칙 명시 | 측정된 EXP 없음. 기준값(출하 epoch_20~100·best, Phase-1 best) 재채점 → RESEARCH §3 | 1/4h 1.02 · 1/4d 1.13 · LPIPS 0.00078 (2026-10-02) |
| P3 | 2026-10-03 | 2단계 합격선: 두 카메라 채점, 격자 제외 선명도·평탄부 노이즈·오버슈트 게이트 추가 (RESEARCH §1) | 1단계 7런 재채점 (`results/261002_exp_results/exps_two_cameras.json`) | 격자 제외 선명도 0.004 · 평탄부 노이즈 0.009 · 오버슈트 0.19 %p · cam98 1/4d 9.8 |
| P4 | 2026-10-04 | 소유자·팀장 기준(에지 대비): G1b 격자 절대 진폭, G3 → over_lr, G5 목표치 상향, G7 저대비 질감(보고), best ≤ ep20이면 epoch_100, 통과 런이 없으면 보드 A/B. 기준값은 06-1·09-1을 본 뒤 정함 (RESEARCH §1) | Phase-1, 00-1, 00-2, 06-1, 09-1 재채점 (`results/release_v3/eval/scores/p4_reference.json`, 기존 표와 소수 셋째 자리까지 일치) | FLIR 저대비 선명도 0.004 · fold 0.004(cam92)/0.001(cam98)/0.005(FLIR even) DN · over_lr 0.10 %p · 에지 폭 0.01 px (00-1 vs 00-2) |
