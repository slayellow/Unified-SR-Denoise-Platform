# src — 코드 지도

이 폴더는 학습, 평가, 양자화 스크립트가 가져다 쓰는 라이브러리 코드입니다. 혼자 실행하는 파일은 없습니다. 진입점은 `tools/`(학습, 평가, 추론, export)와 `examples/`(AIMET, DLC 변환, 채점)입니다. 대부분은 저장소 루트를 `sys.path`에 넣은 뒤 `from src.models import build_model`처럼 가져옵니다. `make_val_camera.py`, `score_eo_chart.py`, `score_grid_artifact.py`는 넣지 않으므로 `docker/run.sh -- env PYTHONPATH=. python3 ...`로 돌립니다(빼면 `ModuleNotFoundError: No module named 'src'`). pytest는 `pytest.ini`의 `pythonpath = .`로 같은 효과를 냅니다.

모든 실행은 Docker 이미지 `usdp` 안에서 `docker/run.sh`로 합니다. 호스트의 `python3`(3.8)에는 다른 프로젝트용 torch 2.4.1만 있고 accelerate, pyiqa, aimet이 없으므로 작업 환경이 아닙니다. `run.sh`의 옵션, 거절 메시지, GPU 고르기는 [docker/README.md](../docker/README.md)에 있습니다.

| 찾는 것 | 문서 |
|---|---|
| 처음부터 보드까지 한 바퀴, 매일 쓰는 명령 | [README.md](../README.md) |
| `train.py`·`evaluate.py`·`inference.py`·`export.py`의 옵션, 출력, resume 규칙 | [tools/README.md](../tools/README.md) |
| config 키 전체, Phase-1/Phase-2 config, 데이터 레시피, AIMET JSON | [configs/README.md](../configs/README.md) |
| AIMET 1~7단계, 평가 스크립트, 검증 세트 만들기 | [examples/README.md](../examples/README.md) |
| DLC 변환, 보드 스크립트, DLC 이름 | [examples/deployment/README.md](../examples/deployment/README.md) |
| 실험과 릴리스 기록 방법 | [docs/experiments/README.md](../docs/experiments/README.md) |

## 1. 폴더 구성

| 파일 | 주요 이름 | 하는 일 |
|---|---|---|
| `models/__init__.py` | `build_model` | `config['model']` → `nn.Module` |
| `models/bicubicplusplus.py` | `BicubicPlusPlus`, `ECBConv` | 출하 모델. SR v3 x2·x3·x4와 BicubicPPDenoise[v1](`scale: 1`) |
| `models/svfocussrnet.py` | `SVFocusSRNet` | 출하하지 않음. 배포 소프트닝용 디노이즈 넷으로 씀(5.4절) |
| `data/datasets.py` | `SRDataset`, `DenoiseDataset`, `PairedDataset`, `apply_configured_degradation` | 학습 쌍 즉석 합성, 미리 만든 검증 쌍 읽기 |
| `data/degradations.py` | `random_mixed_kernels`, `random_add_poisson_noise`, `apply_camera_model` | 블러 커널과 포아송 노이즈(BasicSR에서 가져와 수정), EO 카메라 모델 |
| `engine/trainer.py` | `Trainer`, `Validator` | 학습 루프, 검증, 체크포인트 |
| `engine/checkpoint.py` | `load_model`, `check_model_config`, `normalize_state_dict` | tools 공용 체크포인트 로더 |
| `engine/schedulers.py` | `KneeLRScheduler` | Bicubic++ 공식 구현의 knee 학습률 스케줄 |
| `engine/visualize.py` | `render_results`, `render_train_batch`, `render_dataset_stats`, `render_probe` | run 폴더의 `plots/` 그림 |
| `losses/losses.py` | `UnifiedLoss`와 손실 10종 | config의 `loss:` 블록으로 손실을 조합 |
| `metrics/__init__.py` | `METRIC_SPEC`, `create_psnr`, `create_ssim`, `create_lpips`, `create_niqe` | 저장소 전체가 쓰는 지표 정의 하나 |
| `aimet/quant_sim.py` | `prepare_deploy_model`, `fold_and_equalize`, `build_quantsim`, `DEFAULT_QUANTSIM_CONFIG` | QuantSim을 만드는 유일한 입구 |
| `aimet/utils.py` | `evaluate_model`, `create_sampled_data_loader`, `AutoQuantDatasetWrapper`, `apply_mmp_from_json`, `save_active_results_to_csv` | AIMET 스크립트 공용 도구. import하면 AIMET winnow를 패치함(7.2절) |

## 2. 학습 한 번이 지나가는 길

1. `tools/train.py`가 train config를 읽고, `data_config_path`(또는 `--data_config`)의 데이터 config를 `config['data_config']`에 붙입니다.
2. `task`만 보고 데이터셋을 고릅니다. `sr`이면 `SRDataset`, `denoise`면 `DenoiseDataset`입니다. 검증 세트는 둘 다 `PairedDataset(data.val_hr_root, data.val_lr_root)`입니다.
3. `build_model(config['model'])`로 모델을 만듭니다. `train.pretrained_path`가 있으면 그 체크포인트를 `strict=False`로 읽고 `[Finetune] matched=… missing=… unexpected=…`를 찍습니다.
4. run 폴더(work_dir)를 정하고 config 스냅숏을 씁니다(5.8절).
5. `Trainer.fit`이 돕니다. 파인튜닝이면 먼저 사전학습 가중치를 검증해 기준선(baseline)을 잡고, 그 뒤 epoch마다 학습 → 검증 → CSV → scheduler → 체크포인트 순서로 갑니다(5.1절).
6. 체크포인트는 **학습형**(ECB 가지가 펼쳐진 형태)으로 저장됩니다. 배포형으로 접는 일(`switch_to_deploy`)은 읽는 쪽이 합니다. `load_model`(tools/evaluate.py, inference.py, export.py), `prepare_deploy_model`(AIMET 스크립트), `6_apply_qat.py`, 소프트닝 넷 로더(`Trainer._load_denoise_net`)가 그 자리입니다.
7. `examples/aimet_quantization/6_apply_qat.py`가 `qat_model.onnx`와 `qat_model.encodings`를 쓰고, `examples/deployment/gen_convert_all.py`가 만든 `convert_all.sh`가 DLC로 바꿉니다. DLC의 HTP 캐시는 출력 레이어 `final_act` 기준으로 만들어집니다.

## 3. 모델 (`src/models`)

### 3.1 `build_model`

`build_model(config)`은 `config['name']`을 소문자로 바꿔 if/elif로 고릅니다. 지금 있는 이름은 `svfocussrnet`과 `bicubicplusplus` 둘이고, 그 밖의 이름은 `ValueError: Unknown model name: …`으로 멈춥니다. 두 모델 공통으로 `scale`(기본 2)과 `dim`(생성자의 `n_feats`)을 읽습니다.

두 모델은 모르는 키를 다르게 다룹니다.
- `bicubicplusplus`: `name`, `scale`, `dim`을 뺀 모든 키를 생성자에 그대로 넘깁니다. 키에 오타가 있으면 `TypeError: BicubicPlusPlus.__init__() got an unexpected keyword argument …`로 바로 멈춥니다. 오타 난 키 대신 기본값으로 조용히 학습하는 일을 막으려고 일부러 이렇게 했습니다.
- `svfocussrnet`: `n_resblocks`(기본 8), `dim`(기본 32), `use_advanced_rep`(기본 false)만 읽고 나머지 키는 조용히 버립니다.

### 3.2 `BicubicPlusPlus` 옵션

| config 키 | 기본값 | 뜻 |
|---|---|---|
| `scale` | 2 (`build_model`) | 1 이상의 정수. 1이면 같은 해상도의 디노이저 |
| `dim` | 32 | 특징 채널 수 |
| `in_channels`, `out_channels` | 3 | |
| `bias` | `false` | `use_rep`와 함께 켜면 `ValueError`. 상수항은 0 패딩 경계에서 3x3로 접히지 않습니다 |
| `negative_slope` | 0.1 | LeakyReLU 기울기 |
| `act` | `leaky_relu` | 또는 `relu`. `relu`는 HTP 지연 측정용으로만 있고, 출하 체크포인트는 전부 LeakyReLU(0.1)입니다 |
| `clamp_output` | `true` | 마지막에 `final_act = nn.Hardtanh(0, 1)`을 둡니다. 배포 경로가 이 이름에 의존합니다(3.6절) |
| `upsample_mode` | `legacy` | `legacy`, `auto`, `deconv` (아래 표) |
| `mid_feats` | 16 | `auto`가 확대를 두 단으로 나눌 때(x4) 중간 채널 수 |
| `icnr_init` | `true` | `deconv`의 ConvTranspose를 nearest 확대처럼 시작하도록 초기화 |
| `achromatic_output` | `false` | IR용. forward에서 출력 3채널을 평균합니다. `scale: 1`에서는 거부 |
| `use_rep` | `false` | ECB 재매개변수화(3.3절). legacy SR에서는 거부 |
| `n_body` | 2 | stem 뒤 3x3 conv 개수. SR은 2만, `scale: 1`은 2 이상 |

생성자가 거부하는 조합은 이렇습니다: 정수가 아니거나 1보다 작은 scale, 모르는 `upsample_mode`나 `act`, `scale: 1`과 `deconv`, `scale: 1`과 `achromatic_output`, SR에서 `n_body`가 2가 아님, `use_rep`와 legacy SR, `use_rep`와 `bias`.

| `upsample_mode` | 구조 | 쓰는 이유 |
|---|---|---|
| `legacy` | stride-2 stem, 몸통은 절반 해상도, 마지막에 PixelShuffle(2·scale) | MAC은 적지만 HTP에서 DepthToSpace 비용이 blocksize²에 비례합니다. x4는 PixelShuffle(8) 하나가 17.5M cycles로 네트워크의 98 %였습니다 |
| `auto` | stem stride 1, PixelShuffle(scale). scale이 합성수면 두 단(x4 = PixelShuffle(2) 두 번) | shuffle 비용이 4~5.5배 줄어듭니다 |
| `deconv` | stem stride 1, ConvTranspose2d(kernel 2f, stride f) 하나 | DepthToSpace를 아예 피합니다. x3에서 PixelShuffle(3) 9.18 ms → 5.24 ms |

`scale: 1`일 때 출력은 `clamp(x + residual)`이고, residual 마지막 층은 0.01배로 시작해 처음에는 항등 함수에 가깝습니다. `legacy`는 절반 해상도 몸통과 PixelShuffle(2), `auto`는 전체 해상도 몸통입니다. 절반 해상도 형태는 입력 높이와 너비가 짝수여야 합니다(홀수면 입력을 더하는 곳에서 실패). forward에 크기 검사를 넣지 않은 것은 torch.fx/AIMET 추적에 데이터 의존 분기를 만들지 않기 위해서입니다.

출하 config의 모델 블록(전부 `dim: 32`, `use_rep: true`, `clamp_output: true`):

| 모델 | `upsample_mode` | 그 밖 |
|---|---|---|
| SR x2, x4 (EO, IR) | `auto` | IR Phase-2 config에만 `achromatic_output: true` |
| SR x3 (EO, IR) | `deconv` | 위와 같음 |
| BicubicPPDenoise[v1] | `legacy` | `scale: 1`, `n_body: 4` |

출하 EO x2(`results/release_v3/checkpoints/EO_x2.pth`)는 Phase-1 x2 `best.pth`와 md5가 같습니다(`tests/test_release_golden.py`). 그래서 그 모델 블록은 `configs/train/BicubicPlusPlus/bicubicplusplus_2x.yaml`의 것입니다(5.6절의 기준선 관문).

### 3.3 재매개변수화 (`ECBConv`)

`use_rep: true`이면 stride 1인 3x3 conv가 학습 중에는 `ECBConv`입니다. 가지는 conv3x3, conv1x1, 1x1 뒤에 고정된 Sobel-x·Sobel-y·Laplacian depthwise 커널을 붙인 세 가지, 그리고 입출력 채널이 같으면 identity입니다. 모든 가지가 선형이고 bias가 없으므로 그 합은 정확히 3x3 conv 하나입니다. 그래서 배포 그래프, 지연, DLC는 `use_rep: false`와 똑같고, 학습 때만 폭이 넓어집니다. legacy의 stride-2 stem은 고정 커널을 stride로 접을 수 없어서 평범한 conv로 남습니다.

### 3.4 `switch_to_deploy`와 체크포인트의 두 형태

`BicubicPlusPlus.switch_to_deploy()`는 여러 번 불러도 결과가 같습니다(idempotent). 하는 일은 순서대로 둘입니다.
1. 모든 `ECBConv`를 접고, 그 자리에 안쪽의 `nn.Conv2d`를 다시 넣습니다. 접은 뒤 state_dict 키는 `conv0.conv3x3.weight`가 아니라 `conv0.weight`가 되어, `use_rep: false` 모델과 같고 출하 `.encodings`의 이름과도 맞습니다.
2. `achromatic_output`이면 `_fold_achromatic`이 마지막 층의 세 출력 필터 묶음을 평균해 가중치에 접고 플래그를 끕니다. 평균을 forward에 남기면 ONNX에 ReduceMean·Expand 등 6개 노드가 붙고 HTP에서 float로 떨어져 x3가 2.746 ms에서 10.401 ms가 됩니다. `6_apply_qat._check_achromatic_fold`는 접은 뒤 회색 입력에 남은 색이 1e-3 DN을 넘으면 export를 멈춥니다.

이 함수는 AIMET 사슬에서 **가장 먼저** 돌아야 합니다. QuantSim이 만들어지는 순간의 그래프가 양자화기 집합과 `.encodings`의 모든 이름을 정하기 때문입니다. `prepare_deploy_model`이 이 순서를 지킵니다(7.1절).

Trainer는 체크포인트를 항상 학습형으로 저장합니다. 학습형과 배포형 state_dict는 서로 바꿔 넣을 수 없습니다. strict 로드(`load_model`, `6_apply_qat.py`)는 키가 맞지 않으면 멈추고, `tools/train.py`의 `pretrained_path` 로드는 `strict=False`라서 하나도 맞지 않을 때만 멈추고, 일부만 맞으면 `[Finetune]` 줄의 숫자로만 드러납니다(5.6절).

IR 무채색 출력은 Phase-2 학습 때부터 `achromatic_output: true`로 켜야 합니다. forward의 평균이 세 채널을 묶어 공유 필터 하나를 학습시키기 때문에 채널이 정확히 같아집니다. 이 플래그 없이 학습한 모델을 나중에 접거나, 접은 모델을 더 학습하면 채널이 다시 갈라집니다(접은 뒤 100 스텝 만에 0.0029 DN).

### 3.5 `SVFocusSRNet`

출하하지 않지만 지우지 않습니다. RepBlock(3x3+1x1+identity) 또는 `use_advanced_rep: true`의 RepBlock_Advanced(3x3+1x1+1x3+3x1+AvgPool+identity)를 쌓고, 활성화는 `Hardtanh(0, 1)`입니다. SR은 고정된 nearest 확대(`AnchorOp`)에 PixelShuffle 잔차를 더하고, `scale: 1`은 입력 skip 디노이저입니다. 마지막 모듈 이름도 `final_act`입니다. 지금 쓰임은 Phase-1과 EO Phase-2의 배포 소프트닝 넷(`checkpoints/csuav_deploy/train_svfocusdenoise_adv_x1_dim32_block3_..._260309/best.pth`) 하나입니다.

### 3.6 새 모델이 지켜야 할 계약

| 계약 | 이유 (어기면 보이는 증상) |
|---|---|
| `forward(x)`는 `(N, 3, H, W)`, RGB, [0, 1] float를 받아 텐서 **하나** `(N, 3, H·s, W·s)`를 돌려준다 | `Trainer.train_epoch`과 `Validator.run`은 tuple을 풀지 않으므로 tuple이면 손실 계산에서 멈춥니다. 크기가 틀리면 Validator가 한 번 경고하고 공통 크기로 잘라서 점수를 내므로, 크기 오류가 그럴듯한 숫자 뒤에 숨습니다 |
| 마지막 모듈 속성 이름은 `final_act` (`nn.Hardtanh(0, 1)`) | `gen_convert_all.py`가 `--set_output_layers final_act`(기본값)로 HTP 캐시를 만들고, 보드 gst 스크립트의 `BPP_LAYER` 기본값도 `final_act`입니다. 캐시의 출력층과 gst `layers=`가 다르면 보드가 조용히 온라인 prepare로 돌아가 CMA 절감이 사라집니다. 이름을 바꾸려면 `gen_convert_all.py --output_layer`와 보드 스크립트 변수를 함께 바꿉니다 |
| `switch_to_deploy()`가 학습 전용 가지를 모두 없애고, 여러 번 불러도 같다 | AIMET 스크립트와 tools가 `hasattr`로 찾아 부릅니다. 없거나 덜 접으면 학습형 그래프가 그대로 양자화되고 export됩니다 |
| forward에 데이터에 따라 갈리는 분기가 없다 | AIMET `prepare_model`이 torch.fx로 추적합니다 |
| 접기 전후 출력이 같다 | `tests/test_reparam_equivalence.py`가 무작위 가중치로 fp32와 fp64 모두 확인합니다. 갓 초기화된 가중치는 틀린 접기도 맞아 보이게 할 수 있습니다 |
| LeakyReLU를 쓰면 `DEFAULT_QUANTSIM_CONFIG`를 그대로 쓴다 | 내장 `htp_v73` 설정은 Conv+LeakyRelu를 묶어 Conv 출력 인코딩을 빼고, 변환기가 그 op들을 조용히 float로 떨어뜨립니다(7.1절) |

## 4. 데이터 (`src/data`)

### 4.1 데이터셋

`task`만 데이터셋을 정합니다(`tools/train.py`의 `resolve_dataset_kind`). `sr`, `denoise` 외의 값은 멈추고, `task: denoise`는 `model.scale: 1`이어야 합니다. 데이터 config의 `dataset_type`은 출력만 되고, train config의 `data.type`(eo/ir)과 데이터 config의 `crop_size`는 어떤 코드도 읽지 않습니다. 크롭 크기는 `train.patch_size`입니다.

세 클래스 모두 `{'lr', 'hr', 'path'}`를 돌려주고, 이미지는 RGB, `(3, H, W)`, [0, 1] float입니다. 학습 루트(`data.train_root`)는 문자열 또는 목록이고, 그 아래를 재귀적으로 훑어 `.png/.jpg/.jpeg`(대소문자 무시)를 모읍니다.

| 클래스 | 크롭 | 크롭보다 작은 이미지 | 쌍 만들기 | 증강 |
|---|---|---|---|---|
| `SRDataset` | `patch_size`는 **LR** 패치. HR 크롭은 `patch_size × scale` (출하 SR config의 288/192/144는 모두 HR 576 px) | 건너뛰고 무작위로 다른 이미지를 뽑음. 확대하지 않음 | `clean_prob` 확률로 열화 없는 쌍(축소 보간은 `stage2.target_resize.interpolations`에서 무작위), 그 밖은 열화 파이프라인 | `use_flip`, `use_rot` (데이터 config 최상위, 기본 off) |
| `DenoiseDataset` | `patch_size` 정사각, 같은 해상도 | bicubic으로 키움 | `camera_model.enabled`면 `camera_model.prob` 확률로 카메라 모델 쌍(그 안에서도 `clean_prob`이면 LR을 정답으로 바꿈). 그 밖은 `clean_prob`이면 LR=HR, 아니면 열화 파이프라인(scale 1) | 없음 |
| `PairedDataset` | 없음, 전체 이미지 | 해당 없음 | `hr_dir`, `lr_dir`(문자열 또는 목록)의 **최상위만** 훑어 파일 stem으로 짝지음 | 없음 |

`PairedDataset`은 stem이 겹치거나, HR과 LR의 stem 목록이 하나라도 다르거나, 쌍이 없으면 멈춥니다. 예전에는 정렬 순서로 짝지어서, 파일 하나가 빠지면 뒤의 모든 쌍이 어긋난 채로 숫자가 나왔습니다. 검증 LR 폴더는 미리 만들어 둬야 합니다(`examples/dataset_preparation/make_val_lr.py`, Denoise는 `make_val_camera.py`; [examples/README.md](../examples/README.md)).

`SRDataset`이 작은 이미지를 키우지 않는 이유는, 키우면 정답 자체가 bicubic 확대가 되어 모델이 없애야 할 흐림을 배우기 때문입니다. `SRDataset`은 `camera_model.enabled`를 거부하고, 두 학습 데이터셋 모두 빈 데이터 config를 거부합니다. 빈 config는 "열화 없음"이 아니라 "모든 op가 기본값으로 켜짐"이기 때문입니다.

### 4.2 열화 파이프라인

`apply_configured_degradation(img_hr, cfg, scale_factor)`가 uint8 BGR HR 패치를 받아 LR을 돌려줍니다. op 순서와, config에 키가 없을 때의 기본값은 다음과 같습니다(`_DEGRADATION_PIPELINE_DEFAULTS`와 같은 값).

| 단계 | op | 기본 enabled | 기본 prob | 주요 기본값 |
|---|---|---|---|---|
| stage1 | `blur` | on | 0.5 | 6종 커널 혼합, `sigma` [0.2, 3.0] |
| stage1 | `resize` | on | 0.7 | `scale_min` 0.5, `scale_max` 1.0 |
| stage1 | `gaussian_noise` | on | 0.5 | sigma 1~5, `gray_prob` 0.5 |
| stage1 | `poisson_noise` | on | 0.5 | scale 0.05~3.0 |
| stage1 | `jpeg` | on | 0.5 | quality 85~95 |
| stage1 | `sinc` | on | 0.1 | |
| stage2 | `blur` | on | 0.5 | `sigma` [0.2, 1.5] |
| stage2 | `target_resize` | on | 항상 | HR // scale로 축소, 보간은 CUBIC/LINEAR/LANCZOS4 중 무작위 |
| stage2 | `ir_noise` | on | 하위 op별 | `non_uniformity` p 0.5, `vertical_line` p 0.5 (`sigma_dn` 0.75, `strong_frac` 0.03, `strong_dn` 6.0), `thermal_noise` p 1.0, sigma 5~20 |
| stage2 | `common_noise` | on | 0.0 | sigma 1~10 |
| stage2 | `hot_pixels` | **off** | 0.5 | density 0.001~0.01 |
| stage2 | `final_jpeg` | on | 0.8 | quality 50~90 |

각 op는 `enabled`가 거짓이면 난수를 뽑지 않고 넘어갑니다(`cfg.get('enabled', 기본) and random.random() < prob`). float 연산 뒤 uint8로 돌아올 때는 버림이 아니라 반올림합니다. 버림은 합성 픽셀마다 평균 −0.5 DN의 치우침을 만들었습니다.

데이터셋을 만들 때 `log_active_degradations`가 실제로 적용될 값을 찍습니다. 아래는 2026-10-19 EO x2 Phase-2 smoke run의 로그입니다.

```text
[Degradation] SRDataset(train) stage1 active: blur(p=0.08), gaussian_noise(p=0.1), poisson_noise(p=0.1), jpeg(p=0.12)
[Degradation] SRDataset(train) stage1 off: resize, sinc
[Degradation] SRDataset(train) stage2 active: blur(p=0.05), target_resize(always), common_noise(p=0.05), hot_pixels(p=0.2), final_jpeg(p=0.08)
[Degradation] SRDataset(train) stage2 off: ir_noise
[Degradation] SRDataset(train) clean_prob=0.3
[Degradation] SRDataset(train) camera_model off
```

모든 run을 시작할 때 이 줄들과 `plots/train_batch.png`를 봅니다. 함정이 셋 있습니다.
- config에서 빠진 op는 기본값으로 **돕니다**. 특히 `stage2.ir_noise`는 기본이 켜짐이고 그 안의 `thermal_noise`는 확률 1.0이라, EO 데이터 config가 이것을 끄지 않으면 열화 파이프라인을 거치는 모든 EO 샘플에 열 잡음이 들어갑니다.
- `WARNING: … keys not read by the pipeline`은 stage 바로 아래의 모르는 키만 알려 줍니다. 그보다 깊은 단계의 오타나 옛 키는 조용히 무시됩니다. 예를 들어 `vertical_line`의 `intensity_min`/`intensity_max`(그리고 함수 인자 `intensity_range`, `alpha`)는 예전 세로줄 모양의 키이고, 지금 코드는 `enabled`, `prob`, `sigma_dn`, `strong_frac`, `strong_dn`만 읽습니다.
- 모르는 보간 이름은 조용히 `cv2.INTER_CUBIC`이 됩니다.

### 4.3 카메라 모델 (EO Denoise)

`apply_camera_model(img_bgr, cfg, rng=None)`은 uint8 BGR 패치를 받아 `(lr, target)`을 돌려줍니다. 광학 가우시안 블러 → BT.601 YCbCr → `noise_modes`의 스펙트럼 모양 luma·chroma 잡음 → (선택) hot pixel → 밝기에 따른 샤프닝 → NV12 8비트 순서입니다. `target`은 `camera`(기본, 카메라의 잡음 없는 출력), `optics`, `source` 중 하나입니다. `validate_camera_model_cfg`가 `DenoiseDataset`을 만들 때 블록을 검사하므로, 잘못된 값은 DataLoader 워커 안이 아니라 메인 프로세스에서 멈춥니다. 값의 근거와 시험은 [docs/modules/20261008-MOD-CameraModel.md](../docs/modules/20261008-MOD-CameraModel.md)에 있습니다.

## 5. 학습 엔진 (`src/engine`)

### 5.1 `Trainer.fit`의 순서

1. (파인튜닝일 때만) 기준선 검증. 사전학습 가중치로 검증 세트를 돌려 `best_metric_val`을 채웁니다. 콘솔에 `[Validation][Epoch 0] …`과 `[Trainer] Baseline from pretrained weights: lpips=… best.pth is only written if this is beaten.`이 찍히고, `training_log.csv`에는 들어가지 않습니다. resume에서는 다시 하지 않고 체크포인트의 값을 이어받습니다.
2. epoch마다:
   1. `criterion.set_epoch(epoch)`로 warm-up 가중치를 정하고, 아직 오르는 중이면 `[Trainer] Epoch N loss warm-up: …`을 찍습니다.
   2. 학습. 배치마다 배포 소프트닝(5.4절) → `model(lr)` → `criterion(sr, hr, lr)` → backward → `train.grad_clip`이 0보다 크면 gradient clipping → optimizer step → gradient가 동기화된 스텝에서만 EMA 갱신. 10 스텝마다 TensorBoard에 씁니다.
   3. 검증. `ema.start_epoch`부터는 EMA 가중치로 검증합니다.
   4. CSV 한 줄을 쓰고 `plots/results.png`를 다시 그립니다.
   5. `scheduler.step()` (epoch당 한 번).
   6. `best_attempt.pth`, `best.pth`, 조기 종료 판정(5.6절). 조기 종료가 걸리면 여기서 루프를 나갑니다.
   7. `epoch % train.save_interval == 0`이면 `plots/probe_ep<N>.png`와 `epoch_<N>.pth`.
   8. `last.pth`.

장치는 Accelerate가 정합니다(`self.device = accelerator.device`). 그래서 `tools/train.py --device`는 학습 장치를 바꾸지 않습니다. `docker/run.sh --gpu N`을 주면 그 GPU가 컨테이너 안에서 `cuda:0`으로 보이고, `--gpu`를 빼면 GPU가 없어 CPU로 돕니다. Accelerator에 mixed precision 인자를 주지 않으므로 fp32로 학습합니다. scheduler는 `accelerator.prepare`에 넣지 않습니다. 넣으면 다중 프로세스에서 프로세스 수만큼 스텝되고, `total_steps` 속성을 가진 `KneeLRScheduler`가 accelerate의 OneCycle 보호 장치에 걸려 멈춥니다(`tests/test_scheduler_prepare.py`).

### 5.2 optimizer와 scheduler

| config | 값 | 읽는 키 (기본값) |
|---|---|---|
| `train.optimizer.type` | `Adam` (기본) | `betas` (0.9, 0.999), `train.weight_decay` (0) |
| | `AdamW` | 위와 같음 |
| | `SGD` | `momentum` (0.9), `train.weight_decay` (0) |
| `train.scheduler.type` | `KneeLR` | `peak_lr` (`train.lr`), `warmup_steps` (0), `explore_steps` (0), `total_steps` (`train.epochs`), `min_lr` (0) |
| | `CosineAnnealingLR` | `T_max` (`train.epochs`), `eta_min` (0) |
| | `StepLR` (기본) | `step_size`, `gamma`. scheduler 블록에 없으면 `train` 블록, 거기도 없으면 50 / 0.5 |

모르는 type은 둘 다 `ValueError`입니다. scheduler는 epoch 단위로 스텝하므로 `total_steps`와 `T_max`의 단위는 epoch입니다. 출하 config는 이 값을 모두 적어 두었으므로 `--epochs`를 바꿔도 학습률 곡선은 줄어들지 않습니다.

### 5.3 EMA

| 키 | 기본값 | 뜻 |
|---|---|---|
| `train.ema.enabled` | `false` | 출하 config는 모두 켬 |
| `train.ema.decay` | 0.999 | |
| `train.ema.start_epoch` | 2 | 이 epoch부터 검증, 그림, 체크포인트가 EMA 가중치를 씀 |

EMA를 켜면 평균 가중치를 판정하고 저장하며, optimizer는 계속 실제(live) 가중치를 움직입니다. 저장되는 `model_state_dict`는 EMA 가중치이고, live 가중치는 `live_state_dict`에 따로 들어가 resume이 궤적을 이어 갑니다. 그래서 `ema.start_epoch` 이후에 저장된 체크포인트를 평가하거나 export하면 EMA 모델을 쓰게 되고, 이것이 `training_log.csv`의 숫자와 같은 모델입니다.

### 5.4 배포 소프트닝 (`train.deploy_softening`)

| 키 | 기본값 | 뜻 |
|---|---|---|
| `denoise_prob` | 0.0 (끔) | 배치 안에서 샘플별로 이 비율을 디노이즈 넷에 통과시킴 |
| `noise_sigma` | [2, 15] | 넷에 넣기 전에 더하는 가우시안 sigma(DN). `null`이면 더하지 않음 |
| `denoise_checkpoint` | 없음 | `denoise_prob`이 0보다 크면 필수. 없거나 경로가 틀리면 `FileNotFoundError` |
| `denoise_model` | 체크포인트의 `config.model` | 체크포인트에 config가 없을 때만 필요 |

보드에서 SR 모델이 받는 입력은 디노이즈를 거쳐 학습 LR보다 훨씬 부드럽습니다(2026-09-08 실측, 절반 나이퀴스트 위 에너지 0.0065 대 0.0227). 그래서 학습 배치의 일부를 실제 배포 디노이즈 넷(접은 뒤 고정)에 통과시켜, 디노이즈를 거친 입력과 거치지 않은 입력을 한 모델이 함께 배우게 합니다. 학습에만 쓰고 검증에는 쓰지 않습니다. 출하 config에서는 Phase-1(모든 scale)과 EO Phase-2가 0.5로 켜고, IR Phase-2와 Denoise는 끕니다. 켜지면 `[Trainer] Deployment softening: 50% of each batch through best.pth`가 찍힙니다.

### 5.5 검증과 지표

`Validator.run`은 `model.eval()` 상태로 검증 세트를 돕니다. 손실은 학습과 같은 `criterion`(그 epoch의 warm-up 가중치)으로, 지표는 [0, 1]로 자른 텐서로 계산합니다. `{loss, psnr, ssim, lpips, niqe}`를 돌려주고 `[Validation][Epoch N] Loss: … | PSNR: … | SSIM: … | LPIPS: … | NIQE: …`를 찍습니다. accelerator를 쓰므로 `Trainer` 밖에서 단독으로는 돌지 않습니다.

지표 정의는 `src/metrics` 하나뿐이고, 체크포인트에 기록되는 이름은 `METRIC_SPEC = "y-ycbcr/crop=scale"`입니다.

| 함수 | 정의 |
|---|---|
| `create_psnr(scale, device)` | pyiqa PSNR, 휘도(BT.601 limited range), 테두리 `scale` 픽셀 제외 |
| `create_ssim(scale, device)` | pyiqa SSIM, 같은 조건 |
| `create_lpips(device)` | pyiqa LPIPS, RGB, 테두리 제외 없음 |
| `create_niqe(scale, device)` | 출력만 보는 무참조 NIQE, 테두리 `scale` 제외 |

같은 정의를 Validator, `tools/evaluate.py`, `src/aimet/utils.evaluate_model`(PSNR만), `visualize.render_probe`가 씁니다. 2026-10-19 smoke run에서 Validator의 epoch 1 값(PSNR 31.0827, SSIM 0.8686, LPIPS 0.1772, NIQE 4.5472)과 `tools/evaluate.py`가 같은 `last.pth`로 낸 값이 같았습니다. 이 정의로 바꾼 뒤의 숫자는 예전 로그보다 약 2 dB 높습니다. 모델이 아니라 자가 바뀐 것입니다. pyiqa 가중치는 `~/.cache/torch/hub/pyiqa`에 있고 `run.sh`가 이 폴더를 붙이므로 `--offline`에서도 됩니다.

검증 LR 세트는 config마다 다릅니다(Phase-1 x2는 `val/LRx2_v3`, EO Phase-2 x2는 `val/LRx2_weaknoise_eo_v3` 등). 서로 다른 config의 best 값을 바로 비교하지 말고, 같은 `--hr_dir`/`--lr_dir`로 `tools/evaluate.py`를 돌려 비교합니다([tools/README.md](../tools/README.md)).

### 5.6 best, best_attempt, last, epoch_N과 조기 종료

| 파일 | 언제 쓰나 |
|---|---|
| `best.pth` | 감시 지표가 `best_metric_val`보다 `min_delta` 넘게 좋아졌을 때. 처음부터 학습하면 기준은 ±inf, 파인튜닝이면 기준선(사전학습 가중치의 검증값) |
| `best_attempt.pth` | 이 run 안에서 감시 지표가 가장 좋았던 epoch. 기준선과 `min_delta`를 보지 않음 |
| `epoch_<N>.pth` | `N % train.save_interval == 0`일 때 |
| `last.pth` | 매 epoch 끝. 조기 종료가 걸린 epoch는 저장 전에 루프를 나가므로, 그때 `last.pth`는 CSV 마지막 줄보다 한 epoch 앞 |

검증 세트가 없으면(`train.allow_no_validation: true`) `best.pth`, `best_attempt.pth`, 조기 종료가 모두 없습니다.

| 키 (`train.early_stopping`) | 기본값 | 뜻 |
|---|---|---|
| `monitor` | `loss` | `loss`, `psnr`, `ssim`, `lpips`, `niqe` 중 하나. 그 밖은 `ValueError` |
| `mode` | `min` | `min` 또는 `max` |
| `patience` | 10 | best가 아닌 epoch마다 카운터 +1, best면 0으로. 이 값에 닿으면 멈춤 |
| `min_delta` | 0.0 | `best.pth` 판정에만 씀 |

출하값은 Phase-1이 `psnr`/`max`, Phase-2와 Denoise가 `lpips`/`min`에 `min_delta` 1e-4입니다. 함정이 둘 있습니다.
- `mode`가 지표 방향과 맞는지 검사하지 않습니다. `psnr`에 `min`을 주면 가장 나쁜 epoch가 best로 남습니다.
- `monitor: loss`와 어떤 손실의 `warmup_epochs`를 함께 쓰면 Trainer를 만들 때 `ValueError`입니다. warm-up 중에는 손실 가중치가 epoch마다 달라서 검증 손실끼리 비교할 수 없기 때문입니다. 같은 이유로, 기준선 검증 손실은 warm-up 전의 **전체** 가중치로 계산되어 epoch 1의 손실과 비교할 수 없습니다(smoke run: 기준선 0.0339, epoch 1 0.0242인데 PSNR은 31.3186에서 31.0827로 떨어짐).

**Phase-2 기준선 관문.** 파인튜닝 run은 사전학습 가중치를 이겨야만 `best.pth`를 씁니다. 2026-10-19 EO x2 smoke run(1 epoch)은 기준선 `lpips=0.167224`, epoch 1 `0.1772`로 이기지 못해 `best.pth` 없이 `best_attempt.pth`와 `last.pth`만 남았습니다. 고장이 아니라 설계대로입니다. 기준선이 없던 때에는 사전학습보다 나쁜 epoch 1이 best로 저장될 수 있었습니다. 출하 EO x2가 Phase-1 체크포인트(ep113) 그대로인 것도 v3 Phase-2가 기준선 LPIPS 0.1672를 넘지 못했기 때문입니다([results/release_v3/MANIFEST.md](../results/release_v3/MANIFEST.md)).

**사전학습 로드 확인.** `tools/train.py`는 `train.pretrained_path`를 `strict=False`로 읽고, 하나도 맞지 않을 때만 멈춥니다. smoke run의 EO x2는 `[Finetune] matched=32 missing=0 unexpected=0`이었습니다. `missing=0 unexpected=0`이 아니면 일부만 읽힌 것입니다. 흔한 원인은 배포형(접힌) 체크포인트를 `use_rep: true` 모델에 넣는 경우입니다. 이때 x2·x4(`auto`)는 모든 층이 ECB라 하나도 맞지 않아 `RuntimeError: No weights matched from …`로 멈추고, x3(`deconv`)와 Denoise는 ECB가 아닌 층 하나만 맞아 `matched=1`로 학습이 시작됩니다. 경로가 없으면 `FileNotFoundError`로 멈춥니다. Phase-2 config는 `checkpoints/bicubicpp_deploy/` 아래의 출하 Phase-1 가중치에서 출발하므로, 새 Phase-1 run을 이어 붙이려면 이 경로를 바꿉니다.

### 5.7 체크포인트 내용

| 키 | 내용 |
|---|---|
| `epoch` | 저장한 epoch |
| `model_state_dict` | 판정하고 출하하는 가중치(학습형). `ema.start_epoch`부터는 EMA |
| `ema_used` | 위가 EMA인지 |
| `live_state_dict` | EMA일 때 optimizer가 실제로 있던 가중치, 아니면 `None` |
| `optimizer_state_dict`, `scheduler_state_dict` | resume용 |
| `best_val_loss`, `best_metric_val` | `best.pth` 판정 기준 |
| `best_attempt`, `early_stop_counter` | resume이 이어받는 상태(2026-10-16 이후 체크포인트) |
| `metric_spec` | `y-ycbcr/crop=scale` |
| `git_sha` | 짧은 sha. 추적 중인 파일이 HEAD와 다르면 `-dirty`. 추적하지 않는 새 파일은 `-dirty`를 만들지 않으므로, 남길 만한 실험을 돌리기 전에 새 모듈을 커밋합니다 |
| `config` | 실제로 쓴 전체 config(오버라이드 반영, `data_config` 포함) |

한 체크포인트의 장부는 이렇게 봅니다.

```bash
docker/run.sh -- python3 -c "import torch; c = torch.load('checkpoints/<run>/best.pth', map_location='cpu', weights_only=False); print({k: c.get(k) for k in ('epoch', 'ema_used', 'best_metric_val', 'best_attempt', 'early_stop_counter', 'metric_spec', 'git_sha')}); print(c['config']['model'])"
```

`load_model`(tools)과 `6_apply_qat.py`는 `check_model_config`로 `--config`의 `model` 블록이 체크포인트에 저장된 `model` 블록과 **정확히** 같은지 봅니다. 기본값과 같은 키를 하나 더 적어도 다르다고 보고 `ValueError: … was trained with another model config; differing keys: …`로 멈춥니다. 이 검사가 있는 이유는 `achromatic_output`에 가중치가 없어서, IR 체크포인트를 EO config로 열어도 strict 로드는 통과하고 회색 입력에 색을 칠하기 때문입니다. 그래서 평가와 export에는 그 run의 `train_config.yaml`을 씁니다. config가 없는 옛 체크포인트는 검사하지 않습니다. 키 앞의 `module.`(DDP)과 `_orig_mod.`(torch.compile)는 `normalize_state_dict`가 떼어 냅니다.

### 5.8 run 폴더 (work_dir)

work_dir은 `--resume`이면 그 체크포인트의 폴더, 아니면 `--work_dir`, 아니면 `checkpoints/<train.save_name>`, 그것도 없으면 `checkpoints/<name>_<task>_x<scale>_<YYYYmmdd_HHMMSS>`입니다. 상대 경로는 저장소 루트 기준입니다(`run.sh`가 컨테이너의 작업 폴더를 저장소로 둡니다). 2026-10-19 smoke run의 실제 폴더는 이렇습니다.

```text
results/smoke_20261019/x2_eo_p2/
├── train_config.orig.yaml   # --config verbatim copy (comments kept)
├── data_config.orig.yaml    # data config verbatim copy
├── train_config.yaml        # effective config (overrides applied); resume reads this
├── data_config.yaml         # effective data config; resume reads this
├── logs/                    # TensorBoard events
├── training_log.csv
├── plots/                   # train_batch.png, dataset_stats.png, results.png
├── best_attempt.pth
└── last.pth                 # no best.pth: epoch 1 did not beat the baseline
```

`save_interval` 이상 돈 run에는 `epoch_<N>.pth`와 `plots/probe_ep<N>.png`가, resume한 run에는 `train_config.resume_ep<N>_<stamp>.yaml`이 더 생깁니다. `6_apply_qat.py`가 QAT(`--qat_epochs`가 0보다 큼)를 돌리면 Trainer는 `<output_dir>/qat_checkpoints/`에 씁니다.

같은 config를 `--work_dir` 없이 다시 돌리면 `checkpoints/<save_name>`을 다시 씁니다. `training_log.csv`는 헤더 없이 이어 붙고, `best.pth`와 `last.pth`는 덮어써지며, 처음부터 학습하는 run은 기준이 inf에서 다시 시작하므로 epoch 1이 `best.pth`가 됩니다. 실험에는 항상 `--work_dir`을 줍니다. resume 규칙(거부되는 옵션 등)은 [tools/README.md](../tools/README.md)에 있습니다.

### 5.9 로그와 그림

| 출력 | 내용 |
|---|---|
| `training_log.csv` | `epoch, train_loss, val_loss, psnr, ssim, lpips, niqe, lr`. 이어 쓰기(append). 기준선 검증과 손실 항목별 값은 없음 |
| `logs/` (TensorBoard) | 10 스텝마다 `Loss/Total`, `Train/LR`, `Loss/<손실 키>`(가중치를 곱하기 전 값, warm-up으로 가중치가 0인 epoch에는 0). epoch마다 `Loss/Val`, `Metrics/PSNR`, `Metrics/SSIM`, `Metrics/LPIPS`, `Metrics/NIQE`. 기준선(epoch 0)도 여기 있음 |
| `plots/train_batch.png` | 학습 시작 때 열화된 LR｜HR 8쌍. 데이터 config가 틀리면 여기서 바로 보임 |
| `plots/dataset_stats.png` | 이미지 수, 해상도 분포, 밝기 |
| `plots/results.png` | 매 epoch 다시 그리는 손실, 지표, 학습률 곡선 |
| `plots/probe_ep<N>.png` | `save_interval`마다 검증 이미지 처음·가운데·끝의 bicubic｜SR｜GT와 PSNR (판정한 가중치로 그림) |

`train.plots: false`로 그림을 끕니다. 그림 그리기가 실패해도 `[Plots] … skipped`만 찍고 학습은 계속합니다. 시작 때의 그림은 난수 상태를 보존한 채 그리므로, 같은 seed의 run은 그림을 켜든 끄든 결과가 같습니다. `src/engine/visualize.py` 머리말은 `checkpoints/<save_name>/plots/`라고 적혀 있지만 실제 위치는 `<work_dir>/plots/`입니다. 손실 항목별 값은 `losses.py` 주석과 달리 CSV에 들어가지 않고 TensorBoard에만 있습니다.

`docker/run.sh`는 포트를 열지 않으므로 컨테이너 안에서 띄운 tensorboard는 브라우저에서 열 수 없습니다. 이벤트 파일은 저장소 bind mount를 통해 호스트에 있으므로, 호스트의 `~/.local/bin/tensorboard --logdir <work_dir>/logs`로 읽습니다. `--name` 작업의 콘솔 출력은 저장소 루트의 `logs/<NAME>.log`로, run 폴더 안의 `logs/`(TensorBoard)와 다른 폴더입니다.

## 6. 손실 (`src/losses`)

`UnifiedLoss(config['loss'])`가 켜진 항목만 모읍니다. 각 항목은 `{enabled, weight, …}`이고 모두 기본으로 꺼져 있습니다. 하나도 켜지지 않으면 첫 forward에서 `ValueError`입니다. 모르는 키는 조용히 무시되므로 `gradient_varience` 같은 오타는 그 항 없이 학습합니다. `UnifiedLoss`의 docstring 예시는 옛 목록이니 아래 표를 봅니다.

| 키 | 클래스 | 기본 weight | 옵션 (기본값) |
|---|---|---|---|
| `mse` | `nn.MSELoss` | 1.0 | |
| `l1` | `nn.L1Loss` | 1.0 | |
| `charbonnier` | `CharbonnierLoss` | 1.0 | `eps` (1e-3) |
| `edge` | `EdgeLoss` (Sobel) | 0.05 | `mode` `gray`/`mean`/`rgb` (`gray`), `threshold` (0.2), `signed` (false) |
| `ssim` | `SSIMLoss` (1 − pyiqa SSIM, RGB) | 0.1 | |
| `perceptual` | `PerceptualLoss` (VGG19) | 0.01 | `layers` {VGG19 features 인덱스: 가중치} (`{34: 1.0}`), `gray` (false) |
| `color_consistency` | `ColorConsistencyLoss` | 0.05 | `loss_type` `l1`/`charbonnier` (`l1`) |
| `hot_pixel_masked` | `HotPixelMaskedLoss` | 0.1 | `threshold` (0.15), `base_loss` (`charbonnier`), `kernel_size` (3). LR 입력도 받음 |
| `tv` | `TVLoss` | 0.1 | |
| `gradient_variance` | `GradientVarianceLoss` | 0.05 | `patch` (8), `mode` (`mean`) |

어떤 항목이든 `warmup_epochs: n`을 줄 수 있습니다. 그 항의 가중치는 `weight × min(1, (epoch − 1) / n)`이라 epoch 1에서 정확히 0이고 epoch n+1에서 목표값입니다. 가중치가 0인 항은 계산하지 않습니다. warm-up이 있는 이유는 이렇습니다. Phase-2는 Phase-1 손실의 최적점에서 출발하므로, 보조 손실을 한 번에 켜면 첫 검증이 출발점보다 나빠지고 patience가 그때부터 줄어 아직 좋아지는 중인 run이 멈췄습니다(`tests/test_loss_warmup.py`).

IR(무채색)에는 `edge.mode: mean`, `perceptual.gray: true`, `gradient_variance.mode: mean`을 씁니다. 채널 평균은 세 채널에 같은 gradient를 주어 색을 밀 수 없습니다. 반면 BT.601 휘도 가중치(`gray`)는 그 자체가 색 방향이라, 무채색 입력에서 gradient의 49 %가 색을 밀었습니다(`mean`의 색 gradient가 정확히 0인지는 `tests/test_losses.py`가 확인). `edge.signed: true`는 미분의 부호까지 비교해 링잉(overshoot)에 벌점을 줍니다. 출하 레시피별 손실 조합은 [configs/README.md](../configs/README.md)에 있습니다.

VGG19와 LPIPS 가중치는 `~/.cache/torch/hub`에 있어 `--offline`에서도 됩니다. 학습 중 torchvision이 찍는 `'pretrained' is deprecated` 경고는 무해합니다.

## 7. AIMET 도구 (`src/aimet`)

### 7.1 `quant_sim.py`

QuantSim은 이 파일로만 만듭니다. 순서는 고정입니다.

```text
switch_to_deploy -> prepare_model -> eval -> fold_all_batch_norms -> equalize_model (CLE) -> QuantizationSimModel
```

이 순서가 `.encodings`와 `.onnx`의 이름을 정하므로, 순서를 바꾸면 둘이 조용히 어긋납니다.

| 함수 | 하는 일 |
|---|---|
| `prepare_deploy_model(model)` | `switch_to_deploy`(있으면) → AIMET `prepare_model` → `eval()` |
| `fold_and_equalize(model, dummy_input, apply_cle=True)` | BN 접기와 (선택) CLE |
| `build_quantsim(model, dummy_input, *, quant_scheme='post_training_tf_enhanced', default_output_bw=8, default_param_bw=8, config_file=DEFAULT_QUANTSIM_CONFIG, prepare=True, fold=True, apply_cle=True, **quantsim_kwargs)` | 위 단계를 거쳐 `QuantizationSimModel`을 돌려줌. 이미 prepare나 fold를 했으면 `prepare=False`, `fold=False`. 나머지 kwargs는 그대로 넘김 |

`DEFAULT_QUANTSIM_CONFIG`는 `configs/aimet/htp_v73_no_leakyrelu_supergroup.json`입니다. AIMET 내장 `htp_v73`은 Conv+LeakyRelu를 supergroup으로 묶어 Conv 출력 양자화기를 끕니다. qairt-converter 2.41은 그 융합을 하지 않으므로 Conv 출력에 인코딩이 없고, Bicubic++의 연산 8개 중 6개가 float로 떨어집니다. 변환은 성공으로 끝나므로 로그의 `Following OPs fallback to float:` 뒤 목록을 읽어야만 보입니다. 저장소 config는 내장본에서 LeakyRelu supergroup 3개만 뺀 것입니다(14개 → 11개). LeakyReLU 모델에는 다른 config를 주지 않습니다. AIMET 스크립트 3~7의 `--quantsim_config` 기본값이 이것입니다.

### 7.2 `utils.py`

| 이름 | 하는 일 |
|---|---|
| `evaluate_model(model, loader, device, title="Model", scale=None)` | `src.metrics`의 PSNR만 냄. `scale`을 안 주면 첫 배치의 HR/LR 크기 비로 정함. AIMET 스크립트가 찍는 FP32 / Pre-QAT / Post-QAT 숫자 |
| `save_active_results_to_csv(results, output_dir, filename="quant_results.csv")` | dict 목록을 CSV로 씀 |
| `AutoQuantDatasetWrapper(dataset)` | `item['lr']`만 돌려줌(라벨 없는 보정 데이터) |
| `create_sampled_data_loader(dataset, num_samples, batch_size=32)` | 무작위 부분집합 DataLoader, `num_workers` 4 |
| `apply_mmp_from_json(sim, config_path)` | `{"layers": {"점.이름": {"input_bw", "output_bw", "param_bw"}}}`를 층별로 적용. 모르는 층 이름은 경고만 찍고 넘어감. 출하 `mmp_noprune_w16a16.json`은 `layers`가 비어 있음 |

이 모듈을 import하면 `aimet_torch.common.winnow.mask.Mask`가 프로세스 전체에서 패치됩니다. DepthToSpace, PixelShuffle, AddOp에서 마스크 전파를 멈추고, Hardtanh를 1:1 연결로 다룹니다. 채널 pruning(AIMET 2단계)을 위한 것이며, 이미지의 aimet-torch 2.29에서 적용됩니다. export 로그 첫 줄의 `[Info] Monkey Patch Applied: Compatible with (in_channels, out_channels) signature.`가 그 표시입니다.

AIMET 단계별 사용법과 출하 export 명령은 [examples/README.md](../examples/README.md)에 있습니다.

## 8. 체크리스트

공통 규칙입니다. 새 모듈은 `docs/modules/<YYYYMMDD>-MOD-<Module>.md`(요구 → 설계 → 상세 설계 → 단위 시험 케이스 → 결과)를 먼저 쓰고, 시험 함수 이름에 TC id를 넣습니다(예: `test_TC_BPPD_02_4_ClampIsTheDefault`). 이 저장소에서 마지막으로 모델을 더한 사례가 [docs/modules/20261012-MOD-BicubicPPDenoise.md](../docs/modules/20261012-MOD-BicubicPPDenoise.md)입니다. 문서와 코드는 같은 커밋에 넣고, 커밋 제목은 `[YYYYMMDD] <Component> <Verb> <what>`(Verb는 Add, Update, Fix, Remove, Refactor, Init, Finish 중 하나)입니다. 구조나 레시피를 비교하는 실험은 [docs/experiments/README.md](../docs/experiments/README.md)의 방식으로 기록합니다.

### 8.1 새 모델 추가

1. `src/models/<name>.py`에 `nn.Module`을 씁니다. 3.6절의 계약을 지킵니다: 입력 `(N, 3, H, W)` [0, 1], 출력 텐서 하나 `(N, 3, H·s, W·s)`, 마지막 모듈 `self.final_act = nn.Hardtanh(0.0, 1.0)`, 데이터 의존 분기가 없는 forward.
2. 학습 전용 가지가 있으면 `switch_to_deploy()`를 만듭니다. 여러 번 불러도 같아야 하고, 접은 뒤의 모듈 트리와 state_dict 키가 배포할 그래프와 같아야 합니다. 접을 것이 없어도 아무것도 하지 않는 `switch_to_deploy()`를 둡니다(`tests/test_reparam_equivalence.py`가 `hasattr`을 확인합니다).
3. IR 무채색을 지원하려면 `achromatic_output` 속성을 두고, `switch_to_deploy()`에서 마지막 층에 접은 뒤 `False`로 돌립니다. `6_apply_qat._check_achromatic_fold`가 config의 `model.achromatic_output`을 보고 모델 속성과 회색 입력 출력을 확인합니다.
4. `src/models/__init__.py`에 import, `__all__`, `build_model`의 `elif name == '<name>':` 분기를 더합니다. `bicubicplusplus`처럼 나머지 키를 생성자에 그대로 넘기면 config 오타가 바로 드러납니다.
5. config의 `model:` 블록을 만듭니다([configs/README.md](../configs/README.md)).
6. 테스트를 더합니다.
   - `tests/test_build_model_contract.py`의 `CASES`에 최소 config와 scale을 넣습니다(출력 모양과 유한값 확인).
   - `tests/test_reparam_equivalence.py`의 `CASES`에 넣습니다(접기 전후 출력 차이 fp32 1e-3, fp64 1e-10 이내).
   - `final_act`가 `nn.Hardtanh`인지 확인하는 시험을 더합니다(`tests/test_bicubicpp_denoise.py`의 `test_TC_BPPD_02_4_ClampIsTheDefault`가 본보기).
   - 생성자가 거부해야 하는 조합이 있으면 그 시험도 더합니다.
7. 테스트 수가 늘었으므로 `docker/accept.sh`의 `BASE_TESTS`를 같은 커밋에서 고칩니다(9절).
8. 배포 경로를 한 번 끝까지 돌립니다: 짧은 학습 → `6_apply_qat.py --qat_epochs 0` → `gen_convert_all.py` → `convert_all.sh`. 변환 로그의 `[ WARNING ] Following OPs fallback to float.`가 마침표로 끝나는지(목록 없음) 봅니다. 명령은 [README.md](../README.md)와 [examples/deployment/README.md](../examples/deployment/README.md)에 있습니다.
9. 배포 쪽 결합을 확인합니다. `gen_convert_all.py`는 run 폴더를 `x<scale>_<eo|ir>[_1024]` 이름으로만 찾고, DLC 이름을 `BicubicPP_<EO|IR>_<W>x<H>_x<s>_<version>.dlc`로 고정해 짓습니다. 보드 gst 스크립트도 이 이름 규칙과 `final_act`를 기본으로 씁니다. 다른 구조를 이 규칙으로 내보내려면 새 `--version` 접미사를 쓰고, 그렇지 않으면 생성기와 보드 스크립트를 함께 고칩니다. export 입력 크기는 보드 해상도와 같은 고정값입니다([examples/deployment/README.md](../examples/deployment/README.md)).

### 8.2 `BicubicPlusPlus`를 고칠 때

출하본의 재현을 `tests/test_release_golden.py`가 고정합니다. `__init__`의 모듈 생성 순서나 초기화를 바꾸면 seed 고정 초기 가중치 시험(TC-REL-03-1)이 깨지고, forward나 `switch_to_deploy`를 바꾸면 배포 출력 시험(TC-REL-01-2)과 export 시험(TC-REL-02-1)이 깨집니다. 의도한 변경이면 같은 커밋에서 기록값을 다시 재고, 이유를 커밋과 [docs/modules/20261015-MOD-ReleaseGolden.md](../docs/modules/20261015-MOD-ReleaseGolden.md) 5절에 적습니다. 새 옵션의 기본값은 지금 동작과 같아야 출하 체크포인트가 그대로 읽힙니다(`act`의 기본이 `leaky_relu`인 것과 같은 이유).

### 8.3 새 손실 추가

1. `src/losses/losses.py`에 `nn.Module`을 씁니다. `forward(pred, target)`은 스칼라 텐서를 돌려주고, `pred`와 `target`은 `(N, 3, H, W)` [0, 1]입니다. 고정 커널은 `register_buffer`로 둡니다(Trainer가 `UnifiedLoss(...).to(device)`로 옮깁니다). 사전학습 네트워크는 `eval()`로 두고 `requires_grad=False`로 얼립니다.
2. `UnifiedLoss.__init__`에 블록을 더합니다. **`self.target_weights = dict(self.weights)` 줄보다 위에** 넣어야 warm-up이 이 항을 압니다.

   ```python
   cfg_new = config.get('new_key', {})
   if cfg_new.get('enabled', False):
       self.loss_funcs['new_key'] = NewLoss(option=cfg_new.get('option', 1.0))
       self.weights['new_key'] = float(cfg_new.get('weight', 0.05))
   ```

3. LR 입력이 필요하면 `forward`에서 `hot_pixel_masked`처럼 이름으로 따로 부릅니다. 나머지 항은 `loss_fn(pred, target)`으로 불립니다.
4. `warmup_epochs`와 TensorBoard `Loss/new_key`는 따로 할 일 없이 동작합니다.
5. 새 항은 검증 손실(`val_loss`)도 바꿉니다. 손실 구성이 다른 run끼리 `val_loss`를 비교하지 않습니다.
6. config에 켠 항목이 실제로 들어갔는지 확인합니다. 키 오타는 조용히 무시되기 때문입니다.

   ```bash
   docker/run.sh -- python3 -c "import yaml; from src.losses.losses import UnifiedLoss; c = yaml.safe_load(open('<config>.yaml')); print(sorted(UnifiedLoss(c.get('loss', {})).loss_funcs))"
   ```

7. 새 사전학습 가중치가 필요하고 `~/.cache/torch`에 없으면, 첫 실행은 `--offline` 없이 해서 내려받습니다. `--offline`은 컨테이너의 네트워크를 끕니다.
8. `tests/test_losses.py`에 시험을 더합니다: 알려진 입력에서의 값, 같은 이미지에서 0, backward 뒤 gradient가 유한한지. IR에 쓸 손실이면 무채색 입력에서 채널별 gradient가 같은지도 봅니다(`test_edge_mean_mode_has_zero_chroma_gradient`가 본보기). `BASE_TESTS`도 고칩니다.
9. 출하 config에 넣는다면 키와 근거를 [configs/README.md](../configs/README.md)와 `docs/CONFIG_RATIONALE.md`에 적습니다.

### 8.4 새 열화 op 추가

1. `apply_configured_degradation`의 알맞은 자리에 `if c.get('enabled', 기본) and random.random() < c.get('prob', p):` 꼴로 넣습니다. uint8 BGR을 받아 uint8 BGR을 돌려주고, float에서 돌아올 때는 반올림합니다(`apply_float_op`).
2. `_DEGRADATION_PIPELINE_DEFAULTS`에 같은 기본값으로 더합니다. 빠지면 `[Degradation] … active/off` 줄에 나오지 않고, 그 키가 `keys not read by the pipeline` 경고로 찍힙니다.
3. 기본은 `enabled: False`로 둡니다. 꺼진 op는 난수를 뽑지 않으므로 출하 레시피의 학습 쌍이 그대로이고, `tests/test_data_recipe_golden.py`의 기록값 20개가 유지됩니다. 기본이 켜짐이면 그 op를 적지 않은 모든 레시피가 바뀌므로 같은 커밋에서 기록값을 다시 잽니다.
4. 그 op만 켠 config로 출력 범위나 통계를 확인하는 시험을 `tests/test_sr_data.py`에 더하고, golden 시험을 돌립니다(9절).

### 8.5 새 데이터 넣기

새 이미지 폴더는 코드가 아니라 config 일입니다. `data.train_root`에 목록으로 더하고, 검증 쌍은 새 폴더에 미리 만듭니다([examples/README.md](../examples/README.md)). 데이터는 컨테이너에서 보여야 합니다. 저장소 안이나 `/mnt` 아래면 그대로 보이고, 다른 곳이면 `docker/run.sh --mount`로 붙입니다([docker/README.md](../docker/README.md)).

새 Dataset 클래스가 정말 필요하면 `tools/train.py`의 `resolve_dataset_kind`와 데이터셋 생성 분기를 함께 고칩니다. 클래스는 `{'lr', 'hr', 'path'}`를 돌려줘야 하고, 시작 그림이 `dataset[i]`와 (있으면) `dataset.image_paths`를 씁니다. `tests/test_dataset_routing.py`에 새 경로의 시험을 더합니다.

## 9. 테스트 돌리기

`src`를 고친 뒤에는 이것을 돌립니다.

```bash
docker/run.sh --offline -- env PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q -rs -p no:cacheprovider \
    tests/test_build_model_contract.py tests/test_reparam_equivalence.py tests/test_bicubicpp_denoise.py \
    tests/test_trainer.py tests/test_loss_warmup.py tests/test_losses.py tests/test_config_guards.py \
    tests/test_scheduler_prepare.py tests/test_augmentation.py tests/test_camera_model.py \
    tests/test_sr_data.py tests/test_dataset_skip.py tests/test_dataset_routing.py
```

모델, `switch_to_deploy`, `quant_sim.py`, 열화 코드를 건드렸으면 출하본 재현 시험도 돌립니다.

```bash
docker/run.sh --offline -- env PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q -rs -p no:cacheprovider \
    tests/test_release_golden.py tests/test_data_recipe_golden.py
```

통과는 요약 줄이 `N passed, M warnings in …s`처럼 `passed`만 세고 `failed`, `skipped`, `error`가 없는 것입니다. 그 뒤에 찍히는 `sys:1: DeprecationWarning: builtin type swigvarlink has no __module__ attribute` 한 줄은 무해합니다(2026-10-20 첫 명령: `170 passed, 20 warnings in 154.05s`). `-rs`는 skip과 그 이유를 보여 줍니다. golden 시험은 git이 추적하지 않는 출하 파일이 없거나 OpenCV/NumPy 버전이 다르면 skip하므로, 이 서버에서는 skip이 0이어야 합니다. `PYTHONDONTWRITEBYTECODE=1`과 `-p no:cacheprovider`는 저장소에 `__pycache__`와 `.pytest_cache`를 남기지 않게 합니다(`docker/accept.sh`도 같은 형태로 돌리고, 실행 뒤 새로 생긴 파일을 기록합니다). 전체 시험은 파일 목록을 뺀 같은 명령입니다.

**`BASE_TESTS`.** `docker/accept.sh`는 pytest가 정확히 `BASE_TESTS + (docker 시험 수)`개 통과해야 PASS로 봅니다. `BASE_TESTS=280`은 `tests/test_docker_run.py`와 `tests/test_docker_image.py` 밖의 시험 수입니다. 그 밖의 시험을 더하거나 빼면 같은 커밋에서 이 값을 고칩니다. 새 값은 아래 명령의 마지막 줄 `N tests collected`입니다.

```bash
docker/run.sh --offline -- env PYTHONDONTWRITEBYTECODE=1 python3 -m pytest --collect-only -q -p no:cacheprovider \
    --ignore=tests/test_docker_run.py --ignore=tests/test_docker_image.py
```

## 10. 증상으로 찾기

| 증상 | 원인 | 볼 곳 |
|---|---|---|
| `TypeError: BicubicPlusPlus.__init__() got an unexpected keyword argument …` | `bicubicplusplus` 모델 블록의 키 오타 | 3.1 |
| `ValueError: … was trained with another model config; differing keys: …` | 평가나 export의 `--config` 모델 블록이 체크포인트와 다름 | 5.7 |
| Phase-2 run에 `best.pth`가 없음 | 기준선을 이기지 못함(설계대로) | 5.6 |
| `[Finetune] … missing=N` (N > 0) | 사전학습 체크포인트의 형태(학습형/배포형)나 모델 블록이 다름 | 3.4, 5.6 |
| `ValueError: early_stopping.monitor='loss' with a loss warm-up …` | `monitor: loss`와 `warmup_epochs`를 함께 씀 | 5.6 |
| 손실 항이 효과가 없고 TensorBoard에 `Loss/<키>`가 없음 | 손실 키 오타(조용히 무시) | 6, 8.3 |
| EO 학습 그림에 세로줄이나 열 잡음 | 데이터 config가 `stage2.ir_noise`를 끄지 않음 | 4.2 |
| `ValueError: PairedDataset stem mismatch …` | 검증 HR/LR 폴더의 파일 stem이 다름. 하위 폴더의 파일은 읽지 않음 | 4.1 |
| `[Validator] Shape mismatch detected …` | 출력 크기가 HR과 다름(x3의 나누어떨어지지 않는 폭 등). 새 모델이면 forward를 확인 | 3.6 |
| `--device`를 바꿔도 학습 장치가 그대로 | 장치는 Accelerate가 정함. `docker/run.sh --gpu`로 고름 | 5.1 |
| 변환 로그의 `Following OPs fallback to float:` 뒤에 op 목록 | quantsim config가 LeakyRelu supergroup을 가진 내장본 | 7.1 |
| `last.pth`가 CSV 마지막 epoch보다 하나 앞 | 조기 종료가 걸린 epoch는 저장 전에 루프를 나감 | 5.6 |
