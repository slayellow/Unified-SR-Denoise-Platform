# Unified-SR-Denoise-Platform

Super Resolution(초해상화) 및 Image Denoising(이미지 노이즈 제거) 통합 운용 플랫폼입니다. 
다양한 모델 아키텍처 학습은 물론, 추론, 모델 압축 및 양자화(QAT/PTQ) 파이프라인(Qualcomm AIMET)까지 MLOps 관점의 워크플로우를 모두 지원합니다.

## 주요 특징 (Key Features)
- **통합 모델 관리**: 초해상화(SVFocusSRNet, MambaIR 등)와 노이즈제거(SVFocusDenoiseNet) 모델들을 하나의 파이프라인과 코드베이스에서 통일감 있게 통제합니다.
- **분산 학습 (Distributed Training)**: `Hugging Face Accelerate` 코어 엔진으로 설계되어, 복잡한 DDP 세팅 변경 없이 Single-GPU 및 Multi-GPU 환경을 유연하게 전환할 수 있습니다.
- **최적화 워크플로우 제공**: AIMET 기반의 모델 압축(SVD, Pruning)부터 혼합 정밀도 분석, QAT(양자화 인지 학습)를 거쳐 실제 기기 탑재용 최적화를 위한 과정이 완벽히 내장되어 있습니다.

## 지원되는 모델 (Supported Models)
### Super Resolution
- `SVFocusSRNet`
- `MambaIR` / `MambaIRv2`
- `QuickSRNet`

### Denoising
- `SVFocusDenoiseNet`

---

## 디렉토리 구조 (Directory Structure)

```text
.
├── configs/            # 모델 설정(YAML), 학습 하이퍼파라미터 및 AIMET 셋업(JSON) 설정 파일 모음
├── src/
│   ├── engine/         # Trainer, Validator 등 학습 및 검증 코어 엔진 (Accelerate 적용)
│   ├── models/         # 플랫폼에서 제공하는 SR 및 Denoise 네트워크 아키텍처 스크립트 모음
│   ├── data/           # Dataset, DataLoader 및 전처리(물리 기반 핫픽셀, 노이즈 시뮬레이션 등) 모듈
│   └── losses/         # L1, Charbonnier, Perceptual, Contrastive 등 커스텀 손실 함수 모듈
├── tools/              # 학습(`train.py`), 추론(`inference.py`), 전처리, ONNX 내보내기(`export.py`) 등 유틸 스크립트
└── examples/           # AIMET 기반 모델 압축, 분석, QAT(양자화 인지 학습) 파이프라인용 스크립트 모음
```

---

## 환경 설정 및 요구사항 (Requirements)
- Python 3.8+
- PyTorch 2.x
- Hugging Face `accelerate`
- `pyiqa` (이미지/비디오 퀄리티 지표 측정용)
- (선택) Qualcomm AIMET (양자화 및 모델 압축 기능 사용 시)

---

## 🚀 모델 학습 가이드 (Training Guide)

이 플랫폼의 학습 스크립트(`tools/train.py`)는 `accelerate`를 활용해 구축되었기 때문에 로컬/분산 환경 제어가 매우 쉽습니다.

### 1) 환경 초기화 (최초 1회 필수)
터미널에서 아래 명령어를 실행하여 학습 환경을 설정합니다. 한 번 셋업해두면 추후 분산 학습 시 분산 환경 옵션을 일일이 부여할 필요가 없어 매우 편리합니다.
```bash
accelerate config
```
명령어 입력 시 나오는 질문들에 대해 아래 가이드라인을 참고하여 방향키와 Enter로 선택해 주세요(Multi-GPU 및 1 Node 서버 기준 추천 답변).
- **In which compute environment are you running?** 👉 `This machine`
- **Which type of machine are you using?** 👉 `multi-GPU`
- **How many different machines will you use?** 👉 `1` (물리적 단일 서버 기준)
- **Should distributed operations be checked while running for errors?** 👉 `NO`
- **Do you wish to optimize your script with torch dynamo?** 👉 `NO`
- **Do you want to use DeepSpeed / FullyShardedDataParallel / Megatron-LM?** 👉 모두 `NO`
- **How many GPU(s) should be used for distributed training?** 👉 `2` (또는 추후 터미널 실행 시 명시할 것이므로 임의 입력 무방)
- **Do you wish to use FP16 or BF16 (mixed precision)?** 👉 `fp16` 또는 `no` (VRAM 단축 및 속도 향상을 원하시면 `fp16` 권장)

### 2) Single-GPU (단일 GPU 학습)
가장 기본적으로 GPU 1대를 지정하여 모델을 학습하는 방법입니다. GPU 번호를 할당(`CUDA_VISIBLE_DEVICES`)하여 사용하는 것이 권장됩니다.
```bash
# 특정 GPU(예: 0번) 1대만 명시적으로 사용하여 학습
CUDA_VISIBLE_DEVICES=0 accelerate launch tools/train.py --config configs/train/SVFocusSRNet/svfocussrnet_2x.yaml
```

### 3) Multi-GPU (분산 학습) 진행하기
여러 대의 GPU를 하나로 묶어 빠르게 학습하거나 큰 배치 사이즈(Batch Size)를 원활히 처리하고 싶을 때 사용합니다. `Trainer` 내부적으로 DDP(Distributed Data Parallel)가 자동으로 적용됩니다.

```bash
# 플랫폼 내 가용한 2대의 GPU(예: 3번, 4번)를 모두 할당하여 병렬 학습
CUDA_VISIBLE_DEVICES=3,4 accelerate launch --num_processes=2 tools/train.py --config configs/train/MambaIR/mambairv2_2x.yaml
```

> **💡 VRAM 및 OOM (Out Of Memory) 최적화 팁**
> Multi-GPU 학습 시 Mamba나 Transformer 계열과 같은 무거운 모델에서 OOM이 발생할 우려가 있습니다.
> 이 때는 사용하시는 `xxx.yaml` 파일에서 **`patch_size`를 256에서 128로 절반(면적은 1/4)이 되도록 줄이고**,
> 줄어든 Batch Size를 상쇄하기 위해 **`gradient_accumulation_steps`를 늘리면(예: 1 -> 4)** 실질적인 배치를 유지하며 가벼운 VRAM으로 거대 모델을 학습할 수 있습니다!

---

## 🔍 추론 및 배포 (Inference & Deployment)

### 모델 추론 (Inference)
학습이 끝난 최고 성능의 모델 가중치(`best.pth`)를 사용하여, 폴더 내에 담겨있는 여러 Image 또는 Video에 일괄 추론을 수행할 수 있습니다.
```bash
python tools/inference.py \
    --config configs/train/SVFocusSRNet/svfocussrnet_2x.yaml \
    --checkpoint checkpoints/train_svfocussrnet_sr_x2/best.pth \
    --input ./data/test_images/ \
    --output_dir ./results/output/
```

### ONNX Export (배포용 모델 변환)
추론 성능을 타겟 Edge Device(예: NPU, DSP 칩셋) 등에 연동하기 위해 형태를 `ONNX`로 변환합니다. `Trainer` 내부의 Export 로직이 적용됩니다.
```bash
python tools/export.py \
    --config configs/train/SVFocusSRNet/svfocussrnet_2x.yaml \
    --checkpoint checkpoints/train_svfocussrnet_sr_x2/best.pth \
    --output_dir results/onnx/
```

---

## 🛠 모델 경량화 및 양자화 파이프라인 (AIMET)

최적화와 상용 탑재를 위해 `examples/` 디렉토리에 위치한 단계별 스크립트를 사용합니다. 해당 스크립트들은 Qualcomm AIMET의 기능들과 완벽히 연결됩니다.

1. `examples/1_measure_fp32_model.py` : FP32 베이스라인 성능(PSNR/SSIM 등) 파악
2. `examples/2_compress_svd_pruning.py` : SVD 및 Pruning 등의 모델 압축 진행 및 재학습
3. `examples/3_analyze_quant_sensitivity.py` : 각 레이어별 최소/최대 민감성(PTQ 민감도) 스캔
4. `examples/4_apply_quant_analyzer.py` : PTQ 적용 후의 성능 손실(MSE 추이) 분석기 실행
5. `examples/5_apply_manual_mixed_precision.py` : 앞선 분석 결과를 종합하여 Mixed Precision(혼합 정밀도) 레이어 셋업 지정
6. `examples/6_apply_qat.py` : 최종 혼합된 모델 스킨을 기반으로 QAT(양자화 노이즈 저감 인식 학습) 파이프라인 진행
