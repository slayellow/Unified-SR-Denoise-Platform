# Configs

이 디렉토리는 Unified SR Denoise Platform의 모든 설정 파일을 포함하고 있습니다. YAML 포맷을 사용하여 모델 아키텍처, 학습 하이퍼파라미터, 양자화(AIMET) 파라미터 및 데이터 마이그레이션 구성을 정의합니다.

## 디렉토리 구조

- **`train/`**: 처음부터 모델을 학습(Train from scratch)하기 위한 설정 파일들이 위치합니다.
    - 예: `quicksrnet.yaml`, `svfocussrnet.yaml`
    - 구성 내용: 사용 모델(Scale 옵션 등), 손실 함수(Loss), 옵티마이저(Optimizer), 스케줄러, Epoch 수 등

- **`finetune/`**: SVD, Channel Pruning, QAT 등 모델 압축 및 양자화 이후 파인튜닝을 위한 설정 파일들이 위치합니다.
    - 예: `quicksrnet_ir.yaml`, `svfocussrnet_finetune.yaml`
    - 학습률(Learning Rate)이 낮게 설정되어 있으며, 가중치 미세 조정에 초점이 맞춰져 있습니다.

- **`data/`**: 학습 및 평가에 사용될 데이터셋 경로 및 전처리 옵션(Patch size, augmentation 유무 등) 데이터 로더 환경을 정의합니다.
    - 예: `sr.yaml` (Super Resolution 용), `denoise.yaml` (Denoising 용)

- **`aimet/`**: AIMET(양자화/압축 툴킷) 특화 설정이 일부 적용될 수 있는 구조를 지원합니다.

## 사용 방법

스크립트 실행 시 `--config` 와 `--data_config` 인자로 해당 YAML 파일 경로를 전달하여 사용합니다.
```bash
python tools/train.py --config configs/train/QuickSRNet/quicksrnet_2x.yaml --data_config configs/data/sr_train.yaml
```
