# SRC (Source Code)

이 디렉토리는 플랫폼의 핵심 소스 코드(로직, 모델, 데이터 처리)를 포함하고 있습니다. PyTorch 및 AIMET 기반의 딥러닝 컴포넌트들이 정의되어 있습니다.

## 디렉토리 구조

- **`models/`**: 목표에 특화된 다양한 딥러닝 모델 아키텍처가 구현되어 있습니다.
    - `svfocussrnet.py`, `lrcsr.py`, `quicksrnet.py` 등
    - NPU/Edge 배포를 위해 **Structural Reparameterization** 기법이 다수 적용되어 있습니다 (`switch_to_deploy` 함수 포함).

- **`data/`**: 데이터셋 로딩 및 전처리 모듈이 포함되어 있습니다.
    - `datasets.py`: `SRDataset`, `DenoiseDataset`, `PairedDataset` 등 PyTorch Dataset 인터페이스를 상속합니다.

- **`engine/`**: 모델 학습 및 평가 파이프라인(Trainer)을 관리합니다.
    - `trainer.py`: Epoch 반복, 손실 계산, 역전파, 검증(Validation) 로직, TensorBoard 로깅 등을 캡슐화한 `Trainer` 클래스가 있습니다.

- **`losses/`**: 커스텀 손실 함수가 정의되어 있습니다.
    - L1, L2, SSIM 등 모델 최적화에 필요한 함수들을 관리할 수 있습니다.

- **`aimet/`**: AIMET을 활용한 양자화 및 압축 파이프라인 전용 유틸리티가 위치합니다.
    - `utils.py`: 양자화된 모델을 NPU 형태에 맞게 분석하고 보정하기 위한 커스텀 Wrappers (`AutoQuantDatasetWrapper` 등)와 편의 함수가 존재합니다.

## 활용 원칙

스크립트(`tools/`, `examples/`)에서는 `src` 내부를 모듈로서 임포트하여(ex. `from src.models import build_model`) 코드를 재사용합니다. 직접적으로 `src` 내부의 파일을 단독 실행하지 않습니다.
