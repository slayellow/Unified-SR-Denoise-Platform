# Tools

이 디렉토리는 Unified SR Denoise Platform의 핵심 실행 스트립트들을 포함하고 있습니다. 모델 학습, 평가, 파인튜닝, 압축(ONNX 등) 파이프라인의 엔드포인트를 제공합니다.

## 포함된 스크립트

- **`train.py`**: 밑바닥부터 새로운 모델을 학습하거나 (Train from scratch), 특정 Pre-trained 가중치를 불러와 이어서 학습하는 메인 트레이닝 스크립트입니다.
    - 전체 아규먼트 (Arguments):
        - `--config`: 메인 학습 설정 파일 경로 (필수)
        - `--data_config`: 데이터셋 설정 파일 경로 (설정 파일의 내용을 덮어씀)
        - `--model`: 사용할 모델 이름 (설정 파일 덮어쓰기)
        - `--task`: 태스크 종류 (`sr` 또는 `denoise`, 설정 파일 덮어쓰기)
        - `--scale`: 업스케일링 팩터 (설정 파일 덮어쓰기)
        - `--work_dir`: 체크포인트 및 로그 저장 디렉토리
        - `--resume`: 이어서 학습할 체크포인트(.pth) 경로
        - `--epochs`: 학습 진행 에폭(epochs) 수 (설정 파일 덮어쓰기)
        - `--batch_size`: 배치 사이즈 (설정 파일 덮어쓰기)
        - `--lr`: 학습률 (Learning rate, 설정 파일 덮어쓰기)
        - `--device`: 실행 장치 지정 (기본값: cuda)

- **`evaluate.py`**: `.pth` 체크포인트를 불러들어 검증 데이터셋에 대해 성능(PSNR, SSIM 등)을 평가합니다. 배포용 모델(`switch_to_deploy`)의 성능 측정에도 사용됩니다.
    - 전체 아규먼트 (Arguments):
        - `--config`: 설정 파일 경로 (필수)
        - `--checkpoint`: 평가할 모델 가중치 파일 경로 (필수)
        - `--hr_dir`: 검증에 사용할 High-Resolution 이미지 폴더 (설정 파일 덮어쓰기)
        - `--lr_dir`: 검증에 사용할 Low-Resolution 이미지 폴더 (설정 파일 덮어쓰기)
        - `--device`: 실행 장치 지정 (기본값: cuda)
        - `--save_dir`: 결과 이미지 저장 폴더 (기본값: results)
        - `--save_images`: 모델이 추론한 결과 이미지를 파일로 저장할지 여부 (플래그)

- **`export.py`**: 모델 체크포인트를 ONNX 포맷으로 변환(Export)합니다.
    - 전체 아규먼트 (Arguments):
        - `--config`: 모델 파라미터 로드를 위한 설정 파일 경로 (필수)
        - `--checkpoint`: 변환할 가중치 경로 (생략 시 초기화된 가중치로 빈 모델 Export)
        - `--output`: 출력 ONNX 파일을 저장할 디렉토리 (기본값: ./results)
        - `--height`: 더미(Dummy) 입력 이미지 세로 길이 (기본값: 256)
        - `--width`: 더미(Dummy) 입력 이미지 가로 길이 (기본값: 256)
        - `--opset`: 내보낼 ONNX Opset 버전 (기본값: 17)
        - `--device`: 추론 추출에 사용할 장치 (기본값: cpu)
        - `--sim`: onnx-simplifier 패키지를 사용하여 ONNX 그래프 구조를 단순화할지 여부 (플래그)

- **`inference.py`**: 이미지 또는 디렉토리를 대상으로 학습된 모델을 통해 추론(Inference) 결과를 도출하고 파일로 저장합니다. 
    - 전체 아규먼트 (Arguments):
        - `--input`: 추론할 원본 이미지 파일 또는 이미지가 담긴 디렉토리 (필수)
        - `--output`: 추론된 이미지가 저장될 디렉토리 (기본값: results/inference)
        - `--config`: 모델 구조를 생성하기 위한 메인 설정 파일 (필수)
        - `--checkpoint`: 모델 가중치 파일 경로 (필수)
        - `--device`: 추론 실행 장치 (기본값: cuda)
        - `--fp16`: 16비트 부동소수점(Half-precision) 모드로 연산을 가속할지 여부 (플래그)

- **`run_aimet.py`**: (선택적) AIMET 워크플로우를 간편하게 실행하기 위한 종합 래퍼 스크립트입니다.

## 기본 흐름 (Pipeline)

1. `train.py` 를 통해 기준(Base) 모델 확보
2. `evaluate.py` 를 통해 목표 성능(PSNR) 검증
3. 모델 압축/양자화가 필요하다면 `examples/` 의 스크립트 활용
4. 타겟 하드웨어용 릴리즈 모델 생성 시 `export.py` 로 `.onnx` 추출
