# Examples (Model Compression & Quantization Toolkit)

이 디렉토리는 Qualcomm AIMET을 활용한 **실전 NPU/Edge 디바이스 배포용 모델 최적화 파이프라인** 예제 코드들을 순차적으로 제공합니다. 
과정은 단계별(1~6)로 번호가 매겨져 있으며, 각 번호를 순서대로 실행하며 최적화된 양자화 모델을 도출합니다.

## 파이프라인 오버뷰

### 단계 1: 기준 성능 측정 (FP32 Baseline)
- **`aimet_quantization/1_measure_fp32_model.py`**
- **목적**: 학습이 완료된 32비트 부동소수점(FP32) 모델의 정확도(PSNR)를 측정하여, 이후 압축/양자화 과정에서의 성능 손실(Drop)을 비교하기 위한 기준점(Baseline)을 잡습니다. (내부적으로 자동으로 `switch_to_deploy` 모드로 측정합니다.)
- **전체 아규먼트 (Arguments)**:
    - `--config`: 메인 모델 설정 파일 경로 (필수)
    - `--data_config`: 데이터 파이프라인 설정 파일 경로 (예: `configs/data/sr_train.yaml`)
    - `--checkpoint`: FP32 모델 검증용 가중치 파일 경로 (필수)
    - `--device`: 실행 장치 지정 (기본값: cuda)

### 단계 2: 경량화 (압축)
- **`aimet_quantization/2_compress_svd_pruning.py`**
- **목적**: 불필요한 레이어/채널을 가지치기(Channel Pruning) 하거나 랭크 분해(Spatial SVD)를 통해 파라미터 수와 MACs 연산량을 줄입니다. 이 과정 후 짧은 파인튜닝(Fine-tuning)이 동반됩니다.
- **전체 아규먼트 (Arguments)**:
    - `--config`: 메인 모델 설정 파일 경로 (필수)
    - `--data_config`: 데이터 설정 파일 경로 (예: `configs/data/sr_train.yaml`)
    - `--checkpoint`: 초기 모델 가중치 파일 경로 (필수)
    - `--output_dir`: 압축된 모델 및 로그가 저장될 경로 (기본값: results/compression)
    - `--width`: SVD/Pruning 분석용 더미 입력 가로 (기본값: 640)
    - `--height`: SVD/Pruning 분석용 더미 입력 세로 (기본값: 360)
    - `--use_svd`: Spatial SVD(특이값 분해) 압축을 활성화하는 플래그
    - `--use_pruning`: Channel Pruning(채널 가지치기) 압축을 활성화하는 플래그
    - `--ignore_layers`: 프루닝 대상에서 제외할 레이어 이름 혹은 패턴 (리스트 형태)
    - `--target_ratio`: 압축 후 남길 파라미터/MACs 타겟 비율 (0.0~1.0, 기본값: 0.1)
    - `--num_comp_ratio_candidates`: Greedy Search 탐색 시도 후보 수 (기본값: 10)
    - `--calib_batches`: 민감도를 평가할 때 사용할 데이터 배치 갯수 (기본값: 500)
    - `--ft_epochs`: 각 압축 스테이지 후 복구 파인튜닝을 진행할 에폭 수 (기본값: 20)
    - `--ft_lr`: 복구 파인튜닝 시 사용할 학습률 (기본값: 1e-5)
    - `--dataset_ratio`: 복구 파인튜닝 시 사용할 훈련 데이터셋의 비율 (기본값: 0.4)
    - `--device`: 실행 장치 지정 (기본값: cuda)

### 단계 3: 양자화 민감도 분석
- **`aimet_quantization/3_analyze_quant_sensitivity.py`**
- **목적**: FP32 모델에 `W4A4`, `W8A8`, `W8A4` 등 다양한 양자화 비트(Bit-width)와 전략(TF, Percentile 등)을 적용해보고, 가장 방어가 잘 되는 조합(Sensitivity)을 시각화합니다.
- **전체 아규먼트 (Arguments)**:
    - `--config`: 메인 설정 파일 경로 (필수)
    - `--data_config`: 데이터 설정 파일 경로 (예: `configs/data/sr_train.yaml`)
    - `--checkpoint`: 가중치 파일 경로 (필수)
    - `--output_dir`: 결과 CSV 및 그래프 이미지가 저장될 경로 (기본값: results/sensitivity)
    - `--calib_batches`: 양자화 파라미터 도출에 쓰일 샘플 배치 수 (기본값: 500)
    - `--device`: 실행 장치 지정 (기본값: cuda)
    - `--use_bn_folding`: 모델 내 Batch Normalization 레이어를 Fusing 할지 활성화하는 플래그
    - `--use_cle`: Cross Layer Equalization(계층 간 가중치 평탄화 기법)을 적용하는 플래그
    - `--width`: 더미 입력 가로 해상도 (기본값: 640)
    - `--height`: 더미 입력 세로 해상도 (기본값: 360)

### 단계 4: 계층별 노이즈 분석 (QuantAnalyzer)
- **`aimet_quantization/4_apply_quant_analyzer.py`**
- **목적**: AIMET의 `QuantAnalyzer`를 사용하여, 네트워크의 얕은 층부터 깊은 층까지 각 층(Layer)별로 양자화를 한 번씩 껐다 켜보며 어떤 레이어가 정확도 하락(MSE 증가)의 주범인지 찾아냅니다. 결과로 `.json`과 시각화 파일이 도출됩니다.
- **전체 아규먼트 (Arguments)**:
    - `--config`: 메인 설정 파일 경로 (필수)
    - `--data_config`: 데이터 설정 파일 경로 (예: `configs/data/sr_train.yaml`)
    - `--checkpoint`: 입력 파일 위치 (필수)
    - `--output_dir`: 분석 결과 HTML 및 JSON이 저장될 경로 (기본값: results/quant_analyzer)
    - `--scheme`: AIMET 내부 양자화 계산 스킴 방식 (기본: `post_training_tf_enhanced`)
    - `--param_bw`: 모든 파라미터(Weight)에 적용해 볼 테스트 비트 수 (기본값: 8)
    - `--output_bw`: 모든 활성화값(Activation)에 적용해 볼 테스트 비트 수 (기본값: 8)
    - `--width`: 입력 가로 해상도 (기본값: 640)
    - `--height`: 입력 세로 해상도 (기본값: 360)
    - `--calib_batches`: 노이즈(MSE) 수치를 도출하기 위한 샘플 데이터 수 (기본: 500)
    - `--device`: 실행 장치 지정 (기본값: cuda)

### 단계 5: 혼합 정밀도 인코딩 (Manual Mixed Precision)
- **`aimet_quantization/5_apply_manual_mixed_precision.py`**
- **목적**: 단계 4에서 발굴한 취약 레이어들(High MSE)만 선별적으로 높은 정밀도(INT8)로 고정하고, 나머지 대부분의 레이어는 고속 처리를 위해 INT4로 유지하는 형태의 양자화 파라미터(Encoding)를 확정합니다.
- **전체 아규먼트 (Arguments)**:
    - `--config`: 메인 설정 파일 경로 (필수)
    - `--data_config`: 데이터 설정 파일 경로 (예: `configs/data/sr_train.yaml`)
    - `--checkpoint`: 입력 가중치 경로 (필수)
    - `--output_dir`: 최종 .encodings 파일이 추출될 위치 (기본값: results/mmp)
    - `--param_bw`: 기본 가중치 비트 (기본값: 8)
    - `--output_bw`: 기본 활성화 비트 (기본값: 4, 특정 레이어만 우회됨)
    - `--scheme`: 기반 계산 스킴 지정 (기본값: `post_training_tf_enhanced`)
    - `--width`: 입력 폭 (기본값: 640)
    - `--height`: 입력 높이 (기본값: 360)
    - `--calib_batches`: 교정 샘플 데이터 수 (기본값: 500)
    - `--mmp_config`: 혼합 정밀도를 적용할 커스텀 레이어 구성 JSON 경로 (생략 시 하드코딩 예시 적용)
    - `--device`: 실행 장치 지정 (기본값: cuda)

### 단계 6: 양자화 인지 학습 (Quantization-Aware Training, QAT)
- **`aimet_quantization/6_apply_qat.py`**
- **목적**: 단계 5에서 확정된 인코딩 파라미터(혼합 정밀도/8비트)들을 바탕으로, 딥러닝 망 자체를 파인튜닝하여 양자화로 인해 발생하는 수학적인 노이즈/손실에 모델이 적응하도록 훈련합니다. 이 과정이 완료되면 비로소 실제 NPU에 올릴 수 있는 최종 `.encodings`와 모델 체크포인트가 생성됩니다. 
- **전체 아규먼트 (Arguments)**:
    - `--config`: 메인 설정 파일 경로 (필수)
    - `--data_config`: 데이터 설정 파일 경로 (예: `configs/data/sr_train.yaml`)
    - `--checkpoint`: 파인튜닝을 시작할 FP32 모델 가중치 (필수)
    - `--output_dir`: QAT 완료 `.pth` 와 `.encodings` 가 저장될 경로 (기본값: results/qat)
    - `--scheme`: QAT 학습에 적합한 양자화 범위 추적 스킴 (기본값: `training_range_learning_with_tf_init`)
    - `--param_bw`: 가중치 훈련 양자화 비트 단위 (기본값: 8)
    - `--output_bw`: 활성화값 훈련 양자화 비트 단위 (기본값: 8)
    - `--qat_epochs`: QAT 학습 진행 에폭(epochs) 수 (기본값: 30)
    - `--qat_lr`: QAT 학습에 사용될 미세 학습률 (기본값: 5e-5)
    - `--train_data_ratio`: 전체 학습 데이터 중, QAT 훈련에 투입할 데이터 비율 (기본값: 0.3)
    - `--encodings`: (선택) 외부에서 미리 구해진 초기 Encoding 파라미터
    - `--mmp_config`: (선택) 혼합 정밀도 환경이 필요할 경우 JSON 주입
    - `--width`: 데이터 이미지 폭 (기본값: 640)
    - `--height`: 데이터 이미지 높이 (기본값: 360)
    - `--calib_batches`: 인코딩 미지정 시 자체 캘리브레이션용 배치 수 (기본값: 500)
    - `--device`: 실행 장치 (기본값: cuda)

### 부가 (추론)
- **`aimet_quantization/7_inference_qat.py`**
- QAT까지 끝난 모델을 불러들여 가상의 장비 환경하에서 최종 결과 이미지를 얻어냅니다.

## 주의 사항 (Reparameterization 구조)
`svfocussrnet` 등 다중 분류 훈련 기술이 들어간 모델들은 1, 2, 3, 4, 5, 6 단계를 진행할 때 모두 내부적으로 **학습용 갈래(Branches)를 제거한 단일 3x3 Conv 모드(`switch_to_deploy()`)로 일관되게 분석/압축/양자화 되어야 수학적 오차가 발생하지 않습니다.**