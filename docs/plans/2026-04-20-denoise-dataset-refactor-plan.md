# DenoiseDataset 독립 구축 및 로더 호환성 개선 계획

> For Hermes: Use subagent-driven-development skill to implement this plan task-by-task.

목표: `DenoiseDataset`가 더 이상 `SRDataset`를 상속하지 않고도 독립적으로 동작하게 만들고, `denoise_generic_baseline.yaml` 같은 denoise 전용 data config를 직접 사용할 수 있게 한다. 또한 현재 train config를 크게 바꾸지 않고 `dataset_type: denoise`만 반영해도 dataset 선택과 동작에 문제가 없도록 한다.

아키텍처:
현재 구조는 `DenoiseDataset(SRDataset)`로 되어 있어 SR용 crop/scale/degradation 가정이 denoise에도 그대로 섞여 있다. 이를 공통 유틸/공통 베이스 + `SRDataset` / `DenoiseDataset` 병렬 구조로 바꾸고, `tools/train.py`는 `task`만이 아니라 `data_config.dataset_type`도 읽어 dataset 클래스를 선택하도록 바꾼다. 이렇게 하면 generic baseline과 sensor-specific denoise config를 모두 `DenoiseDataset` 경로에서 직접 사용할 수 있다.

Tech Stack: Python 3.11, PyTorch Dataset/DataLoader, OpenCV, YAML config, existing `src/data/datasets.py`

---

## 0. 현재 문제 정의

현재 확인된 문제:
- `src/data/datasets.py:510-520`에서 `DenoiseDataset`는 `SRDataset`를 그대로 상속하고 `__getitem__`도 그대로 재사용한다.
- `SRDataset.__getitem__`은 `hr_h = patch_size * scale_factor` 가정을 가진다.
- `SRDataset.degradation_pipeline()`은 SR용 resize/downsample 개념을 전제로 만들어져 있고, denoise는 `scale_factor=1` 꼼수로 재사용 중이다.
- `tools/train.py:133-150`은 dataset class를 `config['task']`만 보고 선택한다. `data_config['dataset_type']`는 현재 학습 코드에서 사용되지 않는다.
- 따라서 현재 denoise baseline도 구조적으로는 “SRDataset scale=1 재사용” 상태다.

설계 목표:
1. `DenoiseDataset`는 독립 클래스여야 한다.
2. 공통 image loading / crop / augmentation / tensor 변환은 재사용 가능해야 한다.
3. SR/denoise 각각의 patch 의미와 degradation 의미가 분리되어야 한다.
4. `data_config.dataset_type: denoise`를 넣으면 dataset 선택이 자동으로 denoise로 가야 한다.
5. 기존 SR 학습 경로는 깨지지 않아야 한다.

---

## 1. 최종 설계 요약

### 1-1. 클래스 구조

현재:
- `SRDataset(Dataset)`
- `DenoiseDataset(SRDataset)`

변경 후:
- `BaseImageDataset(Dataset)` 또는 내부 공통 helper 집합
- `SRDataset(BaseImageDataset)`
- `DenoiseDataset(BaseImageDataset)`

핵심 원칙:
- inheritance는 “공통 입출력 처리”까지만 사용
- degradation semantics는 각 dataset이 직접 가진다

### 1-2. 공통으로 뽑을 책임

`src/data/datasets.py` 내부 공통 유틸 후보:
- 이미지 파일 수집
- 이미지 로드 실패 시 재시도
- random crop 좌표 선택
- flip/rot augmentation
- BGR→RGB float tensor 변환
- clean pair 확률 처리용 공통 helper

### 1-3. SRDataset가 계속 맡을 책임
- scale_factor > 1 기준 HR/LR patch 관계
- SR용 degradation pipeline
- target resize로 LR 해상도 생성

### 1-4. DenoiseDataset가 새로 직접 맡을 책임
- denoise patch는 HR/LR 동일 해상도 patch
- denoise용 degradation pipeline 직접 호출
- `patch_size`를 그대로 최종 patch 크기로 사용
- generic baseline / sensor-specific denoise config를 직접 읽음

### 1-5. train loader 선택 규칙

우선순위:
1. `config['data_config'].get('dataset_type')`
2. 없으면 기존 `config['task']`

매핑 규칙:
- `dataset_type in {'sr', 'super_resolution'}` → `SRDataset`
- `dataset_type in {'denoise', 'dn'}` → `DenoiseDataset`
- `dataset_type in {'guide', 'guided_sr'}` → `GuidedSRDataset`
- fallback → 기존 `task`

이렇게 하면 현재 설정에서 `dataset_type: denoise`만 바꿔도 dataset class 선택이 바뀐다.

---

## 2. 변경 대상 파일

### 수정
- `src/data/datasets.py`
- `tools/train.py`
- 필요 시 `configs/data/denoise.yaml`
- 필요 시 `configs/data/denoise_generic_baseline.yaml`
- 필요 시 `configs/README.md`
- 필요 시 `tools/README.md`

### 생성 후보
- `tests/data/test_denoise_dataset.py`
- `tests/data/test_dataset_type_resolution.py`

---

## 3. 호환성 요구사항

반드시 유지할 것:
- dataset output dict 키는 기존과 동일:
  - `{'lr': ..., 'hr': ..., 'path': ...}`
- trainer/criterion 호출은 수정 없이 그대로 동작
- validation set은 기존 paired loader 사용 가능
- 기존 SR config는 아무 수정 없이 계속 동작
- 기존 denoise train config는 최소 수정 또는 `dataset_type: denoise` 추가만으로 동작

추가로 보장할 것:
- `denoise_generic_baseline.yaml`은 `DenoiseDataset` 경로에서 자연스럽게 동작
- denoise에서 `scale_factor`는 더 이상 crop semantics를 좌우하지 않음

---

## 4. 구현 단계 계획

### Task 1: 현재 dataset 공통 책임 분리 설계 주석 추가

Objective: `datasets.py` 내부에서 SR 전용 책임과 공통 책임의 경계를 먼저 문서화한다.

Files:
- Modify: `src/data/datasets.py`

Step 1: `SRDataset` / `DenoiseDataset` 위에 책임 구분 주석 추가
Step 2: 공통 helper로 뽑을 함수 목록을 TODO 주석으로 명시
Step 3: syntax 확인

Run:
```bash
# 주석: datasets.py 문법 확인
python3 -m py_compile src/data/datasets.py
```

Commit:
```bash
git add src/data/datasets.py
git commit -m "refactor: annotate dataset responsibilities before split"
```

### Task 2: 공통 image helper 추출

Objective: 이미지 수집/로드/crop/tensor 변환 공통 로직을 베이스 계층 또는 helper 함수로 분리한다.

Files:
- Modify: `src/data/datasets.py`
- Test: `tests/data/test_denoise_dataset.py`

Step 1: failing test 작성
- image collection / tensor shape / path 반환 여부 검증

Step 2: 공통 helper 구현
- 예시 구조:
```python
class BaseImageDataset(Dataset):
    """Shared utilities for SR/Denoise datasets."""

    def __init__(self, dataset_root, patch_size, is_train, config=None):
        super().__init__()
        self.patch_size = patch_size
        self.is_train = is_train
        self.cfg = config or {}
        self.image_paths = self._collect_image_paths(dataset_root)

    def _collect_image_paths(self, dataset_root):
        ...

    def _read_image_or_retry(self, index):
        ...

    def _to_tensor(self, image_bgr):
        ...
```

Step 3: test 통과 확인

Run:
```bash
# 주석: dataset unit test 실행
pytest tests/data/test_denoise_dataset.py -v
```

Commit:
```bash
git add src/data/datasets.py tests/data/test_denoise_dataset.py
git commit -m "refactor: extract shared image dataset helpers"
```

### Task 3: SRDataset를 BaseImageDataset 기반으로 재구성

Objective: SRDataset가 공통 helper를 사용하면서 기존 동작을 유지하게 만든다.

Files:
- Modify: `src/data/datasets.py`
- Test: `tests/data/test_denoise_dataset.py`

Step 1: SRDataset 생성자 변경
```python
class SRDataset(BaseImageDataset):
    def __init__(self, dataset_root, scale_factor=2, patch_size=128, is_train=True, config=None):
        super().__init__(dataset_root=dataset_root, patch_size=patch_size, is_train=is_train, config=config)
        self.scale_factor = scale_factor
```

Step 2: SR 전용 crop semantics 유지
- `patch_size`는 LR patch 기준
- HR patch는 `patch_size * scale_factor`

Step 3: SR path regression test 실행

Run:
```bash
pytest tests/data/test_denoise_dataset.py -k sr -v
python3 -m py_compile src/data/datasets.py
```

Commit:
```bash
git add src/data/datasets.py tests/data/test_denoise_dataset.py
git commit -m "refactor: keep SRDataset behavior on shared base"
```

### Task 4: DenoiseDataset를 독립 구현으로 교체

Objective: DenoiseDataset가 더 이상 SRDataset를 상속하지 않고 자체 crop/degradation semantics를 가진다.

Files:
- Modify: `src/data/datasets.py`
- Test: `tests/data/test_denoise_dataset.py`

Step 1: failing test 작성
검증할 것:
- `DenoiseDataset`는 `patch_size x patch_size` 크기의 `lr/hr`를 반환한다.
- `scale_factor` 변화가 crop size 의미를 바꾸지 않는다.
- `clean_prob`가 denoise path에서도 동작한다.

Step 2: DenoiseDataset 구현
예시 구조:
```python
class DenoiseDataset(BaseImageDataset):
    """Independent denoise dataset with same-resolution synthetic degradation."""

    def __init__(self, dataset_root, scale_factor=1, patch_size=256, is_train=True, config=None):
        super().__init__(dataset_root=dataset_root, patch_size=patch_size, is_train=is_train, config=config)
        self.scale_factor = 1

    def degradation_pipeline(self, img_hr_patch):
        # denoise 전용 degradation 경로
        # 초기 구현은 기존 SR degradation pipeline 코드를 최대한 재사용하되
        # target_resize/scale semantics는 denoise에 맞게 정리
        ...

    def __getitem__(self, index):
        path, img = self._read_image_or_retry(index)
        img_hr_patch = self._crop_same_resolution_patch(img)
        img_lr_patch = self._build_noisy_patch(img_hr_patch)
        return {
            'lr': self._to_tensor(img_lr_patch),
            'hr': self._to_tensor(img_hr_patch),
            'path': path,
        }
```

Step 3: 기존 generic baseline config로 unit test 통과 확인

Run:
```bash
pytest tests/data/test_denoise_dataset.py -k denoise -v
python3 -m py_compile src/data/datasets.py
```

Commit:
```bash
git add src/data/datasets.py tests/data/test_denoise_dataset.py
git commit -m "refactor: make DenoiseDataset independent from SRDataset"
```

### Task 5: denoise degradation config와 dataset_type 연결

Objective: `dataset_type: denoise`가 실제 dataset class 선택에 반영되게 한다.

Files:
- Modify: `tools/train.py`
- Test: `tests/data/test_dataset_type_resolution.py`

Step 1: failing test 작성
- `task: sr`여도 `data_config.dataset_type == denoise`면 `DenoiseDataset`가 선택되는지 검증
- fallback으로 `dataset_type` 없으면 기존 task 경로를 따르는지 검증

Step 2: dataset resolver 추가
예시:
```python
def resolve_dataset_type(config):
    dataset_type = config.get('data_config', {}).get('dataset_type')
    if dataset_type:
        return dataset_type.lower()
    return config['task'].lower()
```

Step 3: `main()`에서 if/elif 분기 대신 resolver 사용

Run:
```bash
pytest tests/data/test_dataset_type_resolution.py -v
python3 -m py_compile tools/train.py
```

Commit:
```bash
git add tools/train.py tests/data/test_dataset_type_resolution.py
git commit -m "feat: resolve dataset class from data_config dataset_type"
```

### Task 6: denoise data config 기본값 정리

Objective: generic denoise baseline이 DenoiseDataset 의미에 맞게 명시되도록 한다.

Files:
- Modify: `configs/data/denoise_generic_baseline.yaml`
- Modify: `configs/data/denoise.yaml`
- Modify: `configs/README.md`

Step 1: `dataset_type: denoise` 명시 유지/추가
Step 2: denoise config에서 SR 전용으로 오해될 수 있는 키 설명 정리
Step 3: README에 “denoise는 DenoiseDataset로 직접 간다” 반영

권장 설명 문구:
- `patch_size`는 denoise에서 최종 입력/정답 patch 크기다.
- `scale_factor`는 denoise crop semantics에 영향을 주지 않는다.
- `dataset_type: denoise`가 있으면 `tools/train.py`는 `DenoiseDataset`를 선택한다.

Run:
```bash
python3 -m py_compile tools/train.py src/data/datasets.py
```

Commit:
```bash
git add configs/data/denoise_generic_baseline.yaml configs/data/denoise.yaml configs/README.md
git commit -m "docs: align denoise configs with independent DenoiseDataset"
```

### Task 7: end-to-end smoke test

Objective: 실제 config 기준으로 train loader까지 문제 없이 올라가는지 확인한다.

Files:
- No new files required

Step 1: denoise generic baseline smoke test
```bash
# 주석: 실제 학습 대신 dataset/dataloader 초기화까지 확인
python3 tools/train.py \
  --config configs/train/Denoise/svfocusdenoise_block2_basic.yaml \
  --data_config configs/data/denoise_generic_baseline.yaml \
  --epochs 1 \
  --batch_size 1 \
  --device cpu
```

Step 2: current config 최소 수정 시나리오 test
- `configs/train/Denoise/svfocusdenoise_block2_basic.yaml`의 `task`를 그대로 둔 상태에서도
- `data_config.dataset_type=denoise`가 dataset class 선택을 override하는지 확인

Step 3: SR regression smoke test
```bash
python3 tools/train.py \
  --config configs/train/SVFocusSRNet/svfocussrnet_2x_large_data.yaml \
  --epochs 1 \
  --batch_size 1 \
  --device cpu
```

Commit:
```bash
git add -A
git commit -m "test: verify independent DenoiseDataset and loader compatibility"
```

---

## 5. 설계 세부 결정

### 결정 1: `DenoiseDataset`는 inheritance가 아니라 parallel class로 간다
이유:
- denoise는 same-resolution semantics가 핵심이다.
- SR의 `patch_size -> hr_h = patch_size * scale_factor` 구조는 의미가 다르다.
- scale=1 꼼수는 계속 유지보수 비용을 만든다.

### 결정 2: degradation pipeline 코드는 “완전 복붙”보다 공통 helper + dataset별 orchestration으로 나눈다
이유:
- blur/noise/jpeg primitive는 공유 가능
- 하지만 resize / target_resize / scale semantics는 dataset별로 다르다.

### 결정 3: dataset 선택은 `task` 단독이 아니라 `dataset_type` 우선으로 바꾼다
이유:
- 사용자가 원하는 “type만 Denoise로 바뀌어도 동작” 요구를 만족하려면 이게 필요하다.
- backward compatibility도 유지 가능하다.

### 결정 4: output schema는 바꾸지 않는다
이유:
- trainer/loss/model 경로를 흔들지 않기 위함
- 현재 학습 루프는 `lr`, `hr`, `path` dict를 전제로 동작한다.

---

## 6. 검증 체크리스트

- [ ] `DenoiseDataset`가 더 이상 `SRDataset`를 상속하지 않는가?
- [ ] `DenoiseDataset`는 `patch_size x patch_size` same-resolution patch를 반환하는가?
- [ ] generic baseline data config가 `DenoiseDataset` 경로에서 바로 동작하는가?
- [ ] `data_config.dataset_type=denoise`가 dataset class 선택에 반영되는가?
- [ ] 기존 SR training path가 깨지지 않는가?
- [ ] trainer/criterion output contract가 그대로인가?
- [ ] 문서가 실제 동작과 맞는가?

---

## 7. 구현 후 기대 상태

최종적으로는 아래가 성립해야 한다.

1. `DenoiseDataset`는 독립 클래스다.
2. generic baseline과 sensor-specific denoise config 모두 `DenoiseDataset`를 직접 사용한다.
3. `tools/train.py`는 `dataset_type` 기반으로 dataset class를 선택할 수 있다.
4. 현재 denoise config는 최소 수정으로 동작한다.
5. 이후 `denoise_mc_g105_v1.yaml` 같은 adaptation config 설계도 dataset semantics 혼동 없이 진행 가능하다.

---

## 8. 권장 후속 작업

이 계획 구현 후 바로 이어서 할 것:
1. `denoise_mc_g105_v1.yaml` 초안 추가
2. day/night + low/high zoom mixing 규칙 반영
3. origin/deploy 실패 모드에 맞춘 sensor-specific degradation 항목 추가
4. loss 구성 generic vs adaptation 분리 검토
