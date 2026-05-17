# NAFNet Denoise Teacher 설계

작성일: 2026-05-16

목적:
- MC-G105 denoise-priority v2 pipeline이 유효하다고 확인된 이후 학습할 Teacher 모델을 준비한다.
- Student는 QCS8550 운용용 `SVFocusDenoiseNet dim32/block2`로 유지한다.
- Teacher는 배포 모델이 아니라 MTKD용 high-capacity restoration target으로 사용한다.

---

## 1. 결론

첫 Teacher 후보는 NAFNet이 적합하다.

이유:
- image denoising/restoration benchmark에서 검증된 구조다.
- encoder-decoder 구조라 shallow student보다 넓은 receptive field를 가진다.
- NAFBlock은 depth-wise convolution, SimpleGate, simplified channel attention을 사용해 noise/detail 분리에 유리하다.
- Restormer보다 첫 repo 통합과 학습 디버깅 부담이 낮다.
- output/residual/high-frequency distillation으로 student에 연결하기 쉽다.

근거 자료:
- [Simple Baselines for Image Restoration, ECCV 2022](https://arxiv.org/abs/2204.04676).
- [Official NAFNet implementation](https://github.com/megvii-research/NAFNet).
- Official SIDD denoise config는 width32/width64 모두 `enc=[2,2,4,8]`, `middle=12`, `dec=[2,2,2,2]` 구조를 사용한다.

## 2. SVFocusDenoiseNet과 역할 차이

`SVFocusDenoiseNet dim32/block2`:
- QCS8550 실시간 운용용 student.
- 얕고 작다.
- Hardtanh와 residual identity 초기화로 안정적이고 빠르다.
- 다만 넓은 context와 복잡한 sensor noise 제거 capacity는 제한된다.

`NAFNet width64`:
- 배포용이 아니라 Teacher용.
- multi-scale encoder-decoder로 foliage/shadow/chroma mottle 같은 구조적 noise를 더 넓은 context에서 본다.
- 더 큰 capacity로 student가 직접 학습하기 어려운 clean target을 제공한다.

## 3. 구현 파일

추가:
- `src/models/nafnet.py`
- `configs/train/Denoise/nafnet_mc_g105_phase1_denoise_priority_v2_teacher.yaml`

수정:
- `src/models/__init__.py`

등록 이름:
- `nafnet`
- `nafnet_denoise_teacher`

두 이름은 구조가 다른 모델을 의미하지 않는다.
`nafnet_denoise_teacher`는 MTKD/denoise 실험에서 의도를 명확히 하기 위한 alias다.
현재 구현에서는 `nafnet`과 `nafnet_denoise_teacher` 모두 같은 `NAFNetDenoiseTeacher` factory로 연결된다.
이후 task별 기본값을 분리할 필요가 생기면 alias를 별도 preset으로 확장할 수 있다.

## 4. 기본 Teacher Config

기본값:
- width: 64
- encoder blocks: `[2, 2, 4, 8]`
- middle blocks: `12`
- decoder blocks: `[2, 2, 2, 2]`
- input/output: RGB, scale 1
- output: input residual, train-time clamp off

학습 data config:
- `configs/data/denoise_mc_g105_phase1_denoise_priority_v2.yaml`

학습 train config:
- `configs/train/Denoise/nafnet_mc_g105_phase1_denoise_priority_v2_teacher.yaml`

## 5. Params 기준

코드 기준 parameter count:

| model | params |
| --- | ---: |
| `SVFocusDenoiseNet dim32/block2/basic` | 31,619 |
| `SVFocusDenoiseNet dim32/block8/basic` | 93,443 |
| `NAFNet width16 compact` | 2,688,499 |
| `NAFNet width24 compact` | 6,006,123 |
| `NAFNet width32 compact` | 11,454,819 |
| `NAFNet width32 SIDD-style` | 29,159,715 |
| `NAFNet width64 SIDD-style` | 115,982,915 |

현재 추가한 teacher config는 `NAFNet width64 SIDD-style` 기준이다.
이는 active student인 `SVFocusDenoiseNet dim32/block2`보다 약 3,668배 큰 capacity를 가진다.

Teacher는 배포 대상이 아니므로 QCS8550 latency보다 복원 품질과 KD target 품질을 우선한다.
다만 params가 클수록 항상 최종 student 품질이 좋아지는 것은 아니다. Teacher의 복원 capacity 상한은 올라가지만, Teacher가 over-smoothing 또는 색 틀어짐을 만들면 KD target도 같이 오염된다. 따라서 width64 Teacher도 real probe 기준으로 Teacher output 자체를 먼저 검증해야 한다.

학습 서버 기준:
- GPU: RTX A6000 48GB x 2 사용 가정
- per-process batch size: 8
- gradient accumulation: 4
- expected effective batch: `8 x 2 GPUs x 4 accum = 64`

권장 실행:

```bash
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --num_processes=2 tools/train.py \
  --config configs/train/Denoise/nafnet_mc_g105_phase1_denoise_priority_v2_teacher.yaml \
  --data_config configs/data/denoise_mc_g105_phase1_denoise_priority_v2.yaml
```

OOM이 발생하면 `batch_size: 6`, `gradient_accumulation_steps: 6`으로 낮춰 effective batch를 72 근처로 유지한다. 안정적으로 동작하고 VRAM 여유가 크면 `batch_size: 12`, `gradient_accumulation_steps: 3`도 후보로 볼 수 있다.

## 6. 실행 조건

지금 바로 Teacher 학습을 시작하지 않는다.

먼저 확인할 것:
- v2 student가 deploy 대비 noise 제거 방향으로 개선되는가.
- foliage/shadow noise, hot pixel, chroma mottle이 줄어드는가.
- sign text, pole, wire edge가 deploy처럼 과하게 뭉개지지 않는가.

이 조건이 통과되면 같은 v2 degradation으로 NAFNet teacher를 학습한다.

## 7. 이후 MTKD 연결

초기 MTKD는 feature distillation보다 output/residual 중심으로 시작한다.

권장:
- output KD: `student_output` vs `teacher_output`
- residual KD: `(input - student_output)` vs `(input - teacher_output)`
- edge/high-frequency KD: teacher가 보존한 edge/detail을 student가 따라가도록 유도

feature KD는 2차 단계로 미룬다. NAFNet과 SVFocusDenoiseNet의 feature hierarchy가 다르기 때문에 alignment cost가 크다.
