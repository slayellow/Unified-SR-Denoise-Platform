# NAFNet + Restormer Dual-Teacher MTKD 설계

작성일: 2026-05-16
갱신일: 2026-05-18

목적:
- MC-G105 deploy-hybrid v3 pipeline 검증 후 `SVFocusDenoiseNet dim32/block2` student를 개선하기 위한 MTKD 방향을 고정한다.
- Teacher는 `NAFNet width64`와 `Restormer dim48` 두 개로 시작한다.
- Teacher는 모두 offline 학습/추론용이며 QCS8550 배포 대상이 아니다.

---

## 1. 결론

NAFNet과 Restormer를 단순 평균 teacher로 쓰지 않는다.
두 teacher의 역할을 분리하고, GT/HR supervised loss를 항상 anchor로 유지한다.

역할:
- NAFNet: output/residual denoise 안정성, chroma mottle, hot pixel, local grain 제거
- Restormer: edge/frequency/context 보존, foliage/shadow 영역의 구조적 noise 판단
- HR/GT: teacher 오류를 막는 최종 anchor

초기 실험 순서:
1. v3 supervised student가 deploy보다 나은지 real probe와 비행시험으로 확인한다.
2. NAFNet teacher를 같은 v3 degradation으로 학습한다.
3. Restormer teacher를 같은 v3 degradation으로 학습한다.
4. teacher output을 real probe 기준으로 비교한다.
5. NAFNet-only KD, Restormer-only KD, dual-teacher KD를 ablation한다.

## 2. Teacher Training Config

NAFNet:
- active config: `configs/train/Denoise/nafnet_mc_g105_phase1_deploy_hybrid_v3_teacher.yaml`
- historical config: `configs/train/Denoise/nafnet_mc_g105_phase1_denoise_priority_v2_teacher.yaml`
- model: `nafnet_denoise_teacher`
- width: 64
- params: 115,982,915
- 2-GPU start: `batch_size=8`, `gradient_accumulation_steps=4`
- effective batch: `8 x 2 x 4 = 64`

Restormer:
- current prepared config: `configs/train/Denoise/restormer_mc_g105_phase1_denoise_priority_v2_teacher.yaml`
- note: Restormer도 v3 기준으로 학습할 경우 별도 `deploy_hybrid_v3` config를 추가한다.
- model: `restormer_denoise_teacher`
- dim: 48
- params: 26,111,668
- 2-GPU start: `batch_size=4`, `gradient_accumulation_steps=8`
- effective batch: `4 x 2 x 8 = 64`

둘 다 data config는 다음을 명시한다.

```bash
--data_config configs/data/denoise_mc_g105_phase1_deploy_hybrid_v3.yaml
```

## 3. MTKD Loss 초안

초기 loss:

```text
L_total =
  1.0 * L_sup(student, hr)
  + L_output_kd
  + L_residual_kd
  + L_edge_kd
  + L_frequency_kd
```

초기 weight:

| KD term | NAFNet | Restormer |
| --- | ---: | ---: |
| output KD | 0.20 | 0.15 |
| residual KD | 0.30 | 0.40 |
| edge KD | 0.04 | 0.08 |
| frequency KD | 0.04 | 0.06 |

Residual KD:

```text
teacher_residual = noisy_input - teacher_output
student_residual = noisy_input - student_output
```

Teacher disagreement가 큰 영역에서는 KD weight를 낮추고 supervised loss를 우선한다.
초기 구현은 feature KD 없이 output/residual/edge/frequency 중심으로 시작한다.

## 4. Teacher Output 검증 기준

Teacher metric만 보고 채택하지 않는다.
다음 real probe ROI를 반드시 본다.

- foliage/shadow: 2D/3D-like noise가 줄어드는가
- sign text: 글자가 뭉개지지 않는가
- pole/wire: 얇은 edge가 유지되는가
- flat wall/road: chroma mottle, hot pixel이 줄어드는가
- dark region: black crushing 또는 color drift가 생기지 않는가

Teacher가 색을 틀거나 texture hallucination을 만들면 해당 teacher의 KD weight를 낮춘다.

## 5. 구현 범위

현재 구현 완료:
- `src/models/nafnet.py`
- `src/models/restormer.py`
- `src/models/__init__.py` registry
- NAFNet teacher train config
- Restormer teacher train config
- online MTKD loss: `src/losses/kd_losses.py`
- online MTKD trainer: `src/engine/kd_trainer.py`
- KD entrypoint: `tools/train_kd.py`
- KD config: `configs/train/Denoise/mtkd_svfocusdenoise_mc_g105_phase1_denoise_priority_v2_nafnet_restormer.yaml`

다음 구현 대상:
1. v3 supervised student checkpoint 경로를 KD config의 `train.pretrained_path`에 입력
2. NAFNet teacher checkpoint 경로를 KD config의 `kd.teachers.nafnet.checkpoint`에 입력
3. Restormer teacher checkpoint 경로를 KD config의 `kd.teachers.restormer.checkpoint`에 입력
4. v3 기준 KD config를 별도 파일로 분리
5. NAFNet-only, Restormer-only ablation config 분리

KD 구현은 v3 supervised student와 teacher output이 real probe에서 유효하다는 것이 확인된 뒤 진행한다.

## 6. Online MTKD 실행

현재 random degradation 학습 컨셉에서는 teacher output cache를 기본 경로로 쓰지 않는다.
매 iteration마다 새로 생성된 noisy input을 frozen teacher와 student가 같이 본다.

```text
HR -> random degradation -> noisy
noisy -> frozen NAFNet teacher
noisy -> frozen Restormer teacher
noisy -> trainable SVFocusDenoiseNet student
loss -> student only update
```

실행 전 KD config에서 세 경로를 채워야 한다.

```yaml
train:
  pretrained_path: /path/to/supervised_v3_student/best.pth

kd:
  teachers:
    nafnet:
      checkpoint: /path/to/nafnet_teacher/best.pth
    restormer:
      checkpoint: /path/to/restormer_teacher/best.pth
```

실행:

```bash
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --num_processes=2 tools/train_kd.py \
  --config configs/train/Denoise/mtkd_svfocusdenoise_mc_g105_phase1_denoise_priority_v2_nafnet_restormer.yaml \
  --data_config configs/data/denoise_mc_g105_phase1_deploy_hybrid_v3.yaml
```

현재 KD config 파일명은 아직 v2 기준이므로, v3 Teacher checkpoint가 준비되면 `deploy_hybrid_v3` 기준 config로 분리한다.
