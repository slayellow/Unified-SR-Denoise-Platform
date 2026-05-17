# Restormer Denoise Teacher 설계

작성일: 2026-05-16

목적:
- NAFNet과 함께 MC-G105 Denoise MTKD에 사용할 두 번째 Teacher 모델을 준비한다.
- Student는 계속 QCS8550 운용용 `SVFocusDenoiseNet dim32/block2`로 유지한다.
- Restormer는 배포용이 아니라 context-aware denoise teacher로 사용한다.

---

## 1. 결론

Restormer는 NAFNet과 다른 성격의 Teacher로 가치가 있다.

NAFNet:
- gated convolution 기반의 stable restoration teacher
- local/mid-frequency denoise, chroma mottle, hot pixel, grain 제거 reference에 적합

Restormer:
- transformer 기반의 context-aware restoration teacher
- foliage/shadow noise, low-frequency illumination/chroma instability, repetitive texture 판단에 적합

두 모델을 단순 평균하는 것보다 역할을 나누어 KD에 사용하는 방식이 맞다.

근거 자료:
- [Restormer: Efficient Transformer for High-Resolution Image Restoration, CVPR 2022](https://arxiv.org/abs/2111.09881).
- [Official Restormer repository](https://github.com/swz30/Restormer).
- Official real denoising config는 `dim=48`, `num_blocks=[4,6,6,8]`, `heads=[1,2,4,8]`, `num_refinement_blocks=4`, `ffn_expansion_factor=2.66`, `LayerNorm_type=BiasFree`를 사용한다.

## 2. 구조 요약

Restormer는 4-level hierarchical encoder-decoder 구조다.

핵심 block:
- MDTA: Multi-Dconv Head Transposed Self-Attention
- GDFN: Gated-Dconv Feed-Forward Network
- Overlapped patch embedding
- PixelUnshuffle/PixelShuffle 기반 down/up sampling
- skip connection과 refinement block

### MDTA

일반 spatial self-attention은 고해상도에서 비용이 크다.
Restormer는 channel 방향 cross-covariance attention으로 long-range interaction을 효율적으로 처리한다.

MC-G105 관점:
- shadow/foliage noise
- 넓은 영역의 chroma drift
- shading/illumination instability

이런 context-aware 판단에 유리하다.

### GDFN

FFN 내부에 depthwise 3x3 conv와 gate를 넣는다.
Transformer가 global/context만 보다가 local edge/detail을 잃는 문제를 완화한다.

MC-G105 관점:
- sign text
- pole/wire edge
- foliage texture

보존 여부를 teacher output에서 확인해야 한다.

## 3. 구현 파일

추가:
- `src/models/restormer.py`
- `configs/train/Denoise/restormer_mc_g105_phase1_denoise_priority_v2_teacher.yaml`
- `docs/plans/2026-05-16-dual-teacher-mtkd-denoise-plan.md`

수정:
- `src/models/__init__.py`

등록 이름:
- `restormer`
- `restormer_denoise_teacher`

두 이름은 같은 구조를 가리킨다.
`restormer_denoise_teacher`는 MTKD/denoise 실험 의도를 명확히 하기 위한 alias다.

## 4. 기본 Teacher Config

기본값:
- dim: 48
- num blocks: `[4, 6, 6, 8]`
- refinement blocks: `4`
- heads: `[1, 2, 4, 8]`
- FFN expansion: `2.66`
- LayerNorm: `BiasFree`
- output: input residual, train-time clamp off

학습 data config:
- `configs/data/denoise_mc_g105_phase1_denoise_priority_v2.yaml`

학습 train config:
- `configs/train/Denoise/restormer_mc_g105_phase1_denoise_priority_v2_teacher.yaml`

## 5. Params / 학습 서버 기준

Restormer official real denoising style:
- params: 26,111,668

비교:
- `SVFocusDenoiseNet dim32/block2/basic`: 31,619 params
- `NAFNet width64 SIDD-style`: 약 116.0M params
- `Restormer dim48 official real-denoise style`: 26,111,668 params

Restormer는 params는 NAFNet width64보다 작지만 attention activation memory 부담이 있다.

학습 서버 기준:
- GPU: RTX A6000 48GB x 2 사용 가정
- per-process batch size: 4
- gradient accumulation: 8
- expected effective batch: `4 x 2 GPUs x 8 accum = 64`

권장 실행:

```bash
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --num_processes=2 tools/train.py \
  --config configs/train/Denoise/restormer_mc_g105_phase1_denoise_priority_v2_teacher.yaml \
  --data_config configs/data/denoise_mc_g105_phase1_denoise_priority_v2.yaml
```

OOM이 없고 VRAM 여유가 충분하면 `batch_size: 6`, `gradient_accumulation_steps: 6`도 시도할 수 있다.
OOM이 나면 `batch_size: 2`, `gradient_accumulation_steps: 16`으로 낮춘다.

## 6. 실행 조건

Restormer Teacher 학습도 v2 supervised student가 deploy보다 나은 방향이라는 근거가 나온 뒤 시작한다.

확인할 것:
- v2 student가 foliage/shadow noise를 실제로 줄이는가.
- hot pixel/chroma mottle이 줄어드는가.
- sign text/pole/wire edge가 deploy처럼 과하게 뭉개지지 않는가.

Restormer teacher output도 real probe에서 별도 확인해야 한다.
Teacher가 edge ringing, color drift, texture hallucination을 만들면 KD target으로 부적합하다.

## 7. NAFNet + Restormer MTKD 방향

초기 dual-teacher KD는 역할 분리형으로 간다.

권장 역할:
- NAFNet: residual/output denoise 안정성
- Restormer: edge/frequency/context 보존
- HR: 최종 supervised anchor

초기 weight:

| KD term | NAFNet | Restormer |
| --- | ---: | ---: |
| output KD | 0.20 | 0.15 |
| residual KD | 0.30 | 0.40 |
| edge KD | 0.04 | 0.08 |
| frequency KD | 0.04 | 0.06 |

Teacher disagreement가 큰 영역에서는 HR supervised loss를 우선한다.
단순 output 평균은 피한다.
