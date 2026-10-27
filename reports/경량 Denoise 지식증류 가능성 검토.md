# 합성 GT가 있으면 증류 이득은 사라진다

더 강한 teacher를 학습해 4층 BicubicPPDenoise[v1]에 증류하는 일은 구조적으로 막히는 곳이 없다. 그러나 지금처럼 합성 카메라 모델 쌍과 정확한 clean GT로 학습하는 한, 실기 품질이 측정할 수 있을 만큼 오를 가능성은 낮다. GT가 있는 pixel regression에서 출력 KD는 문헌상 대개 **0.00~+0.05 dB**이고 가장 큰 값도 RCAN x4의 +0.12 dB다. FitNet식 특징 KD는 **최대 -1.50 dB**로 오히려 해롭다. 크기가 가장 가까운 13K FSRCNN 학생은 일반 KD로 **+0.01~+0.08 dB**, 전용 teacher를 설계해도 **+0.19 dB**에 그쳤다. **+0.67~+1.58 dB**라는 큰 denoising 이득은 모두 세 조건이 겹친 곳에서 나왔다: GT가 부족하거나 잡음이 섞인 실데이터, 1.6~2.0M급 학생, 다른 변경이 함께 들어간 ablation. 사내 측정도 같은 쪽을 가리킨다. MAC이 약 8배인 full-resolution 모델은 합성 val에서 +0.8 dB였지만 실기 차트에서는 지표마다 엇갈렸고, 미세 텍스처 보존은 0.46으로 학생과 같았다. teacher의 합성 우위가 실기로 이어지지 않고, 텍스처를 막는 것은 용량이 아니라 손실 함수라는 뜻이다. 따라서 32/64층 동일 계열 teacher나 NAFNet·Restormer·SwinIR급 teacher(학생의 약 290~1000배)는 만들 이유가 없다. KD를 쓴다면 유일하게 그럴듯한 형태는 GT가 없는 실기 EO 프레임에 teacher가 라벨을 다는 pseudo-label 방식이다. 이때 teacher는 합성 PSNR이 아니라 실기 차트에서 원하는 지표가 실제로 나은 모델로 고르며, 가장 구체적인 후보는 사내 측정에서 chroma가 가장 좋았던 full-resolution 구조다(체크포인트는 지워져 다시 학습해야 한다). ECB 재파라미터화와 W16A16은 float에서 증류한 뒤 fold하고 변환하는 순서만 지키면 걸림돌이 아니다. 반면 stride-2 + PixelShuffle 구조의 위상(shift) 의존성은 KD로 고칠 수 없고 구조 변경이나 shift-consistency 손실로 다뤄야 한다. 이 보고서는 문헌 검토이며 어떤 실험도 돌리지 않았다. 아래에서 "사내 측정"은 소유자가 이미 가진 결과, "추론"과 "분석"은 이 검토의 판단이다.

## GT가 있으면 출력 KD는 0~0.1 dB에서 멈춘다

분류에서 KD가 통하는 이유는 teacher의 soft label이 정답 one-hot에 없는 클래스 간 유사도를 싣기 때문이다. pixel regression에는 그런 "dark knowledge"가 거의 없다. DUKD는 SR teacher의 출력이 GT에 "barely extra information"만 더한 "noisy approximations to the ground-truth distribution"이라고 정리한다. 그래서 출력을 직접 맞추면 지식이 거의 넘어가지 않고 "may even mislead the student model"이라고 쓴다 ([DUKD](https://arxiv.org/html/2309.14162v4)). 2018년 경량 SISR 연구도 연속값 GT를 쓰는 회귀에서는 "the output of teacher network fails to provide more valuable information than the ground truth"라고 같은 결론을 냈다 ([Adaptive Importance Learning](https://arxiv.org/pdf/1806.01576)). 이유는 식으로 보면 분명하다(추론). MSE에서 E‖s−y‖² = E‖s−E[y|x]‖² + E‖y−E[y|x]‖²이고 둘째 항은 학생과 무관하다. 따라서 조건부 평균을 완벽히 맞히는 teacher도 GT와 같은 최적해를 준다. teacher가 바꾸는 것은 목표의 분산, 즉 gradient 잡음뿐이다. Charbonnier에서는 기준이 조건부 중앙값으로 바뀔 뿐 논리는 같다. 이 프로젝트는 GT가 clean source이고 합성 생성기가 새 쌍을 끝없이 만들 수 있으므로, KD가 줄여 줄 분산이 거의 없다.

숫자도 그렇게 나온다. 아래 표의 차이는 모두 같은 논문 안에서 같은 학생을 GT만으로 학습한 결과와 비교한 값이다.

| 조건 | 학생 규모 | 보고된 Δ (dB, GT-only 대비) | 출처 |
|---|---|---|---|
| SR 출력 KD, EDSR teacher 43M | 2.70M | Urban100 x2/x3/x4 +0.02 / +0.01 / 0.00 | [DUKD](https://arxiv.org/html/2309.14162v4) |
| SR 출력 KD, SwinIR teacher 11.9M | 1.24M | Set5 x4 -0.04 | [DUKD](https://arxiv.org/html/2309.14162v4) |
| SR 출력 KD, RCAN teacher 15.6M (조사 범위 최대) | 5.17M | Urban100 x4 +0.12 | [DUKD](https://arxiv.org/html/2309.14162v4) |
| SR FitNet 특징 KD | 2.70M | Urban100 x2 -1.50, x4 -0.81 | [DUKD](https://arxiv.org/html/2309.14162v4) |
| 약 28배 압축, 직접 KD / FAKD | 1.52M | x4 -0.01 / -0.23 (2단계 TA KD만 +0.28) | [MiPKD](https://arxiv.org/html/2404.02573) |
| 초소형, 일반 KD | FSRCNN 13K | Set5 x2 +0.01~+0.08 | [PISR](https://arxiv.org/abs/2007.07524) |
| 초소형, privileged teacher 전용 설계 | FSRCNN 13K | x2 +0.18~+0.19, x4 +0.06 | [PISR](https://arxiv.org/abs/2007.07524) |
| Challenge, 특징+출력 KD / teacher 강화 GT | DIPNet 243k | +0.05 / +0.06 | [DIPNet](https://arxiv.org/abs/2304.07018) |
| Challenge, 출력+affinity KD (NTIRE 2025 1위) | 131k | +0.01~+0.04 | [DSCLoRA](https://arxiv.org/abs/2504.11271) |
| INT8, MobileOne reparam + PixelShuffle, 출력 KD | 약 82k | +0.09 | [Nguyen et al.](https://arxiv.org/html/2604.20291v1) |
| GT 없는 입력에 teacher가 라벨 (data upcycling) | 1.24~5.17M | +0.24~+0.58 | [DUKD](https://arxiv.org/html/2309.14162v4) |
| 실 잡음 denoising, 학습 쌍 650개, 다른 변경과 섞임 | 1.96M | +1.58 | [Kayani et al.](https://arxiv.org/html/2605.03680v1) |
| mobile denoising, 학습 영상 약 2,400장 | 1.67M | +0.67 (LPIPS 0.126 → 0.099) | [LiteKD-Net](https://arxiv.org/html/2608.05739) |
| RAW denoising, 출력 KD | 약 90 KB | RAW +0.145 / RGB +0.63 | [FAN-KD](https://arxiv.org/abs/2103.01524) |

여러 기법을 직접 재실행해 비교한 논문들은 같은 결론을 낸다. DUKD는 기존 SR KD가 "limited benefits"를 주고 일부는 KD 없이 학습한 학생보다 나쁘다고 했다. MiPKD와 DCKD도 같은 취지로 쓴다 ([DUKD](https://arxiv.org/html/2309.14162v4); [MiPKD](https://arxiv.org/html/2404.02573); [DCKD](https://arxiv.org/html/2412.08939v2)). 가장 긍정적인 MTKD조차 KD·AT·FAKD 재실행 결과는 -0.05~+0.35 dB로 모델마다 엇갈렸다 ([MTKD](https://arxiv.org/html/2404.09571v1)).

학생이 작다고 KD 이득이 커지지는 않는다. 41k에 가장 가까운 문헌 기준점은 PISR의 13K FSRCNN이다. 10배 큰 FSRCNN-L teacher로 일반 KD를 하면 Set5 x2 +0.01~+0.08 dB, B100 +0.01~+0.03 dB였다. teacher가 HR(GT)을 입력으로 보는 privileged 구조, 가중치 이전, VID 손실을 모두 써도 +0.18~+0.19 dB(x4는 +0.06 dB)였고, 저자들은 "there are no distillation losses effective to regression tasks"라고 썼다 ([PISR](https://arxiv.org/abs/2007.07524)). 효율 challenge의 수치도 같은 크기다. NTIRE 2023 DIPNet은 HAT teacher에서 특징+출력 KD로 +0.05 dB, teacher가 다듬은 GT로 +0.06 dB를 얻었다 ([DIPNet](https://arxiv.org/abs/2304.07018)). NTIRE 2025 우승작 EMSR(DSCLoRA, 131k)은 +0.01~+0.04 dB였다 ([DSCLoRA](https://arxiv.org/abs/2504.11271)). 구조가 가장 비슷한 2026년 INT8 SR 모델(약 82k, 32채널 MobileOne reparam 8블록, global skip, PixelShuffle)은 Mamba teacher의 출력 KD로 +0.09 dB였다. 다만 이 논문은 abstract와 표의 최종 수치가 서로 달라 가중치를 낮춰 읽어야 한다 ([Nguyen et al.](https://arxiv.org/html/2604.20291v1)).

이 학생과 거의 같은 형태(pixel-unshuffle, 32채널 RepBlock 3개, pixel-shuffle)인 NTIRE 2023 RTSR의 RUNet도 HAT-L 출력 KD를 썼지만 ablation이 없다. 같은 대회 보고서의 핵심 아이디어 목록(reparameterization, pixel shuffle/unshuffle, multi-stage 학습)에는 KD가 없고, Bicubic++도 KD 없이 우승했다 ([NTIRE 2023 RTSR](https://openaccess.thecvf.com/content/CVPR2023W/NTIRE/papers/Conde_Efficient_Deep_Models_for_Real-Time_4K_Image_Super-Resolution._NTIRE_2023_CVPRW_2023_paper.pdf); [Bicubic++](https://arxiv.org/abs/2305.02126)). challenge 팀이 KD를 쓰는 이유는 고정된 PSNR 문턱을 0.0x dB 차이로 넘기 위해서이지, KD가 큰 품질 레버여서가 아니다(추론).

큰 이득이 나온 사례는 조건이 다르다. MAI 2026 실 잡음 denoising(Kayani et al.)은 41.6M teacher에서 1.96M 학생으로 36.08 → 37.66 dB(+1.58)를 얻었다. 그러나 학습 쌍이 650개뿐이었고, teacher 손실 가중치를 GT의 9배로 둔 이유는 "to prevent the student from overfitting to residual sensor noise present in the raw ground truth data"였다. 같은 단계에 512/1024 crop 미세조정도 더해져서 +1.58 dB 전부를 KD 몫으로 볼 수 없다 ([Kayani et al.](https://arxiv.org/html/2605.03680v1)). LiteKD-Net의 +0.67 dB도 약 2,400장 학습에 1.67M 학생이다 ([LiteKD-Net](https://arxiv.org/html/2608.05739)). RAW denoising FAN-KD는 단순 출력 KD로 RGB에서 +0.63 dB를 얻었지만 RAW에서는 +0.145 dB였고, 저자들은 효과가 "especially in settings with limited training datasets"라고 밝혔다 ([FAN-KD](https://arxiv.org/abs/2103.01524)). transformer 압축 계열 SLKD는 SIDD에서 3.5~7.9M 학생이 teacher와 0.28~0.46 dB 이내라고 보고한다. 그러나 scratch baseline이 없어 이를 KD 이득으로 읽을 수 없고, 같은 저자의 다른 논문과 ablation 수치가 똑같아 신뢰도도 낮다 ([SLKD](https://arxiv.org/html/2501.09268v1)). SR에서 일관된 이득을 낸 유일한 계열인 DUKD(+0.24~+0.58 dB)도 메커니즘은 data upcycling, 즉 teacher가 GT 없는 확대·증강 입력에 라벨을 다는 것이다 ([DUKD](https://arxiv.org/html/2309.14162v4)). 공통점은 하나다. **KD는 데이터셋에 없는 라벨을 teacher가 채워 줄 때만 값을 한다.** 그런데 이 프로젝트의 합성 생성기는 이미 정확한 GT가 붙은 입력을 무한히 만드는 upcycler다. 그래서 합성 영역에서는 DUKD 메커니즘이 이미 포화돼 있다(추론). GT가 없는 곳은 실기 영역뿐이다.

작은 이득은 믿기도 어렵다. 같은 EDSR c64b32 학생의 scratch baseline이 논문에 따라 26.07 dB와 26.21 dB로 다르다. RCAN 5.2M 학생은 32.32 dB와 32.63 dB, 비슷한 크기의 SwinIR 경량 학생은 32.05 dB와 32.76 dB다. baseline이 주장된 KD 이득만큼 흔들리는 것이다 ([CSD](https://ar5iv.labs.arxiv.org/html/2105.11683); [DUKD](https://arxiv.org/html/2309.14162v4); [MTKD](https://arxiv.org/html/2404.09571v1)). 2026년의 엄격한 CIFAR 재검증은 ResNet-18 학생의 KD 이득을 0과 구분할 수 없다고 했다. 가중치는 낮지만 고전적 KD 이득이 실험 절차의 산물일 수 있다는 경고다 ([Student Capacity Moderates KD](https://arxiv.org/abs/2605.31191)). 사내 측정에 대입하면 판단이 더 분명해진다. 합성 val **+0.8 dB**였던 8배 MAC 모델도 실기 차트에서 고르게 낫지 않았다(사내 측정). 그렇다면 합성 +0.1 dB 이하인 KD 효과가 실기 차트 지표에 드러날 가능성은 낮다.

## teacher는 크기보다 학생이 닿을 수 있는지로 고른다

분류 문헌은 teacher가 커질수록 학생이 좋아지다가 다시 나빠진다는 데 대체로 동의한다. TAKD는 2층 plain CNN 학생에 4/6/8/10층 teacher를 붙였다. teacher 정확도는 계속 올랐지만 "the trained student accuracy first increases and then decreases"였다. 가장 좋은 중간 teacher(TA)는 크기가 중간인 모델이 아니라, 정확도가 학생과 teacher의 중간인 모델이었다 ([TAKD](https://arxiv.org/abs/1902.03393)). Cho & Hariharan은 ImageNet ResNet-18 학생에 ResNet-18/34/50 teacher를 붙였는데, 학생 오류가 30.57/30.79/30.95%로 teacher가 클수록 커졌다. 단계적 증류도 직접 증류와 "almost exactly the same"이었고, 결론은 "the solution the large teacher has found is simply not in the solution space of the small student"였다 ([Cho & Hariharan](https://arxiv.org/abs/1910.01348)). 반대 증거도 있다. Beyer et al.은 같은 증강 view를 teacher와 학생에 함께 주는 function matching과 9600 epoch에 이르는 긴 학습으로 큰 격차를 메웠다 ([Beyer et al.](https://arxiv.org/abs/2106.05237)). 다만 TAKD가 꼽은 세 요인 가운데 pixel regression으로 넘어오는 것은 "학생이 흉내 낼 수 있는가" 하나뿐이다. soft label의 온도 효과는 넘어오지 않는다(추론).

저수준 비전도 같은 그림이다. MiPKD는 약 28배 압축(EDSR 43M → 1.52M)에서 "Directly distilling the student model by the teacher model yields negative effects"라고 했다. TA를 둔 2단계 KD만 +0.28 dB였다 ([MiPKD](https://arxiv.org/html/2404.02573)). PISR에서는 13K 학생에게 57.60 dB짜리 무제한 teacher보다, hourglass로 성능을 묶은 37.70 dB teacher가 더 나은 학생을 만들었다 ([PISR](https://arxiv.org/abs/2007.07524)). IDC도 EDSR·SwinIR·MambaIRv2 조합에서 "a stronger teacher does not always yield a superior student"를 확인했다 ([IDC](https://arxiv.org/abs/2505.21099)). 출력 KD에서는 teacher 계열도 거의 상관없다. DUKD에서 RCAN 학생에 EDSR teacher를 붙이든 SwinIR teacher를 붙이든 Urban100 x4는 26.59 dB로 같았다 ([DUKD](https://arxiv.org/html/2309.14162v4)). 41.6M teacher는 41k 학생의 약 1000배다. NAFNet(17.1M)·Restormer(26.1M)·SwinIR(11.9M)도 약 290~640배로, MiPKD가 직접 KD가 해로워지는 것을 본 28배를 한참 넘는다.

동일 계열을 깊게 만드는 방안은 깊이 포화를 따져야 한다. 가장 가까운 비교 대상은 이 학생처럼 몸체 전체를 절반 해상도에서 돌리고 PixelShuffle(2)로 끝나는 x2 SR 경량망이다. SESR(16채널)의 Urban100 층당 이득은 3→5층에서 +0.205 dB, 5→7층에서 +0.08 dB, 7→11층에서 +0.058 dB로 빠르게 준다 ([SESR](https://arxiv.org/abs/2103.09404)). Bicubic++도 x3에서 몸체 2→4층이 +0.15 dB였다 ([Bicubic++](https://arxiv.org/abs/2305.02126)). ECBSR 표를 보면 이 크기 범위에서는 같은 parameter를 들일 때 폭보다 깊이가 조금 싸다. M4C16 → M10C16(parameter 2.4배)과 M10C16 → M10C32(3.9배)가 Urban100에서 똑같이 +0.47 dB였다 ([ECBSR](https://www4.comp.polyu.edu.hk/~cslzhang/paper/MM21_ECBSR.pdf)). 고전 plain denoiser는 일찍 멈췄다. DnCNN은 수용영역 35×35에 맞춰 17층에서 멈췄다. FFDNet은 절반 해상도에서 15층×64채널을 썼고, 컬러에서는 12층×96채널로 오히려 폭을 늘렸다 ([DnCNN](https://arxiv.org/abs/1608.03981); [FFDNet](https://arxiv.org/abs/1710.04026)). VDSR은 20층을 학습하는 데 residual 학습, 높은 학습률, gradient clipping이 모두 필요했다 ([VDSR](https://arxiv.org/abs/1511.04587)). 사내 측정의 2→4층 큰 개선(flat noise 3.00 → 1.99 DN, halo 5.0 → 2.6 %)은 이 곡선의 가파른 아래쪽과 맞는다. SR 표에서 외삽하면(추론, denoising 실측 아님) 32채널 8층 teacher는 4층보다 합성 +0.3~0.5 dB, 16층은 거기서 +0.2~0.3 dB를 더할 것이다. long skip 하나뿐인 32/64층 plain 몸체는 이득이 작고 최적화 위험만 커진다.

깊이를 늘린 teacher가 더 잘하는 부분의 상당수는 학생이 원리적으로 닿을 수 없다(추론). 학생의 수용영역은 full-resolution 기준 약 23 px이고, 8/16/32/64층 teacher는 약 39/71/135/263 px다(stem 3 px에 절반 해상도 3×3층마다 4 px를 더한 계산). 23 px 밖 문맥에 기대는 teacher의 판단은 4층 학생이 아무리 흉내 내도 재현되지 않는다. teacher가 학생보다 나은 이유가 넓은 문맥일수록 증류로 옮겨지는 몫은 줄어든다.

사내 측정을 한데 모으면 teacher 선택에 바로 쓸 수 있는 그림이 나온다(문헌 아님, 소유자 보유 결과).

| 모델 (사내 측정) | 합성 val PSNR | flat noise (DN) | halo (%) | chroma 잡음 비 | 미세 텍스처 보존 |
|---|---|---|---|---|---|
| half-res 몸체 2층 | – | 3.00 | 5.0 | 0.80 | 0.36 |
| half-res 몸체 4층 (배포 학생) | 35.01 dB | 1.99 | 2.6 | 0.84 | 0.46 |
| full-res 8-RepBlock (MAC 약 8배) | 35.80 dB | 2.32 | 1.5 | 0.52 | 0.46 |
| full-res 2층 ReLU | – | 2.46 | 2.7 | 0.41 (최고) | 0.45 |

배포 학생은 이와 별도로 입력을 1 px 이동하면 경계가 약 3 DN 달라진다(사내 측정). 출처는 `results/denoise_release_v1/eval/scores/fp32_summary.md`(서버 FP32, 10배 차트). full-res 두 모델(EXP-08-1, EXP-11-1)의 체크포인트는 2026-10-13 실험 산출물 정리 때 지워졌다. teacher로 쓰려면 같은 레시피로 다시 학습해야 한다.

이 표에서 teacher 선택에 대한 교훈 세 가지가 나온다. 첫째, 합성 PSNR은 teacher 품질의 대리 지표가 못 된다. +0.8 dB 모델은 halo·chroma는 낫지만 flat noise는 나쁘고 텍스처는 같다. 둘째, full-res 모델의 장점은 용량보다 귀납 편향에서 온다. 2층짜리 full-res ReLU 모델이 chroma에서 가장 좋았다는 것은 chroma가 깊이의 문제가 아니라는 뜻이다. 셋째, 같은 합성 쌍으로 학습한 teacher는 학생과 같은 합성-실기 격차를 물려받는다. teacher가 실기에서 나은 점이 있다면 그것은 데이터가 아니라 구조에서 오고, 지표마다 다르다. 그리고 출력 KD는 장점과 단점을 함께 옮긴다.

| teacher 후보 | 규모 (학생 대비, 추정) | 학생 도달성 | 특징 KD | 판정 |
|---|---|---|---|---|
| 동일 계열 8층, 32채널 | 약 78k (1.9배) | 높음 (stem·grid·tail 동일) | 블록 출력 1:1 정렬, adapter 불필요 | 새로 만든다면 최선. 다만 합성 쌍 KD의 이득은 작다 |
| 동일 계열 16층, 32채널 | 약 152k (3.7배) | 중간 (TA 크기) | 동일 | 8층보다 더 얻는 것이 적다 |
| 동일 계열 32/64층 | 약 299k / 594k (7~14배) | 낮음 | 동일 | 학습 위험에 비해 이득 없음, 비추천 |
| 사내 full-res 8-RepBlock | MAC 약 8배 | 낮음 (해상도 위상이 다름) | 해상도 불일치, 출력 KD만 | halo·chroma에 한정한 출력 target 후보. flat noise는 옮기면 안 된다 |
| 사내 full-res 2층 ReLU | 작음 | 동작이 단순 | 출력 KD만 | chroma 전용 target 후보로 가장 구체적 |
| NAFNet / Restormer / SwinIR | 11.9~26.1M (약 290~640배) | 매우 낮음 | 사실상 불가 | 비추천 |

## KD가 못 고치는 세 가지: 텍스처, 위상 의존성, 화소당 폭

첫째는 텍스처다. Blau & Michaeli는 왜곡과 지각 품질이 "at odds with each other"이고, 이 관계가 PSNR만이 아니라 "any distortion measure"에서 성립함을 보였다 ([Blau & Michaeli](https://arxiv.org/abs/1711.06077)). 같은 Charbonnier + edge + SSIM으로 학습한 teacher는 학생과 같은 평균 쪽 해에 있다. 그 출력을 목표에 섞으면 학생을 더 평균 쪽으로 끌 수는 있어도 텍스처를 더해 줄 수는 없다(추론). 사내에서 8배 MAC 모델과 학생의 미세 텍스처 보존이 0.46으로 같았던 것은, 이 한계가 용량이 아니라 손실에 있다는 직접 증거다(사내 측정). teacher의 pseudo-clean 출력이 과평활된다는 관찰도 있다. D-BSN의 출력은 "may suffer from the over-smoothing effect"였다 ([Wu et al.](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123490341.pdf)). QualiTeacher는 품질 낮은 pseudo-label을 무조건 믿으면 "forces the student model to learn undesirable artifacts"라고 경고한다 ([QualiTeacher](https://arxiv.org/abs/2603.08030)).

KD로 텍스처를 바꾼 사례는 GAN teacher에 판별기 보조 증류를 붙인 경우뿐이다. Real-ESRGAN x4의 1/4채널 학생이 FID 62.94 → 50.49로 나아졌지만 teacher(45.62)와는 4.9 FID 차이가 남았고, PSNR은 보고하지 않았다 ([SDAKD](https://arxiv.org/html/2510.03870v1)). EO 감시 영상의 hallucination 위험까지 생각하면 41k 학생에게 권할 경로가 아니다(추론). 텍스처를 움직이려면 teacher가 아니라 학생의 목적함수를 바꿔야 한다. 문헌에는 teacher 없이 쓸 수 있는 후보가 두 가지 있다. MTKD ablation에서 wavelet 손실은 L1 증류보다 약 +0.3 dB 높았다(GT만 쓰는 학생에서는 미검증) ([MTKD](https://arxiv.org/html/2404.09571v1)). CSD에서는 이득 +0.27 dB 가운데 +0.23 dB가 teacher 없이 GT를 positive로 쓴 대조 손실 변형에서 이미 나왔다 ([CSD](https://ar5iv.labs.arxiv.org/html/2105.11683)).

둘째는 위상 의존성이다. Zhang(ICML 2019)은 strided conv가 표본화 정리를 무시하며, 등변성은 stride의 정수배만큼 이동할 때만 유지된다고 정리했다 ([Zhang 2019](https://arxiv.org/abs/1904.11486)). stride-2 stem + PixelShuffle(2) 학생은 주기 2의 등변성만 가진다. 사내 측정의 "1 px 이동 시 경계 약 3 DN 변화"는 이 구조에서 오는 현상이다. 흔한 기대는 shift-등변적인 full-res teacher가 학생에게 등변성을 가르쳐 준다는 것이다. 그러나 GT 자체가 이미 완벽하게 등변이다. 입력이 1 px 움직이면 GT도 1 px 움직인다. teacher target은 GT가 주는 것 이상의 등변성 신호를 주지 않는다. 따라서 남는 위상 잔차는 구조나 최적화의 한계이고, KD로 줄어들 이유가 없다(분석).

표현력 자체가 막혀 있지는 않다(분석). 3×3 stride-2 stem은 2×2 polyphase 블록 전체를 보고, 32채널은 4위상×RGB 12값보다 많다. 따라서 선형 shift-invariant 연산은 polyphase로 정확히 표현할 수 있다. 그러나 4위상이 공유하는 절반 해상도 LeakyReLU 때문에, 비선형 동작을 위상마다 똑같이 만드는 것은 근사만 가능하다. 이를 직접 다루는 수단은 문헌에 있다. blur-pool stem은 이동 안정성을 29.0 dB에서 30.1~34.4 dB로 올리지만, 출력 total variation을 7.48에서 최저 5.84로 깎아 고주파를 잃는다 ([Zhang 2019](https://arxiv.org/abs/1904.11486)). Bicubic++에서는 DWT 다운샘플이 strided conv보다 PSNR이 +0.055~0.062 dB 높았다 ([Bicubic++](https://arxiv.org/abs/2305.02126)). sub-pixel conv의 초기화 checkerboard는 ICNR식 초기화로 없앨 수 있다 ([Aitken et al.](https://arxiv.org/abs/1707.02937)). 이동시킨 입력 쌍 사이의 출력 차이를 직접 벌하는 shift-consistency 손실은, GT 손실이 간접적으로만 주는 신호를 명시적으로 준다(추론). 큰 strided U-Net(GS-DRUNet)은 10회 translation 평균의 PSNR이 단일 추론과 같을 만큼 등변성을 스스로 배웠다 ([Renaud et al.](https://arxiv.org/abs/2412.05343)). 41k 모델도 그런지는 근거가 없다.

셋째는 학생의 화소당 폭이다. 절반 해상도 몸체의 32채널이 출력 12값(4위상×RGB)을 만들므로 출력 화소당 채널은 약 8개다. FFDNet이 컬러에서 채널을 64에서 96으로 늘린 것도 같은 맥락이다 ([FFDNet](https://arxiv.org/abs/1710.04026)). 사내에서 full-res 2층 ReLU 모델의 chroma가 0.41로 가장 좋았다는 것은, chroma가 깊이가 아니라 해상도·폭 쪽의 문제일 수 있다는 뜻이다(가설, 미검증). 이 가설에는 두 갈래가 있고 KD 판단이 여기서 갈린다. chroma 한계가 학생의 표현력(화소당 폭)에서 온다면 어떤 teacher를 써도 옮겨지지 않는다. 반대로 합성 chroma 잡음 모델이 실기와 달라서 생긴 일반화 문제이고, full-res 모델이 귀납 편향 덕에 실기 chroma를 더 잘 다루는 것이라면 이야기가 다르다. chroma 평활화는 선형에 가까운 저역 연산이라 학생도 표현할 수 있으므로, 실기 프레임 위의 출력 KD로 옮겨질 수 있다(추론). 어느 쪽인지는 아직 측정된 적이 없고, 이 미지수가 이 보고서 권고의 핵심 전제다.

teacher의 실패도 함께 옮겨진다. teacher가 못 지운 halo나 sharpening 잔차, 색 이동, hot pixel 처리 습관은 그대로 학생의 목표가 된다(추론). full-res 8-RepBlock 모델을 통째로 teacher로 쓰면 halo·chroma의 장점과 함께 flat noise 2.32 DN이라는 단점도 학생에게 간다(사내 측정에 근거한 추론).

## ECB 재파라미터화와 W16A16은 순서만 지키면 걸림돌이 아니다

재파라미터화와 KD는 충돌하지 않는다. challenge 실무는 일관되게 확장된 학습 그래프에서 증류하고, 수렴한 뒤 fold한다. DIPNet은 RRFB로 증류한 뒤 RLFB로 되돌리고 가지치기했다 ([DIPNet](https://arxiv.org/abs/2304.07018)). NTIRE 2024의 CMVG·HiSR, NTIRE 2023의 RUNet, NTIRE 2025의 mbga가 모두 reparam 블록과 KD를 함께 썼다 ([NTIRE 2024 ESR](https://arxiv.org/abs/2404.10343); [NTIRE 2025 ESR](https://arxiv.org/html/2504.10686v1)). 조사한 논문 가운데 KD가 reparam 학생을 해쳤다는 보고는 없었다. ECB fold는 정확한 선형 합성이므로 블록 출력이 fold 전후로 같다. 따라서 출력 KD든 블록 출력에 거는 특징 KD든 목표가 fold 때문에 바뀌지 않는다(분석). 참고로 ECB 자체의 이득은 ECBSR M4C16에서 plain 대비 Set5/Urban100 +0.09/+0.16 dB였다 ([ECBSR](https://www4.comp.polyu.edu.hk/~cslzhang/paper/MM21_ECBSR.pdf)). 이 학생에게 기대되는 KD 이득과 같은 크기이고, 이미 확보한 몫이다.

특징 KD는 이 학생에게 비용 대비 이득이 나쁘다. FitNet식 회귀기는 SR에서 최대 -1.50 dB였다 ([DUKD](https://arxiv.org/html/2309.14162v4)). MiPKD는 FAKD가 깊이 압축에서는 돕고 폭 압축에서는 해친다며, 기존 기법이 특정 teacher-student 구조에 묶여 있다고 지적한다 ([MiPKD](https://arxiv.org/html/2404.02573)). DIPNet의 4지점 특징 KD는 243k 학생에서 +0.05 dB였다 ([DIPNet](https://arxiv.org/abs/2304.07018)). 동일 계열 8/16층 teacher라면 stem·grid·채널이 같아 블록 출력을 adapter 없이 1:1로 맞출 수 있다. 하지만 4층 학생의 어느 층을 teacher의 어느 층에 맞출지는 임의로 정해야 한다. full-res teacher의 특징을 절반 해상도 학생에 맞출 때는 PixelUnshuffle(2) + 1×1 adapter가 정보를 잃지 않는 유일한 선택이다(추론). 그러나 이를 시험한 저수준 논문은 찾지 못했다. pooling adapter는 PixelShuffle이 만들어야 할 위상별 고주파를 정확히 버린다. 채널 수가 다르면 특징 KD를 아예 생략한 팀(BlingBling)도 있었다 ([NTIRE 2024 ESR](https://arxiv.org/abs/2404.10343)).

W16A16에는 KD가 회복할 손실이 거의 없다. Qualcomm QuickSRNet은 AIMET W8A8에서 per-channel QAT로 FP16 대비 0.04~0.10 dB 이내까지 갔고, input-to-output 연결이 양자화 견고성을 크게 높인다고 보고했다 ([QuickSRNet](https://arxiv.org/abs/2303.04336)). MAI 2022 주최 측은 대부분의 경우 FP32 → INT8 손실이 0.1 dB 미만이라고 적었다 ([MAI 2022](https://arxiv.org/abs/2211.05910)). 이 학생은 global input residual을 가진 16비트 모델이므로 손실은 더 작을 것이다. 사내 기록에도 W16 QAT 학습이 사실상 흔적기관이었다고 남아 있다(사내 기록). 양자화 증류가 큰 효과(+0.43~+0.95 dB)를 낸 2DQuant는 2~4비트의 큰 하락을 메운 경우다 ([2DQuant](https://arxiv.org/html/2406.06649v1)).

순서는 문헌이 정해 준다. QDR은 QAT와 KD를 함께 돌리면 재구성 손실과 증류 손실이 양자화 잡음 속에서 "tug-of-war"를 벌인다고 경고한다 ([QDR](https://arxiv.org/abs/2603.09624)). Nguyen et al.은 teacher 증류를 float 단계에 두고, QAT는 fold된 deploy 그래프에서 BN 재보정·weight clipping과 함께 했다 ([Nguyen et al.](https://arxiv.org/html/2604.20291v1)). 이 프로젝트에 대입하면 float의 ECB 확장 그래프에서 증류하고, fold한 뒤, 기존 W16A16 변환 경로를 그대로 탄다. 배포 그래프와 HTP 변환은 바뀌지 않는다.

손실 가중치의 관례도 정해져 있다. GT 손실을 주 손실로 두고 KD를 작게 얹는 것이 표준이다. DSCLoRA는 재구성 1, pixel KD 0.1, affinity KD 0.01을 썼고, Nguyen et al.은 KD 0.03에 teacher 오차 기반 픽셀 가중치를 붙였다 ([DSCLoRA](https://arxiv.org/abs/2504.11271); [Nguyen et al.](https://arxiv.org/html/2604.20291v1)). teacher를 GT보다 9배 무겁게 둔 Kayani의 설정은 GT에 잔류 센서 잡음이 있다는 근거로만 정당화됐다 ([Kayani et al.](https://arxiv.org/html/2605.03680v1)). teacher는 거의 모든 사례에서 고정(frozen)이다. 예외인 mbga는 학생에 SPAB 3개를 더한 teacher를 함께 학습한 뒤, 추가 블록을 점진적으로 제거해 학생 구조로 수렴시켰다 ([NTIRE 2025 ESR](https://arxiv.org/html/2504.10686v1)).

## 남는 길은 실기 프레임 pseudo-label이며 teacher는 실기 차트로 고른다

합성 GT가 이미 완벽한 teacher 역할을 하는 이상, 새 정보가 들어올 수 있는 곳은 GT가 없는 실기 EO 프레임뿐이다. 선례는 있다. Wu et al.(ECCV 2020)은 (합성 잡음, 실제 clean) 쌍에 λ = 0.1 가중의 (실제 잡음, teacher pseudo-clean) 쌍을 더했다. 실제 쌍 없이 DND sRGB 37.93 dB를 얻어 CBDNet(38.05 dB)에 0.12 dB까지 다가갔고, 두 쌍이 "complementary"라고 했다 ([Wu et al.](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123490341.pdf)). 다만 그 학생이 teacher를 0.5~1.0 dB 넘은 것은 teacher가 중심 화소를 못 보는 blind-spot 구조라는 핸디캡을 안고 있었기 때문이다. 큰 teacher의 pseudo-label이 작은 학생을 GT 학습 이상으로 끌어올린다는 증거는 아니다(추론).

위험도 문서화돼 있다. mean-teacher식 반복은 확증 편향을 만들고 L1이 틀린 라벨에 과적합한다 ([Semi-UIR](https://openaccess.thecvf.com/content/CVPR2023/html/Huang_Contrastive_Semi-Supervised_Learning_for_Underwater_Image_Restoration_via_Reliable_Bank_CVPR_2023_paper.html)). NR-IQA로 신뢰도를 매기면 학생이 IQA를 "reward hacking"하는 문제가 생긴다 ([QualiTeacher](https://arxiv.org/abs/2603.08030)). 회귀 KD에서는 teacher 오차가 클 때 모방을 약하게 하는 가중이 표준적인 안전장치다 ([Saputra et al.](https://arxiv.org/abs/1908.00858)). 성공한 방법은 모두 같은 안전장치를 썼다. GT 쌍을 주 손실로 유지하고, pseudo 쌍의 가중치를 낮추고, 반복 없이 한 번만 라벨링하거나 신뢰 bank를 두었다.

teacher를 더 안전하게 쓰는 방법도 있다. Pseudo-ISP는 사전학습 denoiser의 pseudo-clean 출력을 직접 목표로 쓰지 않았다. 대신 (실제 입력, pseudo-clean) 쌍으로 pseudo ISP와 신호 의존 잡음 모델을 학습하고, 그것으로 clean 영상에서 새 잡음 입력을 합성했다 ([Pseudo-ISP](https://arxiv.org/abs/2103.10234)). 이 프로젝트로 옮기면, teacher의 실기 출력으로 측정 카메라 모델(halo 형상, chroma 잡음, hot pixel 통계)을 점검·재보정하고 학습은 계속 진짜 clean GT로 하는 형태가 된다. 이렇게 하면 teacher의 오류는 잡음 모델 parameter에만 영향을 주고, 학생 목표의 텍스처는 오염시키지 않는다(추론). 엄밀히는 KD가 아니지만, 합성-실기 격차가 이미 알려진 문제인 만큼 teacher를 만든 김에 얻을 수 있는 가장 값싼 부산물이다.

| 선택지 (teacher × KD 변형) | 문헌상 기대 효과 | 이 학생에 대한 판정 |
|---|---|---|
| 동일 계열 8/16층 → 합성 쌍 출력 KD | 0~+0.1 dB, seed 잡음 수준 | 기술적으로 쉽지만 실기 개선 기대 낮음 |
| 동일 계열 → 특징 KD | ±0.1 dB, FitNet은 -1.5 dB까지 | 비추천 |
| teacher 강화 GT (DIPNet식) | +0.06 dB, 텍스처 손실 위험 | source GT에 잔류 잡음·압축 흔적이 있을 때만 |
| data upcycling (증강 입력에 teacher 라벨) | SR에서 +0.24~+0.58 dB | 합성 생성기가 이미 무한한 GT를 주므로 중복 |
| 32/64층, NAFNet·Restormer·SwinIR teacher | 격차 28배 이상에서 직접 KD는 음수 위험 | 비추천 |
| GAN/perceptual teacher + 적대적 KD | 텍스처 개선, teacher와 격차 남음 | EO hallucination 위험, 비추천 |
| 양자화 KD (W16) | 회복할 손실 없음 | 비추천 |
| 실기 프레임 pseudo-label, chroma 한정 target, full-res teacher | 직접 근거 없음 (Wu et al. λ = 0.1 선례) | 가장 그럴듯한 유일한 KD, 조건부 |
| teacher 실기 출력으로 카메라 모델 재보정 (Pseudo-ISP식) | 정량 근거 미확보 | KD는 아니지만 가장 안전한 teacher 활용 |

권고는 이렇다. 32/64층 동일 계열 teacher와 NAFNet·Restormer·SwinIR급 teacher는 만들지 않는다. 격차가 28배를 넘으면 직접 KD가 해로워진다는 결과와, 출력 KD에서는 teacher 계열이 무관하다는 결과가 함께 이 teacher들의 존재 이유를 없앤다. 합성 쌍 위의 출력·특징 KD는 기대값이 seed 잡음 수준이므로, 실기 개선이 목적이라면 하지 않는다.

KD를 한 번 시도한다면 가장 그럴듯한 형태는 축 선택형 pseudo-label KD다. teacher는 사내에서 측정한 full-res 구조, 즉 chroma 0.41인 2층 ReLU 모델이나 halo 1.5 %·chroma 0.52인 8-RepBlock 모델이다. 두 체크포인트는 지워졌으므로 같은 레시피로 다시 학습해야 한다. GT 없는 실기 EO 프레임에서 chroma 성분에만 teacher 목표를 건다. 합성 GT 손실은 주 손실로 유지하고, pseudo 항은 Wu et al.의 λ ≈ 0.1 수준으로 낮게 둔다. 라벨링은 오프라인에서 한 번으로 끝낸다. 휘도(luma)는 teacher를 따르지 않게 해서 full-res 모델의 나쁜 flat noise가 옮겨오지 않게 한다. 이 설계는 어떤 논문도 시험하지 않은 추론이다. 가장 가까운 근거는 MTKD가 보인 multi-teacher 집계의 이득 정도다 ([MTKD](https://arxiv.org/html/2404.09571v1)).

이 설계에는 전제 조건이 두 가지 있다. 하나는 teacher가 실기 차트에서 그 지표가 실제로 낫다는 것이다. 2층 ReLU 모델은 chroma가 0.41로 가장 좋지만 flat noise는 2.46 DN으로 학생(1.99)보다 나쁘고, halo는 2.7 %로 학생(2.6 %)과 비슷하다(사내 측정). luma를 teacher에 묶지 않아야 하는 이유가 이것이다. 다른 하나는 앞 절에서 본 chroma 한계가 표현력이 아니라 일반화의 문제라는 것이다. 새 teacher를 꼭 만들어야 한다면 학생이 닿을 수 있는 동일 계열 8층(최대 16층) 32채널이 맞다. 그리고 증류하기 전에 그 teacher가 실기 차트에서 학생보다 나은지부터 확인해야 한다.

같은 노력을 들일 다른 곳과도 비교해야 한다. 지연 예산이 허락한다면 몸체 층을 하나 더하는 쪽이 더 확실하다. SESR의 얕은 구간에서는 층 하나가 Urban100 약 +0.2 dB로 초소형 학생의 어떤 KD 이득보다 컸고, 사내 2→4층 결과도 같은 방향이다 ([SESR](https://arxiv.org/abs/2103.09404)). 텍스처에는 주파수·대조 손실이, 위상 잔차에는 DWT/blur-pool stem이나 shift-consistency 손실이 KD보다 직접적인 수단이다.

어떤 경로든 판정 장치를 먼저 갖춰야 한다. 기대 효과가 0.1 dB 이하라면 seed 간 변동과 같은 크기다. 그러니 GT-only baseline을 여러 seed로 학습해 잡음 바닥을 먼저 재고, 판정은 합성 val PSNR이 아니라 실기 차트 지표와 보드 화면으로 해야 한다. 근거는 두 가지다. 합성 잡음에서 잘하던 기법이 실제 사진에서는 BM3D에 밀렸고 ([Plötz & Roth](https://arxiv.org/abs/1707.01313)), teacher에 더 가까워진다고 일반화가 나아지지는 않는다 ([Stanton et al.](https://arxiv.org/abs/2106.05945)). pseudo-label을 쓴다면 라벨 풀에 없는 실기 장면으로 평가해야 한다. 그렇지 않으면 학생은 품질이 아니라 teacher 충실도로 채점된다. 사내 기록에 남은 두 가지 사실은 KD에 대한 반대 증거가 아니라 점검 항목으로 읽어야 한다. 사내 차트 채점기의 첫 판(P1)은 잡음에 점수가 부풀려지는 편향을 보였고, 이는 P1.1에서 고쳤다(완벽한 denoiser 오라클 시험 추가). 2026년 6월의 KD 시도는 세 가지 이유로 무효였다: 존재하지 않는 데이터 설정으로 학습된 teacher, 사실상 학습되지 않은 학생, 항등 함수에 만점을 준 채점기. 따라서 다음 시도에서는 teacher의 학습 설정, 같은 schedule로 학습한 GT-only 학생 대비 수렴, 채점기가 항등 출력을 낮게 매기는지를 먼저 확인해야 한다.

## 결론

이 검토로 질문의 축이 바뀐다. "더 강한 teacher"는 이 학생에게 의미 있는 변수가 아니다. 합성 영역에서는 정확한 clean GT가 이미 가장 좋은 teacher다. teacher가 가진 여분의 용량은 대부분 학생이 닿지 못하는 문맥(23 px 밖)과, 손실 함수 때문에 학생이 만들지 못하는 텍스처에 쓰인다. teacher의 가치를 정하는 것은 크기가 아니라 어떤 데이터에 라벨을 다는지, 그리고 실기에서 어느 지표가 나은지다. 그래서 이 프로젝트에서 KD는 압축 도구라기보다 합성-실기 격차를 건너는 도구로 다시 정의된다.

사내 측정은 학생의 실기 한계가 용량보다 귀납 편향(해상도와 위상)과 손실 함수에 있음을 가리킨다. 그 둘은 KD보다 구조와 손실 변경으로 더 직접 다룰 수 있다. KD에 남는 좁은 역할은, 사내에서 측정한 full-res 구조의 특정 실기 동작(chroma)을 배포 가능한 4층 half-res 구조로 옮기는 것이다. 그 가능성은 하나의 미지수, 즉 chroma 한계가 표현력의 문제인지 일반화의 문제인지에 달려 있다.
