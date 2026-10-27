# Teacher choice, teacher-student capacity gap, and depth/width scaling for a tiny plain-CNN denoiser (KD feasibility, literature to 2026)

Scope note. The student under review is a Bicubic++-style same-resolution denoiser of about 41k parameters. Its layers are a stride-2 3x3 stem (3->32), 4 plain 3x3 32->32 convolutions at half resolution with LeakyReLU(0.1) and ECB reparameterization, one long skip, a 3x3 conv to 12 channels, PixelShuffle(2), and a global input residual. My own arithmetic (inference, not from a paper) gives about 0.9k + 36.9k + 3.5k = 41.2k weights, about 10.3k MAC per full-resolution output pixel, and a receptive field of about 23x23 full-res pixels (3 + 5 half-res 3x3 layers x 4 px). With 8/16/32/64 body layers the receptive field grows to roughly 39/71/135/263 px, and the parameters to about 78k/152k/299k/594k.

Structural analogy used below (inference). Like ECBSR, SESR and FSRCNN at x2 SR, the student runs its whole body at half the output resolution and ends in PixelShuffle(2). Their x2 depth/width tables are therefore the closest published scaling data for this student, but they measure SR and not denoising.

## Q1. Does a much larger or deeper teacher distil better or worse into a tiny student? (capacity-gap literature and whether it transfers to regression/low-level tasks)

### Takeaway
In classification, the evidence agrees that student accuracy first rises and then falls as the teacher grows. The cause is that the student cannot reach the teacher's solution, not that the teacher is wrong. Whether teacher assistants or early stopping fix this is disputed. In SR, plain output (pixel) KD from a big teacher gives about 0 dB, and feature KD often hurts. When papers do report gains in low-level KD, the gains come from extra or cleaner targets, a "student-reachable" teacher, or dedicated losses, not from teacher size. No paper sweeps teacher size for a ~40k-parameter denoiser.

### Cited Findings
**Classification capacity-gap results**
- TAKD (Mirzadeh et al., AAAI 2020) trains a plain-CNN student with 2 conv layers using "similar but larger teachers of size 4, 6, 8, and 10" on CIFAR-10/100. As the teacher grows its own accuracy rises, but "the trained student accuracy first increases and then decreases." — [TAKD, arXiv 1902.03393](https://arxiv.org/abs/1902.03393)
- TAKD names three competing factors: (1) a better teacher gives better supervision; (2) "the student does not have the sufficient capacity or mechanics to mimic her behavior"; (3) the teacher's logits become less soft. Factor 3 is specific to softmax classification. — [TAKD](https://arxiv.org/abs/1902.03393)
- TAKD Table 1, NOKD / BLKD / TAKD:
  - plain CNN, S=2, TA=4, T=10: CIFAR-10 70.16 / 72.57 / 73.51; CIFAR-100 41.09 / 44.57 / 44.92.
  - ResNet, S=8, TA=20, T=110: CIFAR-10 88.52 / 88.65 / 88.98.
  - ResNet-14 student, ImageNet: 65.20 / 66.60 / 67.36.

  In Table 2 (S=2, T=10), TA=8/6/4 gives 72.75/73.15/73.51 on CIFAR-10. The best TA is the one whose accuracy is "close to the average performance of teacher and student", not the one midway in size. — [TAKD](https://arxiv.org/abs/1902.03393)
- TAKD also finds that for a fixed size, "a better network (in terms of accuracy) is a better teacher" (teachers trained for 5/10/15/100 epochs). Distilled TAs beat TAs trained from scratch. — [TAKD](https://arxiv.org/abs/1902.03393)
- Cho & Hariharan (ICCV 2019) state that "larger models do not often make better teachers", attribute this to "mismatched capacity", and say "small students are unable to mimic large teachers."
  - ImageNet, ResNet-18 student: scratch 30.24% error; teachers ResNet-18/34/50 (teacher error 30.24/26.70/23.85) give student error 30.57/30.79/30.95.
  - With early-stopped KD (ESKD) the student error is 29.01/29.16/29.35. Larger teachers are still worse, and test KD loss still rises with teacher size.

  — [On the Efficacy of KD, arXiv 1910.01348](https://arxiv.org/abs/1910.01348)
- Cho & Hariharan, CIFAR-10 test KD error (student-teacher disagreement):
  - WRN16-1 student: rises from 6.32% (WRN16-3 teacher) to 7.01% (WRN16-8).
  - WRN28-1 student: rises from 4.05% (WRN28-3) to 4.81% (WRN28-8).

  — [Cho & Hariharan](https://arxiv.org/abs/1910.01348)
- Contradiction with TAKD: Cho & Hariharan find that step-wise distillation (WRN16-8 -> WRN16-3 -> WRN16-1) "performs almost exactly the same as directly using the large model". Their conclusion is that "the solution the large teacher has found is simply not in the solution space of the small student. The only solution is to find a teacher whose discovered solution is in fact reachable by the student." — [Cho & Hariharan](https://arxiv.org/abs/1910.01348); contradicts [TAKD](https://arxiv.org/abs/1902.03393)
- DIST (Huang et al., NeurIPS 2022), Table 3. ResNet-18 student (scratch 69.76%) with vanilla KD:
  - ResNet-34 teacher -> 71.21%
  - ResNet-50 -> 71.35%
  - ResNet-101 -> 71.09%
  - ResNet-152 -> 71.12%

  With a correlation-based relational loss (Pearson correlation) the trend becomes monotonic: 72.07 / 72.12 / 72.08 / 72.24. The authors attribute the problem to the "discrepancy of predictions between the student and a stronger teacher", and say "the representational capacity limits the student's performance". — [DIST, arXiv 2205.10536](https://arxiv.org/abs/2205.10536)
- Beyer et al. (CVPR 2022) show that a big gap can be bridged with "function matching": the same augmented view goes to teacher and student, mixup is aggressive, and training is very long. A ResNet-50 student of BiT-M-R152x2 reaches 82.8% ImageNet top-1 after 9600 distillation epochs. Fixed-teacher (precomputed-target) setups "achieve significantly lower distillation loss" but overfit and generalize worse. — [A good teacher is patient and consistent, arXiv 2106.05237](https://arxiv.org/abs/2106.05237)
- Stanton et al. (NeurIPS 2021) find that student-teacher fidelity is low "even in cases when the student has the capacity to perfectly match the teacher". They identify "difficulties in optimization as a key reason". — [Does KD Really Work?, arXiv 2106.05945](https://arxiv.org/abs/2106.05945)
- Two papers report that a weaker or less optimized teacher can be better:
  - Yuan et al. (CVPR 2020): "a poorly-trained teacher with much lower accuracy than the student can still improve the latter significantly". — [Revisiting KD via Label Smoothing Regularization, arXiv 1909.11723](https://arxiv.org/abs/1909.11723)
  - SFTN (Park et al., NeurIPS 2021): teacher and student accuracy are "rather inversely correlated" in their ablation, and "a loosely optimized teacher model may be more effective". — [Learning Student-Friendly Teacher Networks, arXiv 2102.07650](https://arxiv.org/abs/2102.07650)
- A 2026 single-author CIFAR-10 re-study with a strict protocol (validation-selected, 5 seeds) found that no KD gain for ResNet-18 students is distinguishable from zero. The only significant gains were +0.19/+0.21 pp for ResNet-34 students. The student stem fix was "worth +5.5 to +7.2 pp — more than 25x the largest KD gain". This is low-weight evidence, but it is a caution that classic KD gains can be protocol artefacts. — [Student Capacity Moderates KD Effectiveness, arXiv 2605.31191 (2026)](https://arxiv.org/abs/2605.31191)

**Low-level (SR, denoising) evidence**
- DUKD argues that dark knowledge barely exists in pixel regression: "there is barely such benefit in SR tasks that reconstructs image pixels", because teacher outputs are "noisy and inaccurate approximations" of the HR ground truth, which "shades" the teacher's information. — [Data Upcycling KD for SR, arXiv 2309.14162](https://arxiv.org/abs/2309.14162)
- DUKD Table 2, EDSR x4 (teacher c256b32 at 43.09M, student c64b32 at 2.70M), Urban100 PSNR:
  - Scratch 26.21, logits-KD 26.21
  - FitNet 25.40, AT 25.97, RKD 26.20, FAKD 26.18
  - CSD 26.34, DUKD 26.45

  At x2 the same pattern holds (scratch 31.96, KD 31.98, FitNet 30.46, DUKD 32.53). — [DUKD](https://arxiv.org/abs/2309.14162)
- MiPKD (Li et al., ICLR 2025) states that before their work "none of the existing SRKD methods have discussed ... whether a stronger teacher consistently benefits the capacity-limited student". At about 28x compression (EDSR c256b32 -> c64b16, x4), Urban100 PSNR:
  - Teacher 26.87, NOKD 25.61, KD 25.60, FitNet 25.52, FAKD 25.38
  - Two-stage TA pipelines: FAKD+CSD 25.79; MiPKD+MiPKD 25.89 (best)

  The authors conclude that "Directly distilling the student model by the teacher model yields negative effects on its performance. Two-stage KD with an intermediate teaching-assistant (TA) model [is] necessary in such case." — [MiPKD, arXiv 2404.02573](https://arxiv.org/abs/2404.02573)
- MiPKD Fig. 1 (text) also reports that in depth compression "there are barely KD methods outperforming vanilla logits-KD". For compounded depth+width compression, "almost all KD underperforms training without KD." — [MiPKD](https://arxiv.org/abs/2404.02573)
- PISR (Lee et al., ECCV 2020) uses a 13K-parameter FSRCNN student, x2, Set5. A teacher capped by an hourglass bottleneck (teacher PSNR 37.70) gives a better student (37.22) than an unconstrained HR-input teacher with 57.60 dB (student 37.19). The authors explain that the hourglass "limits the performance of the teacher ... reducing the performance gap between the teacher and the student". The full method gains only +0.18 dB Set5 over the 37.15 baseline (Urban100 30.05 -> 30.24). — [PISR, arXiv 2007.07524](https://arxiv.org/abs/2007.07524)
- Instance Data Condensation (2025) ran a cross-architecture teacher/student study (EDSR, SwinIR, MambaIRv2). It found "a stronger teacher does not always yield a superior student; in some cases, a student model can even outperform another student trained with a more powerful teacher", and interprets teacher outputs as "a form of regularization". The detailed table is in their supplementary, which I did not read. — [IDC, arXiv 2505.21099](https://arxiv.org/abs/2505.21099)
- Real-image denoising for mobile NPUs (CVPRW/MAI 2026):
  - Setup: a 41.6M-parameter dense CNN teacher (37.71 dB) and a 1.96M U-Net student with 3x3 conv, ReLU and nearest upsampling.
  - Result: the student goes from 36.08 dB without KD to 37.66 dB with KD (gap 1.63 -> 0.05 dB). The paired gap is 0.08 dB (95% CI [-0.00, 0.16]).
  - Loss weights: lambda_gt = 100 on L2-to-GT, lambda_distill = 900 on L2-to-teacher, lambda_L1 = 50 on L1-to-GT. The heavy teacher weight is meant "to prevent the student from overfitting to residual sensor noise present in the raw ground truth data".
  - Confounds: the student was also fine-tuned with progressive crops up to 1024x1024, so the gain is not a clean KD ablation. — [Real Image Denoising with KD for Mobile NPUs, arXiv 2605.03680](https://arxiv.org/abs/2605.03680)

### Inferences
- **What transfers to pixel regression.** Of TAKD's three factors, only the mimicability factor (2) carries over to L1/L2 pixel losses. The temperature/softness factor (3) and the inter-class dark knowledge DIST exploits have no direct analogue. With clean GT, an output-KD target E_teacher[x|y] differs from GT mainly by the teacher's error. That is why output KD in SR comes out at about 0 dB (DUKD, MiPKD).
- **When KD should help this denoiser.**
  - (a) GT targets are noisy or imperfect, e.g. real camera pairs. The teacher then acts as a lower-variance target (MAI 2026).
  - (b) Teacher pseudo-labels on extra unlabeled inputs, such as real camera frames without GT, augmented or upcycled crops, or long function-matching schedules (DUKD, IDC, Beyer).
  - In neither case does the gain come from teacher size itself.
- **Capacity gap for this student.** The student is about 41k parameters. A 41.6M-parameter teacher would be a ~1000x gap, far beyond the ~28x where MiPKD already sees direct KD hurt. Classification and SR evidence both predict that the largest teacher is not the best one for this student. The owner's own data already shows the 8x-MAC full-resolution model is not uniformly better on real charts, so that model's synthetic +0.8 dB is not evidence that it is a good teacher.
- **Recommended teacher shape.** A teacher "reachable by the student" (Cho & Hariharan; PISR's hourglass-capped teacher) is the safer default. In this family that means a same-family 8- or 16-layer body at 32 channels, at most about 2-4x the student's parameters. The TAKD heuristic (pick a TA whose accuracy sits between student and teacher) can be applied directly to the 4/8/16 ladder.

### Gaps
- I found no SR or denoising paper that sweeps teacher depth or size for a fixed tiny (<100k-parameter) student and reports the student's PSNR curve. TAKD's plain-CNN 2-vs-4/6/8/10 sweep is the closest structural analogue, and it is classification.
- TAKD Figure 2 values (student accuracy per teacher size) appear only as a plot. I could not extract the numbers.
- I could not read the IDC supplementary, so its cross-teacher table numbers are missing.
- The MAI 2026 KD gain is confounded by training-schedule changes, and the paper is a challenge-workshop report.

## Q2. Where does depth scaling saturate for plain CNN denoisers/SR nets, and is width or depth the better lever for small budgets?

### Takeaway
For half-resolution-body plain CNNs at x2 SR (the closest analogue), most of the depth gain comes in the first few layers. Each added layer is worth about 0.2 dB (Urban100) going from 3 to 5 layers at 16 channels, then about 0.06 dB per layer beyond 7. Classic plain denoisers settle at 15-20 layers at 64-96 channels. At equal parameters, ECBSR data mildly favour depth over width in the 10-100k range. Width matters more for colour images.

### Cited Findings
**VDSR, DnCNN, FFDNet (depth choices)**
- VDSR (Kim et al., CVPR 2016) trains plain 3x3, 64-channel networks of depth 5 to 20 and states "In most cases, performance increases as depth increases". They chose depth 20. SRCNN failed to gain beyond 3 layers, which VDSR attributes to unconverged training (learning rate 1e-5). VDSR needed residual learning and high learning rates with gradient clipping to train at depth 20. The figure values are not in the text. — [VDSR, arXiv 1511.04587](https://arxiv.org/abs/1511.04587)
- DnCNN (Zhang et al., IEEE TIP 2017) sets depth from receptive field. It uses a 35x35 receptive field, "the corresponding depth of 17", for Gaussian denoising at a known noise level, chosen to match EPLL's effective 36x36 patch size. It uses depth 20 for general or blind tasks. — [DnCNN, arXiv 1608.03981](https://arxiv.org/abs/1608.03981)
- FFDNet (Zhang et al., IEEE TIP 2018) is a half-resolution plain denoiser. It runs on four downsampled sub-images ("reversible downsampling"), so "it is not necessary to employ the dilated convolution to further increase the receptive field". It uses 15 layers x 64 channels for grayscale and 12 layers x 96 channels for colour: fewer layers but wider for colour. — [FFDNet, arXiv 1710.04026](https://arxiv.org/abs/1710.04026)

**ECBSR, SESR, Bicubic++, PlainUSR (depth and width tables)**
- ECBSR (Zhang et al., ACM MM 2021), Table 1, x2, Y-channel, M = number of ECB blocks, C = channels:

  | Model | Params | Set5 | Urban100 |
  |---|---|---|---|
  | M4C8 | 2.80K | 36.93 | 29.68 |
  | M4C16 | 10.20K | 37.33 | 30.31 |
  | M10C16 | 24.30K | 37.55 | 30.78 |
  | M10C32 | 94.70K | 37.76 | 31.25 |
  | M16C64 | 596.00K | 37.90 | 31.71 |

  At x4: M4C8 30.52 -> M16C64 31.92 Set5. — [ECBSR (ACM MM 2021)](https://www4.comp.polyu.edu.hk/~cslzhang/paper/MM21_ECBSR.pdf)
- ECBSR Table 3 (M4C16, x2): the full ECB (expand-squeeze + Sobel + Laplacian branches) gives 37.33/30.31 Set5/Urban100. A plain-conv baseline gives 37.24/30.15, so reparameterization adds about +0.09/+0.16 dB. RepVGG-block, AC-block and DB-block reach only 37.24-37.29. — [ECBSR](https://www4.comp.polyu.edu.hk/~cslzhang/paper/MM21_ECBSR.pdf)
- SESR (Bhardwaj et al., MLSys 2022), x2, f = channels, m = layers:

  | Model | Params | Set5 | Urban100 |
  |---|---|---|---|
  | M3 (f=16) | 8.91K | 37.21 | 29.92 |
  | M5 (f=16) | 13.52K | 37.39 | 30.33 |
  | M7 (f=16) | 18.12K | 37.47 | 30.49 |
  | M11 (f=16) | 27.34K | 37.58 | 30.72 |
  | XL (f=32, m=11) | 105.37K | 37.77 | 31.16 |

  The authors say SESR-M11 reaches "very similar or better PSNR than the VDSR network, which has 97x more MACs". — [SESR, arXiv 2103.09404](https://arxiv.org/abs/2103.09404)
- Bicubic++ (Bayazit et al., CVPRW 2023), Table 1. Setting: x3 SR, 32 channels, strided-conv downsampling, LeakyReLU, Y-PSNR on 48 DIV2K validation images.

  | Change | Y-PSNR | Runtime |
  |---|---|---|
  | m=2, R=1 | 30.282 dB | 3.58 ms |
  | m=4, R=1 | 30.433 dB | 4.84 ms |
  | m=2, R=2 | 30.410 dB | — |
  | Bicubic | 29.334 dB | — |
  | ReLU -> LeakyReLU (T->A) | +0.011 dB | — |
  | Channels 24 -> 32 (Q->T) | +0.114 dB | — |

  The authors kept m=2 because the challenge score weights runtime. By PSNR, DWT downsampling beat strided conv (30.219 vs 30.157 at 24 ch; 30.326 vs 30.271 at 32 ch). Strided conv won on score, not PSNR. — [Bicubic++, arXiv 2305.02126](https://arxiv.org/abs/2305.02126)
- PlainUSR (arXiv 2024), Table 4, x2. It lists an "ECBSR-M4C32" of 41.5K parameters, the student's size class, at 37.39 Set5 / 30.69 Urban100. ECBSR-M6C40 (92.4K) is listed at 37.61/31.09. Its own U-Net-style (multi-scale) plain models give: PlainUSR-T 48.7K 37.47/30.53; -S 57.2K 37.64/31.06; -M 148.9K 37.79/31.53; -L 707.1K 38.07/32.53.

  Caveat: PlainUSR's ECBSR-M4C16 row (37.18/30.18, 11.5K) is 0.13-0.15 dB below ECBSR's own M4C16 numbers, while its M16C64 row equals the original. Its ECBSR rows are therefore not fully consistent with the source. — [PlainUSR, arXiv 2409.13435](https://arxiv.org/abs/2409.13435)

### Inferences
- **Diminishing returns per layer (SESR, f=16, x2).** Urban100 gains per added layer are +0.205 dB (3->5), +0.08 dB (5->7) and +0.058 dB (7->11). Set5 gains are +0.09, +0.04 and +0.028 dB. Bicubic++ shows 2->4 body layers is worth +0.15 dB (x3), about 16% of the model's whole gain over bicubic. This is consistent with the owner's 2->4 jump being large on real charts, and it predicts much smaller gains from 4->8 and very small gains beyond 16.
- **Depth vs width at equal parameters (ECBSR, x2).**
  - From M4C16, +2.4x params by depth (M10C16) gives +0.22 Set5 / +0.47 Urban100.
  - From M10C16, +3.9x params by width (M10C32) gives +0.21 / +0.47.
  - So per parameter, depth was cheaper in this range.
  - Cross-source: M10C32 (94.7K, ECBSR paper) 37.76/31.25 vs M6C40 (92.4K, PlainUSR table) 37.61/31.09 also favours depth at equal parameters. The caveat is that the two numbers come from different papers.
  - On hardware, width at 32 channels is often "free" up to a vector-width sweet spot. Bicubic++ Fig. 4 found runtime "sweet spots" by channel count, so the parameter-optimal lever may not be the latency-optimal one.
- **Rough expectations for same-family teachers at 32 channels (my extrapolation from SR x2 tables, not measured for denoising).**
  - 8 body layers: about +0.3-0.5 dB synthetic over 4 layers.
  - 16 layers: a further +0.2-0.3 dB.
  - 32 or 64 plain layers with only one long skip: diminishing returns and growing optimization risk. VDSR needed residual learning, gradient clipping and high learning rates at 20 layers. DnCNN/FFDNet stopped at 12-20 layers with 64-96 channels.
  - A 64-layer 32-channel plain body is very unlikely to be worth training as a teacher.
- **Receptive field and the chroma result.** For denoising, receptive field (DnCNN's criterion) matters, as does colour width (FFDNet uses 96 channels for colour). The student's ~23 px receptive field is below DnCNN's 35 px. The 4-layer half-res body also has only 32 channels for 12 outputs (4 phases x RGB), about 8 channels per output pixel. My hypothesis is that this per-pixel width shortfall, rather than depth, explains why a shallow full-resolution model removed chroma noise better. This is untested.

### Gaps
- I found no published depth/width ablation for a plain CNN denoiser in the 10-100k-parameter range. All small-model tables are SR (x2-x4) on synthetic bicubic data.
- VDSR's Figure 3 depth-vs-PSNR values were not extractable from text.
- I did not verify IRCNN's or RepVGG-style restorers' depth ablations separately.
- No paper relates synthetic-validation PSNR gains to real-chart noise or halo metrics as depth varies. The owner's observation (+0.8 dB synthetic, no uniform real-chart gain) has no literature counterpart I could find.

## Q3. Same-family deeper teacher vs different-architecture teacher (transformer / NAFNet / full-resolution) for low-level KD; is "the best teacher not the most accurate"?

### Takeaway
With output-level (pixel) KD, the teacher's architecture family barely matters once accuracy is similar. EDSR and SwinIR teachers gave identical gains to an RCAN student. Feature-level KD effectively requires the same family. Several papers show a weaker but student-reachable teacher beating a stronger one. No restoration study shows a transformer or full-resolution teacher beating a same-family deeper teacher for a tiny CNN.

### Cited Findings
- DUKD Table 5 (heterogeneous KD, x4 RCAN student at 5.17M), BSD100 / Urban100 PSNR:
  - Scratch 27.64 / 26.37
  - EDSR teacher 27.71 / 26.59
  - SwinIR teacher 27.72 / 26.59

  "Conventional feature-based KD or self-distillation methods are inapplicable to the cross-architecture setting." — [DUKD, arXiv 2309.14162](https://arxiv.org/abs/2309.14162)
- MTKD (ECCV 2024) uses an RCAN_lightweight student (5.2M) at x4, Urban100. Compact 26.34 vs KD 26.49 vs MTKD (SwinIR+RCAN+EDSR teachers) 27.08. The SwinIR-only variant reached 26.70 and SwinIR+EDSR 26.79, but those use MTKD's aggregation network and wavelet loss, so the teacher's family is confounded with the method. — [MTKD, arXiv 2404.09571](https://arxiv.org/abs/2404.09571)
- Instance Data Condensation (2025) found, across EDSR, SwinIR and MambaIRv2 teacher-student combinations, that a stronger teacher does not always give a better student. — [IDC, arXiv 2505.21099](https://arxiv.org/abs/2505.21099)
- PISR's best teacher shares the student's architecture by construction: "For the decoder, we use the same architecture as the student network". The student is initialized from the teacher decoder's weights (+0.01 dB over no transfer, Table 1). The bottleneck-limited teacher was better than an unconstrained one. — [PISR, arXiv 2007.07524](https://arxiv.org/abs/2007.07524)
- NTIRE 2025 Efficient SR challenge:
  - The overall winner, EMSR, distilled a SPAN-based student using spatial-affinity feature KD plus a pixel-level teacher loss.
  - Team mbga used a same-backbone teacher, "the student's backbone + three extra SPAB blocks". It was co-trained by self-distillation, after which "the extra SPAB blocks are gradually removed ... finally resulting in an architecture identical to the original student model".

  — [NTIRE 2025 ESR Challenge Report, arXiv 2504.10686](https://arxiv.org/abs/2504.10686)
- Beyer et al. distilled across families (ResNet teacher -> MobileNet v3 student) successfully with function matching. They describe their approach as applying to "architectures of different families entirely". — [Beyer et al., arXiv 2106.05237](https://arxiv.org/abs/2106.05237)
- Classification-side "less accurate teacher can be better" evidence:
  - Cho & Hariharan: the optimal teacher for WRN16-1 was WRN16-3, not WRN16-8. — [arXiv 1910.01348](https://arxiv.org/abs/1910.01348)
  - SFTN: "maximizing performance of teacher models may be suboptimal for knowledge distillation". — [arXiv 2102.07650](https://arxiv.org/abs/2102.07650)
  - Tf-KD: poorly-trained teachers still help. — [arXiv 1909.11723](https://arxiv.org/abs/1909.11723)
- In monocular depth estimation (dense regression), a transformer-teacher -> CNN-student paper notes that "due to capacity and architecture gaps, the student is difficult to imitate teacher's features". Their fix is a "ghost decoder" (a copy of the student's decoder) on the teacher side. This is known only from the search-result summary; I did not read the full paper. — [Promoting CNNs with Cross-Architecture KD for Monocular Depth, arXiv 2404.16386](https://arxiv.org/abs/2404.16386)
- A 2025 dual-teacher SR KD paper (Knowledge-Based Systems) reports that a "bad" pretrained teacher that produces unreliable artifacts "cannot transfer the ability to enhance visual quality to students". This is known from the abstract and search summary only. — [Reliable Image SR Using Dual-Teacher KD (2025)](https://www.sciencedirect.com/science/article/abs/pii/S0950705125018817)

### Inferences
- **Practical choice.**
  - A same-family deeper teacher (8 or 16 layers) is the lower-risk option. It allows output KD plus optional feature KD at matching resolution and channels, and weight initialization or progressive layer removal (mbga, PISR). Its solution is most likely to be "reachable" by the 4-layer student (Cho & Hariharan).
  - A different-architecture teacher (NAFNet, Restormer, SwinIR) adds value only through better targets on the deployment distribution. With output KD alone, the evidence (DUKD Table 5) says family does not matter once teacher quality is comparable.
- **Why "most accurate" can be the wrong criterion.** For this owner, synthetic-validation PSNR has already been shown to be a weak proxy for real-chart quality. The deciding property of a teacher is its behaviour on real camera inputs: chroma noise removal, halo, texture retention. Synthetic PSNR is secondary. A teacher with high synthetic PSNR but no real-chart advantage (the 8x-MAC full-resolution model) is predicted to transfer its synthetic-domain bias, not real-chart gains.
- **Using the full-resolution chroma result.** The shallow full-resolution model's better chroma removal is a specific behaviour worth transferring. A plausible design is a multi-teacher or per-component target: chroma from the full-resolution teacher, luma and texture from the same-family deeper teacher. MTKD shows multi-teacher aggregation can outperform any single teacher in SR, but this specific split has not been tested.

### Gaps
- I found no restoration paper that compares, for the same tiny CNN student, (a) a deeper same-family teacher vs (b) a transformer or NAFNet teacher of higher PSNR.
- I found no denoising-specific cross-architecture KD ablation. The MAI 2026 and LiteKD-Net mobile denoising papers use a single teacher without varying its family.
- DTKD and the depth-estimation paper were read only via abstract or search summary.

## Q4. Architecture-mismatch issues for feature KD (full-resolution teacher vs half-resolution student; channel adapters)

### Takeaway
In SR, naive feature KD (FitNet, AT) usually hurts, and the better feature KD methods are tied to particular compression settings. A full-resolution teacher and a half-resolution student share no spatial grid inside the body. Output-level KD avoids the problem entirely. If feature KD is used, it should go through a learned or lossless resolution and channel mapping, and its benefit is not guaranteed.

### Cited Findings
- DUKD Table 2 (EDSR x4, Urban100): FitNet 25.40 vs scratch 26.21 (-0.81 dB), AT 25.97 (-0.24 dB). At x2: FitNet 30.46 vs 31.96 (-1.50 dB). — [DUKD, arXiv 2309.14162](https://arxiv.org/abs/2309.14162)
- MiPKD: "FAKD boosts the student model in depth compression but deteriorates the student when applied to a width compression circumstance. CSD improves the student model significantly [in width compression] but is not compatible with depth compression". Feature-based KD imported from high-level vision (RKD, AT, FitNet) "hardly benefit the student model". MiPKD's fix encodes teacher and student features "into a unified latent space ... using separate encoder networks for the teacher and the student" (feature prior mixer), plus a block prior mixer. — [MiPKD, arXiv 2404.02573](https://arxiv.org/abs/2404.02573)
- PISR makes feature KD work by forcing the teacher's decoder to have the student's exact architecture. Feature distillation then gives +0.04 to +0.18 dB on a 13K FSRCNN student (Set5, x2). — [PISR, arXiv 2007.07524](https://arxiv.org/abs/2007.07524)
- In heterogeneous feature KD for dehazing, an SR teacher's features at higher resolution (920x1240) gave a better student than lower-resolution features (460x620): 32.14 vs 31.96 dB PSNR, SSIM 0.9666 vs 0.9508. — [Rich Feature Distillation with Feature Affinity for Dehazing, arXiv 2207.11250](https://arxiv.org/abs/2207.11250)
- The NTIRE 2025 winner (EMSR) used spatial-affinity feature KD, which compares Gram-like affinity matrices rather than raw features, plus pixel-level KD. This form tolerates channel-count differences. — [NTIRE 2025 ESR Report, arXiv 2504.10686](https://arxiv.org/abs/2504.10686)
- A 2026 CIFAR study found feature KD matched or beat logit KD in all four ResNet pairs. That is classification, and with the same family. — [arXiv 2605.31191](https://arxiv.org/abs/2605.31191)

### Inferences
- **Mapping full-res teacher features to the half-res student (options, my design reasoning).** A full-res teacher feature map F_T (C_T x H x W) can be mapped to the student grid (32 x H/2 x W/2) in three ways:
  - (a) Lossless PixelUnshuffle(2) to 4C_T x H/2 x W/2, then a learned 1x1 adapter on the student side (FitNet-style regressor). This is the most information-preserving option.
  - (b) Strided average pooling. This is cheap but throws away exactly the high-frequency, phase-specific information that the student's PixelShuffle has to produce.
  - (c) Affinity or relational matching (FAKD, DIST-style correlation). This avoids channel adapters but still needs a common spatial grid.

  Given DUKD/MiPKD's negative FitNet results in SR, the default should be output-level KD. Feature KD should be added only with a same-family teacher at the same half-resolution grid, where features align one-to-one: the same stem, then the 8/16-layer teacher's matching layer indices.
- **Same-family teachers simplify mismatch handling.** A same-family 8/16-layer teacher shares the stem stride, channel count and PixelShuffle tail. Feature KD then needs no adapters. The tail and stem can also be copied, or the student initialized from teacher layers, as in PISR's weight transfer and mbga's progressive block removal.

### Gaps
- I found no paper that specifically studies distilling features from a stride-1 full-resolution teacher into a stride-2 + PixelShuffle student.
- I found no low-level paper ablating PixelUnshuffle adapters against pooling adapters for feature KD.

## Q5. Can a student learn behaviour its architecture cannot represent (e.g. shift-variance from PixelShuffle / stride-2 stems)? Aliasing and shift-equivariance of strided + PixelShuffle restorers

### Takeaway
A stride-2 stem plus PixelShuffle(2) network is only periodically shift-equivariant (period 2). It can show phase-dependent (checkerboard or grid) behaviour, while a stride-1 full-resolution teacher is fully shift-equivariant. Whether the student can learn near-equivariance is an optimization and data question. Learned near-equivariance is reported for a large strided U-Net denoiser. The anti-aliasing literature shows enforcing it costs some high-frequency content. Nothing in the literature measures this for a ~40k Bicubic++-style net.

### Cited Findings
- Zhang (ICML 2019) states that "commonly used downsampling methods, such as max-pooling, strided-convolution, and average-pooling, ignore the sampling theorem". Literal shift-equivariance "cannot hold when subsampling"; only shifts that are integer multiples of the stride preserve it ("periodic" equivariance). Low-pass filtering before subsampling is the fix, and it "can be applied to existing strided layers".
  - Applied to a pix2pix U-Net (8 down / 8 up), the method improved output stability under input shifts from 29.0 dB (baseline) to 30.1/30.8/31.2/34.4 dB (Rect-2/Tri-3/Bin-4/Bin-5 filters).
  - Total variation of outputs fell from 7.48 to as low as 5.84 (ground truth 7.80), i.e. stronger filtering "decreases the frequency content of generated images". Quality holds up to about Tri-3, then trades off.

  — [Making Convolutional Networks Shift-Invariant Again, arXiv 1904.11486](https://arxiv.org/abs/1904.11486)
- Chaman & Dokmanić (CVPR 2021) note that removing strides restores shift invariance but "leads to a dramatic increase in memory and computation requirements", so they propose adaptive polyphase sampling. Read via search summary. — [Truly shift-invariant CNNs, CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/papers/Chaman_Truly_Shift-Invariant_Convolutional_Neural_Networks_CVPR_2021_paper.pdf)
- Michaeli et al. (CVPR 2023) extend this to fractional shifts. Pointwise non-linearities also create aliasing, which breaks equivariance even without striding. Read via search summary. — [Alias-Free Convnets, arXiv 2303.08085](https://arxiv.org/abs/2303.08085)
- Aitken et al. (2017) show that sub-pixel convolution (PixelShuffle) produces checkerboard artifacts after random initialization even when the kernel size divides the stride. They propose an initialization (ICNR-style: nearest-neighbour-resize-equivalent at init) that removes init-time checkerboards. — [Checkerboard artifact free sub-pixel convolution, arXiv 1707.02937](https://arxiv.org/abs/1707.02937)
- Renaud et al. (2024) tested a GS-DRUNet denoiser, a strided U-Net, on CBSD68. Averaging over 10 random translations gave essentially the same PSNR as a single pass:

  | sigma | Single pass | Translation-averaged |
  |---|---|---|
  | 5/255 | 40.54 | 40.53 |
  | 10/255 | 36.46 | 36.44 |
  | 20/255 | 32.73 | 32.71 |

  The authors conclude "GS-DRUNet has already learned these equivariances". — [Equivariant Denoisers for Image Restoration, arXiv 2412.05343](https://arxiv.org/abs/2412.05343)
- Stanton et al.: low student fidelity is largely an optimization and identifiability problem. Fidelity increased as the distillation dataset was enlarged or augmented (their Fig. 1 and Sec. 5). — [arXiv 2106.05945](https://arxiv.org/abs/2106.05945)
- Bicubic++ found strided-conv downsampling best on runtime-weighted score. By PSNR, DWT downsampling was +0.055-0.062 dB better (Table 1, rows Q/X and T/W). — [Bicubic++, arXiv 2305.02126](https://arxiv.org/abs/2305.02126)

### Inferences
- **Representability (my analysis).** The 3x3 stride-2 stem sees a 3x3 window, which contains a full 2x2 polyphase block of the input. It has 32 output channels, at least the 12 values needed for 4 phases x 3 channels. So the stem itself does not destroy information. In principle the half-res body plus PixelShuffle could emulate a linear shift-invariant operator exactly, as a polyphase implementation.
- **What the student cannot represent exactly.** Non-linear behaviour that must be identical across all four output phases. The LeakyReLU operates on half-res features shared by 4 output phases, so exact equivariance under odd shifts is not guaranteed. The student can only approximate it, and the approximation quality depends on capacity and training data.
- **Expected effect of a full-resolution teacher on the student.** KD from a fully shift-equivariant teacher, using random odd-pixel crops so all phases appear, gives the student a smoother, phase-consistent target. This plausibly reduces phase-dependent grid or checkerboard residue. It cannot add capacity the student lacks.
- **Expected limits.** The teacher's fine texture retention will not transfer if the 4-layer half-res body cannot represent it. That matches the observation that fine-texture retention was identical even for the 8x-MAC model. Behaviours that are mostly linear or low-pass, such as chroma smoothing, are the most transferable.
- **Cheap structural options if phase artifacts matter.** These are more reliable than hoping KD teaches equivariance:
  - a blur-pool or anti-aliased stem (Zhang 2019)
  - ICNR-style PixelShuffle initialization (Aitken 2017)
  - DWT or space-to-depth instead of a strided-conv stem (Bicubic++ shows DWT is PSNR-competitive)
  - shift-consistency augmentation or a consistency loss between shifted inputs

  All of these trade a little sharpness for stability.

### Gaps
- I found no paper that measures shift-equivariance error or phase-dependent artifacts specifically for Bicubic++/ESPCN-style stride-2 + PixelShuffle restorers, or how KD from a stride-1 teacher changes it.
- The DRUNet near-equivariance result is for a large model (tens of millions of parameters, if I recall the DRUNet size correctly; not verified here). There is no evidence on whether a 41k model learns comparable equivariance.
- The Chaman, Michaeli, depth-estimation and DTKD claims come from abstracts or search summaries, not full-text reading.
