# Knowledge distillation (KD) for SR, denoising and low-level restoration: methods and reported gains

Scope note: a literature review only; no experiments were run. "Reported" means a number printed in the cited paper. Deltas marked "(Δ computed)" are my subtraction of two numbers from the same table. Anything under "Inferences" is my reasoning, not a reported fact. Unless stated, PSNR is on the Y channel, in dB, and the student was trained on DIV2K.

Target use case (from the brief): a ~41k-parameter, 32-channel, Bicubic++-style same-resolution denoiser (stride-2 stem, 4 plain 3x3 convs at half resolution, ECB reparameterization, PixelShuffle(2), global input residual), W16A16 on Qualcomm HTP, trained on synthetic ISP-degradation pairs with Charbonnier + edge + SSIM loss.

## Q1. What do output-only (response/pixel) KD methods gain over GT-only training? Is the gain near zero when GT is available?

### Takeaway
Yes, in the SR benchmarks that test it: output (logit/pixel) KD added to an L1-to-GT loss usually gives 0.00 to +0.05 dB, sometimes negative. The one consistent exception in these tables is RCAN at x4 (+0.12 to +0.15 dB on Urban100). Reported large output-KD gains (+0.7 to +1.6 dB) appear only in small-data or mismatched-baseline denoising papers.

### Cited Findings
- DUKD (Zhang et al., arXiv 2309.14162, 2023; same group's follow-up AugKD is ICLR 2025), EDSR teacher 43.09M (c256b32) → student 2.70M (c64b32), Urban100 PSNR Scratch vs KD: x2 31.96 → 31.98, x3 27.99 → 28.00, x4 26.21 → 26.21 (Δ computed: +0.02 / +0.01 / 0.00). Set5 x4 32.29 → 32.30. — [DUKD, arXiv 2309.14162v4](https://arxiv.org/html/2309.14162v4)
- DUKD, RCAN teacher 15.59M (20 groups) → student 5.17M (6 groups), Urban100 Scratch vs KD: x2 32.63 → 32.67, x3 28.59 → 28.62, x4 26.37 → 26.49; Set5 x4 32.31 → 32.45 (Δ computed: +0.04 / +0.03 / +0.12 Urban100; +0.14 Set5 x4). This is the largest output-KD gain in any SR table found. — [DUKD](https://arxiv.org/html/2309.14162v4)
- DUKD, SwinIR teacher 11.90M → student 1.24M: Urban100 x2 32.05 → 32.09, x3 28.20 → 28.23, x4 26.15 → 26.15; Set5 x4 32.31 → 32.27 (Δ computed: KD is -0.04 dB on Set5 x4). — [DUKD](https://arxiv.org/html/2309.14162v4)
- DUKD's explanation, quoted: the teacher's output "contains barely extra information over GT"; "Directly aligning the model outputs hardly transfers knowledge to and may even mislead the student model"; "The teacher model's distribution information is shaded by" the HR ground truth; teacher outputs are "noisy approximations to the ground-truth distribution". — [DUKD](https://arxiv.org/html/2309.14162v4)
- CSD (Wang et al., IJCAI 2021), EDSR+ 0.25x-width student (2.7M) vs 1.0x teacher (43.1M): "J-T1" (joint teacher-student training with an L1 output term) gives Urban100 x4 26.06 vs baseline 26.07, x2 32.26 vs 32.28, which is no gain. — [CSD, ar5iv 2105.11683](https://ar5iv.labs.arxiv.org/html/2105.11683); venue: [IJCAI 2021 proceedings](https://ijcai.org/proceedings/2021/155)
- MiPKD (Li et al., ICLR 2025), EDSR c64b32 student: KD vs Scratch Urban100 x2 31.98 vs 31.96, x4 26.21 vs 26.21. High-compression student c64b16 (1.52M, ~28x smaller than the teacher), x4 Urban100: NOKD 25.61, KD 25.60, FitNet 25.52, FAKD 25.38. — [MiPKD, arXiv 2404.02573](https://arxiv.org/html/2404.02573); venue: [mlanthology ICLR 2025](https://mlanthology.org/iclr/2025/li2025iclr-knowledge/)
- DCKD (arXiv 2412.08939, 2024), restoration beyond SR, logits KD vs Scratch: GoPro deblurring, NAFNet 17.1M → 2.7M: 31.17 → 31.26 (+0.09); Restormer 26.1M → 3.8M: 31.57 → 31.61 (+0.04). Deraining Rain100L (Restormer → 3.8M): 37.47 → 37.17 (Δ computed: -0.30); Rain100H 30.51 → 30.48. — [DCKD, arXiv 2412.08939v2](https://arxiv.org/html/2412.08939v2)
- MTKD (Jiang et al., ECCV 2024), its own KD runs, Urban100: EDSR_baseline 1.5M x2 31.99 → 32.09, x4 26.03 → 26.02; SwinIR_lightweight 0.9M x2 32.76 → 32.79, x4 26.47 → 26.50; RCAN_lightweight 5.2M x4 26.34 → 26.49. — [MTKD, arXiv 2404.09571](https://arxiv.org/html/2404.09571v1); venue: [ECCV 2024 poster](https://eccv2024.ecva.net/virtual/2024/poster/2688)
- DVMSR (arXiv 2405.03008, 2024), 424K-parameter Mamba SR student, x4, output L1-to-teacher plus L1-to-GT at 1:1: Set5 32.04 → 32.12 (+0.08), Set14 28.50 → 28.57 (+0.07). Mid-level feature distillation gave 32.11 / 28.56, so end-level (output) was marginally better. — [DVMSR, arXiv 2405.03008](https://arxiv.org/html/2405.03008)
- Counter-example, small-data denoising. Kayani et al., "Real Image Denoising with Knowledge Distillation for High-Performance Mobile NPUs" (CVPRW 2026, MAI): teacher 41.6M U-Net → student LiteDenoiseNet 1.96M, output-only KD (L2 to teacher weight 900, L2 to GT weight 100, L1 to GT weight 50). Training set: 650 pairs (MAI 2026 denoising dataset). Validation PSNR: no-KD baseline 36.08, KD student 37.66, teacher 37.71 (+1.58 dB from KD). The paper does not say the baseline used the same GT-only recipe, and it has no α ablation. FP16 TFLite runs at 46.1 ms on the Snapdragon 8 Elite NPU at 1088x1920. — [arXiv 2605.03680](https://arxiv.org/html/2605.03680v1); [CVF CVPRW 2026 PDF](https://openaccess.thecvf.com/content/CVPR2026W/MAI/papers/Kayani_Real_Image_Denoising_with_Knowledge_Distillation_for_High-Performance_Mobile_NPUs_CVPRW_2026_paper.pdf)
- AugKD (ICLR 2025, same group as DUKD) argues vanilla KD is "underestimated" for SR. Its claim is that unpaired augmentations, which create extra inputs that only the teacher labels plus a label-consistency regularizer, make output KD effective. I did not obtain its tables (PDF over the fetch size limit). — [AugKD, ICLR 2025 proceedings](https://proceedings.iclr.cc/paper_files/paper/2025/hash/82c1fee06391ef7c22b108b8611cf266-Abstract-Conference.html)

### Inferences
- When the student has unlimited exact GT for the same inputs, output KD gives roughly 0 to +0.05 dB in most published SR settings (EDSR, SwinIR, EDSR_baseline, NAFNet/Restormer deblurring) and can be negative (SwinIR x4 Set5, Rain100L). RCAN x4 (+0.12) is the outlier.
- The project trains on synthetic pairs, so GT is exact and the data supply is effectively unlimited. That is the regime where output KD is near zero in the literature. For denoising (a less ill-posed mapping than x4 SR), the teacher's "dark knowledge" beyond GT should be even smaller. This is my reasoning; no paper tests it directly.
- The large denoising gains (+1.58 dB MAI 2026, +0.67 dB LiteKD-Net in Q4) come from small real-data training sets (650 pairs) or a two-stage recipe whose baseline may be undertrained. Neither transfers directly to an unlimited-synthetic-pair setup.
- The mechanism that does work in DUKD and AugKD is the teacher labelling inputs that have no GT (upcycled or augmented data). For this project the analogue would be a teacher labelling real EO ISP frames that have no clean target. That is the one output-KD scenario with a plausible upside. It is not tested in any found paper for ISP denoising.

### Gaps
- No paper found that tests output KD for a sub-100k-parameter same-resolution denoiser.
- AugKD's numbers (the strongest argument for vanilla KD) could not be read; the PDF exceeded the fetch limit.
- The MAI 2026 paper does not describe the no-KD baseline's training recipe, so its +1.58 dB cannot be attributed to KD alone.

## Q2. Feature-based and other KD signals (FAKD, CSD, PISR, LSFD, MiPKD, DUKD, DCKD, MTKD, ...): what do they report and for which student sizes?

### Takeaway
Methods that move the numbers by 0.2 to 0.6 dB (DUKD, MiPKD, DCKD, MTKD, CSD) all report against "scratch" baselines that are sometimes 0.1 to 0.7 dB below other papers' baselines for the same architecture. Plain feature imitation (FitNet, AT) is harmful in SR. FAKD and RKD are within ±0.1 dB. The best-supported useful signals are data upcycling (teacher labels extra inputs), contrastive terms (largely teacher-independent per the CSD ablation) and frequency/wavelet output losses.

### Cited Findings

Catalogue (Urban100 PSNR unless noted; Δ = method minus same-paper scratch, computed by me from the cited table)

| Method (venue, year) | KD signal | Teacher → student (params) | Reported Δ vs scratch | Source |
|---|---|---|---|---|
| FitNet (as benchmarked) | feature L2 with regressor | EDSR 43.09M → 2.70M | x2 -1.50, x3 -1.38, x4 -0.81 | [DUKD](https://arxiv.org/html/2309.14162v4) |
| AT (as benchmarked) | attention maps | EDSR 43.09M → 2.70M | x2 -0.25, x4 -0.24 | [DUKD](https://arxiv.org/html/2309.14162v4) |
| RKD (as benchmarked) | relational | EDSR → 2.70M; RCAN → 5.17M | 0.00 to -0.03 (EDSR); +0.07 / +0.01 / +0.10 (RCAN x2 / x3 / x4) | [DUKD](https://arxiv.org/html/2309.14162v4) |
| FAKD (ICIP 2020) | feature affinity (2nd-order) | EDSR → 2.70M | x2 +0.08, x3 +0.08, x4 -0.03 | [DUKD](https://arxiv.org/html/2309.14162v4); method: [IEEE Xplore](https://ieeexplore.ieee.org/abstract/document/9190917) |
| FAKD | same | RCAN 15.59M → 5.17M | x2 +0.02, x3 +0.03, x4 +0.13 | [DUKD](https://arxiv.org/html/2309.14162v4) |
| FAKD | same | SwinIR 11.9M → 1.2M | x2 +0.01, x3 -0.02, x4 -0.06 | [DCKD](https://arxiv.org/html/2412.08939v2) |
| CSD (IJCAI 2021) | contrastive self-distillation, channel-split student | EDSR+ 43.1M → 0.25x (2.7M) | own paper (32.28 → 32.55, 26.07 → 26.34): x2 +0.27, x4 +0.27; in DUKD's table (CSD* re-run at x2, paper value at x4, DUKD scratch): x2 +0.30, x4 +0.13 | [CSD](https://ar5iv.labs.arxiv.org/html/2105.11683); [DUKD](https://arxiv.org/html/2309.14162v4) |
| PISR (ECCV 2020) | privileged-info teacher (HR input), VID feature loss | FSRCNN 13K student | x2 +0.19 (Set5 +0.18), x3 +0.18, x4 +0.06 | [PISR](https://arxiv.org/abs/2007.07524) |
| LSFD (arXiv 2021, "in review") | local-selective feature distillation | RCAN 15.59M → 5.17M | x2 +0.39, x3 +0.21, x4 +0.19 (FAKD repro: +0.29 / +0.13 / +0.15) | [LSFD](https://arxiv.org/html/2111.10988v1) |
| LSFD | same, +FFT loss | EDSR 43M → 1.5M | x2 +0.10, x4 +0.08; LSFD+FFT x4 +0.10 | [LSFD](https://arxiv.org/html/2111.10988v1) |
| DUKD (arXiv 2023) | data upcycling: teacher labels zoomed/augmented inputs, plus label consistency | EDSR → 2.70M | x2 +0.57, x3 +0.45, x4 +0.24 (Set5 +0.15 / +0.20 / +0.18) | [DUKD](https://arxiv.org/html/2309.14162v4) |
| DUKD | same | RCAN → 5.17M | x2 +0.24, x3 +0.20, x4 +0.25 | [DUKD](https://arxiv.org/html/2309.14162v4) |
| DUKD | same | SwinIR → 1.24M | x2 +0.58, x3 +0.33, x4 +0.28 | [DUKD](https://arxiv.org/html/2309.14162v4) |
| MiPKD (ICLR 2025) | multi-granularity mixture of priors, feature and block level | EDSR → 2.70M / RCAN → 5.17M / SwinIR → 1.24M | +0.56 / +0.20 / +0.41 (x2); +0.24 / +0.19 / +0.22 (x4) | [MiPKD](https://arxiv.org/html/2404.02573) |
| MiPKD two-stage (teaching assistant) | same | EDSR 43M → c64b16 1.52M | x4 +0.28; direct one-stage "yields negative effects" | [MiPKD](https://arxiv.org/html/2404.02573) |
| DCKD (arXiv 2024) | dynamic contrastive (degraded inputs as negatives) | SwinIR → 1.2M / RCAN → 5.2M | x4 +0.38 / +0.33; DCKD* (+DUKD) +0.47 / +0.38 | [DCKD](https://arxiv.org/html/2412.08939v2) |
| DCKD | same | NAFNet 17.1M → 2.7M (GoPro) / Restormer 26.1M → 3.8M (GoPro) | +0.26 / +0.21 | [DCKD](https://arxiv.org/html/2412.08939v2) |
| MTKD (ECCV 2024) | 3 teachers fused, wavelet-domain output loss | RCAN → 5.2M / EDSR → 1.5M / SwinIR → 0.9M | x4 +0.74 / +0.29 / +0.38 | [MTKD](https://arxiv.org/html/2404.09571v1) |

- MTKD ablation (RCAN_lightweight, x4 Urban100): L1 distillation loss 26.78 vs wavelet loss (full MTKD) 27.08, DCT-based loss 26.98, high-frequency-only DWT loss 26.28, a single SwinIR teacher 26.70. The frequency-domain formulation of the output loss accounts for ~+0.3 dB over L1 in their setup. — [MTKD](https://arxiv.org/html/2404.09571v1)
- MTKD says its compact baselines and the DUKD and CrossKD rows are "taken from their original publications", while KD, AT and FAKD were run by the authors. Its RCAN compact x2 Urban100 baseline is 32.32, whereas DUKD's scratch for the same 5.17M RCAN student is 32.63. — [MTKD](https://arxiv.org/html/2404.09571v1); [DUKD](https://arxiv.org/html/2309.14162v4)
- Baseline mismatch: DUKD's 1.24M SwinIR student scratch x2 Urban100 is 32.05. MTKD's 0.9M SwinIR_lightweight compact baseline is 32.76, and MTKD lists DUKD at 32.63, i.e., below MTKD's no-KD baseline. — [DUKD](https://arxiv.org/html/2309.14162v4); [MTKD](https://arxiv.org/html/2404.09571v1)
- Same-architecture baseline mismatch: CSD's own EDSR+ 0.25x x4 Urban100 baseline is 26.07, while DUKD's scratch for the same c64b32 EDSR student is 26.21. Both report CSD at 26.34. — [CSD](https://ar5iv.labs.arxiv.org/html/2105.11683); [DUKD](https://arxiv.org/html/2309.14162v4)
- CSD ablation (EDSR+ 0.25x, x4 Urban100): baseline 26.07; "W/O T (GT Pos.)", the contrastive loss with GT as positive and no teacher, 26.30; full CSD with teacher 26.34; InfoNCE variant 26.24; joint training with perceptual loss 25.72; teacher and student trained separately 26.15. — [CSD](https://ar5iv.labs.arxiv.org/html/2105.11683)
- PISR loss ablation (FSRCNN 13K student, Set5 x2, baseline 37.15): conventional KD with an FSRCNN-L teacher on LR input: MAE +0.05, FitNet +0.01, AT +0.06, SRKD +0.06, VID-Laplace +0.08. With the privileged-HR teacher: MAE +0.12, FitNet +0.16, AT +0.16, SRKD +0.14, VID-Laplace +0.18. On B100 all variants are +0.01 to +0.07. PISR: "distilling features from the privileged information (i.e., ground-truth HR images) is important, regardless of loss functions"; "there are no distillation losses effective to regression tasks". — [PISR, arXiv 2007.07524](https://arxiv.org/abs/2007.07524)
- PISR, a teacher with weaker capacity (hourglass) helps: a non-hourglass HR-input teacher reaches 57.60 dB but the student gains only +0.04. An hourglass teacher at 37.70 dB gives +0.07, "reducing the performance gap between the teacher and the student", citing teacher-assistant and "efficacy of KD" work. — [PISR](https://arxiv.org/abs/2007.07524)
- LSFD on FitNet in SR: "When FitNet is used directly on SISR, a slight increase in the PSNR scores occurs"; the conventional regressor-based FD method "suffers in the task of super-resolution". — [LSFD](https://arxiv.org/html/2111.10988v1)
- DUKD on transformers: on SwinIR "only KD and DUKD were applicable"; feature KD methods (FitNet, AT, FAKD, CSD) are architecture-tied. MiPKD: "existing KD techniques for SR are tailored to specific teacher-student architectures". — [DUKD](https://arxiv.org/html/2309.14162v4); [MiPKD](https://arxiv.org/html/2404.02573)
- DUKD, perceptual or real-world use: on RealSR, DRealSR and OST300, NIQE for the 1.24M SwinIR student is 5.810 scratch, 5.425 KD, 5.398 DUKD (RealSR); teacher 4.771. — [DUKD](https://arxiv.org/html/2309.14162v4)
- DUKD heterogeneous teachers (RCAN 5.17M student, x4 Urban100, scratch 26.37): EDSR teacher 26.59, SwinIR teacher 26.59, so the teacher architecture barely matters. — [DUKD](https://arxiv.org/html/2309.14162v4)
- Data-free KD for SR (Zhang et al., CVPR 2021): a generator synthesizes inputs and the student is distilled progressively; the abstract reportedly claims only a 0.16 dB PSNR drop on Set5 x2 versus data-driven training. Seen via search snippet only; the CVF page returned 403. — [CVF CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/html/Zhang_Data-Free_Knowledge_Distillation_for_Image_Super-Resolution_CVPR_2021_paper.html)
- NTIRE 2025 Efficient SR: the overall winner EMSR (0.131M params, SPAN-Tiny-based, 26 channels, ConvLoRA merged at inference) used spatial-affinity feature KD plus L1 to teacher output plus L2 to GT. Rank-7 ESPAN (0.192M) used progressive self-distillation. The report says distillation "significantly contributed" but gives no KD ablation numbers. — [NTIRE 2025 ESR report, arXiv 2504.10686](https://arxiv.org/html/2504.10686v1)
- Bicubic++ (the architecture lineage of the project's student; CVPRW 2023, NTIRE RTSR x3 winner): the training pipeline uses global structured pruning and bias removal; the abstract mentions no KD. — [Bicubic++, arXiv 2305.02126](https://arxiv.org/abs/2305.02126)

### Inferences
- Signals that move numbers, best supported first:
  1. Data upcycling or augmentation, where the teacher labels inputs that have no GT (DUKD +0.24 to +0.58 dB). This is the only family with a consistent gain across EDSR, RCAN and SwinIR within one paper's own baseline.
  2. Contrastive terms (CSD, DCKD, +0.2 to +0.4 dB). The CSD ablation suggests most of that gain (+0.23 of +0.27) comes from the contrastive regularizer itself with GT as positive, not from the teacher.
  3. Frequency or wavelet output losses (MTKD ~+0.3 dB over L1 KD; LSFD+FFT only +0.01 to +0.03). MTKD's frequency loss may help a GT-only student too; that was not tested.
- Plain FitNet and AT feature imitation hurt (down to -1.5 dB). RKD and FAKD sit within noise.
- Cross-paper baselines for the same or similar student differ by 0.14 dB (EDSR c64b32: 26.07 vs 26.21), 0.31 dB (RCAN 5.2M: 32.32 vs 32.63) and 0.71 dB (SwinIR 1.24M vs 0.9M: 32.05 vs 32.76). That is the same size as most claimed KD gains. Claimed gains are therefore fragile. A well-tuned GT-only baseline (longer training, better loss) may close much of the gap.
- For the project: feature KD into a 32-channel reparameterized student needs adapter layers. The literature finds FitNet-style adapters harmful in SR, and the training-time ECB branches make "which feature" ambiguous. The cheapest experiments with literature support are a contrastive term with GT positive and ISP-degraded input negative (no teacher needed) and a frequency or wavelet loss term (no teacher needed). Both are my suggestions, not tested in any found paper on this architecture.
- No data on adversarial KD for SR or denoising surfaced in this review.

### Gaps
- FAKD's original ICIP 2020 tables were not retrieved (only the abstract); its numbers above come from DUKD and DCKD re-runs.
- AugKD and the data-free SR KD (CVPR 2021) tables were not readable (fetch limits, 403).
- LSFD's final venue was not verified (arXiv says "in review").
- No adversarial-KD or GAN-distillation numbers for restoration were found.
- Gao et al. (ACCV 2018), the first SR-KD paper, could not be retrieved.

## Q3. Are there systematic benchmarks or "rethinking KD for SR" studies concluding that most KD methods give small or inconsistent gains?

### Takeaway
No standalone "rethinking KD for SR" benchmark paper was found. However, the four papers with the largest multi-method comparison tables (DUKD 2023, MiPKD ICLR 2025, DCKD 2024, MTKD ECCV 2024) all state, in their own re-runs, that existing KD methods give limited or even negative gains for SR.

### Cited Findings
- DUKD (benchmarks KD, FitNet, AT, RKD, FAKD and CSD on EDSR, RCAN and SwinIR at x2/x3/x4): "Existing KD methods have limited benefits and some even result in student models worse than those trained without KD"; FitNet, AT and RKD "while applicable, hardly improve the SR models over training from scratch"; "EDSR models trained with FAKD sometimes underperform the ones trained from scratch". — [DUKD](https://arxiv.org/html/2309.14162v4)
- MiPKD: "Existing KD methods for SR have limited effects, some may even deteriorate the student model"; "AT and FitNet underperform the vanilla student models trained without KD among all settings"; for a ~28x compressed student, "Directly distilling the student model by the teacher model yields negative effects on its performance". — [MiPKD](https://arxiv.org/html/2404.02573)
- DCKD: "Existing KD methods provide limited improvement for the student models", and "in some cases, they even lead to worse performance compared to models trained without KD". — [DCKD](https://arxiv.org/html/2412.08939v2)
- MTKD takes a more positive line: "most of the benchmarked knowledge distillation methods do offer improved performance" over the compact student. In its own tables, KD, AT and FAKD give -0.05 to +0.35 dB on Urban100 (Δ computed), mixed by model. — [MTKD](https://arxiv.org/html/2404.09571v1)
- AugKD (ICLR 2025) states the opposite framing, that vanilla KD has been "underestimated", but its fix is extra teacher-labelled augmented inputs, not a better output loss. — [AugKD](https://proceedings.iclr.cc/paper_files/paper/2025/hash/82c1fee06391ef7c22b108b8611cf266-Abstract-Conference.html)
- A general KD survey (arXiv 2503.12067, 2025) exists but covers SR only through a single contrastive-loss method. No SR-specific KD survey with a unified benchmark was found. — [A Comprehensive Survey on KD, arXiv 2503.12067](https://arxiv.org/abs/2503.12067)

### Inferences
- The consensus across method papers from 2023 to 2025 is that classification-style KD (logits, FitNet, AT, RKD) and plain feature affinity (FAKD) give roughly ±0.1 dB for SR students in the 1 to 5M range. Each new method then reports +0.2 to +0.7 dB over a scratch baseline that it trains itself.
- These statements come from authors motivating a new method, so they may understate baselines' competitors. Their EDSR re-run numbers are consistent: DUKD's and MiPKD's EDSR scratch, KD and FitNet rows are identical, apparently the same runs. Their RCAN x4 scratch rows differ slightly (32.31 vs 32.38 Set5).

### Gaps
- No independent third-party reproduction study (a "NAS-Bench-style" KD benchmark for SR or denoising) was found up to Oct 2026.
- No paper titled "Rethinking knowledge distillation for super-resolution" was found in searches.

## Q4. KD specifically for image denoising (Gaussian and real camera noise, e.g. SIDD/DND): papers and gains for lightweight students

### Takeaway
Denoising-specific KD literature is thin, and few papers have a clean same-recipe no-KD baseline. Reported gains range from about +0.01 to +0.08 dB (in-place or slimmable distillation, SIDD) up to +0.67 to +1.58 dB (2026 mobile-denoising papers with small real training sets and two-stage recipes). Transformer-compression papers (SLKD, SKD) report students within 0.4 to 0.5 dB of the teacher on SIDD but publish no scratch baseline.

### Cited Findings
- MAI 2026 mobile NPU denoising (CVPRW 2026): 41.6M teacher → 1.96M student, output-only KD, 650 training pairs. No-KD 36.08 dB → KD 37.66 dB (teacher 37.71); per-image mean gap to teacher 0.08 dB on 50 validation images. FP16 on NPU; no INT8 or W16A16 experiments. — [arXiv 2605.03680](https://arxiv.org/html/2605.03680v1)
- LiteKD-Net (arXiv 2608.05739, Aug 2026): teacher is a Real-ESRGAN-style RRDB network without upsampling (16.70M params, 1.17T MACs at 256x256); student Lite-RRDB has 1.67M params and 108.28G MACs. Feature-level "Noise-KD" (MSE on body features) is applied in a second stage. Loss is L1 + LPIPS + KD. Training: 2,400 clean images (DocVQA + LSDIR) with synthetic noise, plus real noisy images. Test sets: Mobile AI Denoising and MIDD (not SIDD). PSNR: student no-KD 37.24, student + KD 37.91 (+0.67), teacher 38.14. LPIPS 0.1257 → 0.0987 (teacher 0.0974). — [LiteKD-Net, arXiv 2608.05739](https://arxiv.org/html/2608.05739)
- SLKD (Zhang & Yan, arXiv 2501.09268, 2025), dual teacher (degraded-input teacher and clean-input teacher), SIDD: Restormer 26.1M teacher 40.02 dB → 3.5M student 39.60; Uformer 50.9M teacher 39.89 → 7.9M student 39.61; DRSformer 33.7M teacher 40.03 → 4.6M student 39.57. Competing DCKD students: 39.27 to 39.34; "DFKD" students: 39.22 to 39.27. No scratch-trained student is reported. — [SLKD, arXiv 2501.09268](https://arxiv.org/html/2501.09268v1)
- SKD with multi-dimensional cross-net attention (same authors, arXiv 2501.09321, Jan 2025), SIDD: Res-SKD 39.52, Ufor-SKD 39.46, DRS-SKD 39.62 vs teachers 40.02 / 39.89 / 40.03. Comparison KD methods on SIDD: Wave 39.25 to 39.26, DCKD 39.27 to 39.34. No scratch baseline. Its ablation is labelled GoPro: 32.20 → 33.24. — [SKD, arXiv 2501.09321](https://arxiv.org/pdf/2501.09321)
- Reliability flag: the SKD ablation numbers (32.20/0.924, 32.61/0.929, ..., 33.24/0.937) are identical to the SLKD ablation, which SLKD labels Rain1400 with different loss components. The two papers are by the same authors. — [SKD](https://arxiv.org/pdf/2501.09321); [SLKD](https://arxiv.org/html/2501.09268v1)
- Dynamic Slimmable Denoising Network (DDS-Net, arXiv 2110.08940, 2021): per a search snippet of the paper text, in-place distillation lets the VDNet-based super network beat the no-distillation baseline by 0.08 dB (smallest sub-network) and 0.01 dB (largest) on SIDD. Not verified in full text; the abstract page has no numbers. — [DDS-Net, arXiv 2110.08940](https://arxiv.org/pdf/2110.08940)
- MDRN (CVPRW 2022 NTIRE): the abstract claims its heterogeneous KD strategies "can improve the performance of tiny models or the model under high noise levels". No numbers on the abstract page. — [MDRN, arXiv 2204.13873](https://arxiv.org/abs/2204.13873)
- Feature-Align Network with KD (WACVW 2022): per search snippet, 48.28 dB on DND (raw) with 263x fewer MACs and 17.6x fewer parameters than a SOTA network at 49.12 dB. The KD ablation was not retrievable (CVF 403). — [Feature-Align, WACVW 2022 PDF](https://openaccess.thecvf.com/content/WACV2022W/WACI/papers/Young_Feature-Align_Network_With_Knowledge_Distillation_for_Efficient_Denoising_WACVW_2022_paper.pdf)
- Robust Deep Ensemble for real-world denoising (arXiv 2206.03691): per search snippet, KD fine-tuning of the best Gaussian denoisers (BDE_GD+D) raises DND PSNR from 38.30 to 38.60 dB. Here the "teacher" is an ensemble, and this is not compression. Snippet only. — [arXiv 2206.03691](https://arxiv.org/pdf/2206.03691)
- DCKD (2024) runs SR, deblurring and deraining but no denoising experiments. — [DCKD](https://arxiv.org/html/2412.08939v2)

### Inferences
- No found denoising-KD paper has a student below ~1.5M parameters except possibly MDRN ("tiny models", no numbers). Nothing approaches 41k.
- The two papers with large denoising gains (+0.67, +1.58 dB) train on a few hundred to a few thousand images, on real or mixed noise, and report a single run. Their no-KD baselines are not clearly same-recipe. This matches the DUKD/AugKD reading: KD pays when the teacher supplies labels the dataset lacks (more inputs, better-than-GT targets for real noise), not when exact synthetic GT is plentiful.
- For ISP-output denoising, a plausible KD use is to let a large teacher label real EO frames that have no clean target, possibly after a teacher-side fine-tune. This is a semi-supervised or pseudo-label setup. It is my inference; no found paper tests it on ISP sharpening halos or hot pixels.
- LiteKD-Net's LPIPS gain (0.126 → 0.099) is proportionally larger than its PSNR gain. If the owner's screen-quality judgement tracks perceptual metrics more than PSNR, KD may matter more there. Inference from one paper.

### Gaps
- No Gaussian-noise (BSD68/CBSD68, σ = 15/25/50) KD table with a small student and a no-KD baseline was found.
- No SIDD or DND KD study with a same-recipe scratch baseline for a lightweight CNN student was found.
- No study combines KD with W8/W16 quantization-aware training for a restoration student.

## Q5. How do KD gains depend on student size (tens of thousands of params vs mid-size)?

### Takeaway
There is no systematic sweep. The scattered evidence says very small students do not gain more: PISR's 13K FSRCNN gains +0.06 to +0.19 dB even with a specially designed teacher, and only +0.01 to +0.08 dB with conventional KD. Large teacher-student gaps (about 28x) can make direct KD negative unless a teaching assistant or a capacity-limited teacher is used.

### Cited Findings
- FSRCNN, 13K params (PISR, ECCV 2020), vs a self-reproduced FSRCNN* without distillation: x2 Set5 37.15 → 37.33 (+0.18), Urban100 30.05 → 30.24 (+0.19); x3 Set5 33.15 → 33.31 (+0.16), Urban100 26.49 → 26.67 (+0.18); x4 Set5 30.89 → 30.95 (+0.06), Urban100 24.76 → 24.82 (+0.06). — [PISR](https://arxiv.org/abs/2007.07524)
- Other students with the same PISR framework, Set5 x2 / x3 / x4 (Δ computed vs the same-recipe reproduced baseline): FSRCNN-L (126K) +0.06 / +0.09 / +0.05; VDSR +0.13 / +0.05 / +0.14; IDN +0.05 / +0.09 / -0.02; CARN +0.07 / +0.02 / +0.06. — [PISR](https://arxiv.org/abs/2007.07524)
- Conventional KD into the 13K FSRCNN (teacher FSRCNN-L, 126K, LR input), Set5 x2: +0.01 to +0.08 dB depending on loss; B100: +0.01 to +0.03. — [PISR](https://arxiv.org/abs/2007.07524)
- DVMSR, 424K student, output KD, x4: +0.08 Set5, +0.07 Set14. — [DVMSR](https://arxiv.org/html/2405.03008)
- MiPKD, EDSR c64b16 (1.52M, ~28x compression), x4 Urban100: direct KD, FitNet and FAKD all at or below NOKD (25.61 → 25.60 / 25.52 / 25.38). Only two-stage teaching-assistant KD helps (+0.28). The same family's c64b32 student (2.70M, 16x) gains +0.24 with one-stage MiPKD. — [MiPKD](https://arxiv.org/html/2404.02573)
- EDSR_baseline (1.5M) in MTKD: KD x2 +0.10, x4 -0.01; FAKD x4 +0.08; MTKD x4 +0.29. RCAN_lightweight (5.2M): MTKD x4 +0.74. MTKD ties larger gains over other KD methods to a larger teacher-student gap. — [MTKD](https://arxiv.org/html/2404.09571v1)
- PISR: a deliberately capacity-limited (hourglass) teacher helps the 13K student by "reducing the performance gap between the teacher and the student". — [PISR](https://arxiv.org/abs/2007.07524)
- Scale dependence within one model (DUKD, EDSR 2.70M, Urban100): +0.57 (x2) > +0.45 (x3) > +0.24 (x4). MiPKD on EDSR shows the same trend: +0.56 / +0.42 / +0.24. — [DUKD](https://arxiv.org/html/2309.14162v4); [MiPKD](https://arxiv.org/html/2404.02573)
- NTIRE 2025 ESR: the winning 131K-param model used KD, but no ablation quantifies its share of the gain. — [NTIRE 2025 ESR](https://arxiv.org/html/2504.10686v1)

### Inferences
- For a ~41k student, the closest literature anchor is PISR's 13K FSRCNN. Even its privileged-information method (teacher sees GT; same decoder architecture; weight transfer; VID loss) gives +0.06 to +0.19 dB, and conventional KD from a 10x larger teacher gives at most +0.08 dB. An honest prior for "deeper teacher → distil into the 41k student, with synthetic GT available" is about 0 to +0.1 dB PSNR. Upside cases are inputs without GT (real frames) and perceptual metrics. This is my inference.
- The capacity-gap evidence (MiPKD's negative direct KD at 28x; PISR's preference for a weak teacher) suggests a very large teacher (for example 1M+ params against 41k, ~25x+) would need a teaching assistant or a teacher constrained to a similar half-resolution topology. That adds training complexity for little expected gain.
- The x2 > x4 trend does not extrapolate cleanly to x1 denoising. The x2 gains may reflect that the student is closer to the teacher at x2, not that easier tasks benefit more. Inference only.
- A ~41k student is capacity-bound, not data-bound, when synthetic pairs are unlimited. KD does not add capacity. Reparameterization (already used), a better loss and longer training are the levers the literature implies are comparable or larger. Inference.

### Gaps
- No paper sweeps student size from ~10k to ~5M with a fixed teacher and recipe on SR or denoising.
- No KD result for students in the 20k to 100k range other than FSRCNN (13K) and FSRCNN-L (126K) in PISR.
- No KD result for students with structural reparameterization (ECB, RepVGG-style) that also isolates the KD contribution; NTIRE 2025 entries combine both without ablation.
