# Pitfalls of knowledge distillation in low-level vision, and KD / pseudo-labeling as a synthetic-to-real bridge

Scope note: literature feasibility review only, no experiments. Context: a ~41k-parameter Bicubic++-style same-resolution EO denoiser (half-resolution body, PixelShuffle(2), global input residual, W16A16 on Qualcomm HTP), trained on pairs from a measured synthetic camera model, with a known real-vs-synthetic gap. "Reported" = a number or claim taken from the cited source. "Inference" = my reasoning, not a sourced claim. Today: 2026-10-21.

## Q1. Why output KD adds little in regression when ground truth exists

### Takeaway
Across SR benchmarks, plain output ("logits") KD from a stronger teacher gives about 0.00 to +0.14 dB over the same student trained on ground truth alone, and feature KD (FitNet, AT) is often worse than no KD. The usual explanation is that in pixel regression the teacher's output is a noisy, smoother estimate of the same ground truth, so it carries almost no information the GT does not already carry. The gain that does exist works like target smoothing or variance reduction, not like transferring "dark knowledge".

### Cited Findings
- (2023, arXiv 2309.14162 v3 revised 2024; HKUST(GZ)/Huawei Noah's Ark) DUKD argues that teacher outputs in SR are "noisy approximations to the ground-truth distribution" of HR images. The teacher's knowledge is "shaded by GT", so output KD transfers little. Its PSNR analysis shows existing KD methods make the student only slightly closer to the teacher than training from scratch does — [Data Upcycling Knowledge Distillation for Image Super-Resolution](https://arxiv.org/html/2309.14162v3)
- DUKD tables, reported PSNR in dB. Student trained from scratch vs logits-KD from a stronger teacher, bicubic SR:
  | Pair / scale | Set5 scratch → KD | Urban100 scratch → KD |
  |---|---|---|
  | EDSR (2.70M student) x2 | 38.00 → 38.04 | 31.96 → 31.98 |
  | EDSR x4 | 32.29 → 32.30 | 26.21 → 26.21 |
  | RCAN (5.17M student) x4 | 32.31 → 32.45 | 26.37 → 26.49 |
  | SwinIR (1.24M student) x4 | 32.31 → 32.27 (worse) | 26.15 → 26.15 |

  Feature KD is often below scratch. EDSR x4 FitNet: Set5 31.65, Urban100 25.40 (scratch 32.29 / 26.21). EDSR x4 AT: Set5 32.22, Urban100 25.97. EDSR x4 FAKD is 0.02–0.03 dB below scratch on all four sets. DUKD's own method, which augments the data rather than plain output matching, reaches EDSR x4 Urban100 26.45 (+0.24 dB) — [DUKD](https://arxiv.org/html/2309.14162v3)
- DUKD also says feature-based KD has "marginal performance gains and limited applicability" and needs access to the teacher's architecture — [DUKD](https://arxiv.org/html/2309.14162v3)
- (2018, arXiv 1806.01576) Earlier lightweight-SISR work states that SISR is a regression problem with a continuous GT, so "the output of teacher network fails to provide more valuable information than the ground truth" — [Adaptive Importance Learning for Improving Lightweight Image Super-resolution Network](https://arxiv.org/pdf/1806.01576)
- (ECCV 2020) A teacher that receives the HR image as input (privileged information) "may ... just learn to copy the inputs". The authors needed a specially designed hourglass teacher for the knowledge to be usable — [Learning with Privileged Information for Efficient Image Super-Resolution](https://arxiv.org/pdf/2007.07524)
- (ECCV 2024) MTKD keeps the standard L1 loss between student output and GT and adds a wavelet-domain distillation term against an aggregated multi-teacher target. GT stays the main signal — [MTKD: Multi-Teacher Knowledge Distillation for Image Super-Resolution](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/05614.pdf)
- (ICCV 2019) For pose regression, "dark knowledge" is "not available ... and the teacher prediction is not always accurate". The authors weight the imitation loss by the teacher's own loss ("emphasize the knowledge transfer only when we trust the teacher"; Attentive Imitation Loss). Reported: up to 92.95 % fewer parameters and 2.12x faster, with the student kept "close to the teacher". The abstract gives no accuracy-gain number — [Distilling Knowledge From a Deep Pose Regressor Network (Saputra et al.)](https://arxiv.org/abs/1908.00858)
- (IEEE MIPR 2020) Regression KD has been studied far less than classification KD. The paper proposes a "teacher outlier rejection loss" and a two-head student: one head fits the possibly noisy labels, the other fits the teacher, to correct noisy labels. Its claimed benefit is in the noisy-label regime (tests on MPIIGaze, Multi-PIE and a toy sinusoid). The abstract reports no numbers — [An Efficient Method of Training Small Models for Regression Problems with Knowledge Distillation (Takamoto et al.)](https://arxiv.org/abs/2002.12597)
- (arXiv 2020) Statistical view: the teacher estimates the Bayes class-probability function, which creates "a fundamental bias-variance tradeoff in the student's objective". An approximate teacher helps only to the extent that it is close to the Bayes target — [Why distillation helps: a statistical perspective (Menon et al.)](https://arxiv.org/abs/2005.10419)
- (NeurIPS 2021) Distillation can improve student generalization but "does not typically work as it is commonly understood". A "surprisingly large discrepancy" remains between teacher and student predictions, "even in cases when the student has the capacity to perfectly match the teacher". Optimization difficulty is a key cause, and "more closely matching the teacher paradoxically does not always lead to better student generalization" — [Does Knowledge Distillation Really Work? (Stanton et al.)](https://arxiv.org/abs/2106.05945)
- Stanton et al., full text: LeNet-5 self-distillation on MNIST reaches "over 99% top-1 test agreement". ResNet-56 on CIFAR-100 reaches "nowhere near 99%" even after adding up to 50k GAN samples. In self-distillation, more fidelity lowered student test accuracy. Higher student capacity "has very little effect on fidelity" — [Stanton et al. PDF](https://arxiv.org/pdf/2106.05945)

### Inferences
- Inference (math, not sourced). For an MSE target with teacher t(x) ≈ E[y|x]: E‖s(x)−y‖² = E‖s(x)−E[y|x]‖² + E‖y−E[y|x]‖². The second term does not depend on the student. So a perfect conditional-mean teacher gives the same minimizer as GT. What it changes is gradient variance: it removes the irreducible part of the target. With an L1/Charbonnier loss the teacher sits near the conditional median instead, and the same reasoning applies.
- Inference. In this project the "label noise" that KD helps with is essentially zero: GT is a clean source image and the synthetic pair generator can produce unlimited fresh pairs. So the variance-reduction benefit of KD should be small. The SR evidence (logits-KD +0.00 to +0.14 dB, often within run-to-run noise) is a reasonable prior. For a 41k student, expect at most about 0.1 dB of synthetic PSNR from output KD.
- Inference. The observation that "texture retention is the same across very different capacities" is what you would expect from a loss-limited problem, not a capacity-limited one. A Charbonnier-trained teacher has the same regression-to-the-mean bias, so output KD from it cannot add texture (see Q2).
- Inference. Stanton's result that a student with enough capacity still fails to match the teacher means fidelity to a deep teacher should not be assumed. For a 41k student the shortfall will be larger, and fidelity does not equal quality anyway.

### Gaps
- I found no paper that measures output KD for a tiny (<100k parameter) same-resolution denoiser (not SR) with numbers.
- Stanton's exact CIFAR-100 agreement percentages are only shown in figures; I could not extract the numeric values from the text.
- Menon et al.'s publication venue was not confirmed (arXiv only on the page).

## Q2. Teacher artifact transfer: does the student inherit teacher biases, and can KD improve perceptual texture?

### Takeaway
Yes, the student inherits what the teacher outputs. Pseudo-clean teacher outputs are documented to be over-smoothed, and low-quality pseudo-labels "force the student model to learn undesirable artifacts". A teacher trained with the same pixel loss sits at the low-distortion, low-perception end of the perception-distortion trade-off, so output KD with a pixel loss cannot move the student toward more texture. Texture gains come only when the teacher was trained differently (GAN or perceptual loss) and the distillation itself uses distribution-level or adversarial terms. Even then, the student remains measurably behind the teacher.

### Cited Findings
- (CVPR 2018) "distortion and perceptual quality are at odds with each other". As mean distortion decreases, the chance of telling outputs from real images "must increase", and this holds "for any distortion measure", not only PSNR/SSIM — [The Perception-Distortion Tradeoff (Blau & Michaeli)](https://arxiv.org/abs/1711.06077)
- (ECCV 2020) In the D-BSN unpaired-denoising pipeline, the teacher's denoised outputs on real noisy images "may suffer from the over-smoothing effect". The authors mitigate this by also training on real clean images with synthetic noise. The pseudo-pair term has weight λ = 0.1 against the synthetic-pair term (Eq. 4, both MSE) — [Unpaired Learning of Deep Image Denoising (Wu et al.)](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123490341.pdf)
- (arXiv, Mar 2026) For real-world restoration with Mean-Teacher pseudo-labels: "unconditionally trusting the often imperfect, low-quality PLs forces the student model to learn undesirable artifacts", while discarding them "severely limits data diversity". The proposed fix conditions the student on pseudo-label quality, estimated by an ensemble of no-reference IQA models. The authors claim the student can then "extrapolate to generate results of higher quality than the teacher itself". The abstract gives no numbers — [QualiTeacher: Quality-Conditioned Pseudo-Labeling for Real-World Image Restoration](https://arxiv.org/abs/2603.08030)
- (CVPR 2023) In mean-teacher restoration, consistency loss "might become ineffective when the teacher's prediction is wrong", and an L1 loss "may cause the network to overfit wrong labels, resulting in confirmation bias". The fix is a "reliable bank" of best-ever outputs chosen by an NR-IQA metric plus contrastive regularization — [Contrastive Semi-Supervised Learning for Underwater Image Restoration via Reliable Bank (Semi-UIR)](https://openaccess.thecvf.com/content/CVPR2023/html/Huang_Contrastive_Semi-Supervised_Learning_for_Underwater_Image_Restoration_via_Reliable_Bank_CVPR_2023_paper.html)
- (2025, Knowledge-Based Systems) A "bad" teacher that "produces unreliable artifacts, cannot transfer the ability to enhance visual quality to students". This comes from the abstract snippet only; the full text was not read — [Reliable Image Super-Resolution Using Dual-Teacher Knowledge Distillation](https://www.sciencedirect.com/science/article/abs/pii/S0950705125018817)
- (arXiv, Oct 2025) GAN-SR distillation (Real-ESRGAN x4, GCFSR) is hard because of a "capacity mismatch between the teacher discriminator and the student generator". Reported test FID for Real-ESRGAN x4: teacher 45.62; 1/4-channel student without KD 62.94; with SDAKD 50.49. At 1/2 channels: 58.24 → 49.57. The KD student stays about 4.9 FID behind the teacher. The paper reports FID only, no PSNR or LPIPS — [SDAKD: Student Discriminator Assisted Knowledge Distillation for SR GANs](https://arxiv.org/html/2510.03870v1)

### Inferences
- Inference. If the teacher is trained with the same Charbonnier loss on the same synthetic pairs, its output is a smoother version of GT (Blau & Michaeli; Wu et al. observe this directly for pseudo-clean outputs). Mixing it into the target can only pull the student further toward the mean. Expect no gain in fine-texture retention and possibly a small loss. Run-to-run noise may hide the effect either way.
- Inference. The teacher's specific failure modes on this domain would be copied: halo or sharpening residue it fails to remove, color shifts, hot-pixel handling. The brief says the larger model was "not uniformly better on real charts", which means some of those errors would be distilled into the student.
- Inference. To change texture, change the student's own objective (frequency or perceptual terms, or an adversarial term with a small weight) or the data, not the teacher. A GAN/perceptual teacher plus distribution-level KD is the only KD route with evidence for texture gains (SDAKD). It is unproven at 41k parameters and adds hallucination risk on a security-camera-type EO feed.

### Gaps
- I found no study that measures, for the same pixel loss, whether output KD changes high-frequency texture metrics (MTF, gradient energy) in denoisers. The conclusion above is inference.
- The dual-teacher KBS 2025 paper was not read in full.

## Q3. Pseudo-labeling and self-training on unlabeled real images

### Takeaway
There is a working precedent: train a denoiser on synthetic pairs with real clean images plus a down-weighted set of (real noisy, teacher pseudo-clean) pairs. On DND it got within 0.12 dB of CBDNet without any real paired data. The documented risks are an over-smoothed or artifacted teacher, confirmation bias in iterative or mean-teacher schemes, and a student that is capped by the teacher's real-domain quality. Every successful method adds a guard: GT pairs kept as the anchor, low weight on pseudo pairs, confidence or quality weighting, a reliable bank, or using the pseudo-clean images to fit a noise or ISP model instead of as direct targets.

### Cited Findings
- (ECCV 2020) Two-stage unpaired denoising. A dilated blind-spot network (D-BSN) learns from real noisy images. Then a standard network (MWCNN) is "distilled" on (synthetic noisy, real clean) pairs plus λ = 0.1 × (real noisy, D-BSN pseudo-clean) pairs. The two sets "are complementary, and both benefit the denoising performance". Reported PSNR — [Wu et al., ECCV 2020](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123490341.pdf):
  - BSD68 AWGN σ = 25: D-BSN teacher 28.88, distilled MWCNN 29.38, supervised MWCNN(N2C) 29.41.
  - Heteroscedastic Gaussian (α = 40, δ = 10): D-BSN 29.13, distilled 30.10, N2C 30.16.
  - DND sRGB, PSNR/SSIM: distilled MWCNN(unpaired) 37.93 / 0.9373; GCBD 35.58 / 0.9217; CBDNet 38.05 / 0.9421; VDN (supervised on real pairs) 39.38 / 0.9518; BM3D 34.51 / 0.8507.
  - CC15: 35.90 vs N2S 35.38 and N2V 35.27.
  - The authors say it is "only inferior to deep models specified for real-world noisy photography", meaning CBDNet and VDN.
- (arXiv 2021) Pseudo-ISP: a pre-trained denoiser produces "pseudo clean images for the test images". Those (test, pseudo-clean) pairs are used to "jointly learn the pseudo ISP pipeline and signal-dependent rawRGB noise model", which then synthesizes noisy versions of clean images for adapting the denoiser. Alternating Pseudo-ISP training and denoiser adaptation gives further gains. Qualitatively, it lifts a Gaussian-blur denoiser to be "competitive ... against CBDNet" and improves CBDNet and RIDNet — [Pseudo-ISP: Learning Pseudo In-camera Signal Processing Pipeline from A Color Image Denoiser](https://arxiv.org/abs/2103.10234)
- Iterative self-collaboration and pseudo-pair refinement exist for real-world unsupervised denoising: a noise estimator and denoiser alternate, and the denoiser regenerates pseudo real-noise datasets. These are only listed here; numbers not extracted — [Self-Collaboration, arXiv 2308.06776 (2023)](https://arxiv.org/pdf/2308.06776); [Sparsity is All You Need, arXiv 2503.21377 (2025)](https://arxiv.org/html/2503.21377v1)
- (CVPR 2023) Self-supervised real denoisers distill from multiple blind-spot "teachers" into a student and evaluate on SIDD and DND. Numbers not extracted — [Exploring Efficient Asymmetric Blind-Spots for Self-Supervised Denoising in Real-World Scenarios](https://arxiv.org/pdf/2303.16783)
- Confirmation bias and mitigations in restoration mean-teacher setups: Semi-UIR reliable bank + contrastive regularization (CVPR 2023); QualiTeacher NR-IQA quality conditioning plus a "cropped consistency loss" against "reward hacking of the IQA models" (2026) — [Semi-UIR](https://openaccess.thecvf.com/content/CVPR2023/html/Huang_Contrastive_Semi-Supervised_Learning_for_Underwater_Image_Restoration_via_Reliable_Bank_CVPR_2023_paper.html); [QualiTeacher](https://arxiv.org/abs/2603.08030)
- In regression KD, gating the teacher by its confidence is the standard guard against wrong teacher targets: Attentive Imitation Loss (ICCV 2019) and teacher outlier rejection (MIPR 2020) — [Saputra et al.](https://arxiv.org/abs/1908.00858); [Takamoto et al.](https://arxiv.org/abs/2002.12597)
- (CVPR 2022) Cross-device real-world SR. DADA adapts an SR model trained on paired data from one camera to other cameras' LR images without target pairs, using inter- and intra-domain adversarial adaptation. It reports state-of-the-art results across six real-to-real settings on three cameras. It is UDA, not teacher pseudo-labeling — [Dual Adversarial Adaptation for Cross-Device Real-World Image Super-Resolution](https://arxiv.org/pdf/2205.03524)

### Inferences
- Inference. Note why the D-BSN student beat its teacher by 0.5–1.0 dB. The teacher was architecturally handicapped (a blind-spot net cannot see the center pixel), and the student also had real-clean synthetic pairs. This is not evidence that a larger, unconstrained teacher's pseudo-labels lift a small student above GT training. In this project the teacher has no such handicap to "escape".
- Inference. Pseudo-labels on real EO frames are worth trying only if the teacher is clearly better than the student on real data under a frozen protocol. The brief says the +0.8 dB synthetic teacher was "not uniformly better on real charts", which is a red flag. Distilling it would transfer its real-domain errors, and the student would be capped near the teacher on real data.
- Inference. The ISP problems named in the brief (sharpening halos, temporal 3D NR) mean a teacher that is wrong on halos produces pseudo-targets that keep or invent halos. Per-pixel confidence weighting is hard because there is no real GT. The usable proxies are teacher-ensemble disagreement or NR-IQA, and both can be gamed.
- Inference. The Pseudo-ISP pattern fits this project better than direct pseudo-targets: use teacher pseudo-clean real frames to check or re-fit the measured synthetic camera model (halo profile, chroma noise, hot-pixel statistics), then keep training on synthetic pairs with true clean GT. Errors in the teacher then affect only the noise-model parameters, not the student's target texture.
- Inference. If direct pseudo-pairs are tried anyway, the literature-backed recipe is: GT synthetic pairs as the main loss; pseudo pairs at λ around 0.1 (Wu et al.); a single non-iterated teacher pass, or a reliable bank, to avoid a confirmation loop; confidence or quality weighting.

### Gaps
- Exact Pseudo-ISP PSNR tables on SIDD/DND were not retrieved (PDF over the fetch size limit).
- I found no paper that reports pseudo-label KD from a large teacher into a tiny (<100k) real-world denoiser with real-benchmark numbers.
- No retrieved paper evaluates pseudo-labeling for ISP-processed 8-bit inputs with sharpening halos and temporal NR, the closest match to this camera. SIDD and DND are raw- or sRGB-captured benchmarks without strong ISP sharpening.
- Real-world SR teacher distillation (DASR, Real-ESRGAN-derived pseudo-labels) and test-time adaptation for SR were not retrieved in detail.

## Q4. KD with quantization and with structural reparameterization

### Takeaway
KD is an established tool for recovering low-bit quantization loss in SR. It is worth +0.43 to +0.95 dB at 2–4 bit, but those gains are against large drops. In the closest-matching 2026 design (82k parameters, MobileOne reparam, PixelShuffle, INT8 QAT on the fused graph), teacher-guided output KD gave only +0.09 dB. Reparameterization interacts with quantization more than with KD: the block design determines the INT8 drop, so run QAT and any KD on the fused deploy graph.

### Cited Findings
- (NeurIPS 2024) 2DQuant, a post-training quantization method for SwinIR-light. Its distillation-based calibration (DQC) aligns the quantized model's outputs and intermediate features with the FP model. Reported gains over the same method without DQC (DOBI only), Set5 x2: 4-bit 37.44 → 37.87 (+0.43), 3-bit 36.37 → 37.32 (+0.95), 2-bit 35.25 → 36.00 (+0.75). Urban100 x2 4-bit: 31.29 → 31.84 (+0.55). Remaining 4-bit drop vs FP: Set5 x2 38.15 → 37.87 (−0.28, quoted as "only a 0.28dB decrease"); Urban100 x2 32.76 → 31.84 (−0.92, computed) — [2DQuant: Low-bit Post-Training Quantization for Image Super-Resolution](https://arxiv.org/html/2406.06649v1); [NeurIPS proceedings](https://proceedings.neurips.cc/paper_files/paper/2024/file/832b20b65f655587e9c0447860406a82-Paper-Conference.pdf)
- (arXiv, Apr 2026; MAI 2026 quantized 4K SR challenge) Efficient INT8 SR: about 82k parameters, 8 MobileOne-style reparam blocks at 32 channels, global skip, PixelShuffle x3. INT8 QAT runs on the fused deploy graph after BN recalibration on 64 batches and with weight clipping. Teacher: MambaIRv2Light, output KD only. Per-pixel weight w = clip(exp(−10·e), 0.10, 0.75) from teacher error. Loss = Charbonnier + 0.02–0.015·DCT + 0.03·KD. Reported results — [Efficient INT8 Single-Image Super-Resolution via Deployment-Aware Quantization and Teacher-Guided Training](https://arxiv.org/html/2604.20291v1):
  - Teacher guidance: 29.9114 → 30.0003 dB (+0.09), SSIM 0.853 → 0.856.
  - FP32 → INT8 by block type: RepConv 30.0897 → 29.8492 (−0.24); RepDW 29.3583 → 28.7031 (−0.66); MobileOne 30.1350 → 30.0003 (−0.13).
  - The paper claims, without a direct ablation, that QAT on the multi-branch graph "causes severe performance degradation when branches are analytically fused".
  - Internal inconsistencies to note: the abstract gives 30.006 dB, Table 2 gives 30.13 dB, and the branch count differs between text and Eq. 4.
- (2026, arXiv 2605.22351) QuantSR+ uses quantization-aware distillation between a quantized student and an FP teacher. For architectures that change during training, existing distillation "can only be performed at the output positions" — [QuantSR+](https://arxiv.org/html/2605.22351)
- (AAAI 2024) Plain RepVGG suffers "more than 20% top-1 accuracy drop on ImageNet" under standard INT8 PTQ because "the original design inevitably enlarges quantization error". QARepVGG cuts the drop to "within 2%" at similar FP32 accuracy — [Make RepVGG Greater Again: A Quantization-aware Approach](https://arxiv.org/abs/2212.01593)

### Inferences
- Inference. Quantization KD helps in proportion to the quantization drop. At W16A16 the drop is normally tiny, and project memory notes that W16 QAT has been vestigial. So an FP-teacher-to-W16-student KD has little to recover. Its main value would be a self-consistency regularizer, not quality.
- Inference. Reparameterization and KD do not conflict: distill the multi-branch training graph against the teacher, then fold. Quantization-aware steps (QAT, any KD for quantization) belong on the fused graph, as in the 2026 paper. Pick reparam block types by their measured quantized drop (RepDW is worst there).
- Inference. The 2026 INT8 SR result (+0.09 dB from teacher KD at about 82k parameters, Charbonnier base loss, KD weight 0.03) is the closest published analogue. It agrees with the Q1 prior that KD for a tiny pixel-loss student is worth at most about 0.1 dB of synthetic PSNR.

### Gaps
- I found no paper on KD for W16A16 (16-bit activation) low-level models or on Qualcomm HTP specifically. All retrieved quantization-KD numbers are for INT8 or lower.
- PAMS (ECCV 2020) and its structured-knowledge-transfer loss, and Fully Quantized SR, were not retrieved.

## Q5. Evaluation pitfalls: synthetic PSNR gains vs real-camera quality

### Takeaway
Synthetic-benchmark gains often fail to transfer to real cameras. In 2017, BM3D beat learned methods tuned on synthetic noise on real photographs, and later work still finds that deep denoisers overfit their training noise. KD gains of 0.01–0.1 dB on synthetic validation sit well inside that transfer gap. Better fidelity to the teacher is not better quality. No-reference IQA, which pseudo-label pipelines rely on, can be gamed.

### Cited Findings
- (CVPR 2017) DND benchmark: "recent techniques that perform well on synthetic noise are clearly outperformed by BM3D on photographs with real noise". Real-noise evaluation scenarios "deviate strongly from those commonly used in the scientific literature" — [Benchmarking Denoising Algorithms with Real Photographs (Plötz & Roth)](https://arxiv.org/abs/1707.01313)
- (CVPR 2023) "deep models trained on Gaussian noise may perform poorly when tested on other noise distributions", and deep denoisers "often suffer from a lack of generalization ability". Masked training is proposed as a remedy — [Masked Image Training for Generalizable Deep Image Denoising](https://arxiv.org/abs/2303.13132)
- (NeurIPS 2021) Closer student-teacher matching "paradoxically does not always lead to better student generalization". In the self-distillation experiments, more fidelity lowered test accuracy — [Stanton et al.](https://arxiv.org/abs/2106.05945); [PDF](https://arxiv.org/pdf/2106.05945)
- (CVPR 2023, 2026) Pseudo-label pipelines need a trustworthy no-reference quality signal. Semi-UIR chose its NR-IQA by "empirical analysis based on the monotonicity property". QualiTeacher adds a cropped consistency loss specifically against "reward hacking of the IQA models" — [Semi-UIR](https://openaccess.thecvf.com/content/CVPR2023/html/Huang_Contrastive_Semi-Supervised_Learning_for_Underwater_Image_Restoration_via_Reliable_Bank_CVPR_2023_paper.html); [QualiTeacher](https://arxiv.org/abs/2603.08030)
- (CVPR 2018) Any distortion-optimal estimator is perception-suboptimal, so PSNR-only evaluation systematically favors smoother outputs — [Blau & Michaeli](https://arxiv.org/abs/1711.06077)
- (2023) DUKD's typical logits-KD deltas of +0.01 to +0.05 dB (for example EDSR x2 Urban100 31.96 → 31.98) are smaller than common seed-to-seed variation. DUKD reports single numbers without variance — [DUKD](https://arxiv.org/html/2309.14162v3)

### Inferences
- Inference. Any KD claim for this project needs a frozen evaluation protocol with a measured noise floor: several seeds of the no-KD baseline, because the expected KD effect (≤0.1 dB) is near typical seed variance. It should be judged on real EO captures and charts, not on synthetic-val PSNR alone.
- Inference. Recommended real-domain evaluation, consistent with the sources: (a) real paired or quasi-paired captures where available, such as static scenes with multi-frame averaging as GT, the SIDD/DND idea; (b) real-chart metrics for texture/MTF and halo overshoot, with a check that noise does not inflate sharpness scores (the project already found its chart scorer was noise-biased); (c) side-by-side owner review on the board; (d) NR-IQA used only as a secondary signal, never as the gate.
- Inference. If pseudo-labels are used, evaluate on held-out real scenes that are not in the pseudo-label pool. Otherwise the student is graded on its teacher's targets, which measures fidelity, not quality.

### Gaps
- I found no published protocol for evaluating real-world denoisers on ISP-processed 8-bit surveillance or EO video with temporal NR. Recommendation (a) above is inferred from SIDD/DND practice.
- Seed-variance numbers for tiny SR or denoise students were not found in the retrieved sources.
