from itertools import combinations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.losses.losses import UnifiedLoss, rgb_to_ycbcr


def _as_weight_map(weight: torch.Tensor | None, reference: torch.Tensor) -> torch.Tensor | None:
    if weight is None:
        return None
    while weight.dim() < reference.dim():
        weight = weight.unsqueeze(1)
    if weight.shape[1] == 1 and reference.shape[1] != 1:
        weight = weight.expand(-1, reference.shape[1], -1, -1)
    return weight


def charbonnier_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor | None = None,
    eps: float = 1e-3,
) -> torch.Tensor:
    loss_map = torch.sqrt((pred - target) ** 2 + eps**2)
    weight = _as_weight_map(weight, loss_map)
    if weight is None:
        return loss_map.mean()
    return (loss_map * weight).sum() / (weight.sum() + 1e-8)


def normalize_map(x: torch.Tensor, eps: float = 1e-6, max_value: float = 4.0) -> torch.Tensor:
    reduce_dims = tuple(range(1, x.dim()))
    scale = x.mean(dim=reduce_dims, keepdim=True).clamp_min(eps)
    return torch.clamp(x / scale, 0.0, max_value)


class SobelMagnitude(nn.Module):
    def __init__(self, mode: str = "gray"):
        super().__init__()
        self.mode = mode
        kernel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)
        kernel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)
        self.register_buffer("kernel_x", kernel_x)
        self.register_buffer("kernel_y", kernel_y)

    @staticmethod
    def rgb_to_gray(x: torch.Tensor) -> torch.Tensor:
        return 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "gray":
            x = self.rgb_to_gray(x)
            groups = 1
            kernel_x = self.kernel_x
            kernel_y = self.kernel_y
        else:
            groups = x.shape[1]
            kernel_x = self.kernel_x.repeat(groups, 1, 1, 1)
            kernel_y = self.kernel_y.repeat(groups, 1, 1, 1)

        grad_x = F.conv2d(x, kernel_x, padding=1, groups=groups)
        grad_y = F.conv2d(x, kernel_y, padding=1, groups=groups)
        return torch.abs(grad_x) + torch.abs(grad_y)


def highpass(x: torch.Tensor, kernel_size: int = 5) -> torch.Tensor:
    if kernel_size < 3:
        return x
    if kernel_size % 2 == 0:
        kernel_size += 1
    low = F.avg_pool2d(x, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
    return x - low


def lowpass(x: torch.Tensor, kernel_size: int = 15) -> torch.Tensor:
    if kernel_size < 3:
        return x
    if kernel_size % 2 == 0:
        kernel_size += 1
    return F.avg_pool2d(x, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)


def build_residual_weight(noisy: torch.Tensor, clean: torch.Tensor, cfg: dict) -> torch.Tensor:
    if not cfg.get("enabled", True):
        return torch.ones(noisy.shape[0], 1, noisy.shape[2], noisy.shape[3], device=noisy.device, dtype=noisy.dtype)

    weight = torch.ones(noisy.shape[0], 1, noisy.shape[2], noisy.shape[3], device=noisy.device, dtype=noisy.dtype)

    noise_delta_weight = float(cfg.get("noise_delta_weight", 1.0))
    if noise_delta_weight > 0:
        noise_delta = torch.mean(torch.abs(noisy.detach() - clean.detach()), dim=1, keepdim=True)
        weight = weight + noise_delta_weight * normalize_map(noise_delta, max_value=float(cfg.get("noise_delta_max", 4.0)))

    hot_pixel_weight = float(cfg.get("hot_pixel_weight", 0.0))
    if hot_pixel_weight > 0:
        threshold = float(cfg.get("hot_pixel_threshold", 0.08))
        kernel_size = int(cfg.get("hot_pixel_kernel_size", 3))
        input_max = torch.max(noisy.detach(), dim=1, keepdim=True).values
        local_mean = F.avg_pool2d(input_max, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        hot_mask = ((input_max - local_mean) > threshold).to(noisy.dtype)
        weight = weight + hot_pixel_weight * hot_mask

    chroma_weight = float(cfg.get("chroma_weight", 0.0))
    if chroma_weight > 0:
        _, noisy_cb, noisy_cr = rgb_to_ycbcr(noisy.detach())
        _, clean_cb, clean_cr = rgb_to_ycbcr(clean.detach())
        chroma_delta = torch.abs(noisy_cb - clean_cb) + torch.abs(noisy_cr - clean_cr)
        weight = weight + chroma_weight * normalize_map(chroma_delta, max_value=float(cfg.get("chroma_max", 4.0)))

    return torch.clamp(weight, 0.0, float(cfg.get("max_weight", 8.0)))


def build_teacher_agreement(teacher_outputs: dict[str, torch.Tensor], cfg: dict) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if not cfg.get("enabled", True) or len(teacher_outputs) < 2:
        return None, None

    pair_diffs = []
    for left, right in combinations(teacher_outputs.values(), 2):
        pair_diffs.append(torch.mean(torch.abs(left.detach() - right.detach()), dim=1, keepdim=True))

    disagreement = torch.stack(pair_diffs, dim=0).mean(dim=0)
    alpha = float(cfg.get("alpha", 8.0))
    agreement = torch.exp(-alpha * disagreement)
    agreement = torch.clamp(
        agreement,
        min=float(cfg.get("min", 0.25)),
        max=float(cfg.get("max", 1.0)),
    )
    return agreement.detach(), disagreement.detach()


class OnlineMultiTeacherKDLoss(nn.Module):
    """
    Online multi-teacher KD for restoration/denoise.

    Teachers are expected to be frozen. This loss keeps the supervised HR
    objective as the anchor and adds teacher-output, residual, edge, and
    high-frequency distillation terms.
    """

    def __init__(self, supervised_config: dict, kd_config: dict):
        super().__init__()
        self.supervised_criterion = UnifiedLoss(supervised_config)
        self.kd_config = kd_config if kd_config else {}
        self.teacher_configs = self.kd_config.get("teachers", {})
        self.supervised_weight = float(self.kd_config.get("supervised_weight", 1.0))
        self.edge = SobelMagnitude(mode=self.kd_config.get("edge_mode", "gray"))

    def _term_weight(self, teacher_name: str, term_name: str) -> float:
        teacher_cfg = self.teacher_configs.get(teacher_name, {})
        return float(teacher_cfg.get(f"{term_name}_weight", 0.0))

    def _agreement_for(self, agreement: torch.Tensor | None, term_name: str) -> torch.Tensor | None:
        if agreement is None:
            return None
        apply_to = self.kd_config.get("agreement", {}).get(
            "apply_to",
            ["output", "residual", "edge", "frequency"],
        )
        return agreement if term_name in apply_to else None

    def forward(
        self,
        student_out: torch.Tensor,
        clean: torch.Tensor,
        noisy: torch.Tensor,
        teacher_outputs: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        supervised_loss, supervised_parts = self.supervised_criterion(student_out, clean, noisy)
        total_loss = self.supervised_weight * supervised_loss
        loss_parts: dict[str, torch.Tensor] = {"supervised_total": supervised_loss.detach()}

        for name, value in supervised_parts.items():
            loss_parts[f"supervised_{name}"] = student_out.new_tensor(value)

        agreement, disagreement = build_teacher_agreement(
            teacher_outputs,
            self.kd_config.get("agreement", {}),
        )
        if disagreement is not None:
            loss_parts["teacher_disagreement"] = disagreement.mean()
            loss_parts["teacher_agreement"] = agreement.mean()

        residual_weight = build_residual_weight(
            noisy,
            clean,
            self.kd_config.get("residual_focus", {}),
        )

        for teacher_name, teacher_out in teacher_outputs.items():
            teacher_out = teacher_out.detach()

            output_weight = self._term_weight(teacher_name, "output")
            if output_weight > 0:
                term = charbonnier_loss(
                    student_out,
                    teacher_out,
                    weight=self._agreement_for(agreement, "output"),
                    eps=float(self.kd_config.get("eps", 1e-3)),
                )
                total_loss = total_loss + output_weight * term
                loss_parts[f"kd_{teacher_name}_output"] = term.detach()

            residual_weight_value = self._term_weight(teacher_name, "residual")
            if residual_weight_value > 0:
                student_residual = noisy - student_out
                teacher_residual = noisy - teacher_out
                weight = residual_weight
                agreement_weight = self._agreement_for(agreement, "residual")
                if agreement_weight is not None:
                    weight = weight * agreement_weight
                term = charbonnier_loss(
                    student_residual,
                    teacher_residual,
                    weight=weight,
                    eps=float(self.kd_config.get("eps", 1e-3)),
                )
                total_loss = total_loss + residual_weight_value * term
                loss_parts[f"kd_{teacher_name}_residual"] = term.detach()

            edge_weight = self._term_weight(teacher_name, "edge")
            if edge_weight > 0:
                student_edge = self.edge(student_out)
                teacher_edge = self.edge(teacher_out)
                edge_mask_threshold = float(self.kd_config.get("edge_mask_threshold", 0.0))
                edge_weight_map = None
                if edge_mask_threshold > 0:
                    edge_weight_map = (teacher_edge.detach() > edge_mask_threshold).to(student_out.dtype)
                agreement_weight = self._agreement_for(agreement, "edge")
                if agreement_weight is not None:
                    edge_weight_map = agreement_weight if edge_weight_map is None else edge_weight_map * agreement_weight
                term = charbonnier_loss(
                    student_edge,
                    teacher_edge,
                    weight=edge_weight_map,
                    eps=float(self.kd_config.get("eps", 1e-3)),
                )
                total_loss = total_loss + edge_weight * term
                loss_parts[f"kd_{teacher_name}_edge"] = term.detach()

            frequency_weight = self._term_weight(teacher_name, "frequency")
            if frequency_weight > 0:
                kernel_size = int(self.kd_config.get("frequency_kernel_size", 5))
                student_freq = highpass(student_out, kernel_size=kernel_size)
                teacher_freq = highpass(teacher_out, kernel_size=kernel_size)
                term = charbonnier_loss(
                    student_freq,
                    teacher_freq,
                    weight=self._agreement_for(agreement, "frequency"),
                    eps=float(self.kd_config.get("eps", 1e-3)),
                )
                total_loss = total_loss + frequency_weight * term
                loss_parts[f"kd_{teacher_name}_frequency"] = term.detach()

            tone_luma_weight = self._term_weight(teacher_name, "tone_luma")
            tone_chroma_weight = self._term_weight(teacher_name, "tone_chroma")
            if tone_luma_weight > 0 or tone_chroma_weight > 0:
                kernel_size = int(self.kd_config.get("tone_kernel_size", 15))
                student_y, student_cb, student_cr = rgb_to_ycbcr(student_out)
                teacher_y, teacher_cb, teacher_cr = rgb_to_ycbcr(teacher_out)

                if tone_luma_weight > 0:
                    term = charbonnier_loss(
                        lowpass(student_y, kernel_size=kernel_size),
                        lowpass(teacher_y, kernel_size=kernel_size),
                        eps=float(self.kd_config.get("eps", 1e-3)),
                    )
                    total_loss = total_loss + tone_luma_weight * term
                    loss_parts[f"kd_{teacher_name}_tone_luma"] = term.detach()

                if tone_chroma_weight > 0:
                    student_chroma = torch.cat([student_cb, student_cr], dim=1)
                    teacher_chroma = torch.cat([teacher_cb, teacher_cr], dim=1)
                    term = charbonnier_loss(
                        lowpass(student_chroma, kernel_size=kernel_size),
                        lowpass(teacher_chroma, kernel_size=kernel_size),
                        eps=float(self.kd_config.get("eps", 1e-3)),
                    )
                    total_loss = total_loss + tone_chroma_weight * term
                    loss_parts[f"kd_{teacher_name}_tone_chroma"] = term.detach()

        loss_parts["total"] = total_loss.detach()
        return total_loss, loss_parts
