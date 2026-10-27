import os
import glob
import random
# float -> uint8 casts round instead of truncating. A bare astype floors, which
# biases every synthesised pixel by -0.5 DN on average; measured end-to-end the
# LR came out 0.09 DN (EO) / 0.36 DN (IR) darker than its HR target, a bias the
# model then has to spend capacity undoing. Phase-1 is unaffected because its
# only op is the resize, but Phase-2 runs the noise stack.
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from .degradations import random_mixed_kernels, circular_lowpass_kernel, random_add_poisson_noise
from .degradations import apply_camera_model, validate_camera_model_cfg
import math
from typing import Any

# =========================================================
#  Common Utils
# =========================================================

def apply_float_op(img, func, **kwargs):
    """Wrap a float32 [0,1] operation for uint8 [0,255] image"""
    img_f = img.astype(np.float32) / 255.0
    out_f = func(img_f, **kwargs)
    return np.clip(out_f * 255.0, 0, 255).round().astype(np.uint8)

def get_interpolation(name):
    """Helper to map string config to cv2 constants"""
    if name == "cv2.INTER_LINEAR": return cv2.INTER_LINEAR
    if name == "cv2.INTER_CUBIC": return cv2.INTER_CUBIC
    if name == "cv2.INTER_AREA": return cv2.INTER_AREA
    if name == "cv2.INTER_LANCZOS4": return cv2.INTER_LANCZOS4
    return cv2.INTER_CUBIC

def add_vertical_line_noise(img, prob=0.5, sigma_dn=0.75, strong_frac=0.03,
                            strong_dn=6.0, intensity_range=None, alpha=None):
    """[IR Sensor Noise] Column offsets, shaped like the sensor's actual residual.

    Every column gets its own offset, drawn from N(0, sigma_dn), and a small
    fraction get a larger one on top. That is what the deployed sensor leaves
    behind: measured 2026-09-08 on 600 frames straight off /dev/video2, after
    the sensor's own NUC, the per-column offset had a robust sigma of 0.867 DN
    with 56% of columns past 0.5 DN and 26% past 1.0 DN. It is not a fixed
    pattern -- profiles 18 seconds apart correlate at 0.046 -- so it is drawn
    fresh per sample rather than held as a calibration table.

    The previous shape was a different artefact entirely: 2-8 isolated columns
    displaced by up to +-38 DN (alpha=0.15 of full scale). Nothing on the
    sensor looks like that, and a model trained against it suppressed the real
    residual by only 8% when the measured shape was injected. Strength is
    deliberately left at the measured level rather than raised: more noise
    steers the model toward denoising, and denoising is blur -- the trap this
    project documented on the EO side.

    ``sigma_dn`` is calibrated, not assumed: injecting into RAYSENSE training
    frames and re-measuring with the same estimator used on the sensor, 0.75
    reproduces the sensor's numbers (amplitude 0.87 DN, 56% of columns past
    0.5 DN, 26% past 1.0 DN) to within the spread across images. The value is
    below the sensor's own sigma because uint8 rounding swallows part of a
    sub-DN offset, and because the estimator is calibrated end to end rather
    than on the injected quantity.

    ``intensity_range`` and ``alpha`` are accepted and ignored so that v1/v2
    configs still load; they parameterise the old shape, which no longer
    exists. Callers that set them get the measured shape instead.
    """
    if random.random() > prob:
        return img
    out = img.copy()
    is_uint8 = (out.dtype == np.uint8)
    if is_uint8:
        out = out.astype(np.float32)
    else:
        out = out * 255.0

    w = out.shape[1]
    offsets = np.random.normal(0.0, float(sigma_dn), w).astype(np.float32)
    if strong_frac > 0:
        strong = np.random.random(w) < float(strong_frac)
        offsets[strong] += np.random.uniform(-strong_dn, strong_dn, int(strong.sum()))
    out = out + offsets[None, :, None]

    out = np.clip(out, 0.0, 255.0)
    return out.round().astype(np.uint8) if is_uint8 else (out / 255.0)

def add_non_uniformity_noise(img, prob=0.5, amp_range=(0.1, 0.3)):
    """ [IR Sensor Noise] Vignetting/Shading """
    if random.random() > prob: return img
    out = img.copy()
    is_uint8 = (out.dtype == np.uint8)
    if is_uint8: out = out.astype(np.float32) / 255.0
        
    h, w, c = out.shape
    cx = w // 2 + random.randint(-w//10, w//10)
    cy = h // 2 + random.randint(-h//10, h//10)
    x = np.arange(w); y = np.arange(h)
    X, Y = np.meshgrid(x, y)
    dist_sq = (X - cx)**2 + (Y - cy)**2
    sigma = max(h, w) * random.uniform(0.6, 1.2)
    val_a = random.uniform(amp_range[0], amp_range[1])
    
    noise_map = val_a * np.exp(-dist_sq / (2 * sigma**2))
    noise_map = np.expand_dims(noise_map, axis=2)

    out = out + (noise_map - (val_a / 2))
    out = np.clip(out, 0.0, 1.0)
    return (out * 255).round().astype(np.uint8) if is_uint8 else out

def add_gaussian_noise(img, sigma_range=(1, 10), gray_prob=0.5):
    """ Gaussian Noise """
    sigma = random.uniform(sigma_range[0], sigma_range[1])
    H, W, C = img.shape
    img_float = img.astype(np.float32)
    
    if random.random() < gray_prob:
        noise = np.random.normal(0, sigma, (H, W, 1)).astype('float32')
        noise = np.repeat(noise, C, axis=2)
    else:
        noise = np.random.normal(0, sigma, (H, W, C)).astype('float32')
        
    return np.clip(img_float + noise, 0, 255).round().astype('uint8')

def apply_jpeg(img, quality_range=(30, 95)):
    """ JPEG Compression Artifacts """
    quality = random.randint(quality_range[0], quality_range[1])
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encimg = cv2.imencode('.jpg', img, encode_param)
    return cv2.imdecode(encimg, 1)

def apply_kernel(img, kernel):
    return cv2.filter2D(img, -1, kernel)


def collect_image_paths(dataset_root: Any) -> list[str]:
    """Collect image paths recursively from one or more dataset roots.

    Extensions are matched case-insensitively. Globbing for '*.png' alone made a
    directory of .PNG files come back empty, and the callers only printed a
    warning before training on nothing.
    """
    extensions = ('.png', '.jpg', '.jpeg')
    image_paths: list[str] = []
    if isinstance(dataset_root, str):
        dataset_root = [dataset_root]

    for root_dir in dataset_root:
        for path in glob.glob(os.path.join(root_dir, '**', '*'), recursive=True):
            if path.lower().endswith(extensions):
                image_paths.append(path)

    return sorted(set(image_paths))


def _stem_map(files: list[str], label: str) -> dict[str, str]:
    """Map file stem -> path, refusing to collapse two files onto one stem."""
    mapping: dict[str, str] = {}
    for path in files:
        stem = os.path.splitext(os.path.basename(path))[0]
        if stem in mapping:
            raise ValueError(
                f"{label}: duplicate stem {stem!r} from {mapping[stem]} and {path}"
            )
        mapping[stem] = path
    return mapping


def ensure_min_image_size(img: np.ndarray, min_h: int, min_w: int) -> np.ndarray:
    """Resize an image if it is smaller than the requested spatial size."""
    height, width = img.shape[:2]
    if height >= min_h and width >= min_w:
        return img
    return cv2.resize(img, (max(width, min_w), max(height, min_h)), interpolation=cv2.INTER_CUBIC)


def augment_pair(hr: np.ndarray, lr: np.ndarray, use_flip: bool = False,
                 use_rot: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Dihedral augmentation applied identically to an HR/LR pair.

    One draw drives both images, so they stay pixel-registered. Horizontal flip
    and 90-degree rotation together generate the 8-element dihedral group, which
    is exactly what `use_flip` and `use_rot` in the data configs have always
    promised and never delivered.
    """
    if use_flip and random.random() < 0.5:
        hr, lr = hr[:, ::-1], lr[:, ::-1]
    if use_flip and random.random() < 0.5:
        hr, lr = hr[::-1, :], lr[::-1, :]
    if use_rot and random.random() < 0.5:
        hr, lr = np.rot90(hr, axes=(0, 1)), np.rot90(lr, axes=(0, 1))
    return np.ascontiguousarray(hr), np.ascontiguousarray(lr)


def image_to_tensor(image_bgr: np.ndarray) -> torch.Tensor:
    """Convert a BGR uint8 image into a float RGB tensor in [0, 1]."""
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return torch.from_numpy(np.ascontiguousarray(image_rgb.transpose(2, 0, 1))).float()


# Effective defaults of apply_configured_degradation, in application order.
# op -> (default enabled, default prob). MUST stay in sync with the .get()
# fallbacks in apply_configured_degradation below; prob None means the op has
# no top-level prob gate (target_resize applies whenever enabled, ir_noise
# uses per-sub-op probs).
_DEGRADATION_PIPELINE_DEFAULTS = {
    'stage1': {
        'blur': (True, 0.5),
        'resize': (True, 0.7),
        'gaussian_noise': (True, 0.5),
        'poisson_noise': (True, 0.5),
        'jpeg': (True, 0.5),
        'sinc': (True, 0.1),
    },
    'stage2': {
        'blur': (True, 0.5),
        'target_resize': (True, None),
        'ir_noise': (True, None),
        'common_noise': (True, 0.0),
        'hot_pixels': (False, 0.5),
        'final_jpeg': (True, 0.8),
    },
}


def log_active_degradations(cfg: dict[str, Any], tag: str) -> None:
    """Print the effective degradation pipeline once, at dataset init.

    Mirrors the defaults of apply_configured_degradation so an op whose key is
    missing from the config is reported with the value the pipeline will really
    use. Greppable via the '[Degradation]' prefix. Called from __init__, which
    runs in the main process only, so it prints once per process rather than
    once per dataloader worker.
    """
    d = cfg.get('degradation', {}) or {}
    for stage_name, ops in _DEGRADATION_PIPELINE_DEFAULTS.items():
        stage_cfg = d.get(stage_name, {}) or {}
        active, off = [], []
        for op, (default_enabled, default_prob) in ops.items():
            op_cfg = stage_cfg.get(op, {}) or {}
            if not op_cfg.get('enabled', default_enabled):
                off.append(op)
            elif op == 'ir_noise':
                nu = (op_cfg.get('non_uniformity', {}) or {}).get('prob', 0.5)
                vl = (op_cfg.get('vertical_line', {}) or {}).get('prob', 0.5)
                tn = (op_cfg.get('thermal_noise', {}) or {})
                t_desc = f"p={tn.get('prob', 1.0)}" if tn.get('enabled', True) else 'off'
                active.append(f"ir_noise(non_uniformity={nu},vertical_line={vl},thermal={t_desc})")
            elif default_prob is None:
                active.append(f"{op}(always)")
            else:
                active.append(f"{op}(p={op_cfg.get('prob', default_prob)})")
        print(f"[Degradation] {tag} {stage_name} active: {', '.join(active) or '(none)'}")
        print(f"[Degradation] {tag} {stage_name} off: {', '.join(off) or '(none)'}")
        unknown = sorted(set(stage_cfg) - set(ops))
        if unknown:
            print(f"[Degradation] {tag} WARNING: {stage_name} keys not read by the pipeline: {', '.join(unknown)}")
    print(f"[Degradation] {tag} clean_prob={cfg.get('clean_prob', 0.0)}")
    cam = cfg.get('camera_model', {}) or {}
    if cam.get('enabled', False):
        modes = ', '.join(f"{m.get('name')}(p={m.get('prob')}{', impulses' if m.get('impulses') else ''})"
                          for m in cam.get('noise_modes', []))
        print(f"[Degradation] {tag} camera_model on: prob={cam.get('prob', 1.0)} target={cam.get('target', 'camera')} "
              f"(rest = stage1/stage2 replay, target = HR) "
              f"optics_sigma={cam.get('optics_sigma')} noise_scale={cam.get('noise_scale')} "
              f"hot_pixels={(cam.get('hot_pixels') or {}).get('density', 'off')} modes: {modes}")
    else:
        print(f"[Degradation] {tag} camera_model off")


def apply_configured_degradation(img_hr: np.ndarray, cfg: dict[str, Any], scale_factor: int) -> np.ndarray:
    """Apply the shared on-the-fly degradation pipeline for SR/Denoise datasets."""
    out = img_hr.copy()
    h_hr, w_hr = out.shape[:2]

    d = cfg.get('degradation', {})
    s1 = d.get('stage1', {})
    s2 = d.get('stage2', {})

    c_blur = s1.get('blur', {})
    if c_blur.get('enabled', True) and random.random() < c_blur.get('prob', 0.5):
        kernel_probs = c_blur.get('kernel_probs', [0.45, 0.25, 0.12, 0.03, 0.12, 0.03])
        kernel_sizes = c_blur.get('kernel_sizes', [7, 9, 11, 13, 15, 17, 19, 21])
        sigma_range = c_blur.get('sigma', [0.2, 3.0])
        betag_range = c_blur.get('betag', [0.5, 4.0])
        betap_range = c_blur.get('betap', [1.0, 2.0])
        kernel = random_mixed_kernels(
            kernel_list=['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso'],
            kernel_prob=kernel_probs,
            kernel_size=random.choice(kernel_sizes),
            sigma_x_range=tuple(sigma_range),
            sigma_y_range=tuple(sigma_range),
            rotation_range=(-math.pi, math.pi),
            betag_range=tuple(betag_range),
            betap_range=tuple(betap_range)
        )
        out = apply_float_op(out, apply_kernel, kernel=kernel)

    c_resize = s1.get('resize', {})
    if c_resize.get('enabled', True) and random.random() < c_resize.get('prob', 0.7):
        scale = random.uniform(c_resize.get('scale_min', 0.5), c_resize.get('scale_max', 1.0))
        h, w = out.shape[:2]
        interps = c_resize.get('interpolations', ["cv2.INTER_LINEAR", "cv2.INTER_CUBIC", "cv2.INTER_AREA"])
        interp_name = random.choice(interps)
        out = cv2.resize(out, (int(w * scale), int(h * scale)), interpolation=get_interpolation(interp_name))

    c_noise = s1.get('gaussian_noise', {})
    if c_noise.get('enabled', True) and random.random() < c_noise.get('prob', 0.5):
        out = add_gaussian_noise(
            out,
            sigma_range=(c_noise.get('sigma_min', 1), c_noise.get('sigma_max', 5)),
            gray_prob=c_noise.get('gray_prob', 0.5)
        )

    c_pnoise = s1.get('poisson_noise', {})
    if c_pnoise.get('enabled', True) and random.random() < c_pnoise.get('prob', 0.5):
        out = apply_float_op(
            out,
            random_add_poisson_noise,
            scale_range=(c_pnoise.get('scale_min', 0.05), c_pnoise.get('scale_max', 3.0)),
            gray_prob=c_pnoise.get('gray_prob', 0.5),
            clip=True,
            rounds=False
        )

    c_jpeg = s1.get('jpeg', {})
    if c_jpeg.get('enabled', True) and random.random() < c_jpeg.get('prob', 0.5):
        out = apply_jpeg(out, quality_range=(c_jpeg.get('quality_min', 85), c_jpeg.get('quality_max', 95)))

    c_sinc = s1.get('sinc', {})
    if c_sinc.get('enabled', True) and random.random() < c_sinc.get('prob', 0.1):
        kernel_size = random.choice(c_sinc.get('kernel_sizes', [7, 9, 11, 13, 15, 17, 19, 21]))
        omega_c = np.random.uniform(np.pi / 3, np.pi)
        sinc_kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=0)
        out = apply_float_op(out, apply_kernel, kernel=sinc_kernel)

    c_blur2 = s2.get('blur', {})
    if c_blur2.get('enabled', True) and random.random() < c_blur2.get('prob', 0.5):
        kernel_probs = c_blur2.get('kernel_probs', [0.45, 0.25, 0.12, 0.03, 0.12, 0.03])
        kernel_sizes = c_blur2.get('kernel_sizes', [7, 9, 11, 13, 15, 17, 19, 21])
        sigma_range = c_blur2.get('sigma', [0.2, 1.5])
        betag_range = c_blur2.get('betag', [0.5, 4.0])
        betap_range = c_blur2.get('betap', [1.0, 2.0])
        kernel = random_mixed_kernels(
            kernel_list=['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso'],
            kernel_prob=kernel_probs,
            kernel_size=random.choice(kernel_sizes),
            sigma_x_range=tuple(sigma_range),
            sigma_y_range=tuple(sigma_range),
            rotation_range=(-math.pi, math.pi),
            betag_range=tuple(betag_range),
            betap_range=tuple(betap_range)
        )
        out = apply_float_op(out, apply_kernel, kernel=kernel)

    c_resize2 = s2.get('target_resize', {})
    if c_resize2.get('enabled', True):
        target_h, target_w = h_hr // scale_factor, w_hr // scale_factor
        interps2 = c_resize2.get('interpolations', ["cv2.INTER_CUBIC", "cv2.INTER_LINEAR", "cv2.INTER_LANCZOS4"])
        interp_name = random.choice(interps2)
        out = cv2.resize(out, (target_w, target_h), interpolation=get_interpolation(interp_name))

    c_ir = s2.get('ir_noise', {})
    if c_ir.get('enabled', True):
        non_uniformity_cfg = c_ir.get('non_uniformity', {})
        if non_uniformity_cfg.get('enabled', True) and random.random() < non_uniformity_cfg.get('prob', 0.5):
            amp = (non_uniformity_cfg.get('amp_min', 0.1), non_uniformity_cfg.get('amp_max', 0.3))
            out = add_non_uniformity_noise(out, prob=1.0, amp_range=amp)

        vertical_line_cfg = c_ir.get('vertical_line', {})
        if vertical_line_cfg.get('enabled', True) and random.random() < vertical_line_cfg.get('prob', 0.5):
            # Defaults are the measured post-NUC residual (see the function).
            out = add_vertical_line_noise(
                out, prob=1.0,
                sigma_dn=vertical_line_cfg.get('sigma_dn', 0.75),
                strong_frac=vertical_line_cfg.get('strong_frac', 0.03),
                strong_dn=vertical_line_cfg.get('strong_dn', 6.0),
            )

        # thermal_noise used to ignore its own enabled flag and apply on every
        # frame. Defaults (enabled True, prob 1.0) reproduce that behavior for
        # configs that relied on it; the flags simply work now.
        t_noise = c_ir.get('thermal_noise', {})
        if t_noise.get('enabled', True) and random.random() < t_noise.get('prob', 1.0):
            out = add_gaussian_noise(
                out,
                sigma_range=(t_noise.get('sigma_min', 5), t_noise.get('sigma_max', 20)),
                gray_prob=t_noise.get('gray_prob', 0.9)
            )

    c_common = s2.get('common_noise', {})
    if c_common.get('enabled', True) and random.random() < c_common.get('prob', 0.0):
        out = add_gaussian_noise(
            out,
            sigma_range=(c_common.get('sigma_min', 1), c_common.get('sigma_max', 10)),
            gray_prob=c_common.get('gray_prob', 0.2)
        )

    c_hot = s2.get('hot_pixels', {})
    if c_hot.get('enabled', False) and random.random() < c_hot.get('prob', 0.5):
        h_out, w_out, _ = out.shape
        density = np.random.uniform(c_hot.get('density_min', 0.001), c_hot.get('density_max', 0.01))
        total_defects = int(density * h_out * w_out)
        y_coords = np.random.randint(0, h_out, total_defects)
        x_coords = np.random.randint(0, w_out, total_defects)
        for idx in range(total_defects):
            val = np.random.randint(200, 256)
            color = (val, val, val)
            if np.random.random() < c_hot.get('blob_prob', 0.4):
                radius = np.random.randint(c_hot.get('blob_radius_min', 1), c_hot.get('blob_radius_max', 8) + 1)
                shape_type = np.random.choice(['circle', 'square', 'soft_circle'])
                if shape_type == 'circle':
                    cv2.circle(out, (x_coords[idx], y_coords[idx]), radius, color, -1)
                elif shape_type == 'square':
                    cv2.rectangle(out, (x_coords[idx] - radius, y_coords[idx] - radius), (x_coords[idx] + radius, y_coords[idx] + radius), color, -1)
                else:
                    temp = np.zeros_like(out, dtype=np.float32)
                    cv2.circle(temp, (x_coords[idx], y_coords[idx]), radius, color, -1)
                    if radius > 1:
                        k_size = radius * 2 + 1
                        temp = cv2.GaussianBlur(temp, (k_size, k_size), 0)
                    mask = (temp > 0).astype(np.float32)
                    out = (out * (1 - mask) + temp * mask).round().astype(np.uint8)
            else:
                out[y_coords[idx], x_coords[idx], :] = color

    c_jpeg2 = s2.get('final_jpeg', {})
    if c_jpeg2.get('enabled', True) and random.random() < c_jpeg2.get('prob', 0.8):
        out = apply_jpeg(out, quality_range=(c_jpeg2.get('quality_min', 50), c_jpeg2.get('quality_max', 90)))

    return out

# =========================================================
#  1. Super-Resolution Dataset (Degradation on-the-fly)
# =========================================================

class SRDataset(Dataset):
    """
    Real-ESRGAN Style Degradation Pipeline for Super-Resolution
    Supports configurable degradation via 'config' (the data config dict).
    """
    def __init__(self, dataset_root, scale_factor=2, patch_size=128, is_train=True, config=None):
        super().__init__()
        self.scale_factor = scale_factor
        self.patch_size = patch_size
        self.is_train = is_train
        self.cfg = config if config else {}
        # An empty config is not "no degradation": every op would run on its default.
        if not self.cfg:
            raise ValueError("SRDataset got an empty data config; pass the recipe's data config")
        if (self.cfg.get('camera_model', {}) or {}).get('enabled', False):
            raise ValueError("camera_model is implemented for DenoiseDataset only; SRDataset would ignore it silently")
        log_active_degradations(self.cfg, tag=f"SRDataset({'train' if is_train else 'val'})")

        self.image_paths = collect_image_paths(dataset_root)

        if not self.image_paths:
            raise FileNotFoundError(
                f"SRDataset found no images under {dataset_root}. Check the data root path."
            )

    def __len__(self):
        return len(self.image_paths)

    def degradation_pipeline(self, img_hr):
        return apply_configured_degradation(img_hr=img_hr, cfg=self.cfg, scale_factor=self.scale_factor)

    def __getitem__(self, index):
        path = self.image_paths[index]
        img_hr_raw = cv2.imread(path, cv2.IMREAD_COLOR)
        if img_hr_raw is None: return self.__getitem__(random.randint(0, len(self)-1))
        
        # Random Crop (HR)
        H, W, C = img_hr_raw.shape
        lr_h, lr_w = self.patch_size, self.patch_size
        hr_h, hr_w = lr_h * self.scale_factor, lr_w * self.scale_factor
        
        # An image smaller than the HR crop used to be bicubic-upscaled to fit,
        # which makes the TARGET a bicubic upsample -- band-limited below its own
        # Nyquist and stretched on whichever axis was short, because only the
        # deficient one is grown. The model then learns to reproduce exactly the
        # bicubic softness this project is trying to remove. Measured on LSDIR
        # (median 1128x864) with the old scale-blind patch_size of 288: 13% of
        # samples at x2, 56% at x3, 93% at x4, and a 30% loss of high-frequency
        # energy in the target patch. Skip to another image instead.
        if H < hr_h or W < hr_w:
            if not getattr(self, '_warned_small', False):
                print(f"[SRDataset] HR 크롭 {hr_h}px 보다 작은 이미지는 건너뜁니다 "
                      f"(확대하면 정답이 bicubic 업샘플이 됩니다)")
                self._warned_small = True
            # Redraw at random, in a loop rather than by recursion (the recursive
            # form blew the stack at 978 frames inside a DataLoader worker).
            # LSDIR groups its files by source, so undersized images come in runs;
            # stepping to the next file sent a whole run to one image -- 1,374
            # draws per Phase-1 epoch for OST/tallgrass_019.png. The ordered scan
            # after the random draws only guarantees a fit when few images are
            # large enough.
            n = len(self.image_paths)
            for attempt in range(2 * n - 1):
                cand_index = random.randrange(n) if attempt < n else (index + attempt - n + 1) % n
                cand = cv2.imread(self.image_paths[cand_index], cv2.IMREAD_COLOR)
                if cand is not None and cand.shape[0] >= hr_h and cand.shape[1] >= hr_w:
                    img_hr_raw = cand
                    H, W, C = img_hr_raw.shape
                    break
            else:
                raise RuntimeError(
                    f"HR 크롭 {hr_h}px 를 만족하는 이미지가 데이터셋에 하나도 없습니다.")

        rnd_h = random.randint(0, H - hr_h)
        rnd_w = random.randint(0, W - hr_w)
        img_hr_patch = img_hr_raw[rnd_h:rnd_h+hr_h, rnd_w:rnd_w+hr_w, :]

        # Clean pair: 일정 확률로 열화 없이 clean 그대로 반환 (과도한 denoising 방지)
        clean_prob = self.cfg.get('clean_prob', 0.0)
        
        lr_h, lr_w = hr_h // self.scale_factor, hr_w // self.scale_factor
        
        if clean_prob > 0 and random.random() < clean_prob:
            # The clean pair still has to be built with the SAME downsampling
            # kernels as the degraded one. Hardcoding INTER_CUBIC here made the
            # clean fraction a pocket of the old single-kernel distribution --
            # the one the model was measured to overfit -- so raising clean_prob
            # would have partly undone the kernel randomisation it is paired
            # with. cv2's INTER_CUBIC does not antialias when shrinking, and
            # INTER_AREA is the only one of the four that does.
            interps = ((self.cfg.get('degradation', {}).get('stage2', {})
                        .get('target_resize', {}) or {}).get('interpolations')
                       or ["cv2.INTER_CUBIC"])
            img_lr_patch = cv2.resize(img_hr_patch, (lr_w, lr_h),
                                      interpolation=get_interpolation(random.choice(interps)))
        else:
            img_lr_patch = self.degradation_pipeline(img_hr_patch)

        # Geometric augmentation. use_flip/use_rot have been in the data configs
        # from the start but nothing ever read them -- every deployed model was
        # trained without the free 8x expansion the dihedral group gives a
        # 22.7k-parameter network. One draw drives both images so the pair stays
        # registered, and it runs after the degradation so the LR keeps the
        # artifacts of its own orientation rather than of a rotated one.
        img_hr_patch, img_lr_patch = augment_pair(
            img_hr_patch, img_lr_patch,
            use_flip=self.cfg.get('use_flip', False),
            use_rot=self.cfg.get('use_rot', False))

        # To Tensor
        img_hr = image_to_tensor(img_hr_patch)
        img_lr = image_to_tensor(img_lr_patch)
        
        return {'lr': img_lr, 'hr': img_hr, 'path': path}

# =========================================================
#  2. Denoising Dataset (Synthetic Noise)
# =========================================================

class DenoiseDataset(Dataset):
    """
    Independent denoise dataset with same-resolution clean/noisy pairs.
    It reuses the same degradation config schema as SRDataset, but without
    inheriting SR crop/scale semantics.
    """
    def __init__(self, dataset_root, scale_factor=1, patch_size=256, is_train=True, config=None):
        super().__init__()
        self.scale_factor = 1
        self.patch_size = patch_size
        self.is_train = is_train
        self.cfg = config if config else {}
        # An empty config is not "no degradation": every op would run on its default.
        if not self.cfg:
            raise ValueError("DenoiseDataset got an empty data config; pass the recipe's data config")
        # The camera model's values are measurements; a malformed block must stop
        # training here, in the main process, not inside a DataLoader worker.
        cam = self.cfg.get('camera_model', {}) or {}
        self.camera_cfg = cam if cam.get('enabled', False) else None
        if self.camera_cfg is not None:
            validate_camera_model_cfg(self.camera_cfg)
        log_active_degradations(self.cfg, tag=f"DenoiseDataset({'train' if is_train else 'val'})")

        self.image_paths = collect_image_paths(dataset_root)
        if not self.image_paths:
            raise FileNotFoundError(
                f"DenoiseDataset found no images under {dataset_root}. Check the data root path."
            )

    def __len__(self):
        return len(self.image_paths)

    def degradation_pipeline(self, img_hr):
        return apply_configured_degradation(img_hr=img_hr, cfg=self.cfg, scale_factor=self.scale_factor)

    def __getitem__(self, index):
        path = self.image_paths[index]
        img_hr_raw = cv2.imread(path, cv2.IMREAD_COLOR)
        if img_hr_raw is None:
            return self.__getitem__(random.randint(0, len(self) - 1))

        crop_h = self.patch_size
        crop_w = self.patch_size
        img_hr_raw = ensure_min_image_size(img_hr_raw, min_h=crop_h, min_w=crop_w)
        height, width, _ = img_hr_raw.shape

        rnd_h = random.randint(0, height - crop_h)
        rnd_w = random.randint(0, width - crop_w)
        img_hr_patch = img_hr_raw[rnd_h:rnd_h + crop_h, rnd_w:rnd_w + crop_w, :]

        clean_prob = self.cfg.get('clean_prob', 0.0)
        if self.camera_cfg is not None and random.random() < self.camera_cfg.get('prob', 1.0):
            # Camera pair: camera_model.target picks the target (camera's own
            # noise-free output, the pre-ISP optics image, or the source HR).
            img_lr_patch, img_hr_patch = apply_camera_model(img_hr_patch, self.camera_cfg)
            if clean_prob > 0 and random.random() < clean_prob:
                img_lr_patch = img_hr_patch.copy()
        elif clean_prob > 0 and random.random() < clean_prob:
            img_lr_patch = img_hr_patch.copy()
        else:
            img_lr_patch = self.degradation_pipeline(img_hr_patch)

        img_hr = image_to_tensor(img_hr_patch)
        img_lr = image_to_tensor(img_lr_patch)

        return {'lr': img_lr, 'hr': img_hr, 'path': path}


# =========================================================
#  4. Paired Dataset (Val / Test)
# =========================================================

class PairedDataset(Dataset):
    """
    Simple paired dataset for Validation or Inference
    """
    def __init__(self, hr_dir, lr_dir):
        super().__init__()
        if isinstance(hr_dir, str): hr_dir = [hr_dir]
        if isinstance(lr_dir, str): lr_dir = [lr_dir]
        
        self.hr_files = []
        for d in hr_dir: self.hr_files.extend(glob.glob(os.path.join(d, '*')))
        self.lr_files = []
        for d in lr_dir: self.lr_files.extend(glob.glob(os.path.join(d, '*')))
        
        # Filter only images
        extensions = ('.png', '.jpg', '.jpeg')
        self.hr_files = sorted([f for f in self.hr_files if f.lower().endswith(extensions)])
        self.lr_files = sorted([f for f in self.lr_files if f.lower().endswith(extensions)])

        # Pair by file stem, not by sorted position. Position pairing shifted every
        # pair after the first missing file and only printed a warning, so every
        # number downstream -- best-checkpoint selection, the evaluation tables, the
        # FP32-vs-quantized comparison -- stayed plausible while being meaningless.
        hr_map = _stem_map(self.hr_files, "PairedDataset HR")
        lr_map = _stem_map(self.lr_files, "PairedDataset LR")
        common = sorted(set(hr_map) & set(lr_map))
        if len(common) < max(len(hr_map), len(lr_map)):
            raise ValueError(
                f"PairedDataset stem mismatch: HR={len(hr_map)}, LR={len(lr_map)}, "
                f"matched={len(common)}. HR-only={sorted(set(hr_map) - set(lr_map))[:5]}, "
                f"LR-only={sorted(set(lr_map) - set(hr_map))[:5]}"
            )
        if not common:
            raise FileNotFoundError(
                f"PairedDataset found no image pairs. HR={hr_dir}, LR={lr_dir}"
            )
        self.pairs = [(hr_map[k], lr_map[k]) for k in common]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        hr_path, lr_path = self.pairs[index]
        
        img_hr = cv2.imread(hr_path, cv2.IMREAD_COLOR)
        img_lr = cv2.imread(lr_path, cv2.IMREAD_COLOR)
        
        img_hr = cv2.cvtColor(img_hr, cv2.COLOR_BGR2RGB)
        img_lr = cv2.cvtColor(img_lr, cv2.COLOR_BGR2RGB)
        
        img_hr = img_hr.astype(np.float32) / 255.0
        img_lr = img_lr.astype(np.float32) / 255.0
        
        img_hr = torch.from_numpy(np.ascontiguousarray(img_hr.transpose(2, 0, 1))).float()
        img_lr = torch.from_numpy(np.ascontiguousarray(img_lr.transpose(2, 0, 1))).float()
        
        return {'lr': img_lr, 'hr': img_hr, 'path': hr_path}

