import os
import glob
import random
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from .degradations import random_mixed_kernels, circular_lowpass_kernel, random_add_poisson_noise, random_add_atmospheric_turbulence
from .unprocess import add_unprocess_isp_noise
import math

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

def add_vertical_line_noise(img, prob=0.5, intensity_range=(2, 10), alpha=0.15):
    """ [IR Sensor Noise] Vertical Line Noise """
    if random.random() > prob: return img
    out = img.copy()
    is_uint8 = (out.dtype == np.uint8)
    if is_uint8: out = out.astype(np.float32) / 255.0

    h, w, c = out.shape
    num_lines = random.randint(intensity_range[0], intensity_range[1])
    
    for _ in range(num_lines):
        col = random.randint(0, w - 1)
        noise = random.uniform(-alpha, alpha)
        out[:, col, :] = out[:, col, :] + noise 
        
    out = np.clip(out, 0.0, 1.0)
    return (out * 255).astype(np.uint8) if is_uint8 else out

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
    
    # out = out - noise_map
    out = out + (noise_map - (val_a / 2))
    out = np.clip(out, 0.0, 1.0)
    return (out * 255).astype(np.uint8) if is_uint8 else out

def add_chroma_noise(img, sigma_range=(5, 20), downscale_factor=4):
    """ Chroma (Color Mottling) Noise typically seen in high ISO sensor read noise """
    sigma = random.uniform(sigma_range[0], sigma_range[1])
    H, W, C = img.shape
    img_float = img.astype(np.float32)
    
    h_down, w_down = max(H // downscale_factor, 1), max(W // downscale_factor, 1)
    
    noise = np.random.normal(0, sigma, (h_down, w_down, C)).astype('float32')
    noise_up = cv2.resize(noise, (W, H), interpolation=cv2.INTER_CUBIC)
    
    return np.clip(img_float + noise_up, 0, 255).astype('uint8')

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
        
    return np.clip(img_float + noise, 0, 255).astype('uint8')

def apply_jpeg(img, quality_range=(30, 95)):
    """ JPEG Compression Artifacts """
    quality = random.randint(quality_range[0], quality_range[1])
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encimg = cv2.imencode('.jpg', img, encode_param)
    return cv2.imdecode(encimg, 1)

def apply_kernel(img, kernel):
    return cv2.filter2D(img, -1, kernel)

# =========================================================
#  1. Super-Resolution Dataset (Degradation on-the-fly)
# =========================================================

class SRDataset(Dataset):
    """
    Real-ESRGAN Style Degradation Pipeline for Super-Resolution
    Supports configurable degradation via 'degradation_config'.
    """
    def __init__(self, dataset_root, scale_factor=2, patch_size=128, is_train=True, config=None):
        super().__init__()
        self.scale_factor = scale_factor
        self.patch_size = patch_size
        self.is_train = is_train
        self.cfg = config if config else {}
        
        # Load Defaults if cfg is empty
        if not self.cfg:
            print("[SRDataset] Warning: No degradation config provided. Using defaults.")
            pass # Defaults are handled safely with .get() in pipeline

        self.image_paths = []
        if isinstance(dataset_root, str): dataset_root = [dataset_root]
        for root_dir in dataset_root:
             self.image_paths.extend(glob.glob(os.path.join(root_dir, '**', '*.png'), recursive=True))
             self.image_paths.extend(glob.glob(os.path.join(root_dir, '**', '*.jpg'), recursive=True))
             self.image_paths.extend(glob.glob(os.path.join(root_dir, '**', '*.jpeg'), recursive=True))
        
        if not self.image_paths:
            print(f"[Warning] No images found in {dataset_root}")

    def __len__(self):
        return len(self.image_paths)

    def degradation_pipeline(self, img_hr):
        out = img_hr.copy()
        h_hr, w_hr = out.shape[:2]
        
        # Shortcuts for config sections
        d = self.cfg.get('degradation', {})
        s1 = d.get('stage1', {})
        s2 = d.get('stage2', {})
        
        # --- Stage 1: Blur & Resize & Noise ---
        
        # 1-0-0. Unprocessing ISP Noise (Physical simulation of sensor noise and demosaicing)
        c_unprocess = s1.get('unprocess_noise', {})
        if c_unprocess.get('enabled', False) and random.random() < c_unprocess.get('prob', 0.5):
            out = add_unprocess_isp_noise(
                out,
                read_noise_max=c_unprocess.get('read_noise_max', 0.02),
                shot_noise_max=c_unprocess.get('shot_noise_max', 10.0)
            )

        # [여기에 추가] 1-0. Atmospheric Turbulence (아지랑이 효과)
        c_turb = s1.get('turbulence', {})
        if c_turb.get('enabled', False) and random.random() < c_turb.get('prob', 0.5):
            out = apply_float_op(
                out, 
                random_add_atmospheric_turbulence,
                alpha_range=tuple(c_turb.get('alpha', [5, 15])),
                sigma_range=tuple(c_turb.get('sigma', [3, 7]))
            )
            
        # 1-1. Blur
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
            # 생성된 커널을 이미지에 적용 (핵심!)
            # out = cv2.filter2D(out, -1, kernel)
            out = apply_float_op(out, apply_kernel, kernel=kernel)
        
        # 1-2. Resize (Scale Jittering)
        c_resize = s1.get('resize', {})
        if c_resize.get('enabled', True) and random.random() < c_resize.get('prob', 0.7):
             scale = random.uniform(c_resize.get('scale_min', 0.5), c_resize.get('scale_max', 1.0))
             h, w = out.shape[:2]
             interps = c_resize.get('interpolations', ["cv2.INTER_LINEAR", "cv2.INTER_CUBIC", "cv2.INTER_AREA"])
             interp_name = random.choice(interps)
             interp = get_interpolation(interp_name)
             out = cv2.resize(out, (int(w*scale), int(h*scale)), interpolation=interp)

        # 1-3. Gaussian Noise
        c_noise = s1.get('gaussian_noise', {})
        if c_noise.get('enabled', True) and random.random() < c_noise.get('prob', 0.5):
            out = add_gaussian_noise(out, 
                                     sigma_range=(c_noise.get('sigma_min', 1), c_noise.get('sigma_max', 5)), 
                                     gray_prob=c_noise.get('gray_prob', 0.5))

        # 1-3-2. Poisson Noise
        c_pnoise = s1.get('poisson_noise', {})
        if c_pnoise.get('enabled', True) and random.random() < c_pnoise.get('prob', 0.5):
            #  out = random_add_poisson_noise(out, 
                #  scale_range=(c_pnoise.get('scale_min', 0.05), c_pnoise.get('scale_max', 3.0)), 
                #  gray_prob=c_pnoise.get('gray_prob', 0.5))
            out = apply_float_op(out, random_add_poisson_noise,
                                  scale_range=(c_pnoise.get('scale_min', 0.05), c_pnoise.get('scale_max', 3.0)),
                                  gray_prob=c_pnoise.get('gray_prob', 0.5),
                                  clip=True, rounds=False)
            
        # 1-4. JPEG
        c_jpeg = s1.get('jpeg', {})
        if c_jpeg.get('enabled', True) and random.random() < c_jpeg.get('prob', 0.5):
            out = apply_jpeg(out, quality_range=(c_jpeg.get('quality_min', 85), c_jpeg.get('quality_max', 95)))

        # --- Stage 2: Final Resize & Sensor Noise ---
        # 2-0. Sinc Filter
        c_sinc = s1.get('sinc', {})
        if c_sinc.get('enabled', True) and random.random() < c_sinc.get('prob', 0.1):
            kernel_size = random.choice(c_sinc.get('kernel_sizes', [7, 9, 11, 13, 15, 17, 19, 21]))
            omega_c = np.random.uniform(np.pi / 3, np.pi) # Cutoff frequency
            
            sinc_kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=0)
            # out = cv2.filter2D(out, -1, sinc_kernel)
            out = apply_float_op(out, apply_kernel, kernel=sinc_kernel)

        # 2-1. Blur
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
            # 생성된 커널을 이미지에 적용 (핵심!)
            # out = cv2.filter2D(out, -1, kernel)
            out = apply_float_op(out, apply_kernel, kernel=kernel)

        # 2-2. Target Resize
        target_h, target_w = h_hr // self.scale_factor, w_hr // self.scale_factor
        c_resize2 = s2.get('target_resize', {})
        interps2 = c_resize2.get('interpolations', ["cv2.INTER_CUBIC", "cv2.INTER_LINEAR", "cv2.INTER_LANCZOS4"])
        interp_name = random.choice(interps2)
        out = cv2.resize(out, (target_w, target_h), interpolation=get_interpolation(interp_name))
        
        # 2-3. Sensor Specific Noise (IR or Common)
        c_ir = s2.get('ir_noise', {})
        if c_ir.get('enabled', True):
            non_uniformity_cfg = c_ir.get('non_uniformity', {})
            if random.random() < non_uniformity_cfg.get('prob', 0.5):
                amp = (non_uniformity_cfg.get('amp_min', 0.1), non_uniformity_cfg.get('amp_max', 0.3))
                out = add_non_uniformity_noise(out, prob=1.0, amp_range=amp) # prob handled by outer if

            vertical_line_cfg = c_ir.get('vertical_line', {})
            if random.random() < vertical_line_cfg.get('prob', 0.5):
                ints = (vertical_line_cfg.get('intensity_min', 2), vertical_line_cfg.get('intensity_max', 10))
                out = add_vertical_line_noise(out, prob=1.0, intensity_range=ints)

            # Thermal Noise
            t_noise = c_ir.get('thermal_noise', {})
            out = add_gaussian_noise(out, 
                                    sigma_range=(t_noise.get('sigma_min', 5), t_noise.get('sigma_max', 20)), 
                                    gray_prob=t_noise.get('gray_prob', 0.9))
        # Common/EO Mode
        c_common = s2.get('common_noise', {})
        if c_common.get('enabled', True) and random.random() < c_common.get('prob', 0.0): # Default 0 to skip if not set
            out = add_gaussian_noise(out,
                                    sigma_range=(c_common.get('sigma_min', 1), c_common.get('sigma_max', 10)),
                                    gray_prob=c_common.get('gray_prob', 0.2))

        # 2-3-1. Chroma Noise
        c_chroma = s2.get('chroma_noise', {})
        if c_chroma.get('enabled', False) and random.random() < c_chroma.get('prob', 0.5):
            out = add_chroma_noise(out,
                                   sigma_range=(c_chroma.get('sigma_min', 5), c_chroma.get('sigma_max', 20)),
                                   downscale_factor=c_chroma.get('downscale', 4))


        # 2-3-2. Hot Pixels & Blobs (White Artifact 학습용)
        c_hot = s2.get('hot_pixels', {})
        if c_hot.get('enabled', False) and random.random() < c_hot.get('prob', 0.5):
            h_out, w_out, c_out = out.shape
            density = np.random.uniform(c_hot.get('density_min', 0.001), c_hot.get('density_max', 0.01))
            total_defects = int(density * h_out * w_out)
            y_coords = np.random.randint(0, h_out, total_defects)
            x_coords = np.random.randint(0, w_out, total_defects)
            for idx in range(total_defects):
                val = np.random.randint(200, 256)
                color = (val, val, val)
                if np.random.random() < c_hot.get('blob_prob', 0.4):
                    radius = np.random.randint(c_hot.get('blob_radius_min', 1), c_hot.get('blob_radius_max', 8) + 1)
                    
                    # Randomly choose between soft circle and hard square for optical/digital zoom artifacts
                    shape_type = np.random.choice(['circle', 'square', 'soft_circle'])
                    if shape_type == 'circle':
                        cv2.circle(out, (x_coords[idx], y_coords[idx]), radius, color, -1)
                    elif shape_type == 'square':
                        cv2.rectangle(out, 
                                      (x_coords[idx]-radius, y_coords[idx]-radius), 
                                      (x_coords[idx]+radius, y_coords[idx]+radius), 
                                      color, -1)
                    else: # soft_circle
                        # Draw on a temp mask, blur, and blend
                        temp = np.zeros_like(out, dtype=np.float32)
                        cv2.circle(temp, (x_coords[idx], y_coords[idx]), radius, color, -1)
                        if radius > 1:
                            k_size = radius * 2 + 1
                            temp = cv2.GaussianBlur(temp, (k_size, k_size), 0)
                        mask = (temp > 0).astype(np.float32)
                        out = (out * (1 - mask) + temp * mask).astype(np.uint8)
                else:
                    out[y_coords[idx], x_coords[idx], :] = color

        # 2-4. Final JPEG
        c_jpeg2 = s2.get('final_jpeg', {})
        if c_jpeg2.get('enabled', True) and random.random() < c_jpeg2.get('prob', 0.8):
             out = apply_jpeg(out, quality_range=(c_jpeg2.get('quality_min', 50), c_jpeg2.get('quality_max', 90)))
            
        return out

    def __getitem__(self, index):
        path = self.image_paths[index]
        img_hr_raw = cv2.imread(path, cv2.IMREAD_COLOR)
        if img_hr_raw is None: return self.__getitem__(random.randint(0, len(self)-1))
        
        # Random Crop (HR)
        H, W, C = img_hr_raw.shape
        lr_h, lr_w = self.patch_size, self.patch_size
        hr_h, hr_w = lr_h * self.scale_factor, lr_w * self.scale_factor
        
        # Ensure image is large enough
        if H < hr_h or W < hr_w:
            img_hr_raw = cv2.resize(img_hr_raw, (max(W, hr_w), max(H, hr_h)), interpolation=cv2.INTER_CUBIC)
            H, W, C = img_hr_raw.shape

        rnd_h = random.randint(0, H - hr_h)
        rnd_w = random.randint(0, W - hr_w)
        img_hr_patch = img_hr_raw[rnd_h:rnd_h+hr_h, rnd_w:rnd_w+hr_w, :]

        # Clean pair: 일정 확률로 열화 없이 clean 그대로 반환 (과도한 denoising 방지)
        clean_prob = self.cfg.get('clean_prob', 0.0)
        
        lr_h, lr_w = hr_h // self.scale_factor, hr_w // self.scale_factor
        
        if clean_prob > 0 and random.random() < clean_prob:
            img_lr_patch = cv2.resize(img_hr_patch, (lr_w, lr_h), interpolation=cv2.INTER_CUBIC)
        else:
            img_lr_patch = self.degradation_pipeline(img_hr_patch)
        
        # To Tensor
        img_hr_final = cv2.cvtColor(img_hr_patch, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img_lr_final = cv2.cvtColor(img_lr_patch, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        
        img_hr = torch.from_numpy(np.ascontiguousarray(img_hr_final.transpose(2, 0, 1))).float()
        img_lr = torch.from_numpy(np.ascontiguousarray(img_lr_final.transpose(2, 0, 1))).float()
        
        return {'lr': img_lr, 'hr': img_hr, 'path': path}

# =========================================================
#  2. Denoising Dataset (Synthetic Noise)
# =========================================================

class RealisticNoiseGenerator:
    def __init__(self, config=None):
        self.cfg = config if config else {}
        # Load probs
        probs = self.cfg.get('probs', {})
        self.poisson_prob = probs.get('poisson', 0.8)
        self.hot_pixels_prob = probs.get('hot_pixels', 0.8)
        self.row_noise_prob = probs.get('row_noise', 0.5)
        self.jpeg_prob = probs.get('jpeg', 0.2)

    def add_poisson_gaussian_noise(self, image):
        cfg = self.cfg.get('poisson', {})
        image_float = image.astype(np.float32) / 255.0
        a = np.random.uniform(cfg.get('scale_min', 0.0005), cfg.get('scale_max', 0.002)) 
        b = np.random.uniform(0.00001, 0.0005)
        variance = np.maximum(a * image_float + b, 1e-10)
        noise = np.random.randn(*image.shape) * np.sqrt(variance)
        return np.clip((image_float + noise) * 255.0, 0, 255).astype(np.uint8)

    def add_row_noise(self, image):
        cfg = self.cfg.get('row_noise', {})
        out = image.astype(np.float32)
        h, w, c = out.shape
        sigma = np.random.uniform(cfg.get('sigma_min', 5.0), cfg.get('sigma_max', 25.0))
        
        if np.random.random() < cfg.get('channel_independent_prob', 0.3):
            row_offsets = np.random.normal(0, sigma, (h, 1, c))
        else:
            row_offsets = np.random.normal(0, sigma, (h, 1))
            row_offsets = np.repeat(row_offsets, c, axis=1).reshape(h, 1, c)

        out = out + row_offsets
        return np.clip(out, 0, 255).astype(np.uint8)

    def add_hot_pixels_and_blobs(self, image):
        cfg = self.cfg.get('hot_pixels', {})
        out = np.copy(image)
        h, w, c = image.shape

        density = np.random.uniform(cfg.get('density_min', 0.001), cfg.get('density_max', 0.008)) 
        total_defects = int(density * h * w)

        y_coords = np.random.randint(0, h, total_defects)
        x_coords = np.random.randint(0, w, total_defects)

        for i in range(total_defects):
            if np.random.random() < (1.0 - cfg.get('colored_prob', 0.5)): # If colored_prob is 0.5, then 0.5 chance gray
                val = np.random.randint(200, 256)
                color = (val, val, val)
            else:
                r = np.random.randint(150, 256)
                g = np.random.randint(150, 256)
                b = np.random.randint(150, 256)
                color = (int(r), int(g), int(b))

            if np.random.random() < cfg.get('blob_prob', 0.3):
                radius = np.random.randint(cfg.get('blob_radius_min', 1), cfg.get('blob_radius_max', 4)+1)
                shape_type = np.random.choice(['circle', 'square', 'soft_circle'])
                if shape_type == 'circle':
                    cv2.circle(out, (x_coords[i], y_coords[i]), radius, color, -1)
                elif shape_type == 'square':
                    cv2.rectangle(out, 
                                  (x_coords[i]-radius, y_coords[i]-radius), 
                                  (x_coords[i]+radius, y_coords[i]+radius), 
                                  color, -1)
                else: # soft_circle
                    temp = np.zeros_like(out, dtype=np.float32)
                    cv2.circle(temp, (x_coords[i], y_coords[i]), radius, color, -1)
                    if radius > 1:
                        k_size = radius * 2 + 1
                        temp = cv2.GaussianBlur(temp, (k_size, k_size), 0)
                    mask = (temp > 0).astype(np.float32)
                    out = (out * (1 - mask) + temp * mask).astype(np.uint8)
            else:
                out[y_coords[i], x_coords[i], :] = color

        return out

    def add_jpeg_compression(self, image):
        cfg = self.cfg.get('jpeg', {})
        quality = random.randint(cfg.get('quality_min', 75), cfg.get('quality_max', 95))
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        _, encimg = cv2.imencode('.jpg', img_bgr, encode_param)
        decimg = cv2.imdecode(encimg, 1)
        return cv2.cvtColor(decimg, cv2.COLOR_BGR2RGB)

    def __call__(self, image):
        noisy = image.copy()
        h, w = noisy.shape[:2]

        # Filter transforms based on enabled flag
        active_transforms = []
        
        # Check config and add to active list
        if self.cfg.get('row_noise', {}).get('enabled', True):
            active_transforms.append(('row', self.add_row_noise, self.row_noise_prob))
        
        if self.cfg.get('poisson', {}).get('enabled', True):
            active_transforms.append(('poisson', self.add_poisson_gaussian_noise, self.poisson_prob))
            
        if self.cfg.get('hot_pixels', {}).get('enabled', True):
            active_transforms.append(('hotpixel', self.add_hot_pixels_and_blobs, self.hot_pixels_prob))
            
        if self.cfg.get('jpeg', {}).get('enabled', True):
            active_transforms.append(('jpeg', self.add_jpeg_compression, self.jpeg_prob))
            
        if self.cfg.get('chroma_noise', {}).get('enabled', False):
            c_cfg = self.cfg['chroma_noise']
            prob = c_cfg.get('prob', 0.5)
            sigma = c_cfg.get('sigma', [5, 20])
            downscale = c_cfg.get('downscale', 4)
            active_transforms.append(('chroma', lambda x: add_chroma_noise(x, sigma_range=sigma, downscale_factor=downscale), prob))

        random.shuffle(active_transforms)
        
        for name, func, prob in active_transforms:
            if random.random() < prob:
                noisy = func(noisy)
        return noisy

class DenoiseDataset(SRDataset):
    """
    DenoiseDataset using the advanced degradation pipeline from SRDataset.
    It expects scale_factor=1.
    """
    def __init__(self, dataset_root, scale_factor=1, patch_size=256, is_train=True, config=None):
        super().__init__(dataset_root, scale_factor=1, patch_size=patch_size, is_train=is_train, config=config)

    def __getitem__(self, index):
        # We reuse __getitem__ from SRDataset directly, since it now handles degradation pipeline calling seamlessly.
        return super().__getitem__(index)

# =========================================================
#  3. Guided Super-Resolution Dataset (Paired LR/HR/Guide)
# =========================================================

class GuidedSRDataset(Dataset):
    """
    Guided Super-Resolution Dataset.
    
    두 가지 모드:
    - Paired 모드 (lr_dirs 지정): 미리 생성된 LR 사용 (bicubic 등)
    - Degradation 모드 (lr_dirs=None, config에 degradation 존재): HR에서 on-the-fly LR 생성
    
    Guide(RGB)는 항상 HR 해상도로 제공됩니다.
    """
    def __init__(self, hr_dirs, lr_dirs, guide_dirs, patch_size=128, scale_factor=2, is_train=True, config=None):
        super().__init__()
        self.patch_size = patch_size
        self.scale_factor = scale_factor
        self.is_train = is_train
        self.cfg = config if config else {}

        # Degradation 모드 판단: lr_dirs가 없고 config에 degradation이 있으면 on-the-fly
        self.use_degradation = (lr_dirs is None) and bool(self.cfg.get('degradation'))
        if self.use_degradation:
            # SRDataset의 degradation_pipeline 재사용을 위해 임시 인스턴스 생성
            self._sr_dataset = SRDataset(dataset_root=[], scale_factor=scale_factor, config=self.cfg)
            print("[GuidedSRDataset] Degradation mode: on-the-fly LR from HR")

        extensions = ('.png', '.jpg', '.jpeg')

        # 경로를 리스트로 통일
        if isinstance(hr_dirs, str): hr_dirs = [hr_dirs]
        if isinstance(guide_dirs, str): guide_dirs = [guide_dirs]

        self.pairs = []  # [(hr_path, lr_path_or_None, guide_path), ...]

        if self.use_degradation:
            # Degradation 모드: HR + Guide만 매칭
            for hr_dir, guide_dir in zip(hr_dirs, guide_dirs):
                hr_map = {}
                for f in sorted(glob.glob(os.path.join(hr_dir, '*'))):
                    if f.lower().endswith(extensions):
                        stem = os.path.splitext(os.path.basename(f))[0]
                        hr_map[stem] = f

                guide_map = {}
                for f in sorted(glob.glob(os.path.join(guide_dir, '*'))):
                    if f.lower().endswith(extensions):
                        stem = os.path.splitext(os.path.basename(f))[0]
                        guide_map[stem] = f

                common = sorted(set(hr_map.keys()) & set(guide_map.keys()))
                if len(common) < len(hr_map):
                    print(f"[GuidedSRDataset] {hr_dir}: {len(hr_map)} HR, {len(guide_map)} Guide -> {len(common)} matched")

                for stem in common:
                    self.pairs.append((hr_map[stem], None, guide_map[stem]))
        else:
            # Paired 모드: HR + LR + Guide 매칭
            if isinstance(lr_dirs, str): lr_dirs = [lr_dirs]

            for hr_dir, lr_dir, guide_dir in zip(hr_dirs, lr_dirs, guide_dirs):
                hr_map = {}
                for f in sorted(glob.glob(os.path.join(hr_dir, '*'))):
                    if f.lower().endswith(extensions):
                        stem = os.path.splitext(os.path.basename(f))[0]
                        hr_map[stem] = f

                lr_map = {}
                for f in sorted(glob.glob(os.path.join(lr_dir, '*'))):
                    if f.lower().endswith(extensions):
                        stem = os.path.splitext(os.path.basename(f))[0]
                        lr_map[stem] = f

                guide_map = {}
                for f in sorted(glob.glob(os.path.join(guide_dir, '*'))):
                    if f.lower().endswith(extensions):
                        stem = os.path.splitext(os.path.basename(f))[0]
                        guide_map[stem] = f

                common = sorted(set(hr_map.keys()) & set(lr_map.keys()) & set(guide_map.keys()))
                if len(common) < len(hr_map):
                    print(f"[GuidedSRDataset] {hr_dir}: {len(hr_map)} HR, {len(lr_map)} LR, {len(guide_map)} Guide -> {len(common)} matched")

                for stem in common:
                    self.pairs.append((hr_map[stem], lr_map[stem], guide_map[stem]))

        if not self.pairs:
            print(f"[Warning] GuidedSRDataset: No matched pairs found.")
        else:
            print(f"[GuidedSRDataset] Total {len(self.pairs)} pairs loaded.")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        hr_path, lr_path, guide_path = self.pairs[index]

        img_hr = cv2.imread(hr_path, cv2.IMREAD_COLOR)
        img_guide = cv2.imread(guide_path, cv2.IMREAD_COLOR)

        if self.use_degradation:
            img_lr = None  # on-the-fly로 생성
        else:
            img_lr = cv2.imread(lr_path, cv2.IMREAD_COLOR)

        if img_hr is None or img_guide is None or (not self.use_degradation and img_lr is None):
            return self.__getitem__(random.randint(0, len(self) - 1))

        # Guide(RGB)를 HR(thermal) 해상도에 맞춤 (FLIR 등 HR/RGB 해상도가 다른 경우)
        hr_h, hr_w = img_hr.shape[:2]
        guide_h, guide_w = img_guide.shape[:2]
        if guide_h != hr_h or guide_w != hr_w:
            img_guide = cv2.resize(img_guide, (hr_w, hr_h), interpolation=cv2.INTER_CUBIC)

        if self.is_train:
            # HR 기준으로 랜덤 크롭 좌표 결정
            hr_h, hr_w = img_hr.shape[:2]
            crop_hr = self.patch_size * self.scale_factor

            if hr_h < crop_hr or hr_w < crop_hr:
                new_h = max(hr_h, crop_hr)
                new_w = max(hr_w, crop_hr)
                img_hr = cv2.resize(img_hr, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
                img_guide = cv2.resize(img_guide, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
                if not self.use_degradation and img_lr is not None:
                    img_lr = cv2.resize(img_lr, (new_w // self.scale_factor, new_h // self.scale_factor), interpolation=cv2.INTER_CUBIC)
                hr_h, hr_w = img_hr.shape[:2]

            rnd_h = random.randint(0, hr_h - crop_hr)
            rnd_w = random.randint(0, hr_w - crop_hr)

            img_hr = img_hr[rnd_h:rnd_h + crop_hr, rnd_w:rnd_w + crop_hr]
            img_guide = img_guide[rnd_h:rnd_h + crop_hr, rnd_w:rnd_w + crop_hr]

            if self.use_degradation:
                # Degradation 모드: HR 패치에서 on-the-fly LR 생성
                img_lr = self._sr_dataset.degradation_pipeline(img_hr)
            else:
                # Paired 모드: LR에서 대응 좌표 크롭
                crop_lr = self.patch_size
                lr_rnd_h = rnd_h // self.scale_factor
                lr_rnd_w = rnd_w // self.scale_factor
                img_lr = img_lr[lr_rnd_h:lr_rnd_h + crop_lr, lr_rnd_w:lr_rnd_w + crop_lr]

            # 랜덤 플립/회전 (HR, LR, Guide 동일하게 적용)
            if random.random() < 0.5:
                img_hr = np.flip(img_hr, axis=1).copy()
                img_lr = np.flip(img_lr, axis=1).copy()
                img_guide = np.flip(img_guide, axis=1).copy()
            if random.random() < 0.5:
                img_hr = np.flip(img_hr, axis=0).copy()
                img_lr = np.flip(img_lr, axis=0).copy()
                img_guide = np.flip(img_guide, axis=0).copy()
            if random.random() < 0.5:
                img_hr = np.rot90(img_hr).copy()
                img_lr = np.rot90(img_lr).copy()
                img_guide = np.rot90(img_guide).copy()

        # BGR -> RGB, normalize, to tensor
        img_hr = cv2.cvtColor(img_hr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img_lr = cv2.cvtColor(img_lr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img_guide = cv2.cvtColor(img_guide, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        hr_t = torch.from_numpy(np.ascontiguousarray(img_hr.transpose(2, 0, 1))).float()
        lr_t = torch.from_numpy(np.ascontiguousarray(img_lr.transpose(2, 0, 1))).float()
        guide_t = torch.from_numpy(np.ascontiguousarray(img_guide.transpose(2, 0, 1))).float()

        return {'lr': lr_t, 'hr': hr_t, 'guide': guide_t, 'path': hr_path}

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
        
        if len(self.hr_files) != len(self.lr_files):
            print(f"[Warning] Mismatch in PairedDataset: HR={len(self.hr_files)}, LR={len(self.lr_files)}")
            # Truncate to min
            min_len = min(len(self.hr_files), len(self.lr_files))
            self.hr_files = self.hr_files[:min_len]
            self.lr_files = self.lr_files[:min_len]

    def __len__(self):
        return len(self.hr_files)

    def __getitem__(self, index):
        hr_path = self.hr_files[index]
        lr_path = self.lr_files[index]
        
        img_hr = cv2.imread(hr_path, cv2.IMREAD_COLOR)
        img_lr = cv2.imread(lr_path, cv2.IMREAD_COLOR)
        
        img_hr = cv2.cvtColor(img_hr, cv2.COLOR_BGR2RGB)
        img_lr = cv2.cvtColor(img_lr, cv2.COLOR_BGR2RGB)
        
        img_hr = img_hr.astype(np.float32) / 255.0
        img_lr = img_lr.astype(np.float32) / 255.0
        
        img_hr = torch.from_numpy(np.ascontiguousarray(img_hr.transpose(2, 0, 1))).float()
        img_lr = torch.from_numpy(np.ascontiguousarray(img_lr.transpose(2, 0, 1))).float()
        
        return {'lr': img_lr, 'hr': img_hr, 'path': hr_path}

class GuidedPairedDataset(Dataset):
    """
    Simple paired dataset for Validation or Inference of Guided-SR Models
    """
    def __init__(self, hr_dir, lr_dir, guide_dir):
        super().__init__()
        if isinstance(hr_dir, str): hr_dir = [hr_dir]
        if isinstance(lr_dir, str): lr_dir = [lr_dir]
        if isinstance(guide_dir, str): guide_dir = [guide_dir]

        self.hr_files = []
        for d in hr_dir: self.hr_files.extend(glob.glob(os.path.join(d, '*')))
        self.lr_files = []
        for d in lr_dir: self.lr_files.extend(glob.glob(os.path.join(d, '*')))
        self.guide_files = []
        for d in guide_dir: self.guide_files.extend(glob.glob(os.path.join(d, '*')))
        
        # Filter only images
        extensions = ('.png', '.jpg', '.jpeg')
        self.hr_files = sorted([f for f in self.hr_files if f.lower().endswith(extensions)])
        self.lr_files = sorted([f for f in self.lr_files if f.lower().endswith(extensions)])
        self.guide_files = sorted([f for f in self.guide_files if f.lower().endswith(extensions)])
        
        # We assume file names map 1:1 in sorted order. For robustness, we could map by name, 
        # but sorted order is acceptable if directories are identical in clean setups. 
        # A more robust check:
        import os as _os
        self.pairs = []
        hr_map = {_os.path.splitext(_os.path.basename(f))[0]: f for f in self.hr_files}
        lr_map = {_os.path.splitext(_os.path.basename(f))[0]: f for f in self.lr_files}
        guide_map = {_os.path.splitext(_os.path.basename(f))[0]: f for f in self.guide_files}
        
        common = sorted(set(hr_map.keys()) & set(lr_map.keys()) & set(guide_map.keys()))
        for stem in common:
            self.pairs.append((hr_map[stem], lr_map[stem], guide_map[stem]))
            
        if len(self.pairs) != len(self.hr_files):
            print(f"[Warning] Mismatch in GuidedPairedDataset: HR={len(self.hr_files)}, LR={len(self.lr_files)}, Guide={len(self.guide_files)} -> {len(self.pairs)} matched")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        hr_path, lr_path, guide_path = self.pairs[index]
        
        img_hr = cv2.imread(hr_path, cv2.IMREAD_COLOR)
        img_lr = cv2.imread(lr_path, cv2.IMREAD_COLOR)
        img_guide = cv2.imread(guide_path, cv2.IMREAD_COLOR)
        
        img_hr = cv2.cvtColor(img_hr, cv2.COLOR_BGR2RGB)
        img_lr = cv2.cvtColor(img_lr, cv2.COLOR_BGR2RGB)
        img_guide = cv2.cvtColor(img_guide, cv2.COLOR_BGR2RGB)
        
        img_hr = img_hr.astype(np.float32) / 255.0
        img_lr = img_lr.astype(np.float32) / 255.0
        img_guide = img_guide.astype(np.float32) / 255.0
        
        img_hr = torch.from_numpy(np.ascontiguousarray(img_hr.transpose(2, 0, 1))).float()
        img_lr = torch.from_numpy(np.ascontiguousarray(img_lr.transpose(2, 0, 1))).float()
        img_guide = torch.from_numpy(np.ascontiguousarray(img_guide.transpose(2, 0, 1))).float()
        
        return {'lr': img_lr, 'hr': img_hr, 'guide': img_guide, 'path': hr_path}
