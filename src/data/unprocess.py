import numpy as np
import cv2
import random

def add_unprocess_isp_noise(img, read_noise_max=0.02, shot_noise_max=10.0):
    """
    Simulates physical sensor noise by unprocessing an sRGB image to linear Bayer, 
    adding heteroscedastic noise (shot + read), and reprocessing it through a simple ISP.
    Based on the principles of 'Unprocessing Images for Learned Raw Denoising' (Brooks et al. 2019).
    Optimized for CPU using NumPy and OpenCV.
    
    Args:
        img (np.ndarray): HxWxC uint8 image.
        read_noise_max (float): Maximum allowed standard deviation for read noise.
        shot_noise_max (float): Maximum allowed multiplicative factor for shot noise.
        
    Returns:
        np.ndarray: Noisy HxWxC uint8 image with realistic chroma and luminance noise.
    """
    H, W, C = img.shape
    if C != 3: 
        return img # Ensures it runs only on 3 channel images
    
    # Ensure dimensions are even for Bayer pattern
    pad_h = H % 2
    pad_w = W % 2
    if pad_h or pad_w:
        img_pad = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
    else:
        img_pad = img
        
    H_p, W_p, _ = img_pad.shape
    
    # 1. Unprocessing: Inverse Tone/Gamma (Simplified to fixed gamma inversion)
    # Convert to float [0, 1]
    img_float = img_pad.astype(np.float32) / 255.0
    img_linear = np.power(np.maximum(img_float, 1e-8), 2.2)
    
    # 2 & 3. Add Heteroscedastic Noise Directly to RGB channels
    # Instead of Bayer mosaicing (which causes irreversible edge color fringing via interpolation),
    # we simulate the independent read/shot noise directly on the linear RGB channels.
    read_noise_std = random.uniform(0.001, read_noise_max)
    shot_noise_std = read_noise_std * random.uniform(1.0, shot_noise_max)
    
    # Calculate Variance = (lambda_shot * signal) + lambda_read^2 for all channels
    variance = shot_noise_std * img_linear + (read_noise_std ** 2)
    variance = np.maximum(variance, 1e-10) # Avoid negative variance
    
    # Independent Gaussian noise per pixel and per channel
    # AWGN preserves high-frequency 1-pixel spikes, preventing the model from confusing 
    # noise blobs with thin physical structures (like tree branches or fine textures).
    noise = np.random.normal(0.0, np.sqrt(variance)).astype(np.float32)
    
    rgb_noisy = np.clip(img_linear + noise, 0.0, 1.0)
    
    # 5. Forward Color Correction Matrix (CCM)
    # Simulates random cross-talk and color space conversion, amplifying color noise
    ccm = np.eye(3, dtype=np.float32) + np.random.normal(0.0, 0.1, (3, 3)).astype(np.float32)
    # Normalize rows to preserve luminance roughly
    ccm = ccm / np.sum(ccm, axis=1, keepdims=True)
    
    rgb_noisy = np.dot(rgb_noisy, ccm.T)
    rgb_noisy = np.clip(rgb_noisy, 0.0, 1.0)
    
    # 6. Forward Gamma
    final_rgb = np.power(np.maximum(rgb_noisy, 1e-8), 1.0 / 2.2)
    
    # Revert padding if applied
    if pad_h or pad_w:
        final_rgb = final_rgb[:H, :W, :]
        
    return (final_rgb * 255.0).astype(np.uint8)
