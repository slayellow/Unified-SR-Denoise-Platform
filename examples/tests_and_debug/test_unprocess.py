import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import cv2
from src.data.unprocess import add_unprocess_isp_noise

def test_color_swap():
    # Create a pure blue BGR image
    H, W = 100, 100
    bgr_img = np.zeros((H, W, 3), dtype=np.uint8)
    bgr_img[:, :, 0] = 255  # Pure Blue
    
    noisy_bgr = add_unprocess_isp_noise(bgr_img, read_noise_max=0.01, shot_noise_max=1.0)
    
    avg_b = np.mean(noisy_bgr[:, :, 0])
    avg_g = np.mean(noisy_bgr[:, :, 1])
    avg_r = np.mean(noisy_bgr[:, :, 2])
    
    print(f"Original Input : B={255}, G={0}, R={0}")
    print(f"Noisy Output   : B={avg_b:.2f}, G={avg_g:.2f}, R={avg_r:.2f}")
    
    if avg_b > avg_r and avg_b > avg_g:
        print("✅ SUCCESS: Color channels are preserved correctly (Blue remains max).")
    else:
        print("❌ FAILED: Color channels are swapped or distorted.")

if __name__ == '__main__':
    test_color_swap()
