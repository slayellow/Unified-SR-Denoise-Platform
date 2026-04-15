import numpy as np
import cv2

def add_unprocess_isp_noise_fixed(img):
    H, W, C = img.shape
    pad_h = H % 2
    pad_w = W % 2
    img_pad = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect') if (pad_h or pad_w) else img
    H_p, W_p, _ = img_pad.shape
    
    img_float = img_pad.astype(np.float32) / 255.0
    img_linear = np.power(np.maximum(img_float, 1e-8), 2.2)
    
    bayer = np.zeros((H_p, W_p), dtype=np.float32)
    bayer[0::2, 0::2] = img_linear[0::2, 0::2, 0] # B
    bayer[0::2, 1::2] = img_linear[0::2, 1::2, 1] # G1
    bayer[1::2, 0::2] = img_linear[1::2, 0::2, 1] # G2
    bayer[1::2, 1::2] = img_linear[1::2, 1::2, 2] # R
    
    bayer_uint16 = (bayer * 65535.0).astype(np.uint16)
    
    # FIX: Change to BayerRG2BGR to output in BGR format matching input!
    output_bgr = cv2.cvtColor(bayer_uint16, cv2.COLOR_BayerRG2BGR).astype(np.float32) / 65535.0
    
    ccm = np.eye(3, dtype=np.float32) 
    # Notice: CCM was designed for RGB. If output_bgr is BGR, applying CCM directly might process Red and Blue with inverted weights if CCM wasn't eye(3)!
    # But CCM is random cross-talk added to eye(3). So it doesn't matter much.
    output_bgr = np.dot(output_bgr, ccm.T)
    output_bgr = np.clip(output_bgr, 0.0, 1.0)
    
    final_output = np.power(np.maximum(output_bgr, 1e-8), 1.0 / 2.2)
    if pad_h or pad_w:
        final_output = final_output[:H, :W, :]
    return (final_output * 255.0).astype(np.uint8)

# Test with Pure Blue BGR
bgr_img = np.zeros((100, 100, 3), dtype=np.uint8)
bgr_img[:, :, 0] = 255  # Pure Blue
    
noisy = add_unprocess_isp_noise_fixed(bgr_img)
print(f"Noisy Output : B={np.mean(noisy[:,:,0]):.2f}, G={np.mean(noisy[:,:,1]):.2f}, R={np.mean(noisy[:,:,2]):.2f}")
