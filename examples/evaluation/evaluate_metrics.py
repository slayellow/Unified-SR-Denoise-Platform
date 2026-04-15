import cv2
import os
import pandas as pd
import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# -----------------------------------------------------------------
# 1. NIQE 설정 (Basicsr)
# -----------------------------------------------------------------
try:
    from basicsr.metrics.niqe import calculate_niqe
    NIQE_ENABLED = True
except ImportError:
    try:
        from basicsr.metrics.niqe_metric import calculate_niqe
        NIQE_ENABLED = True
    except ImportError:
        NIQE_ENABLED = False
        print("[Warning] 'basicsr' not found. NIQE will be skipped.")

# -----------------------------------------------------------------
# 2. LPIPS 설정 (Learned Perceptual Image Patch Similarity)
# -----------------------------------------------------------------
try:
    import lpips
    LPIPS_ENABLED = True
    # LPIPS 모델 초기화 (AlexNet 기반이 가장 가볍고 표준적임)
    # GPU가 있으면 GPU로 로드
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loss_fn_alex = lpips.LPIPS(net='alex').to(device)
    print(f"[Info] LPIPS model loaded on {device}")
except ImportError:
    LPIPS_ENABLED = False
    print("[Warning] 'lpips' library not found. LPIPS will be skipped.")

# -----------------------------------------------------------------
# 유틸리티 함수들
# -----------------------------------------------------------------

def calculate_niqe_score(img):
    if not NIQE_ENABLED: return np.nan
    # Basicsr NIQE는 Y channel 변환 등을 내부적으로 처리하거나 옵션으로 받음
    return calculate_niqe(img, crop_border=0, input_order='HWC', convert_to='y')

def calculate_lpips_score(hr_img, sr_img):
    """
    OpenCV 이미지(H, W, C, [0,255])를 받아 LPIPS를 계산
    LPIPS는 입력으로 [-1, 1] 범위의 RGB Tensor (B, C, H, W)를 받음
    """
    if not LPIPS_ENABLED: return np.nan

    # 1. BGR to RGB
    hr_img = hr_img[:, :, ::-1].copy()
    sr_img = sr_img[:, :, ::-1].copy()

    # 2. Normalize to [-1, 1] and Convert to Tensor
    # (H, W, C) -> (C, H, W) -> (1, C, H, W)
    hr_tensor = torch.from_numpy(hr_img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    sr_tensor = torch.from_numpy(sr_img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    
    hr_tensor = (hr_tensor * 2) - 1.0
    sr_tensor = (sr_tensor * 2) - 1.0

    # 3. GPU로 이동
    hr_tensor = hr_tensor.to(device)
    sr_tensor = sr_tensor.to(device)

    # 4. 계산 (Gradient 계산 불필요)
    with torch.no_grad():
        score = loss_fn_alex(hr_tensor, sr_tensor)
    
    return score.item()

def calculate_psnr_ssim(hr_img, sr_img):
    hr_ycc = cv2.cvtColor(hr_img, cv2.COLOR_BGR2YCrCb)
    sr_ycc = cv2.cvtColor(sr_img, cv2.COLOR_BGR2YCrCb)
    
    # 2. 첫 번째 채널인 Y(Luminance, 밝기) 채널만 추출 (2D 배열이 됨)
    hr_y = hr_ycc[:, :, 0]
    sr_y = sr_ycc[:, :, 0]
    
    psnr_score = psnr(hr_y, sr_y, data_range=255)
    try:
        ssim_score = ssim(hr_y, sr_y, data_range=255)
    except TypeError:
        ssim_score = ssim(hr_y, sr_y, data_range=255)
    return psnr_score, ssim_score

def evaluate_models(hr_dir, sr_model_dirs_config):
    results = []
    
    if not os.path.exists(hr_dir):
        print(f"오류: HR 폴더를 찾을 수 없습니다: {hr_dir}")
        return pd.DataFrame()

    hr_filenames = sorted([f for f in os.listdir(hr_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    print(f"\n총 {len(hr_filenames)}개의 HR 이미지 파일을 기준으로 평가를 시작합니다.")

    for i, filename in enumerate(hr_filenames):
        hr_path = os.path.join(hr_dir, filename)
        hr_image = cv2.imread(hr_path) # BGR
        
        if hr_image is None: continue
        
        print(f"[{i+1}/{len(hr_filenames)}] {filename} ... ", end="")

        base_name, extension = os.path.splitext(filename)
            
        for model_name, config in sr_model_dirs_config.items():
            model_dir = config["path"]
            suffix = config.get("suffix", "")
            
            # 메타데이터 (Params, FLOPs, Latency) 가져오기 (없으면 NaN)
            meta_params = config.get("params", "N/A")
            meta_flops = config.get("flops", "N/A")
            meta_latency = config.get("latency", "N/A")

            sr_filename = f"{base_name}{suffix}{extension}"
            sr_path = os.path.join(model_dir, sr_filename)

            if not os.path.exists(sr_path):
                continue
                
            sr_image = cv2.imread(sr_path)
            if sr_image is None or hr_image.shape != sr_image.shape:
                continue

            # --- 메트릭 계산 ---
            psnr_val, ssim_val = calculate_psnr_ssim(hr_image, sr_image)
            niqe_val = calculate_niqe_score(sr_image)
            lpips_val = calculate_lpips_score(hr_image, sr_image)
            
            results.append({
                "filename": filename,
                "model": model_name,
                "PSNR": psnr_val,
                "SSIM": ssim_val,
                "NIQE": niqe_val,
                "LPIPS": lpips_val,
                # 고정 메타데이터 추가
                "Params(M)": meta_params,
                "FLOPs(G)": meta_flops,
                "Latency(ms)": meta_latency
            })
        print("Done.")

    return pd.DataFrame(results)

# -----------------------------------------------------------------
# 메인 실행
# -----------------------------------------------------------------
if __name__ == "__main__":
    
    BASE_PATH = "/mnt/data_server/etc/jshong/SuperResolution/Pretrained_Dataset/val_denoise/"
    HR_DIR = os.path.join(BASE_PATH, "HR")

    # -------------------------------------------------------------
    # ★ 설정 부분: 모델별 경로와 성능 지표(Params, FLOPs 등)를 직접 입력하세요.
    # 이 값들은 모델 구조에 따라 고정된 값이므로, 논문이나 모델 정보에서 가져와 기입합니다.
    # -------------------------------------------------------------
    SR_DIRS_MAP = {
        "v1_DENOISE": {
            "path": os.path.join("/mnt/data_server/etc/jshong/SuperResolution/SR_RESULTS/", 'QuickDenoiseNet_1280x720_v1', 'denoise'),
            "suffix": "",
            "params": 0.0,    # 예: 16.7M
            "flops": 0.0,    # 예: 120.5G (720p 기준)
            "latency": 0.0    # 예: 45.2ms
        },
        "v1_SVFOCUS_DENOISE": {
            "path": os.path.join("/mnt/data_server/etc/jshong/SuperResolution/SR_RESULTS/", 'SVFocusDenoiseNet_1280x720_v1', 'denoise'),
            "suffix": "",
            "params": 0.0,    # 예: 16.7M
            "flops": 0.0,    # 예: 120.5G (720p 기준)
            "latency": 0.0    # 예: 45.2ms
        },
        # "v2_BICUBIC": { 
        #     "path": os.path.join("/mnt/data_server/etc/jshong/SuperResolution/SR_RESULTS/", 'QuickSRNet_Large_IR_640x360_x2_v2', 'val_IR_BICUBIC'),
        #     "suffix": "",
        #     "params": 0.0,    # 예: 16.7M
        #     "flops": 0.0,    # 예: 120.5G (720p 기준)
        #     "latency": 0.0    # 예: 45.2ms
        # },
        # "v2_DEGRADATION": {
        #     "path": os.path.join("/mnt/data_server/etc/jshong/SuperResolution/SR_RESULTS/", 'QuickSRNet_Large_IR_640x360_x2_v2', 'val_IR_Degradation'),
        #     "suffix": "",
        #     "params": 0.0,    # 예: 16.7M
        #     "flops": 0.0,    # 예: 120.5G (720p 기준)
        #     "latency": 0.0    # 예: 45.2ms
        # },
        "v2_SVFOCUS_DENOISE": {
            "path": os.path.join("/mnt/data_server/etc/jshong/SuperResolution/SR_RESULTS/", 'SVFocusDenoiseNet_1280x720_v2', 'denoise'),
            "suffix": "",
            "params": 0.0,    # 예: 16.7M
            "flops": 0.0,    # 예: 120.5G (720p 기준)
            "latency": 0.0    # 예: 45.2ms
        },
        # "v2_DEGRADATION": {
        #     "path": os.path.join("/mnt/data_server/etc/jshong/SuperResolution/SR_RESULTS/", 'SVFocusSRNet_IR_320x180_x4_v2', 'val_IR_Degradation'),
        #     "suffix": "",
        #     "params": 0.0,    # 예: 16.7M
        #     "flops": 0.0,    # 예: 120.5G (720p 기준)
        #     "latency": 0.0    # 예: 45.2ms
        # },
       

        # "[After] QuickSRNet-Large": {
        #     "path": os.path.join("/mnt/data_server/etc/jshong/SuperResolution/260206/", 'QuickSRNet-Large_New'),
        #     "suffix": "",
        #     "params": 0,       # 파라미터 없음
        #     "flops": 0,        # 연산량 무시 가능
        #     "latency": 0.0     # 예시
        # },
        # "[After] RealESRNet_x2plus": {
        #     "path": os.path.join("/mnt/data_server/etc/jshong/SuperResolution/260206/", 'RRDBNet_New'),
        #     "suffix": "",
        #     "params": 0.0,    # 예: 16.7M
        #     "flops": 0.0,    # 예: 120.5G (720p 기준)
        #     "latency": 0.0    # 예: 45.2ms
        # },
        # "SVSRNet": {
        #     "path": os.path.join("/mnt/data_server/etc/jshong/SuperResolution/260206/", 'SVSRNet_New'),
        #     "suffix": "",
        #     "params": 0.0,    # 예: 16.7M
        #     "flops": 0.0,    # 예: 120.5G (720p 기준)
        #     "latency": 0.0    # 예: 45.2ms
        # },
        # "QuickSRNet": {
        #     "path": os.path.join('experiments', 'GPU', 'QuickSRNet'),
        #     "suffix": "",
        #     "params": 0.0,
        #     "flops": 0.0,
        #     "latency": 0.0
        # }
    }
    # -------------------------------------------------------------

    print(">>> 평가 시작 (Metrics: PSNR, SSIM, NIQE, LPIPS)")
    df_results = evaluate_models(HR_DIR, SR_DIRS_MAP)

    if not df_results.empty:
        # 1. 모델별 평균 계산 (숫자형 컬럼만)
        numeric_cols = ["PSNR", "SSIM", "NIQE", "LPIPS"]
        # Params, FLOPs 등은 문자열일 수도 있고 고정값이므로 평균 낼 필요 없이 첫 번째 값을 가져오거나 별도 표기
        
        # 그룹화하여 평균 계산
        df_summary = df_results.groupby("model")[numeric_cols].mean(numeric_only=True)
        
        # 메타데이터(Params 등)는 각 모델별로 동일하므로 첫 번째 행의 값을 가져와서 합침
        meta_cols = ["Params(M)", "FLOPs(G)", "Latency(ms)"]
        df_meta = df_results.groupby("model")[meta_cols].first()
        
        # 최종 요약표 결합
        final_summary = pd.concat([df_summary, df_meta], axis=1)

        print("\n\n=== [최종 모델 성능 비교 요약] ===")
        # 보기 좋게 포맷팅
        pd.options.display.float_format = '{:.4f}'.format
        print(final_summary)
        
        # CSV 저장 (선택)
        # final_summary.to_csv("sr_benchmark_results.csv")
    else:
        print("결과가 없습니다.")