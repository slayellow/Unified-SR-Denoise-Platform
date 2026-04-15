import argparse
import cv2
import os
from pathlib import Path

def process_images(input_dir, scale):
    input_path = Path(input_dir)
    if not input_path.is_dir():
        print(f"Error: 입력 경로가 존재하지 않거나 폴더가 아닙니다: {input_dir}")
        return

    # 출력 폴더 생성 (예: image_folder_x2)
    output_dir = input_path.parent / f"{input_path.name}_x{scale}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 지원하는 이미지 확장자
    valid_extensions = {'.png', '.jpg', '.jpeg', '.bmp'}
    
    image_files = [f for f in input_path.iterdir() if f.suffix.lower() in valid_extensions]
    
    if not image_files:
        print(f"입력 경로에서 이미지 파일({', '.join(valid_extensions)})을 찾을 수 없습니다.")
        return
        
    print(f"총 {len(image_files)}개의 이미지를 x{scale} 다운샘플링합니다.")
    print(f"출력 경로: {output_dir}")
    
    for count, img_path in enumerate(image_files, 1):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"경고: 이미지를 읽을 수 없습니다 - {img_path}")
            continue
            
        h, w = img.shape[:2]
        new_h, new_w = h // scale, w // scale
        
        # Bicubic interpolation을 사용한 다운샘플링
        resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        
        # 저장
        output_path = output_dir / img_path.name
        cv2.imwrite(str(output_path), resized_img)
        
        if count % 10 == 0 or count == len(image_files):
            print(f"진행 중... ({count}/{len(image_files)})")
            
    print("작업이 완료되었습니다.")

def main():
    parser = argparse.ArgumentParser(description="이미지 폴더 내의 파일들을 Bicubic 다운샘플링합니다.")
    parser.add_argument("input_dir", type=str, help="이미지가 포함된 입력 폴더 경로")
    parser.add_argument("--scale", type=int, choices=[2, 4, 8], required=True, 
                        help="다운샘플링 비율 (2, 4, 8 중 선택)")
    
    args = parser.parse_args()
    process_images(args.input_dir, args.scale)

if __name__ == "__main__":
    main()
