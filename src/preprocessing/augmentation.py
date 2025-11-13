import os
import sys
import shutil # 원본 복사를 위해 shutil 임포트
from PIL import Image

# torchvision 설치 확인
try:
    import torchvision.transforms as T
except ImportError:
    print("Error: 'torchvision' 라이브러리를 찾을 수 없습니다.", file=sys.stderr)
    print("스크립트를 실행하기 전에 'pip install torchvision' 명령어로 설치해주세요.", file=sys.stderr)
    sys.exit(1)

transform_brightness = T.ColorJitter(brightness=0.5)
transform_contrast = T.ColorJitter(contrast=0.5)
transform_rotation = T.RandomRotation(degrees=30)
transform_shear = T.RandomAffine(degrees=0, shear=15)

augmentation_list = [
    transform_brightness,
    transform_contrast,
    transform_rotation,
    transform_shear
]

def augment_images():
    preprocessing_dir = os.path.dirname(os.path.abspath(__file__))
    
    # src_dir = ".../ML/src"
    src_dir = os.path.dirname(preprocessing_dir)
    
    # base_dir = ".../ML" (올바른 경로)
    base_dir = os.path.dirname(src_dir)
    
    # 이제 이 경로들이 올바르게 작동합니다.
    raw_train_dir = os.path.join(base_dir,"data", "raw", "train")
    processed_dir = os.path.join(base_dir, "data", "processed")

    print(f"--- 이미지 증강 시작 ---")
    print(f"원본 위치: {raw_train_dir}")
    print(f"저장 위치: {processed_dir}")
    print("-" * 25)

    if not os.path.exists(raw_train_dir):
        print(f" ! 에러: 원본 폴더 '{raw_train_dir}'를 찾을 수 없습니다.", file=sys.stderr)
        return

    for sub_folder_name in os.listdir(raw_train_dir):
        source_folder = os.path.join(raw_train_dir, sub_folder_name)
        dest_folder = os.path.join(processed_dir, sub_folder_name)   
        
        if not os.path.isdir(source_folder):
            continue
            
        print(f"\n[폴더 처리 중: {sub_folder_name}]")
        
        os.makedirs(dest_folder, exist_ok=True)
        
        original_image_files = [
            f for f in os.listdir(source_folder) 
            if f.lower().endswith('.jpg') and not f.startswith('aug_')
        ]
        
        num_originals = len(original_image_files)
        if num_originals == 0:
            print("  ! 경고: 이 폴더에 원본 이미지가 없어 건너뜁니다.")
            continue

        copied_count = 0
        for img_name in original_image_files:
            source_path = os.path.join(source_folder, img_name)
            dest_path = os.path.join(dest_folder, img_name)
            if not os.path.exists(dest_path):
                shutil.copy2(source_path, dest_path)
                copied_count += 1
        
        if copied_count > 0:
            print(f"  > 원본 이미지 {copied_count}장을 'processed' 폴더로 복사했습니다.")
        
        print(f"  > 원본 이미지 {num_originals}장에 대해 각각 10회 증강을 시작합니다...")

        generated_total_count = 0
        
        for original_image_name in original_image_files:
            source_image_path = os.path.join(source_folder, original_image_name)
            base_filename = os.path.splitext(original_image_name)[0]
            
            try:
                with Image.open(source_image_path) as img:
                    img_rgb = img.convert("RGB")
                    
                    for i in range(10): 
                        transform_index = i % len(augmentation_list)
                        chosen_transform = augmentation_list[transform_index]
                        
                        augmented_img = chosen_transform(img_rgb)
                        
                        new_filename = f"{base_filename}_aug_{i:02d}.jpg"
                        save_path = os.path.join(dest_folder, new_filename)
                        
                        augmented_img.save(save_path, "JPEG", quality=90) 
                        generated_total_count += 1

            except Exception as e:
                print(f"  ! 에러: {original_image_name} 처리 중 오류 발생 - {e}")

        print(f"  > 완료: 총 {generated_total_count}장의 증강 이미지를 생성했습니다.")

    print("\n--- 모든 폴더 처리 완료 ---")

if __name__ == "__main__":
    augment_images()