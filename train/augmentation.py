import os
import random
import sys
from PIL import Image

try:
    import torchvision.transforms as T
except ImportError:
    print("Error: 'torchvision' 라이브러리를 찾을 수 없습니다.", file=sys.stderr)
    print("스크립트를 실행하기 전에 'pip install torchvision' 명령어로 설치해주세요.", file=sys.stderr)
    sys.exit(1)

TARGET_IMAGE_COUNT = 40  

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
    train_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"--- 이미지 증강 시작 (대상 폴더: {train_dir}) ---")

    for item_name in os.listdir(train_dir):
        item_path = os.path.join(train_dir, item_name)
        
        if os.path.isdir(item_path):
            print(f"\n[폴더 처리 중: {item_name}]")
            all_files = os.listdir(item_path)
            original_image_files = [
                f for f in all_files 
                if f.lower().endswith(".png") and not f.startswith('aug_')
            ]
            
            existing_aug_files = [
                f for f in all_files 
                if f.lower().endswith(".png") and f.startswith('aug_')
            ]

            num_originals = len(original_image_files)
            num_total_images = num_originals + len(existing_aug_files)
            
            if num_originals == 0:
                print("  ! 경고: 원본 이미지가 없어 이 폴더를 건너뜁니다.")
                continue

            print(f"  > 원본 이미지: {num_originals}장 / 총 이미지: {num_total_images}장")

            # 생성해야 할 이미지 개수 계산
            num_to_generate = TARGET_IMAGE_COUNT - num_total_images

            if num_to_generate <= 0:
                print(f"  > 이미 {num_total_images}장의 이미지가 있어 목표({TARGET_IMAGE_COUNT}장)를 충족합니다. 건너뜁니다.")
                continue
            
            print(f"  > 목표({TARGET_IMAGE_COUNT}장)를 위해 {num_to_generate}장의 새 이미지를 생성합니다...")
            
            generated_count = 0
            # 파일명 중복을 피하기 위해 기존 증강 파일 개수에서 시작
            file_index = len(existing_aug_files) 
            
            while generated_count < num_to_generate:
                try:
                    source_image_index = generated_count % len(original_image_files)
                    source_image_name = original_image_files[source_image_index]
                    
                    source_image_path = os.path.join(item_path, source_image_name)
                    
                    with Image.open(source_image_path) as img:
                        img_rgb = img.convert("RGB")
                        
                        transform_index = generated_count % len(augmentation_list)
                        chosen_transform = augmentation_list[transform_index]
                        
                        augmented_img = chosen_transform(img_rgb)
                        
                        new_filename = f"aug_{file_index:03d}.png"
                        save_path = os.path.join(item_path, new_filename)
                        augmented_img.save(save_path, "PNG", quality=90) 
                        
                        generated_count += 1
                        file_index += 1

                except Exception as e:
                    print(f"  ! 에러: {source_image_name} 처리 중 오류 발생 - {e}")

            print(f"  > 완료: {generated_count}장 생성 완료. (총 {file_index}장의 증강 이미지)")

    print("\n--- 모든 폴더 처리 완료 ---")

if __name__ == "__main__":
    augment_images()