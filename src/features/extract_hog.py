import os
import cv2
import pandas as pd
from skimage.feature import hog
from skimage import exposure
import matplotlib.pyplot as plt
from tqdm import tqdm

MONITORING_ENABLED = False
GRID_SIZE = 8

def extract_hog_features():
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.dirname(current_file_dir)
    base_dir = os.path.dirname(src_dir)
    
    processed_dir = os.path.join(base_dir, "data", "processed")
    
    feature_name = "hog"
    output_dir = os.path.join(base_dir, "data", "features")
    os.makedirs(output_dir, exist_ok=True)
    
    pkl_output_path = os.path.join(output_dir, f"features_{feature_name}.pkl")
    csv_output_path = os.path.join(output_dir, f"features_{feature_name}.csv")

    all_features = []
    labels = []
    image_paths = []

    print(f"--- HOG 피처 추출 시작 ---")
    print(f"이미지 소스: {processed_dir}")

    # --- 2. 'data/processed'의 모든 하위 폴더 순회 ---
    for zone_name in sorted(os.listdir(processed_dir)):
        zone_path = os.path.join(processed_dir, zone_name)
        if not os.path.isdir(zone_path):
            continue
        
        print(f"\n[Zone 처리 중: {zone_name}]")
        
        # [수정] Zone당 한 번 시각화 플래그 제거
        
        for img_name in tqdm(os.listdir(zone_path), desc=zone_name):
            img_path = os.path.join(zone_path, img_name)
            
            try:
                # --- 3. HOG 피처 추출 ---
                img = cv2.imread(img_path)
                img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img_resized = cv2.resize(img_gray, (128, 128)) 
                
                feature_vector, hog_image = hog(img_resized, pixels_per_cell=(GRID_SIZE, GRID_SIZE),
                                     cells_per_block=(2, 2), visualize=True)
                
                # --- [수정] 모든 이미지 모니터링 ---
                if MONITORING_ENABLED:
                    print(f"  > [모니터링] {img_name} HOG 시각화...")
                    
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)
                    ax1.axis('off')
                    ax1.imshow(img_resized, cmap=plt.cm.gray)
                    ax1.set_title(f'Original Image: {img_name}')
                    
                    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
                    ax2.axis('off')
                    ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
                    ax2.set_title('HOG Feature')
                    
                    plt.suptitle(f"HOG Monitoring: {zone_name}")
                    plt.show() # [경고] 여기서 스크립트가 멈춥니다.
                # --- [모니터링 끝] ---

                all_features.append(feature_vector)
                labels.append(zone_name)
                image_paths.append(img_path)

            except Exception as e:
                print(f" ! 에러: {img_path} 처리 중 오류 - {e}")

    if not all_features:
        print("\n추출된 피처가 없습니다. 스크립트를 종료합니다.")
        return

    print(f"\n총 {len(all_features)}개의 피처 추출 완료. 이제 파일을 저장합니다...")

    df_pkl = pd.DataFrame({
        'image_path': image_paths,
        'label': labels,
        'features': all_features
    })
    df_pkl.to_pickle(pkl_output_path)
    print(f"-> PKL 파일 저장 완료: {pkl_output_path}")

    print("CSV 파일 생성 중... (피처 벡터가 많아 시간이 걸릴 수 있습니다)")
    num_features = len(all_features[0])
    feature_columns = [f'{feature_name}_{i}' for i in range(num_features)]
    
    df_features_csv = pd.DataFrame(all_features, columns=feature_columns)
    df_info_csv = pd.DataFrame({'label': labels})
    
    df_csv_final = pd.concat([df_info_csv, df_features_csv], axis=1)
    df_csv_final.to_csv(csv_output_path, index=False)
    print(f"-> CSV 파일 저장 완료: {csv_output_path}")

    print("\n--- 모든 작업 완료 ---")

if __name__ == "__main__":
    extract_hog_features()