import os
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

MONITORING_ENABLED = False
# H(색상), S(채도), V(명도)를 각각 몇 개의 구간으로 나눌지 설정
# (총 8*8*8 = 512차원의 피처 벡터가 생성됨)
HSV_BINS = [8, 8, 8] 

def extract_hsv_features():
    # --- 1. 경로 설정 ---
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.dirname(current_file_dir)
    base_dir = os.path.dirname(src_dir)
    
    processed_dir = os.path.join(base_dir, "data", "processed")
    
    feature_name = "hsv"
    output_dir = os.path.join(base_dir, "data", "features")
    os.makedirs(output_dir, exist_ok=True)
    
    pkl_output_path = os.path.join(output_dir, f"features_{feature_name}.pkl")
    csv_output_path = os.path.join(output_dir, f"features_{feature_name}.csv")

    all_features = []
    labels = []
    image_paths = []

    print(f"--- HSV 3D Histogram 피처 추출 시작 ---")
    print(f"이미지 소스: {processed_dir}")

    # --- 2. 'data/processed'의 모든 하위 폴더 순회 ---
    for zone_name in sorted(os.listdir(processed_dir)):
        zone_path = os.path.join(processed_dir, zone_name)
        if not os.path.isdir(zone_path):
            continue
        
        print(f"\n[Zone 처리 중: {zone_name}]")
        
        for img_name in tqdm(os.listdir(zone_path), desc=zone_name):
            img_path = os.path.join(zone_path, img_name)
            
            try:
                # --- 3. HSV 3D Histogram 피처 추출 ---
                img = cv2.imread(img_path)
                img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                
                # H, S, V 3D 히스토그램 계산
                # H(0~179), S(0~255), V(0~255) 범위
                hist = cv2.calcHist([img_hsv], [0, 1, 2], None, 
                                    HSV_BINS, 
                                    [0, 180, 0, 256, 0, 256])
                
                # 히스토그램 정규화 (0~1 사이 값으로)
                cv2.normalize(hist, hist)
                
                # 1차원 벡터로 변환 (512차원)
                feature_vector = hist.flatten()
                
                # --- 모니터링 ---
                if MONITORING_ENABLED:
                    print(f"  > [모니터링] {img_name} HSV 시각화...")
                    h, s, v = cv2.split(img_hsv)
                    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
                    
                    axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) # Original
                    axes[0].set_title(f'Original: {img_name}')
                    axes[0].axis('off')

                    axes[1].imshow(h, cmap='hsv')
                    axes[1].set_title('Hue Channel')
                    axes[1].axis('off')
                    
                    axes[2].imshow(s, cmap='gray')
                    axes[2].set_title('Saturation Channel')
                    axes[2].axis('off')

                    axes[3].imshow(v, cmap='gray')
                    axes[3].set_title('Value Channel')
                    axes[3].axis('off')
                    
                    plt.suptitle(f"HSV Monitoring: {zone_name}")
                    plt.show()
                # --- 모니터링 끝 ---

                all_features.append(feature_vector)
                labels.append(zone_name)
                image_paths.append(img_path)

            except Exception as e:
                print(f" ! 에러: {img_path} 처리 중 오류 - {e}")

    # --- 4. DataFrame 변환 및 2가지 형식으로 저장 ---
    if not all_features:
        print("\n추출된 피처가 없습니다. 스크립트를 종료합니다.")
        return

    print(f"\n총 {len(all_features)}개의 피처 추출 완료. 이제 파일을 저장합니다...")

    # 4-1. PKL (image_path 포함)
    df_pkl = pd.DataFrame({
        'image_path': image_paths,
        'label': labels,
        'features': all_features
    })
    df_pkl.to_pickle(pkl_output_path)
    print(f"-> PKL 파일 저장 완료: {pkl_output_path}")

    # 4-2. CSV (image_path 제외)
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
    extract_hsv_features()