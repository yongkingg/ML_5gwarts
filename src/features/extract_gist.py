import os
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

MONITORING_ENABLED = False

def extract_orb_features():
    # --- 1. 경로 설정 ---
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.dirname(current_file_dir)
    base_dir = os.path.dirname(src_dir)
    
    processed_dir = os.path.join(base_dir, "data", "processed")
    
    # GIST 대신 ORB (Averaged) 사용
    feature_name = "orb_avg"
    output_dir = os.path.join(base_dir, "data", "features")
    os.makedirs(output_dir, exist_ok=True)
    
    pkl_output_path = os.path.join(output_dir, f"features_{feature_name}.pkl")
    csv_output_path = os.path.join(output_dir, f"features_{feature_name}.csv")

    all_features = []
    labels = []
    image_paths = []

    # ORB 객체 생성 (기본 500개 키포인트)
    try:
        orb = cv2.ORB_create()
    except Exception as e:
        print(f" ! 에러: OpenCV ORB 생성 실패. {e}")
        print("   (opencv-python 라이브러리가 올바르게 설치되었는지 확인하세요)")
        return

    print(f"--- ORB (Averaged) 피처 추출 시작 ---")
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
                # --- 3. ORB 피처 추출 및 평균 집계 ---
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                
                # 키포인트와 디스크립터(피처) 검출
                # 'des'는 (N, 32) 형태의 배열 (ORB는 32바이트)
                kp, des = orb.detectAndCompute(img, None)
                
                if des is not None and len(des) > 0:
                    # N개의 32차원 피처를 평균내어 1개의 32차원 벡터로 만듦
                    # (ORB는 float이 아니므로 float32로 변환 후 평균)
                    feature_vector = np.mean(des.astype(np.float32), axis=0)
                else:
                    # 피처가 검출되지 않으면 32차원 0 벡터로 채움
                    feature_vector = np.zeros(32)
                
                # --- 모니터링 ---
                if MONITORING_ENABLED:
                    print(f"  > [모니터링] {img_name} ORB Keypoints 시각화...")
                    img_color = cv2.imread(img_path)
                    img_with_keypoints = cv2.drawKeypoints(img_color, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                    
                    plt.figure(figsize=(10, 10))
                    plt.imshow(cv2.cvtColor(img_with_keypoints, cv2.COLOR_BGR2RGB))
                    plt.title(f'ORB Keypoints: {img_name} ({len(kp)} points found)')
                    plt.axis('off')
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
    print("CSV 파일 생성 중...")
    num_features = len(all_features[0])
    feature_columns = [f'{feature_name}_{i}' for i in range(num_features)]
    
    df_features_csv = pd.DataFrame(all_features, columns=feature_columns)
    df_info_csv = pd.DataFrame({'label': labels})
    
    df_csv_final = pd.concat([df_info_csv, df_features_csv], axis=1)
    df_csv_final.to_csv(csv_output_path, index=False)
    print(f"-> CSV 파일 저장 완료: {csv_output_path}")

    print("\n--- 모든 작업 완료 ---")

if __name__ == "__main__":
    extract_orb_features()