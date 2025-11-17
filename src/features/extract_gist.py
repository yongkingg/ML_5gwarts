import os
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage.filters import gabor   

MONITORING_ENABLED = False

IMG_SIZE = 256              # 이미지를 256x256으로 리사이즈
N_BLOCKS = 4                # 4x4 spatial grid
ORIENTATIONS_PER_SCALE = [8, 8, 8, 8]   # 각 scale 당 8방향 (총 32 filters)
FREQUENCIES = [0.05, 0.10, 0.20, 0.40]  # 대략적인 저주파 ~ 고주파


def compute_gist_gray(img_gray: np.ndarray) -> np.ndarray:
    # 1) 리사이즈
    img_resized = cv2.resize(img_gray, (IMG_SIZE, IMG_SIZE)).astype(np.float32) / 255.0

    feats = []

    # scale (frequency) 별로
    for freq, n_ori in zip(FREQUENCIES, ORIENTATIONS_PER_SCALE):
        for k in range(n_ori):
            theta = k * np.pi / n_ori   # 방향

            # skimage.filters.gabor → (real, imag)
            real, imag = gabor(img_resized, frequency=freq, theta=theta)
            magnitude = np.sqrt(real ** 2 + imag ** 2)

            # 4x4 블록으로 나누어 각 블록 평균
            h, w = magnitude.shape
            bh, bw = h // N_BLOCKS, w // N_BLOCKS

            for by in range(N_BLOCKS):
                for bx in range(N_BLOCKS):
                    block = magnitude[by * bh:(by + 1) * bh,
                                      bx * bw:(bx + 1) * bw]
                    feats.append(block.mean())

    return np.asarray(feats, dtype=np.float32) 


def extract_gist_features():
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.dirname(current_file_dir)
    base_dir = os.path.dirname(src_dir)

    processed_dir = os.path.join(base_dir, "data", "processed")

    feature_name = "gist"
    output_dir = os.path.join(base_dir, "data", "features")
    os.makedirs(output_dir, exist_ok=True)

    pkl_output_path = os.path.join(output_dir, f"features_{feature_name}.pkl")
    csv_output_path = os.path.join(output_dir, f"features_{feature_name}.csv")

    all_features = []
    labels = []
    image_paths = []

    print(f"--- GIST-like 피처 추출 시작 ---")
    print(f"이미지 소스: {processed_dir}")

    for zone_name in sorted(os.listdir(processed_dir)):
        zone_path = os.path.join(processed_dir, zone_name)
        if not os.path.isdir(zone_path):
            continue

        print(f"\n[Zone 처리 중: {zone_name}]")

        for img_name in tqdm(os.listdir(zone_path), desc=zone_name):
            img_path = os.path.join(zone_path, img_name)

            try:
                img_bgr = cv2.imread(img_path)
                if img_bgr is None:
                    print(f" ! 경고: 이미지를 읽을 수 없습니다: {img_path}")
                    continue

                img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

                gist_vec = compute_gist_gray(img_gray)

                if MONITORING_ENABLED:
                    print(f"  > [모니터링] {img_name} GIST 시각화...")
                    plt.figure(figsize=(10, 4))
                    plt.subplot(1, 2, 1)
                    plt.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
                    plt.title(f"Original: {img_name}")
                    plt.axis("off")

                    plt.subplot(1, 2, 2)
                    plt.plot(gist_vec)
                    plt.title(f"GIST vector (len={len(gist_vec)})")
                    plt.xlabel("dimension")
                    plt.ylabel("value")
                    plt.tight_layout()
                    plt.show()

                all_features.append(gist_vec)
                labels.append(zone_name)
                image_paths.append(img_path)

            except Exception as e:
                print(f" ! 에러: {img_path} 처리 중 오류 - {e}")

    if not all_features:
        print("\n추출된 피처가 없습니다. 스크립트를 종료합니다.")
        return

    print(f"\n총 {len(all_features)}개의 GIST 피처 추출 완료. 이제 파일을 저장합니다...")

    df_pkl = pd.DataFrame({
        "image_path": image_paths,
        "label": labels,
        "features": all_features,
    })
    df_pkl.to_pickle(pkl_output_path)
    print(f"-> PKL 파일 저장 완료: {pkl_output_path}")

    print("CSV 파일 생성 중...")
    num_features = len(all_features[0])
    feature_columns = [f"{feature_name}_{i}" for i in range(num_features)]

    df_features_csv = pd.DataFrame(all_features, columns=feature_columns)
    df_info_csv = pd.DataFrame({"label": labels})

    df_csv_final = pd.concat([df_info_csv, df_features_csv], axis=1)
    df_csv_final.to_csv(csv_output_path, index=False)
    print(f"-> CSV 파일 저장 완료: {csv_output_path}")

    print("\n--- 모든 작업 완료 ---")


if __name__ == "__main__":
    extract_gist_features()
