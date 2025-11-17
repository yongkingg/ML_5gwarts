import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

# ==========================
# 1. 파일 경로 설정
# ==========================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
FEATURE_DIR = os.path.join(BASE_DIR, "data", "features")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "features")

os.makedirs(OUTPUT_DIR, exist_ok=True)

FILES = {
    "gist":        ("features_gist.csv",       128),
    "sift":        ("features_sift_avg.csv",    64),
    "hog":         ("features_hog.csv",        128),
    "hsv":         ("features_hsv.csv",         64),
    "orb":         ("features_orb_avg.csv",     32),  
}

# ==========================
# 2. PCA 함수
# ==========================
def run_pca(feature_name, filename, out_dim):
    path = os.path.join(FEATURE_DIR, filename)
    print(f"\n[Loading] {path}")

    df = pd.read_csv(path)
    if "label" in df.columns:
        df = df.drop(columns=["label"])
    X = df.values

    print(f" - original shape : {X.shape}")

    # PCA 학습
    pca = PCA(n_components=out_dim)
    X_pca = pca.fit_transform(X)

    explained = pca.explained_variance_ratio_.sum()
    print(f" - reduced shape  : {X_pca.shape}")
    print(f" - explained variance ratio: {explained:.4f}")

    # 저장
    out_path = os.path.join(OUTPUT_DIR, f"{feature_name}_pca_{out_dim}.csv")
    pd.DataFrame(X_pca).to_csv(out_path, index=False)

    print(f" - saved to: {out_path}")

    # PCA 모델도 저장하고 싶으면 추가
    # import pickle
    # with open(out_path.replace(".csv", ".pkl"), "wb") as f:
    #     pickle.dump(pca, f)

    return X_pca, explained


# ==========================
# 3. 실행
# ==========================
if __name__ == "__main__":
    results = {}

    for name, (fname, out_dim) in FILES.items():
        X_pca, exp = run_pca(name, fname, out_dim)
        results[name] = exp

    print("\n===== SUMMARY =====")
    for k, v in results.items():
        print(f"{k:6s} → explained variance {v:.4f}")
