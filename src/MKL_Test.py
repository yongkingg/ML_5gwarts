import os
import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm
from skimage.feature import hog
from skimage.filters import gabor
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics.pairwise import linear_kernel, chi2_kernel
import sys
import io
import pickle
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# ============================================================
# 1. Feature Extraction Functions
# ============================================================

def extract_hsv(img_bgr):
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([img_hsv], [0, 1, 2], None, 
                        [8, 8, 8], [0, 180, 0, 256, 0, 256])
    hist += 1e-6
    cv2.normalize(hist, hist)
    return hist.flatten()

def extract_hog(img_bgr):
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    img_resized = cv2.resize(img_gray, (128, 128)) 
    return hog(img_resized, pixels_per_cell=(8, 8),
               cells_per_block=(2, 2), visualize=False)

def extract_sift_avg(img_bgr):
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    try:
        sift = cv2.SIFT_create()
        kp, des = sift.detectAndCompute(img_gray, None)
        if des is not None and len(des) > 0:
            return np.mean(des, axis=0)
        return np.zeros(128)
    except:
        return np.zeros(128)

def extract_orb_avg(img_bgr):
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    try:
        orb = cv2.ORB_create(nfeatures=500)
        kp, des = orb.detectAndCompute(img_gray, None)
        if des is not None and len(des) > 0:
            return np.mean(des.astype(np.float32), axis=0)
        return np.zeros(32)
    except:
        return np.zeros(32)

def compute_gist_gray(img_bgr):
    IMG_SIZE = 256
    N_BLOCKS = 4
    ORIENT = [8, 8, 8, 8]
    FREQ = [0.05, 0.10, 0.20, 0.40]
    
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    img_resized = cv2.resize(img_gray, (IMG_SIZE, IMG_SIZE)).astype(np.float32) / 255.0
    feats = []
    
    for freq, n_ori in zip(FREQ, ORIENT):
        for k in range(n_ori):
            theta = k * np.pi / n_ori
            real, imag = gabor(img_resized, frequency=freq, theta=theta)
            magnitude = np.sqrt(real ** 2 + imag ** 2)
            h, w = magnitude.shape
            bh, bw = h // N_BLOCKS, w // N_BLOCKS
            
            for by in range(N_BLOCKS):
                for bx in range(N_BLOCKS):
                    block = magnitude[by*bh:(by+1)*bh, bx*bw:(bx+1)*bw]
                    feats.append(block.mean())

    return np.asarray(feats, dtype=np.float32)

# ============================================================
# 2. SimpleMKL Helper Functions
# ============================================================

def compute_objective(alpha_y_sv, K_combined_sv):
    dual_term = np.dot(alpha_y_sv, np.dot(K_combined_sv, alpha_y_sv))
    sum_alpha = np.sum(np.abs(alpha_y_sv))
    return sum_alpha - 0.5 * dual_term

def compute_gradient(alpha_y_sv, K_list_sv):
    grad = []
    for K in K_list_sv:
        grad.append(-0.5 * np.dot(alpha_y_sv, np.dot(K, alpha_y_sv)))
    return np.array(grad)

def compute_descent_direction(d, grad):
    M = len(d)
    mu = np.argmax(d)
    
    rgrad = np.zeros(M)
    for m in range(M):
        if m != mu:
            rgrad[m] = grad[m] - grad[mu]
    rgrad[mu] = -np.sum(rgrad[np.arange(M) != mu])
    
    D = -rgrad
    
    for m in range(M):
        if d[m] < 1e-10 and D[m] < 0:
            D[m] = 0
    
    D[mu] = -np.sum(D[np.arange(M) != mu])
    return D

def backtracking_line_search(d, D, grad, K_list_train, y_train, C):
    gamma = 1.0
    alpha = 0.5
    beta = 0.1

    curr_J = compute_current_J(d, K_list_train, y_train, C)
    grad_dot_D = np.dot(grad, D)

    if grad_dot_D > 0:
        return 0

    while True:
        d_new = d + gamma * D
        if np.any(d_new < 0):
            gamma *= alpha
            if gamma < 1e-10:
                return 0
            continue

        d_new /= np.sum(d_new)

        new_J = compute_current_J(d_new, K_list_train, y_train, C)
        if new_J <= curr_J + beta * gamma * grad_dot_D:
            return gamma

        gamma *= alpha
        if gamma < 1e-10:
            return 0

def compute_current_J(d, K_list_train, y_train, C):
    K_comb = np.zeros_like(K_list_train[0])
    for i in range(len(d)):
        K_comb += d[i] * K_list_train[i]

    svm = SVC(kernel='precomputed', C=C)
    svm.fit(K_comb, y_train)

    sv = svm.support_
    if len(sv) == 0:
        return 0

    alpha_y = svm.dual_coef_[0]
    K_sv = K_comb[np.ix_(sv, sv)]

    return compute_objective(alpha_y, K_sv)

# ============================================================
# 3. SimpleMKL Training Loop (with progress bar)
# ============================================================

def simple_mkl_train(K_list_train, y_train, C=1.0, max_iter=100, tol=1e-3):
    M = len(K_list_train)
    N = K_list_train[0].shape[0]

    d = np.ones(M) / M
    svm_model = None

    print("\n📌 SimpleMKL Iteration Progress")
    for it in tqdm(range(max_iter), desc="MKL Optimize"):
        K_comb = np.zeros((N, N))
        for m in range(M):
            K_comb += d[m] * K_list_train[m]

        svm = SVC(kernel='precomputed', C=C)
        svm.fit(K_comb, y_train)

        sv = svm.support_
        if len(sv) == 0:
            svm_model = svm
            break

        alpha_y = svm.dual_coef_[0]
        K_sv = [K[np.ix_(sv, sv)] for K in K_list_train]
        grad = compute_gradient(alpha_y, K_sv)

        Q = -grad
        gap = np.max(Q) - np.dot(d, Q)

        if gap < tol:
            svm_model = svm
            break

        D = compute_descent_direction(d, grad)

        gamma = backtracking_line_search(d, D, grad, K_list_train, y_train, C)
        if gamma == 0:
            svm_model = svm
            break

        d = d + gamma * D
        d[d < 0] = 0
        d /= np.sum(d)

    if svm_model is None:
        svm_model = svm

    return svm_model, d

# ============================================================
# 4. MAIN
# ============================================================

def main():
    print("--- 1. 데이터 준비 ---")

    try:
        current_script_path = os.path.abspath(__file__)
        base_dir = os.path.dirname(current_script_path)
    except:
        base_dir = os.getcwd()

    data_root_dir = os.path.join(base_dir, "data", "processed")
    print(f"데이터 폴더: {data_root_dir}")

    if not os.path.exists(data_root_dir):
        print("❌ processed 폴더 없음")
        return

    feature_extractors = {
        'hsv': extract_hsv,
        'hog': extract_hog,
        'sift_avg': extract_sift_avg,
        'orb_avg': extract_orb_avg,
        'gist': compute_gist_gray
    }

    kernel_functions = {
        'hsv': chi2_kernel,
        'hog': linear_kernel,
        'sift_avg': linear_kernel,
        'orb_avg': linear_kernel,
        'gist': linear_kernel
    }

    feature_names = list(feature_extractors.keys())
    print(f"사용 특징: {feature_names}")

    # ---------------------------------------------------------
    # Load ALL images (with progress)
    # ---------------------------------------------------------
    print("\n--- 2. 이미지 로드 ---")
    all_images = []
    all_labels = []

    zone_list = sorted(os.listdir(data_root_dir))
    for label in tqdm(zone_list, desc="📂 Load Folders"):
        zone_path = os.path.join(data_root_dir, label)
        if not os.path.isdir(zone_path):
            continue

        for img_name in tqdm(os.listdir(zone_path), desc=f" → {label}", leave=False):
            img_path = os.path.join(zone_path, img_name)
            img = cv2.imread(img_path)
            if img is not None:
                all_images.append(img)
                all_labels.append(label)

    print(f"로드 완료: {len(all_images)}장 이미지")

    le = LabelEncoder()
    y_labels = le.fit_transform(all_labels)

    X_train_imgs, X_test_imgs, y_train, y_test = train_test_split(
        all_images, y_labels, test_size=0.3, random_state=42, stratify=y_labels
    )

    del all_images
    del all_labels

    # ---------------------------------------------------------
    # Feature Extraction with Progress Bar
    # ---------------------------------------------------------
    print("\n--- 3. Feature Extraction ---")

    X_train_features = {name: [] for name in feature_names}
    X_test_features = {name: [] for name in feature_names}

    print("Train feature extraction...")
    for img in tqdm(X_train_imgs, desc="✨ Train Extract"):
        for name, func in feature_extractors.items():
            X_train_features[name].append(func(img))

    print("Test feature extraction...")
    for img in tqdm(X_test_imgs, desc="🔍 Test Extract"):
        for name, func in feature_extractors.items():
            X_test_features[name].append(func(img))

    del X_train_imgs, X_test_imgs

    scalers = {}

    # scale transform
    for name in feature_names:
        X_train_features[name] = np.array(X_train_features[name])
        X_test_features[name] = np.array(X_test_features[name])

        scaler = StandardScaler().fit(X_train_features[name])
        X_train_features[name] = scaler.transform(X_train_features[name])
        X_test_features[name] = scaler.transform(X_test_features[name])
        scalers[name] = scaler

    # ---------------------------------------------------------
    # Compute Kernels with Progress Bar
    # ---------------------------------------------------------
    print("\n--- 4. 커널 계산 ---")

    K_list_train = []
    K_list_test = []

    for name in tqdm(feature_names, desc="🔧 Kernel Build"):
        Xtr = X_train_features[name]
        Xte = X_test_features[name]
        kernel = kernel_functions[name]

        if kernel == chi2_kernel:
            minv = Xtr.min()
            if minv <= 0:
                Xtr += -minv + 1e-6
            minv_t = Xte.min()
            if minv_t <= 0:
                Xte += -minv_t + 1e-6

        K_list_train.append(kernel(Xtr, Xtr))
        K_list_test.append(kernel(Xte, Xtr))

    # ---------------------------------------------------------
    # SimpleMKL Train (with progress bar)
    # ---------------------------------------------------------
    print("\n--- 5. SimpleMKL Training ---")

    svm_model, d = simple_mkl_train(K_list_train, y_train, C=1.0, max_iter=200, tol=1e-4)

    print("\n최적 커널 가중치 d:")
    for i, w in enumerate(d):
        print(f"  {feature_names[i]:<10}: {w:.4f}")

    # ---------------------------------------------------------
    # Evaluate MKL
    # ---------------------------------------------------------
    print("\n--- 6. MKL 평가 ---")

    K_test_comb = np.zeros_like(K_list_test[0])
    for i in range(len(d)):
        K_test_comb += d[i] * K_list_test[i]

    y_pred_mkl = svm_model.predict(K_test_comb)
    print(f"MKL Accuracy: {accuracy_score(y_test, y_pred_mkl):.4f}")

    # ---------------------------------------------------------
    # Concatenate Baseline
    # ---------------------------------------------------------
    print("\n--- 7. Concatenated Baseline ---")

    Xtr_concat = np.concatenate([X_train_features[n] for n in feature_names], axis=1)
    Xte_concat = np.concatenate([X_test_features[n] for n in feature_names], axis=1)

    svm_cat = SVC(kernel='linear')
    svm_cat.fit(Xtr_concat, y_train)
    y_pred_cat = svm_cat.predict(Xte_concat)
    print(f"Concat Accuracy: {accuracy_score(y_test, y_pred_cat):.4f}")

    # ---------------------------------------------------------
    # Save Model
    # ---------------------------------------------------------
    print("\n--- 8. 모델 저장 ---")

    model_dir = os.path.join(base_dir, "models")
    os.makedirs(model_dir, exist_ok=True)

    model_data = {
        "svm_model": svm_model,
        "kernel_weights": d,
        "feature_names": feature_names,
        "kernel_functions": kernel_functions,
        "label_encoder": le,
        "scalers": scalers,
        "X_train_features": X_train_features
    }

    save_path = os.path.join(model_dir, "simple_mkl_model.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(model_data, f)

    print(f"저장 완료: {save_path}")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    old_stdout = sys.stdout
    old_stderr = sys.stderr

    try:
        main()
    except Exception as e:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        print("❌ 에러 발생:")
        raise e

    sys.stdout = old_stdout
    sys.stderr = old_stderr

    print("\n--- 🚀 실행 완료 ---\n")
