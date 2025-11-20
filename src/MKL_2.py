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
# 1. Feature Extraction Functions (동일)
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
# 2. MKL 헬퍼 함수 (L1/L2 공통)
# ============================================================

def compute_objective(alpha_y_sv, K_combined_sv):
    """SVM 듀얼 목적 함수 J(d) 계산 (공통)"""
    dual_term = np.dot(alpha_y_sv, np.dot(K_combined_sv, alpha_y_sv))
    sum_alpha = np.sum(np.abs(alpha_y_sv))
    return sum_alpha - 0.5 * dual_term

def compute_gradient(alpha_y_sv, K_list_sv):
    """J(d)의 그래디언트 dJ/dd 계산 (공통)"""
    grad = []
    for K in K_list_sv:
        grad.append(-0.5 * np.dot(alpha_y_sv, np.dot(K, alpha_y_sv)))
    return np.array(grad)

def compute_current_J(d, K_list_train, y_train, C):
    """현재 d에 대한 SVM을 풀고 목적 함수 값 J(d) 반환 (공통)"""
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
# 3. [신규] L2-Norm (Ridge) MKL 헬퍼 및 훈련 함수
# ============================================================

def ridge_line_search(d, D, grad_J_new, K_list_train, y_train, C, lambda_ridge):
    """
    L2-MKL (Ridge)을 위한 Backtracking Line Search
    - J_new(d) = J(d) + lambda * ||d||^2 에 대한 Armijo 조건을 만족하는 스텝(gamma) 탐색
    """
    gamma = 1.0
    alpha = 0.5
    beta = 0.1

    # J_new(d) = J(d) + lambda * ||d||^2
    curr_J = compute_current_J(d, K_list_train, y_train, C)
    curr_J_new = curr_J + lambda_ridge * np.dot(d, d)

    grad_dot_D = np.dot(grad_J_new, D)

    if grad_dot_D > 0:  # 하강 방향이 아니면 중단
        return 0

    while True:
        d_new = d + gamma * D
        d_new[d_new < 0] = 0  # 0 미만은 0으로 투영 (Positivity)

        new_J = compute_current_J(d_new, K_list_train, y_train, C)
        new_J_new = new_J + lambda_ridge * np.dot(d_new, d_new)

        # Armijo 조건: J_new(d + gamma*D) <= J_new(d) + beta * gamma * (grad_J_new . D)
        if new_J_new <= curr_J_new + beta * gamma * grad_dot_D:
            return gamma  # 조건 만족 시 스텝 반환

        gamma *= alpha
        if gamma < 1e-10:
            return 0  # 스텝이 너무 작아지면 중단

def ridge_mkl_train(K_list_train, y_train, C=1.0, max_iter=100, tol=1e-3, lambda_ridge=0.1):
    """
    L2-Norm (Ridge) MKL 훈련 함수
    - 목적 함수: min J(d) + lambda * ||d||^2
    - 제약 조건: d_m >= 0
    """
    M = len(K_list_train)
    N = K_list_train[0].shape[0]

    d = np.ones(M) / M  # 1/M로 시작 (단순 평균)
    svm_model = None

    print("\n📌 L2-Ridge MKL Iteration Progress")
    for it in tqdm(range(max_iter), desc="L2-MKL Optimize"):
        # 1. 현재 d (정규화되지 않음)로 SVM 훈련
        K_comb = np.zeros((N, N))
        for m in range(M):
            K_comb += d[m] * K_list_train[m]

        svm = SVC(kernel='precomputed', C=C)
        svm.fit(K_comb, y_train)

        sv = svm.support_
        if len(sv) == 0:
            svm_model = svm
            break

        # 2. L2 페널티가 적용된 새 그래디언트 계산
        alpha_y = svm.dual_coef_[0]
        K_sv = [K[np.ix_(sv, sv)] for K in K_list_train]
        
        grad_J = compute_gradient(alpha_y, K_sv)  # 원본 그래디언트 dJ/dd
        grad_J_new = grad_J + 2 * lambda_ridge * d  # Ridge 페널티 항의 그래디언트(2*lambda*d) 추가

        # 3. 하강 방향 (Projected Gradient)
        D = -grad_J_new
        
        # d_m=0이고 하강 방향도 음수면, 0으로 고정 (Positivity)
        for m in range(M):
            if d[m] < 1e-10 and D[m] < 0:
                D[m] = 0
        
        # 4. 종료 조건 (그래디언트 크기)
        if np.linalg.norm(D) < tol:
            svm_model = svm
            break

        # 5. 라인 서치 (L2용)
        gamma = ridge_line_search(d, D, grad_J_new, K_list_train, y_train, C, lambda_ridge)
        
        if gamma == 0:
            svm_model = svm
            break

        # 6. d 업데이트 (심플렉스 정규화 X)
        d = d + gamma * D
        d[d < 0] = 0  # 최종 투영

    if svm_model is None:
        svm_model = svm

    # 가중치 d는 크기 자체가 최적화된 결과 (e.g., [0.5, 0.2, 0.1...])
    # 해석을 위해 정규화된 버전도 반환
    d_normalized = d / np.sum(d) if np.sum(d) > 0 else np.ones(M) / M
    
    return svm_model, d, d_normalized

# ============================================================
# 4. MAIN (L2-MKL 버전으로 수정)
# ============================================================

def main():
    print("--- 1. 데이터 준비 ---")

    try:
        current_script_path = os.path.abspath(__file__)
        base_dir = os.path.dirname(current_script_path)
    except:
        base_dir = os.getcwd()

    data_root_dir = os.path.join(base_dir,"../", "data", "processed")
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
    # 2. Load ALL images (with progress)
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
    # 3. Feature Extraction with Progress Bar
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
    # 4. Compute Kernels with Progress Bar
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
    # 5. [수정] L2-MKL Training
    # ---------------------------------------------------------
    print("\n--- 5. L2-Ridge MKL Training ---")

    # lambda_ridge 값을 조절하여 희소성(0)과 조합(분산) 사이의 균형을 맞춤
    # - lambda가 크면: 모든 d가 0에 가까워짐 (강한 L2 페널티)
    # - lambda가 작으면: d가 커질 수 있음 (L1과 유사해질 수 있음)
    LAMBDA_RIDGE = 0.1 # <--- 이 값을 1.0, 0.01 등으로 조절해보세요

    svm_model, d_raw, d_normalized = ridge_mkl_train(
        K_list_train, y_train, C=1.0, max_iter=200, tol=1e-4, lambda_ridge=LAMBDA_RIDGE
    )

    print("\n최적 커널 가중치 d (해석용 정규화):")
    for i, w in enumerate(d_normalized):
        print(f"  {feature_names[i]:<10}: {w:.4f}")

    # ---------------------------------------------------------
    # 6. [수정] L2-MKL 평가
    # ---------------------------------------------------------
    print("\n--- 6. L2-MKL 평가 ---")

    # 중요: 예측 시에는 '정규화되지 않은' raw 가중치(d_raw)를 사용해야 합니다.
    # L2 MKL은 가중치의 '크기' 자체를 학습하기 때문입니다.
    K_test_comb = np.zeros_like(K_list_test[0])
    for i in range(len(d_raw)):
        K_test_comb += d_raw[i] * K_list_test[i]

    y_pred_mkl = svm_model.predict(K_test_comb)
    print(f"L2-MKL Accuracy (lambda={LAMBDA_RIDGE}): {accuracy_score(y_test, y_pred_mkl):.4f}")

    # ---------------------------------------------------------
    # 7. Concatenated Baseline (이전과 동일)
    # ---------------------------------------------------------
    print("\n--- 7. Concatenated Baseline ---")

    Xtr_concat = np.concatenate([X_train_features[n] for n in feature_names], axis=1)
    Xte_concat = np.concatenate([X_test_features[n] for n in feature_names], axis=1)

    svm_cat = SVC(kernel='linear')
    svm_cat.fit(Xtr_concat, y_train)
    y_pred_cat = svm_cat.predict(Xte_concat)
    print(f"Concat Accuracy: {accuracy_score(y_test, y_pred_cat):.4f}")

    # ---------------------------------------------------------
    # 8. [수정] 모델 저장
    # ---------------------------------------------------------
    print("\n--- 8. 모델 저장 ---")

    model_dir = os.path.join(base_dir, "models")
    os.makedirs(model_dir, exist_ok=True)

    model_data = {
        "svm_model": svm_model,
        "kernel_weights": d_raw,  # <--- raw 가중치 저장
        "feature_names": feature_names,
        "kernel_functions": kernel_functions,
        "label_encoder": le,
        "scalers": scalers,
        "X_train_features": X_train_features,
        "mkl_type": "L2_Ridge" # MKL 타입 명시
    }

    save_path = os.path.join(model_dir, "l2_mkl_model.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(model_data, f)

    print(f"저장 완료: {save_path}")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    
    # [수정] print 출력을 위한 리디렉션 제거 (디버깅/확인을 위해)
    # redirected_output = io.StringIO()
    # sys.stdout = redirected_output
    # sys.stderr = redirected_output
    
    try:
        main()
    except Exception as e:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        print("❌ 에러 발생:")
        raise e

    sys.stdout = old_stdout
    sys.stderr = old_stderr

    # output = redirected_output.getvalue()
    # print(output)
    print("\n--- 🚀 실행 완료 ---\n")