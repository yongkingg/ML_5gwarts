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
from sklearn.metrics.pairwise import linear_kernel, rbf_kernel, chi2_kernel
import sys
import io
import pickle
import warnings
import matplotlib.pyplot as plt # 시각화용 (필요시)

# 경고 메시지 무시
warnings.filterwarnings('ignore')

# -----------------------------------------------------------------
# 1. 피쳐 추출 함수 (제공된 5개 스크립트 로직 통합)
# -----------------------------------------------------------------

def extract_hsv(img_bgr):
    """ (8, 8, 8) 빈의 3D HSV 히스토그램 (512-dim)을 반환합니다. """
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([img_hsv], [0, 1, 2], None, 
                        [8, 8, 8], [0, 180, 0, 256, 0, 256])
    # 히스토그램이 0이 되는 것을 방지 (chi2 커널을 위해)
    hist += 1e-6 
    cv2.normalize(hist, hist)
    return hist.flatten()

def extract_hog(img_bgr):
    """ 128x128로 리사이즈된 흑백 이미지에서 HOG 피쳐 (8100-dim)를 반환합니다. """
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    img_resized = cv2.resize(img_gray, (128, 128)) 
    feature_vector = hog(img_resized, pixels_per_cell=(8, 8),
                         cells_per_block=(2, 2), visualize=False)
    return feature_vector

def extract_sift_avg(img_bgr):
    """ SIFT 디스크립터의 평균 (128-dim)을 반환합니다. """
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    try:
        sift = cv2.SIFT_create()
        kp, des = sift.detectAndCompute(img_gray, None)
        if des is not None and len(des) > 0:
            return np.mean(des, axis=0)
        else:
            return np.zeros(128)
    except cv2.error as e:
        print(" ! SIFT_create() 에러. 'opencv-contrib-python'이 필요할 수 있습니다.", file=sys.stderr)
        return np.zeros(128)
    except Exception as e:
        print(f" ! SIFT 처리 중 오류: {e}", file=sys.stderr)
        return np.zeros(128)


def extract_orb_avg(img_bgr):
    """ ORB 디스크립터의 평균 (32-dim)을 반환합니다. """
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    try:
        orb = cv2.ORB_create(nfeatures=500) # 키포인트 수 지정
        kp, des = orb.detectAndCompute(img_gray, None)
        if des is not None and len(des) > 0:
            return np.mean(des.astype(np.float32), axis=0)
        else:
            return np.zeros(32)
    except Exception as e:
        # print(f" ! ORB 생성 실패: {e}", file=sys.stderr) # 주석 처리
        return np.zeros(32)

def compute_gist_gray(img_bgr: np.ndarray) -> np.ndarray:
    """ GIST-like 피쳐 (512-dim)를 반환합니다. """
    IMG_SIZE = 256
    N_BLOCKS = 4
    ORIENTATIONS_PER_SCALE = [8, 8, 8, 8]
    FREQUENCIES = [0.05, 0.10, 0.20, 0.40]
    
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    img_resized = cv2.resize(img_gray, (IMG_SIZE, IMG_SIZE)).astype(np.float32) / 255.0
    feats = []
    for freq, n_ori in zip(FREQUENCIES, ORIENTATIONS_PER_SCALE):
        for k in range(n_ori):
            theta = k * np.pi / n_ori
            real, imag = gabor(img_resized, frequency=freq, theta=theta)
            magnitude = np.sqrt(real ** 2 + imag ** 2)
            h, w = magnitude.shape
            bh, bw = h // N_BLOCKS, w // N_BLOCKS
            for by in range(N_BLOCKS):
                for bx in range(N_BLOCKS):
                    block = magnitude[by * bh:(by + 1) * bh, bx * bw:(bx + 1) * bw]
                    feats.append(block.mean())
    return np.asarray(feats, dtype=np.float32)

# -----------------------------------------------------------------
# 2. SimpleMKL 헬퍼 함수 (simpleMKL.pdf 기반)
# -----------------------------------------------------------------

def compute_objective(alpha_y_sv, K_combined_sv):
    """SVM 듀얼 목적 함수 J(d) 계산 [cite: 215] (Equation 10)"""
    dual_obj_term = np.dot(alpha_y_sv, np.dot(K_combined_sv, alpha_y_sv))
    sum_alpha = np.sum(np.abs(alpha_y_sv)) # dual_coef_는 y*alpha 이므로 sum(alpha)는 sum(abs(alpha_y_sv))
    
    # J(d) = sum(alpha) - 0.5 * ...
    return sum_alpha - 0.5 * dual_obj_term

def compute_gradient(alpha_y_sv, K_list_sv):
    """J(d)의 경사(gradient) 계산 [cite: 229] (Equation 11)"""
    M = len(K_list_sv)
    grad_J = np.zeros(M)
    for m in range(M):
        K_m_sv = K_list_sv[m]
        # dJ/dm = -0.5 * sum(alpha_i*alpha_j*y_i*y_j * K_m(i,j))
        grad_J[m] = -0.5 * np.dot(alpha_y_sv, np.dot(K_m_sv, alpha_y_sv))
    return grad_J

def compute_descent_direction(d, grad_J):
    """축소 경사(Reduced Gradient) 및 투영(Projection) 적용  (Equation 12)"""
    M = len(d)
    # d_m 중 0이 아닌 가장 큰 값을 기준으로 사용 (수치적 안정성)
    mu = np.argmax(d) 
    
    # 1. Reduced Gradient 계산
    reduced_grad = np.zeros(M)
    for m in range(M):
        if m != mu:
            reduced_grad[m] = grad_J[m] - grad_J[mu]
    
    # sum(D_m) = 0 제약조건을 만족시키기 위함
    reduced_grad[mu] = -np.sum(reduced_grad[np.arange(M) != mu])
    
    # 2. Descent Direction (D = -grad)
    D = -reduced_grad
    
    # 3. Projection (Positivity constraints)
    # d_m=0 이고 D_m < 0 (감소 방향)이면, d_m이 음수가 되므로 D_m=0으로 강제
    for m in range(M):
        if d[m] < 1e-10 and D[m] < 0:
            D[m] = 0
            
    # sum(D_m) = 0 제약조건 다시 적용
    D[mu] = -np.sum(D[np.arange(M) != mu])
    return D

def backtracking_line_search(d, D, grad_J, K_list_train, y_train, C):
    """라인 서치(Line Search)로 스텝 사이즈(gamma) 탐색 [cite: 262]"""
    gamma = 1.0 # 최대 스텝
    alpha = 0.5 # 스텝 감소율
    beta = 0.1  # Armijo 조건 체크용
    
    current_J = compute_current_J(d, K_list_train, y_train, C)
    grad_dot_D = np.dot(grad_J, D)
    
    # 하강 방향이 아니면 중단
    if grad_dot_D > 0:
        return 0
        
    while True:
        d_new = d + gamma * D
        
        # 스텝이 너무 커서 d가 음수가 되면 gamma를 줄임
        if np.any(d_new < 0):
            gamma *= alpha
            if gamma < 1e-10: return 0
            continue
            
        d_new /= np.sum(d_new) # 심플렉스 제약 조건 만족 (sum(d)=1)
        
        J_new = compute_current_J(d_new, K_list_train, y_train, C)
        
        # Armijo 조건: J(d + gamma*D) <= J(d) + beta * gamma * (grad_J . D)
        if J_new <= current_J + beta * gamma * grad_dot_D:
            return gamma # 조건 만족 시 스텝 반환
            
        gamma *= alpha # 조건 불만족 시 스텝 감소
        
        if gamma < 1e-10:
            return 0 # 스텝이 너무 작아지면 중단

def compute_current_J(d, K_list_train, y_train, C):
    """현재 d에 대한 SVM을 풀고 목적 함수 값 J(d) 반환 [cite: 197-199]"""
    K_combined = np.zeros_like(K_list_train[0])
    for m in range(len(d)):
        K_combined += d[m] * K_list_train[m]
        
    svm = SVC(kernel='precomputed', C=C, tol=1e-5, probability=True, cache_size=500)
    svm.fit(K_combined, y_train)
    
    sv_indices = svm.support_
    if len(sv_indices) == 0:
        return 0 # 서포트 벡터가 없는 경우
        
    # alpha_y_sv = y_i * alpha_i (for support vectors)
    alpha_y_sv = svm.dual_coef_[0] 
    
    # 서포트 벡터에 해당하는 커널 부분만 추출
    K_combined_sv = K_combined[np.ix_(sv_indices, sv_indices)]
    
    J_d = compute_objective(alpha_y_sv, K_combined_sv)
    return J_d

# -----------------------------------------------------------------
# 3. SimpleMKL 메인 훈련 알고리즘
# -----------------------------------------------------------------
def simple_mkl_train(K_list_train, y_train, C=1.0, max_iter=100, tol=1e-3):
    """SimpleMKL Algorithm 1 구현 """
    M = len(K_list_train) # 커널의 수
    n_train = K_list_train[0].shape[0] # 훈련 샘플 수
    
    # 1. d 초기화
    d = np.ones(M) / M
    svm_model = None 
    
    for i in range(max_iter):
        # 2. 현재 d로 결합된 커널 K 계산
        K_combined_train = np.zeros((n_train, n_train))
        for m in range(M):
            K_combined_train += d[m] * K_list_train[m]
            
        # 3. K로 SVM 훈련 (J(d) 계산을 위해)
        svm = SVC(kernel='precomputed', C=C, tol=1e-5, probability=True, cache_size=500)
        svm.fit(K_combined_train, y_train)
        
        sv_indices = svm.support_
        if len(sv_indices) == 0:
            print(f"반복 {i+1}회: 서포트 벡터가 없습니다. 훈련 중단.")
            svm_model = svm
            break
            
        # y_i * alpha_i
        alpha_y_sv = svm.dual_coef_[0]
        
        # 4. dJ/dm (그래디언트) 계산
        K_list_sv = [K_m[np.ix_(sv_indices, sv_indices)] for K_m in K_list_train]
        grad_J = compute_gradient(alpha_y_sv, K_list_sv)
        
        # 5. 종료 조건 확인 (Duality Gap)
        # Q = -grad_J
        # Gap = max(Q) - (d . Q)
        Q = -grad_J
        gap = np.max(Q) - np.dot(d, Q)
        
        if gap < tol:
            print(f"반복 {i+1}회: 최적해 도달 (Duality Gap < {tol}).")
            svm_model = svm
            break
            
        # 6. 하강 방향 D 계산
        D = compute_descent_direction(d, grad_J)
        
        if np.allclose(D, 0):
            print(f"반복 {i+1}회: 하강 방향이 0, 최적해 도달.")
            svm_model = svm
            break

        # 7. 라인 서치로 스텝 사이즈 gamma 결정
        gamma = backtracking_line_search(d, D, grad_J, K_list_train, y_train, C)
        
        if gamma == 0:
            print(f"반복 {i+1}회: 스텝 사이즈가 0, 최적해 도달.")
            svm_model = svm
            break
            
        # 8. d 업데이트
        d = d + gamma * D
        d[d < 0] = 0      # 음수 방지
        d /= np.sum(d)  # 정규화 (sum(d)=1)
        
        print(f"반복 {i+1}/{max_iter}: Gap={gap:.4f}, d={[round(x, 3) for x in d]}")

    if svm_model is None:
        svm_model = svm
        print("최대 반복 횟수에 도달했습니다.")
        
    return svm_model, d

# -----------------------------------------------------------------
# 4. 메인 실행 함수
# -----------------------------------------------------------------
def main():
    print("--- 1. 데이터 준비 (디렉토리 구조 기반) ---")
    
    # [수정] 스크립트 위치 및 데이터 폴더 경로 설정
    try:
        current_script_path = os.path.abspath(__file__)
        base_dir = os.path.dirname(current_script_path) # ML_5gwarts 폴더
    except NameError:
        # e.g., Jupyter 노트북에서 실행 시 __file__이 없음
        base_dir = os.path.abspath(os.getcwd())
        
    # [수정] 이미지의 'train' 디렉토리 사용
    data_root_dir = os.path.join(base_dir, "train") 
    
    if not os.path.exists(data_root_dir):
        print(f" ! 에러: '{data_root_dir}' 디렉토리를 찾을 수 없습니다.")
        print(f"   예상 스크립트 위치: {base_dir}")
        print("   스크립트가 'ML_5gwarts' 폴더에 있는지, 'train' 폴더가 있는지 확인하세요.")
        return

    # 피쳐 추출 함수 맵핑
    feature_extractors = {
        'hsv': extract_hsv,
        'hog': extract_hog,
        'sift_avg': extract_sift_avg,
        'orb_avg': extract_orb_avg,
        'gist': compute_gist_gray
    }
    
    # 커널 함수 맵핑 (피쳐마다 다른 커널을 지정)
    kernel_functions = {
        'hsv': chi2_kernel, # 히스토그램이므로 chi2
        'hog': linear_kernel,
        'sift_avg': linear_kernel,
        'orb_avg': linear_kernel,
        'gist': linear_kernel
    }
    
    feature_names = list(feature_extractors.keys())
    print(f"사용할 피쳐 디스크립터: {feature_names}")

    # --- 2. 이미지 로드 및 피쳐 추출 ---
    print("\n--- 2. 이미지 로드 및 피쳐 추출 ---")
    
    all_labels = []
    all_images = []
    
    # [수정] 'train' 폴더의 하위 폴더(1F1N, 1F2N...)를 레이블로 사용
    for label_name in sorted(os.listdir(data_root_dir)):
        zone_path = os.path.join(data_root_dir, label_name)
        if not os.path.isdir(zone_path):
            continue
        print(f"  [Zone 로드 중: {label_name}]")
        for img_name in tqdm(os.listdir(zone_path), desc=label_name):
            img_path = os.path.join(zone_path, img_name)
            img_bgr = cv2.imread(img_path)
            if img_bgr is not None:
                # [수정] 원본 이미지를 그대로 추가 (각 추출 함수가 리사이징 담당)
                all_images.append(img_bgr)
                all_labels.append(label_name)
            else:
                print(f" ! 경고: {img_path} 로드 실패")

    if not all_images:
        print(f" ! 에러: '{data_root_dir}' 폴더에서 이미지를 찾을 수 없습니다.")
        return

    # 레이블을 숫자로 변환 (e.g., "1F1N" -> 0)
    le = LabelEncoder()
    y_labels = le.fit_transform(all_labels)
    print(f"\n총 {len(all_images)}개 이미지 로드 완료.")
    print(f"클래스: {le.classes_} ({len(le.classes_)}개)")

    # 훈련 / 테스트 데이터 분리
    X_train_imgs, X_test_imgs, y_train, y_test = train_test_split(
        all_images, y_labels, test_size=0.3, random_state=42, stratify=y_labels
    )
    print(f"훈련 이미지: {len(X_train_imgs)}개, 테스트 이미지: {len(X_test_imgs)}개")
    
    # 메모리 확보 (원본 이미지 리스트 삭제)
    del all_images
    del all_labels

    # 모든 피쳐 추출
    X_train_features = {name: [] for name in feature_names}
    X_test_features = {name: [] for name in feature_names}
    scalers = {} # 피쳐별 스케일러 저장

    print("훈련 데이터 피쳐 추출 중...")
    for img in tqdm(X_train_imgs, desc="Train Extract"):
        for name, func in feature_extractors.items():
            X_train_features[name].append(func(img))
            
    print("테스트 데이터 피쳐 추출 중...")
    for img in tqdm(X_test_imgs, desc="Test Extract"):
        for name, func in feature_extractors.items():
            X_test_features[name].append(func(img))
    
    # 메모리 확보 (훈련/테스트 이미지 리스트 삭제)
    del X_train_imgs
    del X_test_imgs

    # Numpy 배열 변환 및 정규화
    for name in feature_names:
        X_train_features[name] = np.array(X_train_features[name])
        X_test_features[name] = np.array(X_test_features[name])
        
        scaler = StandardScaler().fit(X_train_features[name])
        X_train_features[name] = scaler.transform(X_train_features[name])
        X_test_features[name] = scaler.transform(X_test_features[name])
        scalers[name] = scaler # 스케일러 저장

    # --- 3. 개별 커널 계산 ---
    print("\n--- 3. 개별 커널 계산 ---")
    K_list_train = [] # 훈련 커널 (Train-Train)
    K_list_test = []  # 테스트 커널 (Test-Train)
    
    for name in feature_names:
        print(f"  - {name} 커널 계산 (커널: {kernel_functions[name].__name__})...")
        X_train = X_train_features[name]
        X_test = X_test_features[name]
        
        kernel_func = kernel_functions[name]
        
        # chi2 커널은 음수 값을 처리하지 못함
        if kernel_func == chi2_kernel:
            # StandardScaler로 인해 음수가 된 값을 양수화
            min_val_train = X_train.min()
            if min_val_train <= 0:
                X_train += -min_val_train + 1e-6 # 0 방지를 위해 1e-6 더함
            
            min_val_test = X_test.min()
            if min_val_test <= 0:
                X_test += -min_val_test + 1e-6
        
        K_list_train.append(kernel_func(X_train, X_train))
        K_list_test.append(kernel_func(X_test, X_train)) # <--- 중요: K_test(X_test, X_train)

    # --- 4. SimpleMKL 훈련 및 예측 ---
    print("\n--- 4. SimpleMKL 훈련 시작 ---")
    C_value = 1.0 
    
    final_svm_model, optimal_d = simple_mkl_train(K_list_train, y_train, C=C_value, max_iter=200, tol=1e-4)

    print("-" * 50)
    print(f"훈련 완료.")
    print(f"최적의 커널 가중치 (d):")
    for i in range(len(optimal_d)):
        print(f"  {feature_names[i]:<10}: {optimal_d[i]:.4f}")
    print("-" * 50)

    print("SimpleMKL 예측 수행...")
    
    # K_test_combined = sum(d_m * K_m_test)
    K_combined_test = np.zeros_like(K_list_test[0])
    for m in range(len(optimal_d)):
        K_combined_test += optimal_d[m] * K_list_test[m]
    
    # 훈련된 SVM 모델(final_svm_model)은 'precomputed' 커널을 사용
    y_pred_mkl = final_svm_model.predict(K_combined_test)
    print(f"✅ SimpleMKL 정확도: {accuracy_score(y_test, y_pred_mkl):.4f}")
    print("-" * 50)

    # --- 5. (비교) 나이브한 결합 (Concatenation) 방식 ---
    print("--- 5. (비교) 나이브한 피쳐 결합 (Concatenate) 수행 ---")
    
    X_train_concat = np.concatenate([X_train_features[name] for name in feature_names], axis=1)
    X_test_concat = np.concatenate([X_test_features[name] for name in feature_names], axis=1)
    
    print(f"나이브 결합 피쳐 차원: {X_train_concat.shape[1]}")

    svm_concat = SVC(kernel='linear', C=C_value)
    svm_concat.fit(X_train_concat, y_train)
    y_pred_concat = svm_concat.predict(X_test_concat)

    print(f"✅ 나이브 결합 정확도: {accuracy_score(y_test, y_pred_concat):.4f}")
    print("-" * 50)
    
    # --- 6. 모델 저장 ---
    print("--- 6. 최종 모델 저장 ---")
    model_output_dir = os.path.join(base_dir, "models")
    os.makedirs(model_output_dir, exist_ok=True)
    
    # MKL 모델 저장은 SVM 모델 외에도 많은 정보가 필요함
    mkl_model_data = {
        'svm_model': final_svm_model,       # 훈련된 SVC 객체
        'kernel_weights': optimal_d,      # 커널 가중치 d
        'feature_names': feature_names,   # 피쳐 이름 리스트
        'kernel_functions': kernel_functions, # 사용된 커널 함수 맵
        'label_encoder': le,              # 레이블 인코더
        'scalers': scalers,               # 피쳐별 스케일러
        'X_train_features': X_train_features # 예측 시 K_test 계산에 필요
    }
    
    model_path = os.path.join(model_output_dir, "simple_mkl_model.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(mkl_model_data, f)
    print(f"MKL 모델 데이터 저장 완료: {model_path}")


if __name__ == "__main__":
    # 표준 출력을 io.StringIO 객체로 리디렉션 (print문 캡처용)
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    redirected_output = io.StringIO()
    sys.stdout = redirected_output
    sys.stderr = redirected_output
    
    try:
        main()
    except Exception as e:
        # 에러 발생 시 표준 출력/에러 복원
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        print(f"\n--- 🚫 스크립트 실행 중 에러 발생 ---")
        # 캡처된 출력 인쇄
        output = redirected_output.getvalue()
        print(output)
        # 에러 트레이스백 인쇄
        raise e
    
    # 성공 시 표준 출력/에러 복원
    sys.stdout = old_stdout
    sys.stderr = old_stderr
    
    # 캡처된 모든 출력을 마지막에 한 번에 인쇄
    output = redirected_output.getvalue()
    print("--- 🚀 스크립트 실행 완료 ---")
    print(output)