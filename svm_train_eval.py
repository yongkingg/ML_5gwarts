#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SVM 학습 스크립트 (딥러닝 X)
- features/*.npz 에서 (color, gist, hog, sift) 로드
- SIFT -> BoW(TF-IDF) (codebook.npy, idf.npy 없으면 train으로 생성)
- 히스토그램(Color|HOG|BoW)에는 AdditiveChi2Sampler(χ² 커널 근사) 적용
- GIST는 그대로 표준화
- 두 벡터를 hstack 후 LinearSVC 학습
- 라벨/NPZ 누락 파일은 경고 출력 후 '스킵' (학습에서 제외)
"""

import os, numpy as np, cv2
from pathlib import Path
from typing import Dict, Tuple, List
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.kernel_approximation import AdditiveChi2Sampler
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold
import joblib
import csv

# ===== 경로/설정 =====
ROOT         = "/home/autonav/Desktop/ML"
FEATURES_DIR = f"{ROOT}/features"
TRAIN_DIR    = f"{ROOT}/train"
BOW_DIR      = f"{ROOT}/bow"          # codebook.npy, idf.npy 저장
MODEL_DIR    = f"{ROOT}/models"       # SVM 모델 저장
LABEL_CSV    = f"{ROOT}/labels.csv"   # (flat 구조일 때) filename,label

K                  = 800
MAX_DESC_PER_IMAGE = 2500
MAX_TOTAL_DESC     = 250_000
RANDOM_SEED        = 123
N_FOLDS            = 5                # 교차검증 폴드
# =====================

IMG_EXT = (".jpg",".jpeg",".png",".bmp",".tif",".tiff")

def list_images(d: str) -> List[str]:
    return sorted([str(p) for p in Path(d).glob("*") if p.suffix.lower() in IMG_EXT])

# ---------- NPZ 인덱스 ----------
def build_npz_index(features_dir: str) -> Dict[str, str]:
    idx = {}
    for fp in sorted(Path(features_dir).glob("*.npz")):
        try:
            d = np.load(str(fp), allow_pickle=True)
            if "image_path" not in d.files: 
                continue
            base = Path(str(d["image_path"])).name  # 예: IMG_0731.jpeg
            idx[base] = str(fp)
        except Exception as e:
            print(f"[WARN] skip npz {fp}: {e}")
    if not idx:
        raise RuntimeError(f"No usable NPZ in: {features_dir}")
    return idx

# ---------- NPZ 파트 로딩 ----------
def load_parts(npz_path: str):
    d = np.load(npz_path, allow_pickle=True)
    color = d["color_hist"] if "color_hist" in d.files else None
    gist  = d["gist"]       if "gist"       in d.files else None
    hog   = d["hog"]        if "hog"        in d.files else None
    sift  = d["sift"]       if "sift"       in d.files else None
    def f1(x): return None if x is None else x.astype(np.float32).ravel()
    def f2(x): return None if x is None else x.astype(np.float32)
    return f1(color), f1(gist), f1(hog), f2(sift)

# ---------- BoW ----------
def stack_sift_for_kmeans(npz_paths: List[str]) -> np.ndarray:
    rng = np.random.RandomState(RANDOM_SEED)
    bag=[]; total=0
    for fp in npz_paths:
        try:
            _,_,_, sift = load_parts(fp)
            if sift is None or sift.size==0: 
                continue
            if sift.shape[0] > MAX_DESC_PER_IMAGE:
                sel = rng.choice(sift.shape[0], MAX_DESC_PER_IMAGE, replace=False)
                sift = sift[sel]
            bag.append(sift.astype(np.float32))
            total += sift.shape[0]
            if total >= MAX_TOTAL_DESC:
                break
        except Exception as e:
            print(f"[WARN] kmeans load {fp}: {e}")
    if not bag:
        raise RuntimeError("No SIFT for codebook.")
    X = np.vstack(bag).astype(np.float32)
    rng.shuffle(X)
    if X.shape[0] > MAX_TOTAL_DESC:
        X = X[:MAX_TOTAL_DESC]
    print(f"[INFO] KMeans training set: {X.shape}")
    return X

def train_codebook(X: np.ndarray, k: int) -> np.ndarray:
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-4)
    _, _, centers = cv2.kmeans(X, k, None, crit, 3, cv2.KMEANS_PP_CENTERS)
    return centers.astype(np.float32)  # (k,128)

def bow_hist(desc: np.ndarray, codebook: np.ndarray) -> np.ndarray:
    k = codebook.shape[0]
    h = np.zeros((k,), dtype=np.float32)
    if desc is None or desc.size == 0:
        return h
    x2 = np.sum(desc*desc, axis=1, keepdims=True)
    c2 = np.sum(codebook*codebook, axis=1, keepdims=True).T
    prod = desc @ codebook.T
    dist = x2 + c2 - 2.0*prod
    nn = np.argmin(dist, axis=1)
    for j in nn: h[j]+=1.0
    s = h.sum()
    if s > 0: h /= s
    return h

def compute_idf(train_npz: List[str], codebook: np.ndarray) -> np.ndarray:
    H=[]
    for fp in train_npz:
        _,_,_, sift = load_parts(fp)
        H.append(bow_hist(sift, codebook))
    H = np.vstack(H).astype(np.float32)
    df = (H>0).sum(axis=0).astype(np.float32)
    idf = np.log((H.shape[0]+1.0)/(df+1.0)) + 1.0
    return idf

def tfidf(h: np.ndarray, idf: np.ndarray) -> np.ndarray:
    return np.sqrt(np.maximum(h,0,dtype=np.float32), dtype=np.float32) * idf

# ---------- 라벨 유틸 ----------
def read_labels_csv(csv_path: str) -> Dict[str, str]:
    """utf-8-sig/BOM 안전, 공백 트림, 헤더 검증."""
    mapping = {}
    if not Path(csv_path).exists():
        return mapping
    with open(csv_path, newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise RuntimeError("labels.csv를 읽을 수 없습니다.")
        field_map = {k.lower().strip(): k for k in reader.fieldnames}
        fn_key = field_map.get('filename'); lb_key = field_map.get('label')
        if not fn_key or not lb_key:
            raise RuntimeError("labels.csv 헤더는 'filename,label' 이어야 합니다.")
        for row in reader:
            fn = str(row[fn_key]).strip()
            lb = str(row[lb_key]).strip()
            if fn:
                mapping[fn] = lb
    return mapping

def infer_labels_from_tree(image_paths: List[str]) -> List[str]:
    return [Path(p).parent.name for p in image_paths]

# ---------- 메인 ----------
def main():
    os.makedirs(BOW_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    # 0) train 이미지 수집
    train_imgs = list_images(TRAIN_DIR)
    if not train_imgs:
        raise RuntimeError("No train images in: " + TRAIN_DIR)

    # 1) 라벨 결정: 하위 폴더 구조면 폴더명, 아니면 CSV
    has_subdirs = any(Path(p).parent != Path(TRAIN_DIR) for p in train_imgs)
    if has_subdirs:
        labels_all = infer_labels_from_tree(train_imgs)
        pairs_all  = list(zip(train_imgs, labels_all))
    else:
        mapping = read_labels_csv(LABEL_CSV)
        # 스킵 허용: CSV에 없는 파일은 제외
        pairs_all = []
        skipped = []
        for p in train_imgs:
            bn = Path(p).name
            lab = mapping.get(bn, None)
            if lab is None:
                skipped.append(bn)
            else:
                pairs_all.append((p, lab))
        if skipped:
            print("[WARN] labels.csv에 라벨이 없어 학습에서 제외되는 이미지:")
            for s in skipped: print("  -", s)
        if not pairs_all:
            raise RuntimeError("라벨이 매칭된 학습 이미지가 없습니다. labels.csv를 확인하세요.")

    # 2) NPZ 매칭 (누락 스킵)
    idx = build_npz_index(FEATURES_DIR)
    train_npz, y = [], []
    for p, lab in pairs_all:
        npz = idx.get(Path(p).name, "")
        if not npz:
            print("[WARN] NPZ 없음, 제외:", Path(p).name)
            continue
        train_npz.append(npz); y.append(lab)
    if not train_npz:
        raise RuntimeError("NPZ가 매칭된 학습 이미지가 없습니다. features/를 확인하세요.")

    # 3) 코드북/IDF
    cb_path = Path(BOW_DIR)/"codebook.npy"
    idf_path= Path(BOW_DIR)/"idf.npy"
    if cb_path.exists() and idf_path.exists():
        codebook = np.load(str(cb_path)).astype(np.float32)
        idf      = np.load(str(idf_path)).astype(np.float32)
        print(f"[INFO] loaded codebook/idf: {codebook.shape}, {idf.shape}")
    else:
        Xk = stack_sift_for_kmeans(train_npz)
        codebook = train_codebook(Xk, K)
        np.save(str(cb_path), codebook)
        idf = compute_idf(train_npz, codebook)
        np.save(str(idf_path), idf.astype(np.float32))
        print("[OK] saved codebook/idf")

    # 4) 피처 구성: H = [Color | HOG | BoW(TF-IDF)], G = GIST
    H_list, G_list, y_keep = [], [], []
    for fp, lab in zip(train_npz, y):
        color, gist, hog, sift = load_parts(fp)
        # 필수 파트 없으면 제외
        if color is None or gist is None or hog is None:
            print(f"[WARN] missing parts, 제외: {Path(fp).name}")
            continue
        bow  = bow_hist(sift, codebook)
        bow  = tfidf(bow, idf)
        H = np.concatenate([color, hog, bow]).astype(np.float32)
        G = gist.astype(np.float32)
        H_list.append(H); G_list.append(G); y_keep.append(lab)

    if not H_list:
        raise RuntimeError("학습 가능한 벡터가 없습니다. 피처 추출을 점검하세요.")

    X_H = np.vstack(H_list)
    X_G = np.vstack(G_list)
    y   = y_keep

    # 5) χ² 근사 → 표준화 → LinearSVC
    chi2   = AdditiveChi2Sampler(sample_steps=2, sample_interval=1e-1)
    scaleH = StandardScaler(with_mean=False)   # histogram류는 보통 with_mean=False
    scalerG= StandardScaler(with_mean=True, with_std=True)

    # 교차검증(간단)
    X_Gz = scalerG.fit_transform(X_G)
    skf = StratifiedKFold(n_splits=min(N_FOLDS, len(set(y))), shuffle=True, random_state=RANDOM_SEED)
    scores=[]
    for tr, va in skf.split(X_H, y):
        XH_tr = chi2.fit_transform(X_H[tr]); XH_tr = scaleH.fit_transform(XH_tr)
        XH_va = chi2.transform(X_H[va]);     XH_va = scaleH.transform(XH_va)
        X_tr  = np.hstack([XH_tr, X_Gz[tr]])
        X_va  = np.hstack([XH_va, X_Gz[va]])
        clf   = LinearSVC(C=1.0, class_weight="balanced", max_iter=5000)
        clf.fit(X_tr, [y[i] for i in tr])
        scores.append(clf.score(X_va, [y[i] for i in va]))
    print(f"[CV] mean acc={np.mean(scores):.4f} ± {np.std(scores):.4f}")

    # 최종 학습 및 저장
    XH = chi2.fit_transform(X_H); XH = scaleH.fit_transform(XH)
    X  = np.hstack([XH, scalerG.fit_transform(X_G)])
    clf= LinearSVC(C=1.0, class_weight="balanced", max_iter=5000)
    clf.fit(X, y)

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump({
        "codebook": codebook,
        "idf": idf,
        "chi2": chi2,
        "scaleH": scaleH,
        "scalerG": scalerG,
        "clf": clf,
        "classes_": np.unique(y)
    }, f"{MODEL_DIR}/svm_floorzone.pkl")
    print(f"[OK] saved model -> {MODEL_DIR}/svm_floorzone.pkl")

if __name__ == "__main__":
    main()
