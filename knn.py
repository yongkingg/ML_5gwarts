#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NPZ( color_hist | gist | hog | sift )만 사용해 BoW를 학습/적용하고
test(질의) -> train(갤러리) KNN 검색을 수행한다. (딥러닝 X)

필수: features/*.npz 안에는 최소한 'image_path', 'sift'가 있어야 함.
색/전역 특징도 npz에서 그대로 읽어 결합한다.
"""

import os, glob, numpy as np, cv2
from pathlib import Path
from typing import Dict, Tuple, List

# ==================== 경로/설정 ====================
ROOT          = "/home/autonav/Desktop/ML"
FEATURES_DIR  = f"{ROOT}/features"  # 너가 만든 npz 폴더
TRAIN_DIR     = f"{ROOT}/train"     # 갤러리
TEST_DIR      = f"{ROOT}/test"      # 질의
BOW_DIR       = f"{ROOT}/bow"       # codebook.npy, idf.npy 저장 위치

K                  = 800            # 시각 단어 개수 (300~800 권장)
MAX_DESC_PER_IMAGE = 2500           # 이미지당 SIFT 샘플 상한(학습)
MAX_TOTAL_DESC     = 250_000        # 전체 kmeans 학습용 한도
RANDOM_SEED        = 123
TOP_K              = 5              # KNN 상위 개수
# ==================================================

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

# ---------- NPZ 인덱스 ----------
def build_npz_index(features_dir: str) -> Dict[str, str]:
    """
    features/*.npz 를 훑어 '원본 이미지 basename' -> 'npz 경로' 맵을 만든다.
    (npz 내부의 image_path 키를 사용해 정확 매칭)
    """
    index = {}
    for fp in sorted(Path(features_dir).glob("*.npz")):
        try:
            d = np.load(str(fp), allow_pickle=True)
            if "image_path" not in d.files: 
                continue
            base = Path(str(d["image_path"])).name  # 예: IMG_0731.jpeg
            index[base] = str(fp)
        except Exception as e:
            print(f"[WARN] skip {fp}: {e}")
    if not index:
        raise RuntimeError(f"No usable NPZ in: {features_dir}")
    return index

def list_images(folder: str) -> List[str]:
    return sorted([str(p) for p in Path(folder).glob("*") if p.suffix.lower() in IMG_EXT])

def npz_for_image(img_path: str, idx: Dict[str, str]) -> str:
    base = Path(img_path).name
    return idx.get(base, "")

# ---------- NPZ 로딩 ----------
def load_parts_from_npz(npz_path: str) -> Tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
    """npz에서 (color_hist, gist, hog, sift) 순서로 반환. 없으면 None."""
    d = np.load(npz_path, allow_pickle=True)
    color = d["color_hist"] if "color_hist" in d.files else None
    gist  = d["gist"]       if "gist"       in d.files else None
    hog   = d["hog"]        if "hog"        in d.files else None
    sift  = d["sift"]       if "sift"       in d.files else None
    # 안전 변환
    def asfloat1d(x): 
        return None if x is None else x.astype(np.float32).ravel()
    def asfloat2d(x):
        return None if x is None else x.astype(np.float32)
    return asfloat1d(color), asfloat1d(gist), asfloat1d(hog), asfloat2d(sift)

# ---------- BoW ----------
def stack_sift_for_kmeans(npz_paths: List[str]) -> np.ndarray:
    rng = np.random.RandomState(RANDOM_SEED)
    bag=[]; total=0
    for fp in npz_paths:
        try:
            _,_,_, sift = load_parts_from_npz(fp)
            if sift is None or sift.size == 0: 
                continue
            if sift.shape[0] > MAX_DESC_PER_IMAGE:
                idx = rng.choice(sift.shape[0], MAX_DESC_PER_IMAGE, replace=False)
                sift = sift[idx]
            bag.append(sift.astype(np.float32))
            total += sift.shape[0]
            if total >= MAX_TOTAL_DESC:
                break
        except Exception as e:
            print(f"[WARN] kmeans load {fp}: {e}")
    if not bag:
        raise RuntimeError("No SIFT descriptors found for codebook.")
    X = np.vstack(bag).astype(np.float32)
    rng.shuffle(X)
    if X.shape[0] > MAX_TOTAL_DESC:
        X = X[:MAX_TOTAL_DESC]
    print(f"[INFO] KMeans training set: {X.shape}")
    return X

def train_codebook(X: np.ndarray, k: int) -> np.ndarray:
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-4)
    _, _, centers = cv2.kmeans(X, k, None, crit, 5, cv2.KMEANS_PP_CENTERS)
    return centers.astype(np.float32)  # (k,128)

def bow_hist(desc: np.ndarray, codebook: np.ndarray) -> np.ndarray:
    k = codebook.shape[0]
    h = np.zeros((k,), dtype=np.float32)
    if desc is None or desc.size == 0:
        return h
    x2 = np.sum(desc*desc, axis=1, keepdims=True)           # (N,1)
    c2 = np.sum(codebook*codebook, axis=1, keepdims=True).T # (1,k)
    prod = desc @ codebook.T                                 # (N,k)
    dist = x2 + c2 - 2.0*prod                                # (N,k)
    nn = np.argmin(dist, axis=1)
    for j in nn:
        h[j] += 1.0
    s = h.sum()
    if s > 0: h /= s
    return h

def compute_idf(train_npz_paths: List[str], codebook: np.ndarray) -> np.ndarray:
    H=[]
    for fp in train_npz_paths:
        _,_,_, sift = load_parts_from_npz(fp)
        H.append(bow_hist(sift, codebook))
    H = np.vstack(H).astype(np.float32)
    df = (H > 0).sum(axis=0).astype(np.float32)
    idf = np.log((H.shape[0]+1.0)/(df+1.0)) + 1.0
    return idf  # (K,)

def tfidf_l2(h: np.ndarray, idf: np.ndarray) -> np.ndarray:
    v = np.sqrt(np.maximum(h, 0, dtype=np.float32), dtype=np.float32) * idf
    n = np.linalg.norm(v) + 1e-12
    return v / n

# ---------- 결합 벡터 ----------
def fuse_vector_from_npz(npz_path: str, codebook: np.ndarray, idf: np.ndarray) -> np.ndarray:
    color, gist, hog, sift = load_parts_from_npz(npz_path)
    parts = []
    if color is not None: parts.append(color)
    if gist  is not None: parts.append(gist)
    if hog   is not None: parts.append(hog)
    # SIFT -> BoW(TF-IDF)
    if codebook is not None and idf is not None:
        bow = bow_hist(sift, codebook)
        bow = tfidf_l2(bow, idf)
        parts.append(bow)
    if not parts:
        raise RuntimeError(f"No usable parts in: {npz_path}")
    v = np.concatenate(parts).astype(np.float32)
    v /= (np.linalg.norm(v) + 1e-12)
    return v

# ---------- KNN ----------
def cosine_batch(G: np.ndarray, q: np.ndarray) -> np.ndarray:
    return (G @ q) / ((np.linalg.norm(G,axis=1)+1e-12) * (np.linalg.norm(q)+1e-12))

def main():
    os.makedirs(BOW_DIR, exist_ok=True)

    # 0) NPZ 인덱스 구축
    idx = build_npz_index(FEATURES_DIR)

    # 1) train/test 이미지 목록 -> 해당 npz 매핑
    train_imgs = list_images(TRAIN_DIR)
    test_imgs  = list_images(TEST_DIR)

    train_npz = [npz_for_image(p, idx) for p in train_imgs if npz_for_image(p, idx)]
    test_npz  = [npz_for_image(p, idx) for p in test_imgs  if npz_for_image(p, idx)]

    if not train_npz:
        raise RuntimeError("No train NPZ matched. Check 'image_path' inside npz or filenames.")
    if not test_npz:
        raise RuntimeError("No test NPZ matched. Check 'image_path' inside npz or filenames.")

    # 2) 코드북/IDF 만들거나 불러오기
    cb_path = Path(BOW_DIR)/"codebook.npy"
    idf_path= Path(BOW_DIR)/"idf.npy"
    if cb_path.exists() and idf_path.exists():
        codebook = np.load(str(cb_path)).astype(np.float32)
        idf      = np.load(str(idf_path)).astype(np.float32)
        print(f"[INFO] loaded codebook/idf: {codebook.shape}, {idf.shape}")
    else:
        X = stack_sift_for_kmeans(train_npz)   # train NPZ만으로 코드북 학습
        codebook = train_codebook(X, K)
        np.save(str(cb_path), codebook)
        print(f"[OK] saved codebook: {codebook.shape} -> {cb_path}")

        idf = compute_idf(train_npz, codebook) # train NPZ로 IDF
        np.save(str(idf_path), idf.astype(np.float32))
        print(f"[OK] saved idf: {idf.shape} -> {idf_path}")

    # 3) 갤러리 벡터( train ) 생성
    G_paths, G_vecs = [], []
    for fp in train_npz:
        try:
            v = fuse_vector_from_npz(fp, codebook, idf)
            G_paths.append(fp); G_vecs.append(v)
        except Exception as e:
            print(f"[WARN] skip train {fp}: {e}")
    G = np.vstack(G_vecs).astype(np.float32)

    # 4) 질의별 Top-K
    for qfp in test_npz:
        print("[DEBUG] query npz ->", qfp)   # 매핑 확인
        qv = fuse_vector_from_npz(qfp, codebook, idf)
        sims = cosine_batch(G, qv)
        top = np.argsort(-sims)[:TOP_K]
        print(f"\n[QUERY] {qfp}")
        for r,i in enumerate(top,1):
            # 원본 이미지 경로를 보고 싶으면 npz 내부 image_path를 읽어 출력해도 됨
            print(f"  {r:>2}. sim={sims[i]:.4f}  {G_paths[i]}")

if __name__ == "__main__":
    main()
