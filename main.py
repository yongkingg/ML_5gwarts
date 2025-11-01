"""
Feature Descriptor
    1. Color Histogram -> PCA
    2. GIST            -> PCA
    3. HOG             -> PCA 
    4. SIFT            -> BoW Encoding
"""

"""
    Feature 정규화가 있는지 ? 
    어떻게 합칠 지 .. 
    각 Feature 뽑기
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Classic descriptors extractor (No deep learning):
  1) Color Histogram (HSV)
  2) GIST (Gabor bank + 4x4 pooling)
  3) HOG (OpenCV HOGDescriptor, global)
  4) SIFT (variable-length per image)

- 파라미터는 파일 상단의 CONFIG 섹션에서 고정.
- /home/autonav/Desktop/ML/data 아래의 모든 이미지를 재귀적으로 읽음.
- 각 이미지별 .npz 저장(+ manifest.csv 요약).
"""

import os
import cv2
import csv
import numpy as np
from pathlib import Path
from typing import List, Tuple

# =========================
# CONFIG (필요시 여기만 수정)
# =========================
DATA_ROOT   = "/home/autonav/Desktop/ML/train"     # 입력 이미지 최상위 폴더
OUT_DIR     = "/home/autonav/Desktop/ML/features" # 출력(.npz, manifest.csv) 폴더

# 리사이즈 (W,H)
RESIZE_W, RESIZE_H = 512, 256

# Color Histogram(HSV) bins
H_BINS, S_BINS, V_BINS = 32, 16, 16   # 권장 기본값

# GIST (Gabor bank + grid pooling)
GABOR_KSIZE   = 31
GABOR_SIGMAS  = (1.0, 2.0, 4.0, 8.0)  # 4 scales
GABOR_THETAS  = [i * np.pi / 8.0 for i in range(8)]  # 8 orientations
GIST_GRID     = (4, 4)                # 4x4 pooling (대략 4*8*16 = 512D)

# HOG
HOG_CELL      = (8, 8)
HOG_BLOCK     = (2, 2)                # in cells
HOG_NBINS     = 9

# SIFT
SIFT_NFEATURES = 0   # 0이면 OpenCV 기본
# =========================

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


# ------------------------------
# 유틸
# ------------------------------
def list_images(root: str) -> List[str]:
    """root 하위 모든 이미지 파일 경로를 정렬 반환(재귀)."""
    root = os.path.abspath(root)
    paths = []
    if not os.path.isdir(root):
        return paths
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            if f.lower().endswith(IMG_EXT):
                paths.append(os.path.join(dirpath, f))
    paths.sort()
    return paths

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def resize_and_colors(bgr: np.ndarray, target_size: Tuple[int, int]):
    w, h = target_size
    img = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return img, gray, hsv


# ------------------------------
# 1) Color Histogram (HSV)
# ------------------------------
def color_hist_hsv(hsv: np.ndarray) -> np.ndarray:
    """H:32, S:16, V:16 → concat → L1 → sqrt(Hellinger) → L2."""
    h_hist = cv2.calcHist([hsv], [0], None, [H_BINS], [0, 180]).flatten()
    s_hist = cv2.calcHist([hsv], [1], None, [S_BINS], [0, 256]).flatten()
    v_hist = cv2.calcHist([hsv], [2], None, [V_BINS], [0, 256]).flatten()
    hist = np.concatenate([h_hist, s_hist, v_hist]).astype(np.float32)
    s = hist.sum()
    if s > 0:
        hist /= s
    hist = np.sqrt(hist, dtype=np.float32)              # Hellinger
    hist /= (np.linalg.norm(hist) + 1e-12)              # L2
    return hist  # (H_BINS+S_BINS+V_BINS,)


# ------------------------------
# 2) GIST (Gabor bank + grid pooling)
# ------------------------------
def build_gabor_kernels():
    kernels = []
    for sigma in GABOR_SIGMAS:
        for theta in GABOR_THETAS:
            k = cv2.getGaborKernel((GABOR_KSIZE, GABOR_KSIZE), sigma, theta, 10.0, 0.5, 0.0, ktype=cv2.CV_32F)
            k -= k.mean()  # zero-mean (안정화)
            kernels.append(k)
    return kernels

_GABOR_BANK = build_gabor_kernels()

def gist_descriptor(gray: np.ndarray) -> np.ndarray:
    """가보 응답의 절댓값을 grid pooling으로 평균. log1p + L2."""
    gx, gy = GIST_GRID
    h, w = gray.shape[:2]
    cell_w = w // gx
    cell_h = h // gy

    feats = []
    for ker in _GABOR_BANK:
        resp = cv2.filter2D(gray, cv2.CV_32F, ker)
        resp = np.abs(resp)
        f = []
        for j in range(gy):
            y0 = j * cell_h
            y1 = h if j == gy - 1 else (j + 1) * cell_h
            for i in range(gx):
                x0 = i * cell_w
                x1 = w if i == gx - 1 else (i + 1) * cell_w
                patch = resp[y0:y1, x0:x1]
                f.append(float(patch.mean()))
        feats.append(np.array(f, dtype=np.float32))
    gist = np.concatenate(feats).astype(np.float32)
    gist = np.log1p(gist)                         # dynamic range 완화
    gist /= (np.linalg.norm(gist) + 1e-12)        # L2
    return gist  # 대략 4*len(SIGMAS)*len(THETAS)*len(GRID)=512D


# ------------------------------
# 3) HOG (global)
# ------------------------------
def hog_global(gray: np.ndarray) -> np.ndarray:
    """OpenCV HOGDescriptor 기반 전역 HOG."""
    h, w = gray.shape[:2]
    cell_w, cell_h = HOG_CELL
    block_w, block_h = HOG_BLOCK
    win_size = (w // (cell_w * block_w) * cell_w * block_w,
                h // (cell_h * block_h) * cell_h * block_h)
    if win_size[0] < 64 or win_size[1] < 64:
        win_size = (64, 64)
        gray = cv2.resize(gray, win_size, interpolation=cv2.INTER_AREA)

    block_size   = (block_w * cell_w, block_h * cell_h)
    block_stride = (cell_w, cell_h)
    cell_size    = (cell_w, cell_h)

    hog = cv2.HOGDescriptor(
        _winSize=win_size,
        _blockSize=block_size,
        _blockStride=block_stride,
        _cellSize=cell_size,
        _nbins=HOG_NBINS
    )
    desc = hog.compute(gray)
    if desc is None:
        return np.zeros((0,), dtype=np.float32)
    v = desc.flatten().astype(np.float32)
    v = np.sqrt(np.maximum(v, 0, dtype=np.float32))     # Hellinger
    v /= (np.linalg.norm(v) + 1e-12)                    # L2
    return v


# ------------------------------
# 4) SIFT (variable-length)
# ------------------------------
def create_sift():
    sift = None
    if hasattr(cv2, "SIFT_create"):
        try:
            sift = cv2.SIFT_create(nfeatures=SIFT_NFEATURES)
        except Exception:
            sift = None
    if sift is None and hasattr(cv2, "xfeatures2d"):
        try:
            sift = cv2.xfeatures2d.SIFT_create(nfeatures=SIFT_NFEATURES)
        except Exception:
            sift = None
    return sift

_SIFT = create_sift()

def sift_descriptors(gray: np.ndarray):
    """(kps, desc) 반환. desc는 (N,128) float32 또는 None."""
    if _SIFT is None:
        return None, None
    kps, desc = _SIFT.detectAndCompute(gray, None)
    if desc is None:
        return kps, None
    return kps, desc.astype(np.float32)


# ------------------------------
# 메인 루프
# ------------------------------
def process_image(path: str, out_dir: str, size=(RESIZE_W, RESIZE_H)) -> dict:
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Failed to read image: {path}")
    _, gray, hsv = resize_and_colors(bgr, size)

    feat_color = color_hist_hsv(hsv)
    feat_gist  = gist_descriptor(gray)
    feat_hog   = hog_global(gray)
    kps, desc_sift = sift_descriptors(gray)

    stem = Path(path).stem
    out_npz = os.path.join(out_dir, f"{stem}.npz")
    np.savez_compressed(
        out_npz,
        image_path=path,
        color_hist=feat_color,
        gist=feat_gist,
        hog=feat_hog,
        sift=(desc_sift if desc_sift is not None else np.zeros((0, 128), dtype=np.float32))
    )
    return {
        "image_path": path,
        "out_npz": out_npz,
        "color_dim": int(feat_color.size),
        "gist_dim":  int(feat_gist.size),
        "hog_dim":   int(feat_hog.size),
        "sift_n":    0 if desc_sift is None else int(desc_sift.shape[0]),
    }

def main():
    ensure_dir(OUT_DIR)
    paths = list_images(DATA_ROOT)
    if len(paths) == 0:
        raise RuntimeError(f"No images found under: {DATA_ROOT}")

    manifest_rows = []
    for idx, p in enumerate(paths, 1):
        try:
            rec = process_image(p, OUT_DIR, size=(RESIZE_W, RESIZE_H))
            manifest_rows.append(rec)
            if idx % 20 == 0:
                print(f"[{idx}/{len(paths)}] processed")
        except Exception as e:
            print(f"[WARN] {p}: {e}")

    csv_path = os.path.join(OUT_DIR, "manifest.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["image_path", "out_npz", "color_dim", "gist_dim", "hog_dim", "sift_n"])
        w.writeheader()
        w.writerows(manifest_rows)

    print("\n[Done]")
    print(f"Images    : {len(paths)}")
    print(f"Out dir   : {OUT_DIR}")
    print(f"Manifest  : {csv_path}")
    if _SIFT is None:
        print("Note: SIFT is not available in your OpenCV build. 'sift' arrays will be empty.")
    elif any(r["sift_n"] == 0 for r in manifest_rows):
        print("Note: Some images have 0 SIFT descriptors (texture-less or low-contrast).")

if __name__ == "__main__":
    main()
