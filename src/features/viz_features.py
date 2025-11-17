#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
data/processed/* 에서 임의의 이미지 하나를 골라,
각 Feature Descriptor(HOG, HSV, ORB, SIFT, GIST)를

- 왼쪽 : 이미지 기반 시각화
- 오른쪽: feature vector 크기/분포 그래프

형태로 **디스크립터별 개별 Figure**로 보여주는 스크립트.
"""

from pathlib import Path
import random

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from skimage.feature import hog
from skimage import exposure
from skimage.filters import gabor

# ===== 경로/상수 설정 =====
PROJECT_ROOT  = Path(__file__).resolve().parents[2]          # .../ML
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

IMG_EXTS   = (".jpg", ".jpeg", ".png")
HOG_GRID   = 8

# GIST-like 설정 (extract_gist.py와 호환)
GIST_IMG_SIZE = 256
GIST_N_BLOCKS = 4
GIST_ORIENTATIONS_PER_SCALE = [8, 8, 8, 8]
GIST_FREQUENCIES = [0.05, 0.10, 0.20, 0.40]


# ---------- 공통 유틸 ----------

def pick_random_processed_image() -> Path:
    """data/processed/* 폴더들에서 임의의 이미지를 하나 선택."""
    if not PROCESSED_DIR.exists():
        raise FileNotFoundError(f"processed 폴더가 없습니다: {PROCESSED_DIR}")

    candidates = []
    for subdir in sorted(PROCESSED_DIR.iterdir()):
        if not subdir.is_dir():
            continue
        for ext in IMG_EXTS:
            candidates.extend(subdir.glob(f"*{ext}"))

    if not candidates:
        raise FileNotFoundError(f"{PROCESSED_DIR} 아래에서 찾은 이미지 파일이 없습니다.")

    img_path = random.choice(candidates)
    print(f"[선택된 이미지] {img_path}  (label 폴더: {img_path.parent.name})")
    return img_path


def compute_gist_gray(img_gray: np.ndarray) -> np.ndarray:
    """
    단일 채널(gray) 이미지에서 GIST-like feature 추출.
    - 여러 frequency & orientation의 Gabor 필터 적용
    - 응답을 4x4 블록으로 나누어 각 블록 평균 에너지 계산
    - 반환: 길이 = (#filters * 4 * 4) = 32*16 = 512
    """
    img_resized = cv2.resize(img_gray, (GIST_IMG_SIZE, GIST_IMG_SIZE)).astype(np.float32) / 255.0
    feats = []

    for freq, n_ori in zip(GIST_FREQUENCIES, GIST_ORIENTATIONS_PER_SCALE):
        for k in range(n_ori):
            theta = k * np.pi / n_ori

            real, imag = gabor(img_resized, frequency=freq, theta=theta)
            magnitude = np.sqrt(real**2 + imag**2)

            h, w = magnitude.shape
            bh, bw = h // GIST_N_BLOCKS, w // GIST_N_BLOCKS

            for by in range(GIST_N_BLOCKS):
                for bx in range(GIST_N_BLOCKS):
                    block = magnitude[by * bh:(by + 1) * bh,
                                      bx * bw:(bx + 1) * bw]
                    feats.append(block.mean())

    return np.asarray(feats, dtype=np.float32)


def compute_hsv_descriptor(img_bgr: np.ndarray):
    """
    HSV 3D histogram 기반 descriptor (8x8x8 = 512 dim).
    extract_hsv.py 와 개념을 맞추는 용도.
    """
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist(
        [img_hsv],
        channels=[0, 1, 2],
        mask=None,
        histSize=[8, 8, 8],
        ranges=[0, 180, 0, 256, 0, 256]
    )  # (8,8,8)
    hist = hist.flatten().astype(np.float32)
    hist /= (hist.sum() + 1e-6)  # 정규화
    return img_hsv, hist


# ---------- HOG 시각화 ----------

def visualize_hog(img_rgb, img_gray, label_name):
    img_resized = cv2.resize(img_gray, (128, 128))

    hog_vec, hog_img = hog(
        img_resized,
        pixels_per_cell=(HOG_GRID, HOG_GRID),
        cells_per_block=(2, 2),
        visualize=True
    )
    hog_img_rescaled = exposure.rescale_intensity(hog_img, in_range="image")

    print(f"[HOG] dim = {hog_vec.size}")

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle(f"HOG descriptor – {label_name}", fontsize=14)

    # 왼쪽 위: 원본(리사이즈)
    axes[0, 0].imshow(img_resized, cmap="gray")
    axes[0, 0].set_title("Input (128x128)")
    axes[0, 0].axis("off")

    # 왼쪽 아래: HOG 시각화
    axes[1, 0].imshow(hog_img_rescaled, cmap="gray", vmin=0, vmax=1)
    axes[1, 0].set_title("HOG image")
    axes[1, 0].axis("off")

    # 오른쪽 위: feature vector 앞부분(예: 400개) 라인 플롯
    N = min(400, hog_vec.size)
    axes[0, 1].plot(hog_vec[:N])
    axes[0, 1].set_title(f"HOG vector (first {N} dims of {hog_vec.size})")
    axes[0, 1].set_xlabel("dimension index")
    axes[0, 1].set_ylabel("value")

    # 오른쪽 아래: 전체 분포 히스토그램
    axes[1, 1].hist(hog_vec, bins=40)
    axes[1, 1].set_title("HOG value distribution")
    axes[1, 1].set_xlabel("value")
    axes[1, 1].set_ylabel("count")

    plt.tight_layout(rect=[0, 0, 1, 0.96])


# ---------- HSV 시각화 ----------

def visualize_hsv(img_rgb, img_bgr, label_name):
    img_hsv, hsv_vec = compute_hsv_descriptor(img_bgr)
    print(f"[HSV] dim = {hsv_vec.size} (8x8x8 histogram)")

    h = img_hsv[:, :, 0].astype(np.float32) / 179.0
    s = img_hsv[:, :, 1].astype(np.float32) / 255.0
    v = img_hsv[:, :, 2].astype(np.float32) / 255.0

    fig = plt.figure(figsize=(12, 7))
    fig.suptitle(f"HSV descriptor – {label_name}", fontsize=14)
    gs = gridspec.GridSpec(2, 3, figure=fig, wspace=0.25, hspace=0.25)

    # 왼쪽: 원본 + HSV 채널들
    ax_orig = fig.add_subplot(gs[0, 0])
    ax_h = fig.add_subplot(gs[1, 0])
    ax_s = fig.add_subplot(gs[1, 1])
    ax_v = fig.add_subplot(gs[1, 2])

    ax_orig.imshow(img_rgb)
    ax_orig.set_title("Original RGB")
    ax_orig.axis("off")

    ax_h.imshow(h, cmap="hsv", vmin=0, vmax=1)
    ax_h.set_title("H channel (Hue)")
    ax_h.axis("off")

    ax_s.imshow(s, cmap="gray", vmin=0, vmax=1)
    ax_s.set_title("S channel")
    ax_s.axis("off")

    ax_v.imshow(v, cmap="gray", vmin=0, vmax=1)
    ax_v.set_title("V channel")
    ax_v.axis("off")

    # 오른쪽 상단: HSV descriptor 벡터 (512 dim) 라인 플롯
    ax_vec = fig.add_subplot(gs[0, 1:])
    ax_vec.plot(hsv_vec)
    ax_vec.set_title(f"HSV 3D histogram vector (dim={hsv_vec.size})")
    ax_vec.set_xlabel("dimension index")
    ax_vec.set_ylabel("normalized count")

    plt.tight_layout(rect=[0, 0, 1, 0.95])


# ---------- ORB 시각화 ----------

def visualize_orb(img_rgb, img_gray, label_name):
    orb = cv2.ORB_create(nfeatures=500)
    kp, des = orb.detectAndCompute(img_gray, None)

    if des is not None and len(des) > 0:
        orb_avg = des.astype(np.float32).mean(axis=0)  # (32,)
    else:
        orb_avg = np.zeros(32, dtype=np.float32)
        kp = []

    print(f"[ORB] descriptor dim = {orb_avg.size}  (avg of N={0 if des is None else des.shape[0]} keypoints)")

    img_kp = img_rgb.copy()
    for k in kp:
        x, y = k.pt
        r = max(3, int(k.size / 2))
        cv2.circle(img_kp, (int(x), int(y)), r, (0, 255, 0), 2)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle(f"ORB descriptor – {label_name}", fontsize=14)

    axes[0, 0].imshow(img_rgb)
    axes[0, 0].set_title("Original")
    axes[0, 0].axis("off")

    axes[1, 0].imshow(img_kp)
    axes[1, 0].set_title(f"ORB keypoints (N={len(kp)})")
    axes[1, 0].axis("off")

    axes[0, 1].bar(np.arange(orb_avg.size), orb_avg)
    axes[0, 1].set_title(f"ORB avg descriptor (dim={orb_avg.size})")
    axes[0, 1].set_xlabel("dimension")
    axes[0, 1].set_ylabel("value")

    axes[1, 1].hist(orb_avg, bins=16)
    axes[1, 1].set_title("ORB value distribution")
    axes[1, 1].set_xlabel("value")
    axes[1, 1].set_ylabel("count")

    plt.tight_layout(rect=[0, 0, 1, 0.96])


# ---------- SIFT 시각화 ----------

def visualize_sift(img_rgb, img_gray, label_name):
    try:
        sift = cv2.SIFT_create()
    except AttributeError:
        print("[SIFT] OpenCV 빌드에 SIFT 미지원입니다.")
        return

    kp, des = sift.detectAndCompute(img_gray, None)

    if des is not None and len(des) > 0:
        sift_avg = des.astype(np.float32).mean(axis=0)  # (128,)
    else:
        sift_avg = np.zeros(128, dtype=np.float32)
        kp = []

    print(f"[SIFT] descriptor dim = {sift_avg.size}  (avg of N={0 if des is None else des.shape[0]} keypoints)")

    img_kp = img_rgb.copy()
    for k in kp:
        x, y = k.pt
        r = max(3, int(k.size / 2))
        cv2.circle(img_kp, (int(x), int(y)), r, (255, 0, 255), 2)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle(f"SIFT descriptor – {label_name}", fontsize=14)

    axes[0, 0].imshow(img_rgb)
    axes[0, 0].set_title("Original")
    axes[0, 0].axis("off")

    axes[1, 0].imshow(img_kp)
    axes[1, 0].set_title(f"SIFT keypoints (N={len(kp)})")
    axes[1, 0].axis("off")

    axes[0, 1].plot(sift_avg)
    axes[0, 1].set_title(f"SIFT avg descriptor (dim={sift_avg.size})")
    axes[0, 1].set_xlabel("dimension")
    axes[0, 1].set_ylabel("value")

    axes[1, 1].hist(sift_avg, bins=20)
    axes[1, 1].set_title("SIFT value distribution")
    axes[1, 1].set_xlabel("value")
    axes[1, 1].set_ylabel("count")

    plt.tight_layout(rect=[0, 0, 1, 0.96])


# ---------- GIST 시각화 ----------

def visualize_gist(img_rgb, img_gray, label_name):
    gist_vec = compute_gist_gray(img_gray)  # (512,)
    print(f"[GIST] descriptor dim = {gist_vec.size} (=32 filters x 4x4 blocks)")

    # (필터 수, 4, 4)로 reshape
    n_filters = sum(GIST_ORIENTATIONS_PER_SCALE)
    n_cells = GIST_N_BLOCKS * GIST_N_BLOCKS
    cells = gist_vec.reshape(n_filters, GIST_N_BLOCKS, GIST_N_BLOCKS)

    # 전체 평균 4x4 맵
    gist_map = cells.mean(axis=0)

    # scale별 평균 값 (주파수 대역별 에너지)
    scale_means = []
    start = 0
    for n_ori in GIST_ORIENTATIONS_PER_SCALE:
        end = start + n_ori
        scale_means.append(cells[start:end].mean())
        start = end
    scale_means = np.array(scale_means)

    fig = plt.figure(figsize=(12, 7))
    fig.suptitle(f"GIST-like descriptor – {label_name}", fontsize=14)
    gs = gridspec.GridSpec(2, 2, figure=fig, wspace=0.25, hspace=0.25)

    # 왼쪽 위: 원본 이미지
    ax_orig = fig.add_subplot(gs[0, 0])
    ax_orig.imshow(img_rgb)
    ax_orig.set_title("Original")
    ax_orig.axis("off")

    # 왼쪽 아래: 4x4 GIST block map
    ax_map = fig.add_subplot(gs[1, 0])
    im = ax_map.imshow(gist_map, cmap="magma", interpolation="nearest")
    ax_map.set_title("GIST block map (4x4)")
    ax_map.axis("off")
    fig.colorbar(im, ax=ax_map, fraction=0.046, pad=0.04)

    # 오른쪽 위: 전체 벡터 라인 플롯
    ax_vec = fig.add_subplot(gs[0, 1])
    ax_vec.plot(gist_vec)
    ax_vec.set_title(f"GIST vector (dim={gist_vec.size})")
    ax_vec.set_xlabel("dimension index")
    ax_vec.set_ylabel("value")

    # 오른쪽 아래: scale별 평균 에너지 바 차트
    ax_scale = fig.add_subplot(gs[1, 1])
    x = np.arange(len(scale_means))
    ax_scale.bar(x, scale_means)
    ax_scale.set_xticks(x)
    ax_scale.set_xticklabels([f"S{i+1}" for i in range(len(scale_means))])
    ax_scale.set_title("Average energy per scale")
    ax_scale.set_xlabel("scale (low → high freq)")
    ax_scale.set_ylabel("mean energy")

    plt.tight_layout(rect=[0, 0, 1, 0.95])


# ---------- main ----------

def main():
    img_path = pick_random_processed_image()

    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        raise FileNotFoundError(f"이미지를 읽을 수 없습니다: {img_path}")

    img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    label_name = img_path.parent.name  # 예: 1F1N

    visualize_hog(img_rgb, img_gray, label_name)
    visualize_hsv(img_rgb, img_bgr, label_name)
    visualize_orb(img_rgb, img_gray, label_name)
    visualize_sift(img_rgb, img_gray, label_name)
    visualize_gist(img_rgb, img_gray, label_name)

    # 만들어진 모든 Figure를 한 번에 표시
    plt.show()


if __name__ == "__main__":
    main()
