#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
features 폴더(.npz, manifest.csv) 빠른 점검용 스크립트
- 인자 없이 실행: CONFIG의 FEATURE_DIR만 맞추면 됨
- 전체 파일 개수, SIFT 디스크립터 유무 통계, 차원 정보 요약
- 샘플 1개 파일을 골라 key/shape/일부 값 출력
"""

import os, random, csv, numpy as np
from pathlib import Path

# ======== CONFIG ========
FEATURE_DIR = "/home/autonav/Desktop/ML/features"
# ========================

def main():
    p = Path(FEATURE_DIR)
    if not p.exists():
        raise FileNotFoundError(FEATURE_DIR)

    npz_paths = sorted([str(x) for x in p.glob("*.npz")])
    if not npz_paths:
        raise RuntimeError("NPZ가 없습니다. extract_descriptors_fixed.py를 먼저 실행하세요.")

    # manifest.csv 요약(있으면)
    manifest_path = p / "manifest.csv"
    if manifest_path.exists():
        with open(manifest_path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        print(f"[OK] manifest.csv rows: {len(rows)}")
        if rows:
            print(" 예시 3행:")
            for r in rows[:3]:
                print("  -", {k: r[k] for k in ["image_path","out_npz","color_dim","gist_dim","hog_dim","sift_n"]})
    else:
        print("[INFO] manifest.csv 없음 (문제는 아님).")

    # 전체 통계
    total = len(npz_paths)
    sift_zero = 0
    color_dim = set()
    gist_dim = set()
    hog_dim  = set()

    for fp in npz_paths[: min(500, total)]:  # 처음 500개만 훑어도 충분
        d = np.load(fp, allow_pickle=True)
        color_dim.add(int(d["color_hist"].size))
        gist_dim.add(int(d["gist"].size))
        hog_dim.add(int(d["hog"].size))
        if d["sift"].size == 0 or d["sift"].shape[0] == 0:
            sift_zero += 1

    print("\n=== 전체 요약 ===")
    print(f"NPZ 파일 수      : {total}")
    print(f"SIFT 없는 파일 수 : {sift_zero} ({sift_zero/total*100:.1f}%)")
    print(f"Color dim 다양성  : {sorted(color_dim)}")
    print(f"GIST dim 다양성   : {sorted(gist_dim)}")
    print(f"HOG  dim 다양성   : {'여러 값' if len(hog_dim)>1 else list(hog_dim)}")

    # 샘플 1개 상세 출력
    sample = random.choice(npz_paths)
    d = np.load(sample, allow_pickle=True)
    print("\n=== 샘플 파일 ===")
    print("path:", sample)
    print("keys:", d.files)
    print("color_hist:", d["color_hist"].shape, "gist:", d["gist"].shape, "hog:", d["hog"].shape, "sift:", d["sift"].shape)
    print("color_hist head:", d["color_hist"][:5])
    print("gist head      :", d["gist"][:5])
    print("hog head       :", d["hog"][:5])
    if d["sift"].size:
        print("sift first row :", d["sift"][0][:10])

if __name__ == "__main__":
    main()
