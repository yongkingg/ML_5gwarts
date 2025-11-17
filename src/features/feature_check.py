"""
data/features/ 아래의 features_*.csv 파일을 불러와서

- 샘플(벡터) 개수
- feature 차원 수
- (있다면) 라벨 컬럼별 샘플 개수 분포

를 MatPlotLib 그래프로 시각화해서 보여주는 스크립트.
"""

from pathlib import Path
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ────────────── 경로 설정 ──────────────
# 현재 파일: ML/src/features/feature_check_plot.py
PROJECT_ROOT = Path(__file__).resolve().parents[2]      # .../ML
FEATURE_DIR  = PROJECT_ROOT / "data" / "features"


def guess_feature_type(filename: str) -> str:
    name = filename.lower()
    if "hog" in name:
        return "HOG"
    if "hsv" in name:
        return "HSV"
    if "orb" in name:
        return "ORB (avg)"
    if "sift" in name:
        return "SIFT (avg)"
    if "gist" in name:
        return "GIST"
    return "Unknown"


def show_summary_plot(df: pd.DataFrame, fname: str):
    """하나의 feature 파일에 대해 요약 정보 + 라벨 분포를 플롯으로 표시."""
    num_cols   = df.select_dtypes(include=[np.number]).columns.tolist()
    label_cols = [c for c in df.columns if c not in num_cols]

    n_samples = len(df)
    n_dim     = len(num_cols)

    feature_type = guess_feature_type(fname)

    # ── Figure 하나에 요약 정보 + (있으면) 라벨 분포 bar chart ──
    fig, ax = plt.subplots(figsize=(8, 4))

    if label_cols:
        # 첫 번째 라벨 컬럼 기준으로 분포 그림
        label_col = label_cols[0]
        counts = df[label_col].value_counts().sort_index()
        counts.plot(kind="bar", ax=ax)
        ax.set_xlabel(label_col)
        ax.set_ylabel("Sample Count")
        ax.set_title(
            f"{fname}  |  {feature_type}\n"
            f"samples={n_samples}, dim={n_dim}, label_col='{label_col}'"
        )
    else:
        # 라벨이 없으면 텍스트로만 요약 정보 노출
        ax.axis("off")
        text = (
            f"{fname}  |  {feature_type}\n\n"
            f"샘플(벡터) 개수 : {n_samples}\n"
            f"feature 차원 수 : {n_dim}\n"
            f"(라벨 컬럼 없음)"
        )
        ax.text(
            0.5, 0.5, text,
            ha="center", va="center", fontsize=12, transform=ax.transAxes
        )

    plt.tight_layout()


def main():
    print(f"PROJECT_ROOT : {PROJECT_ROOT}")
    print(f"FEATURE_DIR  : {FEATURE_DIR}")

    pattern = str(FEATURE_DIR / "features_*.csv")
    files = sorted(glob.glob(pattern))

    if not files:
        print("data/features/ 안에 features_*.csv 파일이 없습니다.")
        return

    for path in files:
        df = pd.read_csv(path)
        show_summary_plot(df, Path(path).name)

    # 모든 figure 한 번에 띄우기
    plt.show()


if __name__ == "__main__":
    main()
