#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
data/features/*.csv 에 저장된 피처들을 이용해
층(label)을 분류하는 RandomForestClassifier 학습 스크립트.

- 디스크립터별(HOG, HSV, ORB_avg, SIFT_avg, GIST)로 개별 모델 학습
- 모든 피처를 concat한 통합 모델도 학습
- accuracy, precision, recall, F1-score를 수치 + 시각화로 확인
- 모든 Figure는 마지막에 한 번에 띄운다.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
import joblib


# ===== 경로 설정 =====
PROJECT_ROOT = Path(__file__).resolve().parents[2]      # .../ML
FEATURES_DIR = PROJECT_ROOT / "data" / "features"
MODELS_DIR   = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42


# ===== 유틸 함수들 =====
def load_feature_csv(feature_name: str):
    """
    features_{feature_name}.csv 를 로드해서 (X, y)를 반환.
    CSV 포맷은 다음과 같다고 가정:
        label, {feature_name}_0, {feature_name}_1, ...
    """
    csv_path = FEATURES_DIR / f"features_{feature_name}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV 파일을 찾을 수 없습니다: {csv_path}")

    df = pd.read_csv(csv_path)
    if "label" not in df.columns:
        raise ValueError(f"{csv_path} 에 'label' 컬럼이 없습니다.")

    y = df["label"].values
    X = df.drop(columns=["label"]).values.astype(np.float32)

    print(f"[LOAD] {feature_name:8s} | samples={X.shape[0]}, dim={X.shape[1]}")
    return X, y


def concat_features(feature_names):
    """
    여러 디스크립터를 concat해서 X_all, y_all 을 반환.
    각 CSV의 row 순서가 동일하다고 가정 (feature 추출 코드가 동일한 processed 순회 사용).
    만약 label이 안 맞으면 에러를 발생시켜서 보호.
    """
    X_list = []
    y_ref = None

    for fname in feature_names:
        X, y = load_feature_csv(fname)

        if y_ref is None:
            y_ref = y
        else:
            if not np.array_equal(y_ref, y):
                raise ValueError(
                    f"레이블 순서가 {feature_names[0]} 과 {fname} 사이에서 다릅니다."
                )

        X_list.append(X)

    X_concat = np.concatenate(X_list, axis=1)
    print(
        f"[CONCAT] {' + '.join(feature_names)} | "
        f"samples={X_concat.shape[0]}, dim={X_concat.shape[1]}"
    )
    return X_concat, y_ref


# ===== 시각화 함수 =====
def visualize_metrics(y_test, y_pred, classes, feature_name: str):
    """
    y_test / y_pred (정수 인코딩)와 클래스 이름을 받아서

    - Confusion matrix heatmap
    - 클래스별 F1-score bar chart

    를 하나의 Figure에 그린다.
    (여기서는 fig만 반환하고 show는 하지 않는다.)
    """
    # classification_report 딕셔너리 버전
    report = classification_report(
        y_test,
        y_pred,
        target_names=classes,
        output_dict=True,
        zero_division=0,
    )

    # per-class F1
    f1_scores = [report[c]["f1-score"] for c in classes]

    macro_f1    = report["macro avg"]["f1-score"]
    weighted_f1 = report["weighted avg"]["f1-score"]

    # confusion matrix (정규화)
    labels_idx = np.arange(len(classes))
    cm = confusion_matrix(y_test, y_pred, labels=labels_idx)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"RandomForest metrics – {feature_name}", fontsize=14)

    # ----- (1) Confusion matrix heatmap -----
    ax_cm = axes[0]
    im = ax_cm.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    ax_cm.set_title("Normalized Confusion Matrix")
    ax_cm.set_xlabel("Predicted label")
    ax_cm.set_ylabel("True label")
    ax_cm.set_xticks(labels_idx)
    ax_cm.set_yticks(labels_idx)
    ax_cm.set_xticklabels(classes, rotation=45, ha="right")
    ax_cm.set_yticklabels(classes)

    # 숫자 쓰기
    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            val = cm_norm[i, j]
            ax_cm.text(
                j,
                i,
                f"{val:.2f}",
                ha="center",
                va="center",
                color="white" if val > 0.5 else "black",
                fontsize=8,
            )

    fig.colorbar(im, ax=ax_cm, fraction=0.046, pad=0.04)

    # ----- (2) F1-score bar chart -----
    ax_f1 = axes[1]
    x = np.arange(len(classes))
    ax_f1.bar(x, f1_scores)
    ax_f1.set_xticks(x)
    ax_f1.set_xticklabels(classes, rotation=45, ha="right")
    ax_f1.set_ylim(0, 1.05)
    ax_f1.set_ylabel("F1-score")
    ax_f1.set_title(
        f"Per-class F1 (macro={macro_f1:.3f}, weighted={weighted_f1:.3f})"
    )

    # 막대 위에 수치 표시
    for xi, f1 in zip(x, f1_scores):
        ax_f1.text(xi, f1 + 0.02, f"{f1:.2f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    return fig


# ===== 학습 + 평가 함수 =====
def train_eval_rf(X, y, feature_name: str, save_model: bool = True):
    """
    단일 feature set(X, y)에 대해 RandomForest 학습 및 평가 + 시각화.
    fig 객체를 반환하고, plt.show()는 호출하지 않는다.
    """
    # label을 숫자로 인코딩
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_enc,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y_enc,
    )

    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )

    print(f"\n[TRAIN] RandomForest on {feature_name} ...")
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"[RESULT] {feature_name:12s} accuracy = {acc:.4f}")

    # 텍스트 classification_report도 같이 출력
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # ---- 시각화 (fig만 만들고 반환) ----
    fig = visualize_metrics(y_test, y_pred, le.classes_, feature_name)

    # ---- 모델 저장 ----
    if save_model:
        model_path = MODELS_DIR / f"rf_{feature_name}.joblib"
        joblib.dump(
            {
                "model": clf,
                "label_encoder": le,
                "feature_name": feature_name,
            },
            model_path,
        )
        print(f"[SAVE] 모델 저장 완료: {model_path}")

    return clf, le, acc, fig


# ===== main =====
def main():
    # 최종적으로 한 번에 show 할 figure들을 모아둔다
    all_figs = []

    # 1) 개별 디스크립터별 학습 + 시각화
    single_features = ["hog", "hsv", "orb_avg", "sift_avg", "gist"]

    for fname in single_features:
        X, y = load_feature_csv(fname)
        _, _, _, fig = train_eval_rf(X, y, feature_name=fname, save_model=True)
        all_figs.append(fig)

    # 2) 모든 디스크립터 concat 해서 통합 모델 학습 + 시각화
    try:
        X_all, y_all = concat_features(single_features)
        _, _, _, fig_all = train_eval_rf(
            X_all, y_all, feature_name="all_features", save_model=True
        )
        all_figs.append(fig_all)
    except Exception as e:
        print(f"\n[WARN] 통합 feature 학습 중 오류 발생: {e}")

    # ===== 여기서 한 번에 모든 figure를 띄운다 =====
    if all_figs:
        plt.show()


if __name__ == "__main__":
    main()
