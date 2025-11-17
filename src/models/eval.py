import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report
)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
DATA_FILE = os.path.join(BASE_DIR, "data/features/concat/features_concat_raw.csv")
MODEL_DIR = os.path.join(BASE_DIR, "data/models")

def load_dataset():
    df = pd.read_csv(DATA_FILE)
    y = df["label"].values
    X = df.drop(columns=["label"]).values.astype(np.float32)
    return X, y

def load_models():
    with open(os.path.join(MODEL_DIR, "rf.pkl"), "rb") as f:
        rf = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "svm.pkl"), "rb") as f:
        svm = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "svm_scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)

    return rf, svm, scaler

def compute_metrics(model, X, y, scaler=None):
    if scaler is not None:
        X = scaler.transform(X)

    pred = model.predict(X)

    return {
        "acc": accuracy_score(y, pred),
        "macro_f1": f1_score(y, pred, average="macro"),
        "micro_f1": f1_score(y, pred, average="micro"),
        "cm": confusion_matrix(y, pred),
        "report": classification_report(y, pred)
    }

def plot_f1_with_acc(ax, metrics, title):
    macro = metrics["macro_f1"]
    micro = metrics["micro_f1"]
    acc = metrics["acc"]

    values = [macro, micro]
    bars = ax.bar(["Macro F1", "Micro F1"], values,
                  color=["#4C72B0", "#55A868"])

    ax.set_ylim(0, 1)
    ax.set_ylabel("F1 Score")
    ax.set_title(title, fontsize=13)

    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height / 2,                           
            f"Acc: {acc:.4f}",
            ha='center',
            va='center',
            fontsize=11,
            fontweight='bold',
            color="white" if height > 0.5 else "black"  
        )

def plot_metrics(rf_metrics, svm_metrics, class_names):

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle("Random Forest & SVM Evaluation", fontsize=18, y=0.97)

    plot_f1_with_acc(axes[0][0], rf_metrics, "RF - F1 Scores")

    sns.heatmap(rf_metrics["cm"], annot=True, cmap="Blues", fmt="d",
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[0][1])
    axes[0][1].set_title("RF - Confusion Matrix")

    plot_f1_with_acc(axes[1][0], svm_metrics, "SVM - F1 Scores")

    sns.heatmap(svm_metrics["cm"], annot=True, cmap="Blues", fmt="d",
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[1][1])
    axes[1][1].set_title("SVM - Confusion Matrix")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

def main():
    X, y = load_dataset()
    rf, svm, scaler = load_models()

    rf_metrics = compute_metrics(rf, X, y)
    svm_metrics = compute_metrics(svm, X, y, scaler=scaler)

    print("===== Random Forest Report =====")
    print(rf_metrics["report"])

    print("\n===== SVM Report =====")
    print(svm_metrics["report"])

    class_names = np.unique(y)
    plot_metrics(rf_metrics, svm_metrics, class_names)


if __name__ == "__main__":
    main()
