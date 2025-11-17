import os
import pickle
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
CONCAT_FILE = os.path.join(BASE_DIR, "data/features/concat/features_concat_raw.csv")
MODEL_DIR = os.path.join(BASE_DIR, "data/models")
os.makedirs(MODEL_DIR, exist_ok=True)


# ================================================
# Load Dataset
# ================================================
def load_dataset():
    df = pd.read_csv(CONCAT_FILE)
    y = df["label"].values
    X = df.drop(columns=["label"]).values.astype(np.float32)
    return X, y


# ================================================
# Train Models Using ALL Data
# ================================================
def train_models_full():
    X, y = load_dataset()

    print(f"[INFO] Training on FULL dataset: {X.shape[0]} samples")

    # ================================
    # Random Forest (NO SPLIT)
    # ================================
    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X, y)

    with open(os.path.join(MODEL_DIR, "rf.pkl"), "wb") as f:
        pickle.dump(rf, f)

    print("[SAVE] Random Forest trained on full data.")

    # ================================
    # SVM (Scaled, NO SPLIT)
    # ================================
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    svm = SVC(C=10, kernel="rbf", gamma="scale", probability=True)
    svm.fit(X_scaled, y)

    with open(os.path.join(MODEL_DIR, "svm.pkl"), "wb") as f:
        pickle.dump(svm, f)

    with open(os.path.join(MODEL_DIR, "svm_scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    print("[SAVE] SVM + Scaler trained on full data.")

    print("===== TRAIN FULL DATA DONE =====")


if __name__ == "__main__":
    train_models_full()
