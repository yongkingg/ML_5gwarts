import os
import numpy as np
import pandas as pd

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
FEATURE_DIR = os.path.join(BASE_DIR, "data", "features")
OUTPUT_DIR = os.path.join(FEATURE_DIR, "concat")

os.makedirs(OUTPUT_DIR, exist_ok=True)

FEATURE_FILES = {
    "gist": "features_gist.csv",
    "sift": "features_sift_avg.csv",
    "hog":  "features_hog.csv",
    "hsv":  "features_hsv.csv",
    "orb":  "features_orb_avg.csv",
}

def load_feature_and_label(path):
    df = pd.read_csv(path)
    y = df["label"].values
    df_feat = df.drop(columns=["label"])

    return df_feat.values.astype(np.float32), y


def load_feature_only(path):
    df = pd.read_csv(path)
    if "label" in df.columns:
        df = df.drop(columns=["label"])
    return df.values.astype(np.float32)

def concat_all_features():
    print("\n[Loading features and concatenating...]")
    feat_list = []
    y = None
    first = True
    for name, filename in FEATURE_FILES.items():
        path = os.path.join(FEATURE_DIR, filename)
        print(f" - Loading {name:5s} from {path}")

        if first:
            feat, y = load_feature_and_label(path)
            first = False
        else:
            feat = load_feature_only(path)

        feat_list.append(feat)

    X = np.concatenate(feat_list, axis=1)

    print("\n[Final concatenated shape]")
    print(f"X shape = {X.shape} (samples={X.shape[0]}, dims={X.shape[1]})")
    print(f"y shape = {y.shape}")

    df_out = pd.DataFrame(X)
    df_out.insert(0, "label", y)  

    out_path = os.path.join(OUTPUT_DIR, "features_concat_raw.csv")
    df_out.to_csv(out_path, index=False)

    print(f"\n[Saved concatenated features to]\n{out_path}")

    return X, y

if __name__ == "__main__":
    concat_all_features()
