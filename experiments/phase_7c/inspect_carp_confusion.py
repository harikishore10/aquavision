"""Investigate Carp <-> Tilapia confusion on the 375-sample training pool only.

Identifies out-of-fold errors, visual characteristics, and biological traits
without touching or inspecting any held-out test images.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROJECT_SRC = PROJECT_ROOT / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from fishfarm_ai.transfer_model import TransferLearningClassifier

def main() -> None:
    meta_path = PROJECT_ROOT / "data" / "species_images" / "metadata.csv"
    eval_rep_path = PROJECT_ROOT / "models" / "evaluation_report.json"
    out_dir = PROJECT_ROOT / "experiments" / "phase_7c"
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = pd.read_csv(meta_path)
    with open(eval_rep_path, "r", encoding="utf-8") as f:
        rep = json.load(f)

    test_files = set(rep["species_model"]["metrics"]["partition_manifest"]["test_files"])
    train_df = meta[~meta["filename"].isin(test_files)].reset_index(drop=True)
    print(f"Loaded {len(train_df)} training observations across {train_df['recorded_by'].nunique()} observers.")

    # Extract deep features
    clf = TransferLearningClassifier(backbone_name="efficientnet_b0", random_state=42)
    features = []
    labels = []
    groups = []
    filepaths = []

    for _, r in train_df.iterrows():
        p = PROJECT_ROOT / "data" / "species_images" / r["species"] / r["filename"]
        features.append(clf.extract_features(Image.open(p)))
        labels.append(r["species"])
        groups.append(str(r["recorded_by"]))
        filepaths.append(p)

    X = np.array(features, dtype=np.float32)
    y = np.array(labels)
    g = np.array(groups)
    classes = sorted(list(set(y)))

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.empty(len(y), dtype=object)
    oof_probs = np.zeros((len(y), len(classes)), dtype=np.float64)

    for tr_idx, val_idx in sgkf.split(X, y, g):
        scaler = StandardScaler()
        lr = LogisticRegression(max_iter=1000, random_state=42)
        X_tr = scaler.fit_transform(X[tr_idx])
        lr.fit(X_tr, y[tr_idx])
        X_val = scaler.transform(X[val_idx])
        oof_preds[val_idx] = lr.predict(X_val)
        oof_probs[val_idx] = lr.predict_proba(X_val)

    train_df["predicted"] = oof_preds
    train_df["is_correct"] = (train_df["predicted"] == train_df["species"])
    for idx, c in enumerate(classes):
        train_df[f"prob_{c}"] = oof_probs[:, idx]

    # Analyze Carp misclassifications
    carp_df = train_df[train_df["species"] == "carp"].copy()
    carp_tilapia_errors = carp_df[carp_df["predicted"] == "tilapia"]
    print(f"\nTotal Carp samples: {len(carp_df)}")
    print(f"Carp correct: {carp_df['is_correct'].sum()} ({carp_df['is_correct'].mean():.1%})")
    print(f"Carp misclassified as Tilapia: {len(carp_tilapia_errors)} ({len(carp_tilapia_errors)/len(carp_df):.1%})")

    # Analyze Tilapia misclassifications
    tilapia_df = train_df[train_df["species"] == "tilapia"].copy()
    tilapia_carp_errors = tilapia_df[tilapia_df["predicted"] == "carp"]
    print(f"\nTotal Tilapia samples: {len(tilapia_df)}")
    print(f"Tilapia correct: {tilapia_df['is_correct'].sum()} ({tilapia_df['is_correct'].mean():.1%})")
    print(f"Tilapia misclassified as Carp: {len(tilapia_carp_errors)} ({len(tilapia_carp_errors)/len(tilapia_df):.1%})")

    # Image attribute inspection
    def inspect_image_properties(p: Path) -> dict:
        img_bgr = cv2.imread(str(p))
        if img_bgr is None:
            return {"error": "unreadable"}
        h, w, _ = img_bgr.shape
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        sat_mean = float(hsv[:, :, 1].mean())
        val_mean = float(hsv[:, :, 2].mean())
        # Contrast: std dev of grayscale
        contrast = float(gray.std())
        return {
            "width": w,
            "height": h,
            "aspect_ratio": round(w / max(1, h), 2),
            "laplacian_var_sharpness": round(laplacian_var, 1),
            "mean_saturation": round(sat_mean, 1),
            "mean_brightness": round(val_mean, 1),
            "contrast_std": round(contrast, 1),
        }

    carp_tilapia_records = []
    for _, r in carp_tilapia_errors.iterrows():
        p = PROJECT_ROOT / "data" / "species_images" / r["species"] / r["filename"]
        props = inspect_image_properties(p)
        carp_tilapia_records.append({
            "filename": r["filename"],
            "true_species": "carp",
            "predicted": "tilapia",
            "carp_prob": round(r["prob_carp"], 3),
            "tilapia_prob": round(r["prob_tilapia"], 3),
            "observer": r.get("recorded_by", "unknown"),
            **props
        })

    tilapia_carp_records = []
    for _, r in tilapia_carp_errors.iterrows():
        p = PROJECT_ROOT / "data" / "species_images" / r["species"] / r["filename"]
        props = inspect_image_properties(p)
        tilapia_carp_records.append({
            "filename": r["filename"],
            "true_species": "tilapia",
            "predicted": "carp",
            "tilapia_prob": round(r["prob_tilapia"], 3),
            "carp_prob": round(r["prob_carp"], 3),
            "observer": r.get("recorded_by", "unknown"),
            **props
        })

    analysis_output = {
        "summary": {
            "total_train_samples": len(train_df),
            "carp_total": len(carp_df),
            "carp_correct": int(carp_df["is_correct"].sum()),
            "carp_as_tilapia_count": len(carp_tilapia_errors),
            "tilapia_total": len(tilapia_df),
            "tilapia_correct": int(tilapia_df["is_correct"].sum()),
            "tilapia_as_carp_count": len(tilapia_carp_errors),
        },
        "carp_predicted_as_tilapia": carp_tilapia_records,
        "tilapia_predicted_as_carp": tilapia_carp_records,
    }

    report_file = out_dir / "carp_tilapia_confusion_analysis.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(analysis_output, f, indent=2)
    print(f"\n[OK] Saved confusion analysis to {report_file}")


if __name__ == "__main__":
    main()
