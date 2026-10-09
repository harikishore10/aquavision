import time
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

from sklearn.ensemble import (
    ExtraTreesClassifier,
    RandomForestClassifier,
    HistGradientBoostingClassifier,
    GradientBoostingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    confusion_matrix,
    brier_score_loss,
    classification_report,
)

import sys
PROJECT_SRC = Path(__file__).resolve().parent / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from fishfarm_ai.features import extract_image_features

def run_benchmarks():
    meta = pd.read_csv("data/species_images/metadata.csv")
    print(f"Loaded {len(meta)} records from metadata.csv")

    # Extract features for all images
    print("Extracting features with existing extract_image_features (59 dims)...")
    X_list = []
    y_list = []
    filenames = []
    
    t0 = time.time()
    for _, row in meta.iterrows():
        img = Image.open(row["filepath"])
        feat = extract_image_features(img)
        X_list.append(feat)
        y_list.append(row["species"])
        filenames.append(row["filename"])
    print(f"Feature extraction completed in {time.time() - t0:.2f}s")

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list)
    classes = sorted(list(set(y)))
    print("Classes:", classes)

    # Strictly separate held-out test set (25%, random_state=42)
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )
    print(f"Train/Val pool: {len(y_train_full)} samples | Held-out test set: {len(y_test)} samples")

    # Candidate models to evaluate on validation (5-fold stratified CV on X_train_full)
    candidates = {
        "ExtraTrees (n=150)": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", ExtraTreesClassifier(n_estimators=150, random_state=42)),
        ]),
        "RandomForest (n=150)": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=150, random_state=42)),
        ]),
        "HistGradientBoosting": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", HistGradientBoostingClassifier(random_state=42)),
        ]),
        "GradientBoosting (n=100)": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", GradientBoostingClassifier(random_state=42)),
        ]),
        "LogisticRegression (C=1.0)": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=42)),
        ]),
        "SVC (RBF, prob=True)": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(probability=True, random_state=42)),
        ]),
        "KNeighbors (k=5)": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=5)),
        ]),
        "Calibrated ExtraTrees (sigmoid)": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", CalibratedClassifierCV(
                estimator=ExtraTreesClassifier(n_estimators=150, random_state=42),
                method="sigmoid",
                cv=3,
            )),
        ]),
        "Calibrated LogisticRegression (sigmoid)": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", CalibratedClassifierCV(
                estimator=LogisticRegression(max_iter=1000, random_state=42),
                method="sigmoid",
                cv=3,
            )),
        ]),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = []

    print("\n" + "=" * 80)
    print(f"{'Classifier':<35} | {'Val Acc (mean±std)':<18} | {'Val Macro F1':<14} | {'Infer ms/sample':<10}")
    print("=" * 80)

    for name, pipeline in candidates.items():
        fold_accs = []
        fold_f1s = []
        infer_times = []

        for train_idx, val_idx in cv.split(X_train_full, y_train_full):
            X_tr, X_val = X_train_full[train_idx], X_train_full[val_idx]
            y_tr, y_val = y_train_full[train_idx], y_train_full[val_idx]

            pipeline.fit(X_tr, y_tr)

            t_inf_start = time.perf_counter()
            y_pred = pipeline.predict(X_val)
            t_inf = (time.perf_counter() - t_inf_start) / len(X_val) * 1000  # ms per sample
            infer_times.append(t_inf)

            fold_accs.append(accuracy_score(y_val, y_pred))
            fold_f1s.append(f1_score(y_val, y_pred, average="macro"))

        mean_acc = np.mean(fold_accs)
        std_acc = np.std(fold_accs)
        mean_f1 = np.mean(fold_f1s)
        std_f1 = np.std(fold_f1s)
        mean_infer = np.mean(infer_times)

        print(f"{name:<35} | {mean_acc:.3f} ± {std_acc:.3f}       | {mean_f1:.3f} ± {std_f1:.3f}  | {mean_infer:.2f} ms")

        results.append({
            "name": name,
            "pipeline": pipeline,
            "val_acc": mean_acc,
            "val_f1": mean_f1,
            "infer_ms": mean_infer,
        })

    # Sort candidates by validation Macro F1
    results_sorted = sorted(results, key=lambda r: r["val_f1"], reverse=True)
    best_candidate = results_sorted[0]
    print("\n" + "=" * 80)
    print(f"WINNING MODEL on Validation Data: {best_candidate['name']}")
    print(f"Validation Macro F1: {best_candidate['val_f1']:.4f} | Validation Acc: {best_candidate['val_acc']:.4f}")
    print("=" * 80)

    # Now train winning candidate on the full training set (131 samples)
    winning_pipeline = best_candidate["pipeline"]
    winning_pipeline.fit(X_train_full, y_train_full)

    # Evaluate ONCE on held-out test set
    y_test_pred = winning_pipeline.predict(X_test)
    y_test_probs = winning_pipeline.predict_proba(X_test)
    test_acc = accuracy_score(y_test, y_test_pred)
    test_macro_f1 = f1_score(y_test, y_test_pred, average="macro")
    test_weighted_f1 = f1_score(y_test, y_test_pred, average="weighted")

    print("\n--- FINAL TEST EVALUATION ON HELD-OUT SET (44 unseen images) ---")
    print(f"Test Accuracy:    {test_acc:.4f} ({test_acc:.2%})")
    print(f"Test Macro F1:    {test_macro_f1:.4f}")
    print(f"Test Weighted F1: {test_weighted_f1:.4f}")
    print("\nClassification Report on Held-Out Test Set:")
    print(classification_report(y_test, y_test_pred, target_names=classes, digits=4))

    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_test_pred, labels=classes)
    cm_df = pd.DataFrame(cm, index=[f"True_{c}" for c in classes], columns=[f"Pred_{c}" for c in classes])
    print(cm_df)

    # Also evaluate the baseline ExtraTrees on test set for direct comparison
    baseline = candidates["ExtraTrees (n=150)"]
    baseline.fit(X_train_full, y_train_full)
    base_preds = baseline.predict(X_test)
    print("\n--- BASELINE ExtraTrees on Held-Out Test Set ---")
    print(f"Baseline Test Accuracy: {accuracy_score(y_test, base_preds):.4f}")
    print(f"Baseline Macro F1:      {f1_score(y_test, base_preds, average='macro'):.4f}")


if __name__ == "__main__":
    run_benchmarks()
