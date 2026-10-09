"""Reconcile and Benchmark Fish Species Classifiers on AquaVision Phase 5 Dataset.

Evaluates HistGradientBoosting, RandomForest, and ExtraTrees using 5-fold group-stratified
cross-validation with zero observer overlap.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

PROJECT_SRC = Path(__file__).resolve().parent.parent / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from fishfarm_ai.features import extract_image_features
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix


def main():
    meta = pd.read_csv("data/species_images/metadata.csv")
    print(f"Loaded {len(meta)} metadata rows across {meta['recorded_by'].nunique()} unique observers.")

    X_list = [extract_image_features(Image.open(p)) for p in meta["filepath"]]
    X = np.array(X_list, dtype=np.float32)
    y = np.array(meta["species"])
    groups = np.array(meta["recorded_by"])
    classes = sorted(list(set(y)))
    print(f"Features extracted: X shape {X.shape}, classes: {classes}")

    # Split strictly: 375 train/val, 125 held-out test (0 observer overlap)
    sgkf = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=42)
    train_idx, test_idx = next(sgkf.split(X, y, groups))
    X_train, y_train, groups_train = X[train_idx], y[train_idx], groups[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    models = {
        "HistGradientBoosting": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", HistGradientBoostingClassifier(random_state=42)),
        ]),
        "RandomForest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=150, min_samples_split=3, random_state=42, n_jobs=-1)),
        ]),
        "ExtraTrees": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", ExtraTreesClassifier(n_estimators=150, min_samples_split=3, random_state=42, n_jobs=-1)),
        ]),
    }

    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}

    for model_name, pipeline in models.items():
        fold_accs = []
        fold_macro_f1s = []
        fit_times = []
        infer_times = []
        fold_per_class_p = {c: [] for c in classes}
        fold_per_class_r = {c: [] for c in classes}
        fold_per_class_f1 = {c: [] for c in classes}

        oof_preds = np.empty_like(y_train)

        for fold_num, (f_tr, f_val) in enumerate(cv.split(X_train, y_train, groups_train)):
            f_pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", pipeline.named_steps["clf"].__class__(**pipeline.named_steps["clf"].get_params())),
            ])
            t0 = time.perf_counter()
            f_pipe.fit(X_train[f_tr], y_train[f_tr])
            fit_times.append(time.perf_counter() - t0)

            t_inf = time.perf_counter()
            val_pred = f_pipe.predict(X_train[f_val])
            infer_times.append((time.perf_counter() - t_inf) / len(f_val) * 1000.0)

            oof_preds[f_val] = val_pred

            acc = accuracy_score(y_train[f_val], val_pred)
            mf1 = f1_score(y_train[f_val], val_pred, average="macro", zero_division=0)
            fold_accs.append(acc)
            fold_macro_f1s.append(mf1)

            for c in classes:
                p = precision_score(y_train[f_val] == c, val_pred == c, zero_division=0)
                r = recall_score(y_train[f_val] == c, val_pred == c, zero_division=0)
                f = f1_score(y_train[f_val] == c, val_pred == c, zero_division=0)
                fold_per_class_p[c].append(p)
                fold_per_class_r[c].append(r)
                fold_per_class_f1[c].append(f)

        results[model_name] = {
            "mean_acc": np.mean(fold_accs),
            "std_acc": np.std(fold_accs),
            "mean_macro_f1": np.mean(fold_macro_f1s),
            "std_macro_f1": np.std(fold_macro_f1s),
            "mean_fit_time_s": np.mean(fit_times),
            "mean_infer_ms": np.mean(infer_times),
            "fold_accs": fold_accs,
            "fold_macro_f1s": fold_macro_f1s,
            "per_class_precision": {c: float(np.mean(fold_per_class_p[c])) for c in classes},
            "per_class_recall": {c: float(np.mean(fold_per_class_r[c])) for c in classes},
            "per_class_f1": {c: float(np.mean(fold_per_class_f1[c])) for c in classes},
            "oof_classification_report": classification_report(y_train, oof_preds, output_dict=True, zero_division=0),
            "oof_confusion_matrix": confusion_matrix(y_train, oof_preds, labels=classes).tolist(),
        }

    print("\n" + "=" * 88)
    print(f"{'Classifier':<22} | {'5-Fold CV Accuracy':<18} | {'5-Fold CV Macro F1':<18} | {'Infer ms/samp':<12}")
    print("=" * 88)
    for m_name, res in results.items():
        print(f"{m_name:<22} | {res['mean_acc']:.4f} ± {res['std_acc']:.4f}     | {res['mean_macro_f1']:.4f} ± {res['std_macro_f1']:.4f}     | {res['mean_infer_ms']:.2f} ms")
    print("=" * 88)

    print("\n--- Per-Class Validation F1 Scores (Mean across 5 folds) ---")
    for m_name, res in results.items():
        print(f"\nModel: {m_name}")
        for c in classes:
            print(f"  {c:<10} -> Prec: {res['per_class_precision'][c]:.3f} | Rec: {res['per_class_recall'][c]:.3f} | F1: {res['per_class_f1'][c]:.3f}")

    # Final evaluation of HistGradientBoosting on untouched test set
    hgb_pipe = models["HistGradientBoosting"]
    hgb_pipe.fit(X_train, y_train)
    test_preds = hgb_pipe.predict(X_test)
    test_acc = accuracy_score(y_test, test_preds)
    test_rep = classification_report(y_test, test_preds, output_dict=True, zero_division=0)
    print("\n" + "=" * 88)
    print(f"HELD-OUT TEST EVALUATION (125 samples, 0 observer overlap):")
    print(f"HistGradientBoosting Accuracy: {test_acc:.2%}")
    print(f"HistGradientBoosting Macro F1: {test_rep['macro avg']['f1-score']:.4f}")
    print("=" * 88)


if __name__ == "__main__":
    main()
