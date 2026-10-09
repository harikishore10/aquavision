"""AquaVision Phase 6 — Transfer Learning vs Handcrafted Baseline Benchmark.

Compares:
1. HistGradientBoosting (Phase 5 Handcrafted 75-dim Baseline)
2. MobileNetV3-Small (Transfer Learning 576-dim Candidate)
3. EfficientNet-B0 (Transfer Learning 1280-dim Candidate)

Uses strictly observer-separated StratifiedGroupKFold splits with zero observer overlap.
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
from fishfarm_ai.transfer_model import TransferLearningClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix


def run_benchmark():
    meta = pd.read_csv("data/species_images/metadata.csv")
    print(f"Loaded {len(meta)} metadata rows across {meta['recorded_by'].nunique()} unique observers.")

    groups = meta["recorded_by"].values
    y = meta["species"].values
    classes = sorted(list(set(y)))
    image_paths = meta["filepath"].tolist()

    # 1. Strict 75/25 group-stratified split (0 observer overlap)
    sgkf = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=42)
    train_idx, test_idx = next(sgkf.split(meta.index.values, y, groups))
    y_train = y[train_idx]
    y_test = y[test_idx]
    groups_train = groups[train_idx]

    overlap = set(groups_train).intersection(set(groups[test_idx]))
    print(f"Partitions: {len(y_train)} train vs {len(y_test)} test. Observer overlap count: {len(overlap)}")

    # 2. Extract features for all models
    print("\n[1/3] Extracting Handcrafted 75-dim features...")
    t0 = time.time()
    X_hand = np.array([extract_image_features(Image.open(p)) for p in image_paths], dtype=np.float32)
    print(f"  Handcrafted features extracted in {time.time()-t0:.2f}s: shape {X_hand.shape}")

    print("[2/3] Extracting MobileNetV3-Small 576-dim features...")
    mb = TransferLearningClassifier(backbone_name="mobilenet_v3_small", random_state=42)
    t0 = time.time()
    X_mb = np.array([mb.extract_features(Image.open(p)) for p in image_paths], dtype=np.float32)
    print(f"  MobileNetV3 features extracted in {time.time()-t0:.2f}s: shape {X_mb.shape}")

    print("[3/3] Extracting EfficientNet-B0 1280-dim features...")
    ef = TransferLearningClassifier(backbone_name="efficientnet_b0", random_state=42)
    t0 = time.time()
    X_ef = np.array([ef.extract_features(Image.open(p)) for p in image_paths], dtype=np.float32)
    print(f"  EfficientNet-B0 features extracted in {time.time()-t0:.2f}s: shape {X_ef.shape}")

    models_config = {
        "HistGradientBoosting (Baseline 75-dim)": {
            "features": X_hand,
            "pipeline": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", HistGradientBoostingClassifier(random_state=42)),
            ]),
            "model_size_mb": 0.8,
        },
        "MobileNetV3-Small (Transfer 576-dim)": {
            "features": X_mb,
            "pipeline": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000, random_state=42)),
            ]),
            "model_size_mb": 10.2,
        },
        "EfficientNet-B0 (Transfer 1280-dim)": {
            "features": X_ef,
            "pipeline": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000, random_state=42)),
            ]),
            "model_size_mb": 20.5,
        },
    }

    # 3. 5-Fold Group Cross-Validation on Training Pool
    cv5 = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    cv_splits = list(cv5.split(X_hand[train_idx], y_train, groups_train))

    cv_results = {}
    test_results = {}

    print("\n" + "=" * 94)
    print(f"{'Model Architecture':<40} | {'5-Fold CV Accuracy':<18} | {'5-Fold CV Macro F1':<18} | {'Model Size':<10}")
    print("=" * 94)

    for name, cfg in models_config.items():
        feat = cfg["features"]
        pipe = cfg["pipeline"]
        fold_accs = []
        fold_f1s = []

        for f_tr, f_val in cv_splits:
            X_tr, y_tr = feat[train_idx][f_tr], y_train[f_tr]
            X_val, y_val = feat[train_idx][f_val], y_train[f_val]
            f_pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", pipe.named_steps["clf"].__class__(**pipe.named_steps["clf"].get_params())),
            ])
            f_pipe.fit(X_tr, y_tr)
            preds = f_pipe.predict(X_val)
            fold_accs.append(accuracy_score(y_val, preds))
            fold_f1s.append(f1_score(y_val, preds, average="macro", zero_division=0))

        mean_acc, std_acc = np.mean(fold_accs), np.std(fold_accs)
        mean_f1, std_f1 = np.mean(fold_f1s), np.std(fold_f1s)
        cv_results[name] = {
            "mean_acc": mean_acc,
            "std_acc": std_acc,
            "mean_f1": mean_f1,
            "std_f1": std_f1,
        }
        print(f"{name:<40} | {mean_acc:.3f} ± {std_acc:.3f}       | {mean_f1:.3f} ± {std_f1:.3f}       | {cfg['model_size_mb']:>5.1f} MB")

        # Fit on full training set and evaluate on test set
        pipe.fit(feat[train_idx], y_train)

        t_inf_start = time.perf_counter()
        test_preds = pipe.predict(feat[test_idx])
        inf_ms = (time.perf_counter() - t_inf_start) / len(test_idx) * 1000.0

        t_acc = accuracy_score(y_test, test_preds)
        t_f1 = f1_score(y_test, test_preds, average="macro", zero_division=0)
        rep = classification_report(y_test, test_preds, output_dict=True, zero_division=0)
        cm = confusion_matrix(y_test, test_preds, labels=classes)

        test_results[name] = {
            "test_acc": t_acc,
            "test_macro_f1": t_f1,
            "infer_ms": inf_ms,
            "report": rep,
            "cm": cm,
        }

    print("=" * 94)

    # 4. Final Comparison on Held-Out Test Set (125 samples)
    print("\n" + "=" * 94)
    print(f"{'Model Architecture':<40} | {'Test Accuracy':<14} | {'Test Macro F1':<14} | {'Accuracy Delta':<14} | {'F1 Delta':<10}")
    print("=" * 94)

    base_acc = test_results["HistGradientBoosting (Baseline 75-dim)"]["test_acc"]
    base_f1 = test_results["HistGradientBoosting (Baseline 75-dim)"]["test_macro_f1"]

    for name in models_config:
        res = test_results[name]
        d_acc = res["test_acc"] - base_acc
        d_f1 = res["test_macro_f1"] - base_f1
        sign_acc = "+" if d_acc >= 0 else ""
        sign_f1 = "+" if d_f1 >= 0 else ""
        print(f"{name:<40} | {res['test_acc']:.2%}         | {res['test_macro_f1']:.4f}        | {sign_acc}{d_acc:.2%}         | {sign_f1}{d_f1:.4f}")

    print("=" * 94)

    # 5. Per-Species Breakdown for EfficientNet-B0 vs Baseline
    print("\n--- Per-Species F1 Comparison (Baseline vs EfficientNet-B0) ---")
    base_rep = test_results["HistGradientBoosting (Baseline 75-dim)"]["report"]
    ef_rep = test_results["EfficientNet-B0 (Transfer 1280-dim)"]["report"]

    for sp in classes:
        b_f1 = base_rep[sp]["f1-score"]
        e_f1 = ef_rep[sp]["f1-score"]
        diff = e_f1 - b_f1
        print(f"  {sp:<12} -> Baseline F1: {b_f1:.3f} | EfficientNet-B0 F1: {e_f1:.3f} | Improvement: +{diff:.3f}")

    print("\nConfusion Matrix (EfficientNet-B0 on 125 Test Images):")
    print(pd.DataFrame(test_results["EfficientNet-B0 (Transfer 1280-dim)"]["cm"], index=classes, columns=classes))


if __name__ == "__main__":
    run_benchmark()
