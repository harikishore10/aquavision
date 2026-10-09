"""Evaluate and optimize probability calibration on the 375-sample training pool.

Strictly adheres to:
1. Observer-group separation (StratifiedGroupKFold on recorded_by metadata).
2. Out-of-fold evaluation only (NO in-sample fitting, NO held-out test set access).
3. Evaluates Multiclass Brier score, Log Loss, and ECE before and after calibration.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from PIL import Image
from scipy.optimize import minimize
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROJECT_SRC = PROJECT_ROOT / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from fishfarm_ai.transfer_model import TransferLearningClassifier


def compute_ece(probs: np.ndarray, y_true: np.ndarray, classes: list[str], n_bins: int = 10) -> tuple[float, list[dict]]:
    top_conf = np.max(probs, axis=1)
    pred_labels = np.array([classes[i] for i in np.argmax(probs, axis=1)])
    correct = (pred_labels == y_true)

    bins = np.linspace(0.2, 1.0, n_bins + 1)
    ece = 0.0
    bin_details = []

    for i in range(len(bins) - 1):
        low, high = bins[i], bins[i + 1]
        mask = (top_conf >= low) & (top_conf < high)
        cnt = int(np.sum(mask))
        if cnt > 0:
            bin_acc = float(np.mean(correct[mask]))
            bin_conf = float(np.mean(top_conf[mask]))
            gap = abs(bin_acc - bin_conf)
            ece += (cnt / len(y_true)) * gap
            bin_details.append({
                "bin_low": round(low, 2),
                "bin_high": round(high, 2),
                "count": cnt,
                "mean_conf": round(bin_conf, 4),
                "accuracy": round(bin_acc, 4),
                "gap": round(gap, 4),
            })
    return float(ece), bin_details


def compute_multiclass_brier(probs: np.ndarray, y_true: np.ndarray, classes: list[str]) -> float:
    y_onehot = np.zeros_like(probs)
    for i, lbl in enumerate(y_true):
        y_onehot[i, classes.index(lbl)] = 1.0
    return float(np.mean(np.sum((probs - y_onehot) ** 2, axis=1)))


def main() -> None:
    # 1. Load metadata and partition manifest
    meta_path = PROJECT_ROOT / "data" / "species_images" / "metadata.csv"
    eval_rep_path = PROJECT_ROOT / "models" / "evaluation_report.json"
    
    meta = pd.read_csv(meta_path)
    with open(eval_rep_path, "r", encoding="utf-8") as f:
        rep = json.load(f)
    
    test_files = set(rep["species_model"]["metrics"]["partition_manifest"]["test_files"])
    train_df = meta[~meta["filename"].isin(test_files)].reset_index(drop=True)
    
    print(f"Training observations: {len(train_df)} across {train_df['recorded_by'].nunique()} observers.")
    
    # 2. Extract features with frozen EfficientNet-B0
    clf = TransferLearningClassifier(backbone_name="efficientnet_b0", random_state=42)
    features = []
    labels = []
    groups = []
    for _, r in train_df.iterrows():
        p = PROJECT_ROOT / "data" / "species_images" / r["species"] / r["filename"]
        features.append(clf.extract_features(Image.open(p)))
        labels.append(r["species"])
        groups.append(str(r["recorded_by"]))
    
    X = np.array(features, dtype=np.float32)
    y = np.array(labels)
    g = np.array(groups)
    classes = sorted(list(set(y)))
    
    # 3. 5-Fold StratifiedGroupKFold out-of-fold prediction gathering
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    oof_logits = np.zeros((len(y), len(classes)), dtype=np.float64)
    oof_probs_raw = np.zeros((len(y), len(classes)), dtype=np.float64)
    
    print("\nRunning 5-fold StratifiedGroupKFold on training pool...")
    for fold, (tr_idx, val_idx) in enumerate(sgkf.split(X, y, g)):
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X[tr_idx])
        X_val_s = scaler.transform(X[val_idx])
        
        lr = LogisticRegression(max_iter=1000, random_state=42)
        lr.fit(X_tr_s, y[tr_idx])
        
        # Raw logits from decision function
        logits = lr.decision_function(X_val_s)
        probs = lr.predict_proba(X_val_s)
        
        oof_logits[val_idx] = logits
        oof_probs_raw[val_idx] = probs
        print(f"  Fold {fold+1}: {len(tr_idx)} train, {len(val_idx)} val. Observers: {len(set(g[tr_idx]).intersection(set(g[val_idx])))} overlap")

    # 4. Uncalibrated baseline metrics
    brier_raw = compute_multiclass_brier(oof_probs_raw, y, classes)
    loss_raw = log_loss(y, oof_probs_raw, labels=classes)
    ece_raw, bins_raw = compute_ece(oof_probs_raw, y, classes)
    
    print("\n==========================================")
    print("UNCALIBRATED PROBABILITY METRICS (OOF):")
    print(f"  Multi-class Brier Score : {brier_raw:.4f}")
    print(f"  Log Loss                : {loss_raw:.4f}")
    print(f"  Expected Calibration Err: {ece_raw:.4f} ({ece_raw*100:.2f}%)")
    print("==========================================")

    # 5. Fit Temperature Scaling T on out-of-fold logits
    # Softmax function with temperature
    def temp_softmax(logits: np.ndarray, temp: float) -> np.ndarray:
        scaled = logits / max(temp, 1e-4)
        exp_z = np.exp(scaled - np.max(scaled, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def nll_objective(temp_arr: np.ndarray) -> float:
        t = temp_arr[0]
        probs = temp_softmax(oof_logits, t)
        return log_loss(y, probs, labels=classes)

    res = minimize(nll_objective, x0=[1.5], bounds=[(0.1, 10.0)], method="L-BFGS-B")
    optimal_t = float(res.x[0])
    print(f"\nOptimization converged: Optimal Temperature T = {optimal_t:.4f}")

    # 6. Evaluate Calibrated metrics with optimal T
    oof_probs_cal = temp_softmax(oof_logits, optimal_t)
    brier_cal = compute_multiclass_brier(oof_probs_cal, y, classes)
    loss_cal = log_loss(y, oof_probs_cal, labels=classes)
    ece_cal, bins_cal = compute_ece(oof_probs_cal, y, classes)

    print("\n==========================================")
    print(f"CALIBRATED PROBABILITY METRICS (T = {optimal_t:.4f}):")
    print(f"  Multi-class Brier Score : {brier_cal:.4f} (Delta {brier_cal - brier_raw:+.4f})")
    print(f"  Log Loss                : {loss_cal:.4f} (Delta {loss_cal - loss_raw:+.4f})")
    print(f"  Expected Calibration Err: {ece_cal:.4f} ({ece_cal*100:.2f}%) (Delta {(ece_cal - ece_raw)*100:+.2f}%)")
    print("==========================================")

    print("\nReliability Diagram Comparison (Confidence Bins):")
    print(f"{'Bin':<16} | {'Raw Conf':<9} | {'Raw Acc':<8} | {'Raw Gap':<8} | {'Cal Conf':<9} | {'Cal Acc':<8} | {'Cal Gap':<8}")
    print("-" * 75)
    
    # Format comparison table
    bin_dict_raw = {(b["bin_low"], b["bin_high"]): b for b in bins_raw}
    bin_dict_cal = {(b["bin_low"], b["bin_high"]): b for b in bins_cal}
    all_ranges = sorted(set(bin_dict_raw.keys()).union(set(bin_dict_cal.keys())))
    
    for r in all_ranges:
        r_str = f"[{r[0]:.2f}, {r[1]:.2f})"
        br = bin_dict_raw.get(r, {})
        bc = bin_dict_cal.get(r, {})
        r_conf = f"{br.get('mean_conf', 0):.3f}" if br else "  -  "
        r_acc  = f"{br.get('accuracy', 0):.3f}" if br else "  -  "
        r_gap  = f"{br.get('gap', 0):.3f}" if br else "  -  "
        c_conf = f"{bc.get('mean_conf', 0):.3f}" if bc else "  -  "
        c_acc  = f"{bc.get('accuracy', 0):.3f}" if bc else "  -  "
        c_gap  = f"{bc.get('gap', 0):.3f}" if bc else "  -  "
        print(f"{r_str:<16} | {r_conf:<9} | {r_acc:<8} | {r_gap:<8} | {c_conf:<9} | {c_acc:<8} | {c_gap:<8}")

    # Tier analysis
    def evaluate_tiers(probs_matrix: np.ndarray, label_str: str) -> None:
        print(f"\n--- Uncertainty Tier Distribution: {label_str} ---")
        pred_labels = np.array([classes[i] for i in np.argmax(probs_matrix, axis=1)])
        correct = (pred_labels == y)
        states = []
        for i in range(len(probs_matrix)):
            p_row = probs_matrix[i]
            sorted_p = np.sort(p_row)[::-1]
            p1, p2 = sorted_p[0], sorted_p[1]
            margin = p1 - p2
            entropy = -float(np.sum(p_row * np.log2(np.maximum(p_row, 1e-9))))
            if p1 < 0.40 or margin < 0.10 or entropy > 1.95:
                states.append("UNCERTAIN")
            elif p1 < 0.65 or margin < 0.25:
                states.append("MARGINAL")
            else:
                states.append("CONFIDENT")
        states = np.array(states)
        for st in ["CONFIDENT", "MARGINAL", "UNCERTAIN"]:
            mask = (states == st)
            cnt = int(np.sum(mask))
            if cnt > 0:
                acc = float(np.mean(correct[mask]))
                print(f"  {st:<10}: {cnt:3d} samples ({cnt/len(y)*100:4.1f}%) | Empirical Accuracy: {acc*100:5.1f}% | Error Rate: {(1-acc)*100:5.1f}%")
            else:
                print(f"  {st:<10}:   0 samples")

    evaluate_tiers(oof_probs_raw, "Raw Softmax (Uncalibrated)")
    evaluate_tiers(oof_probs_cal, f"Calibrated (Temperature T = {optimal_t:.4f})")


if __name__ == "__main__":
    main()
