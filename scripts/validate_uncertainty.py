"""Empirical Uncertainty and Out-of-Distribution Validation Script for AquaVision.

Evaluates:
1. Error rates across CONFIDENT, MARGINAL, and UNCERTAIN tiers on 375 out-of-fold samples.
2. Error rates across tiers on 125 unseen held-out test samples (0 observer overlap).
3. Open-set out-of-distribution (OOD) response to non-fish patterns, blank fields, and noise.
"""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

PROJECT_SRC = Path(__file__).resolve().parent.parent / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from fishfarm_ai.features import extract_image_features
from fishfarm_ai.services import SpeciesIdentificationService
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier


def main():
    meta = pd.read_csv("data/species_images/metadata.csv")
    print(f"Loaded {len(meta)} metadata rows across {meta['recorded_by'].nunique()} unique observers.")

    X_list = [extract_image_features(Image.open(p)) for p in meta["filepath"]]
    X = np.array(X_list, dtype=np.float32)
    y = np.array(meta["species"])
    groups = np.array(meta["recorded_by"])

    # Strict 75/25 group-stratified partition
    sgkf = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=42)
    tr_idx, te_idx = next(sgkf.split(X, y, groups))
    X_train, y_train, groups_train = X[tr_idx], y[tr_idx], groups[tr_idx]
    X_test, y_test = X[te_idx], y[te_idx]

    # 5-fold group cross-validation on train pool (375 samples)
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    val_true = []
    val_pred = []
    val_max_prob = []
    val_margin = []
    val_entropy = []
    val_state = []

    for f_tr, f_val in cv.split(X_train, y_train, groups_train):
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", HistGradientBoostingClassifier(random_state=42)),
        ])
        pipe.fit(X_train[f_tr], y_train[f_tr])
        probs = pipe.predict_proba(X_train[f_val])
        preds = pipe.predict(X_train[f_val])

        for i in range(len(preds)):
            p_row = probs[i]
            sorted_p = np.sort(p_row)[::-1]
            p1, p2 = sorted_p[0], sorted_p[1]
            margin = p1 - p2
            entropy = -np.sum(p_row * np.log2(np.maximum(p_row, 1e-9)))

            if p1 < 0.40 or margin < 0.10 or entropy > 1.95:
                st = "UNCERTAIN"
            elif p1 < 0.65 or margin < 0.25:
                st = "MARGINAL"
            else:
                st = "CONFIDENT"

            val_true.append(y_train[f_val][i])
            val_pred.append(preds[i])
            val_max_prob.append(p1)
            val_margin.append(margin)
            val_entropy.append(entropy)
            val_state.append(st)

    vdf = pd.DataFrame({
        "true": val_true,
        "pred": val_pred,
        "max_prob": val_max_prob,
        "margin": val_margin,
        "entropy": val_entropy,
        "state": val_state,
        "correct": np.array(val_true) == np.array(val_pred),
    })

    print("\n================================================================================")
    print("1. VALIDATION UNCERTAINTY BREAKDOWN (375 Out-of-Fold Samples, 0 Observer Overlap)")
    print("================================================================================")
    print(f"Overall Validation Accuracy: {vdf['correct'].mean():.2%}")
    print("\nBy Predicted Uncertainty State:")
    for state_name in ["CONFIDENT", "MARGINAL", "UNCERTAIN"]:
        sub = vdf[vdf["state"] == state_name]
        count = len(sub)
        if count > 0:
            acc = sub["correct"].mean()
            err = 1.0 - acc
            print(f"State: {state_name:<10} | Count: {count:>3} ({count/len(vdf):.1%}) | Accuracy: {acc:.2%} | Error Rate: {err:.2%}")

    # Evaluate on held-out test set (125 unseen samples)
    full_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", HistGradientBoostingClassifier(random_state=42)),
    ])
    full_pipe.fit(X_train, y_train)

    test_probs = full_pipe.predict_proba(X_test)
    test_preds = full_pipe.predict(X_test)

    test_states = []
    test_margins = []
    test_entropies = []
    for i in range(len(test_preds)):
        p_row = test_probs[i]
        sorted_p = np.sort(p_row)[::-1]
        p1, p2 = sorted_p[0], sorted_p[1]
        margin = p1 - p2
        entropy = -np.sum(p_row * np.log2(np.maximum(p_row, 1e-9)))

        if p1 < 0.40 or margin < 0.10 or entropy > 1.95:
            st = "UNCERTAIN"
        elif p1 < 0.65 or margin < 0.25:
            st = "MARGINAL"
        else:
            st = "CONFIDENT"
        test_states.append(st)
        test_margins.append(margin)
        test_entropies.append(entropy)

    tdf = pd.DataFrame({
        "true": y_test,
        "pred": test_preds,
        "correct": y_test == test_preds,
        "state": test_states,
        "prob": test_probs.max(axis=1),
        "margin": test_margins,
        "entropy": test_entropies,
    })

    print("\n================================================================================")
    print("2. HELD-OUT TEST UNCERTAINTY BREAKDOWN (125 Unseen Samples, 0 Observer Overlap)")
    print("================================================================================")
    print(f"Overall Test Accuracy: {tdf['correct'].mean():.2%}")
    print("\nBy Predicted Uncertainty State:")
    for state_name in ["CONFIDENT", "MARGINAL", "UNCERTAIN"]:
        sub = tdf[tdf["state"] == state_name]
        count = len(sub)
        if count > 0:
            acc = sub["correct"].mean()
            err = 1.0 - acc
            print(f"State: {state_name:<10} | Count: {count:>3} ({count/len(tdf):.1%}) | Accuracy: {acc:.2%} | Error Rate: {err:.2%}")

    # 3. Test out-of-distribution patterns using service
    print("\n================================================================================")
    print("3. OUT-OF-DISTRIBUTION (OOD) NON-FISH AND ADVERSARIAL PATTERN STRESS TEST")
    print("================================================================================")
    service = SpeciesIdentificationService()

    ood_cases = {
        "flat_neutral_gray": Image.new("RGB", (128, 128), color=(128, 128, 128)),
        "uniform_water_blue": Image.new("RGB", (128, 128), color=(30, 80, 160)),
        "pure_black": Image.new("RGB", (128, 128), color=(0, 0, 0)),
        "pure_white": Image.new("RGB", (128, 128), color=(255, 255, 255)),
        "random_noise": Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)),
    }

    for name, img in ood_cases.items():
        res = service.predict(img)
        print(f"Pattern: {name:<20} -> Pred: {res.species:<10} | Conf: {res.confidence:.1%} | State: {res.uncertainty_status:<10} | Reliable: {res.is_reliable}")
        if res.warning:
            print(f"   Warning: {res.warning}")

    print("\nCRITICAL OBSERVATION: Machine learning models without open-world rejection mechanisms")
    print("force any image into the 5 supported classes. Predictions must carry visible caveats.")


if __name__ == "__main__":
    main()
