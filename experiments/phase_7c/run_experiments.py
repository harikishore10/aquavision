"""Phase 7C: Targeted Carp Improvement Experiments.

Compares:
  Exp A: Baseline (Frozen EfficientNet-B0 + LogisticRegression)
  Exp B: Training-only augmentation (HorizontalFlip + ColorJitter + Scale)
  Exp C1: Class-weighted LogisticRegression (balanced)
  Exp C2: Custom cost-sensitive Class-weighted LogisticRegression (carp emphasis)
  Exp D: Fine-tuning upper EfficientNet-B0 layers (features.7, features.8 + Head)

Strict Protocol:
  - 5-Fold StratifiedGroupKFold on 375 training pool (0 observer overlap).
  - 125 held-out test images strictly untouched.
  - Preprocessing and classification fitted independently inside each fold.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torchvision import models, transforms

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROJECT_SRC = PROJECT_ROOT / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from fishfarm_ai.transfer_model import TransferLearningClassifier


def get_training_data() -> tuple[list[Path], np.ndarray, np.ndarray, list[str]]:
    meta_path = PROJECT_ROOT / "data" / "species_images" / "metadata.csv"
    eval_rep_path = PROJECT_ROOT / "models" / "evaluation_report.json"
    meta = pd.read_csv(meta_path)
    with open(eval_rep_path, "r", encoding="utf-8") as f:
        rep = json.load(f)
    test_files = set(rep["species_model"]["metrics"]["partition_manifest"]["test_files"])
    train_df = meta[~meta["filename"].isin(test_files)].reset_index(drop=True)

    paths = [PROJECT_ROOT / "data" / "species_images" / r["species"] / r["filename"] for _, r in train_df.iterrows()]
    y = np.array(train_df["species"])
    g = np.array(train_df["recorded_by"])
    classes = sorted(list(set(y)))
    return paths, y, g, classes


# -----------------------------------------------------------------------------
# EXPERIMENT A: Baseline Frozen Features + Standard LogisticRegression
# -----------------------------------------------------------------------------
def run_exp_a(
    X_base: np.ndarray,
    y: np.ndarray,
    g: np.ndarray,
    cv_splits: list[tuple[np.ndarray, np.ndarray]],
    classes: list[str],
) -> dict:
    print("\n--- Running Experiment A: Baseline Frozen EfficientNet-B0 ---")
    fold_accs, fold_f1s = [], []
    fold_carp_f1s, fold_carp_recalls = [], []
    oof_preds = np.empty(len(y), dtype=object)
    t0 = time.perf_counter()

    for fold, (tr_idx, val_idx) in enumerate(cv_splits):
        scaler = StandardScaler()
        clf = LogisticRegression(max_iter=1000, random_state=42)
        X_tr = scaler.fit_transform(X_base[tr_idx])
        clf.fit(X_tr, y[tr_idx])
        X_val = scaler.transform(X_base[val_idx])
        preds = clf.predict(X_val)
        oof_preds[val_idx] = preds

        acc = float(accuracy_score(y[val_idx], preds))
        f1 = float(f1_score(y[val_idx], preds, average="macro", zero_division=0))
        rep = classification_report(y[val_idx], preds, output_dict=True, zero_division=0)
        fold_accs.append(acc)
        fold_f1s.append(f1)
        fold_carp_f1s.append(float(rep.get("carp", {}).get("f1-score", 0.0)))
        fold_carp_recalls.append(float(rep.get("carp", {}).get("recall", 0.0)))

    fit_time = time.perf_counter() - t0
    report = classification_report(y, oof_preds, target_names=classes, output_dict=True, zero_division=0)
    cm = confusion_matrix(y, oof_preds, labels=classes).tolist()

    return {
        "name": "Exp A: Baseline Frozen Features",
        "mean_accuracy": float(np.mean(fold_accs)),
        "std_accuracy": float(np.std(fold_accs)),
        "mean_macro_f1": float(np.mean(fold_f1s)),
        "std_macro_f1": float(np.std(fold_f1s)),
        "mean_carp_f1": float(np.mean(fold_carp_f1s)),
        "std_carp_f1": float(np.std(fold_carp_f1s)),
        "mean_carp_recall": float(np.mean(fold_carp_recalls)),
        "total_time_sec": fit_time,
        "classification_report": report,
        "confusion_matrix": cm,
    }


# -----------------------------------------------------------------------------
# EXPERIMENT B: Training-only Augmentation
# -----------------------------------------------------------------------------
def run_exp_b(
    paths: list[Path],
    y: np.ndarray,
    cv_splits: list[tuple[np.ndarray, np.ndarray]],
    classes: list[str],
) -> dict:
    print("\n--- Running Experiment B: Training-Only Augmentation ---")
    base_clf = TransferLearningClassifier(backbone_name="efficientnet_b0", random_state=42)

    # Mild augmentation preserving species anatomy
    train_aug = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.RandomAffine(degrees=10, scale=(0.95, 1.05)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    def extract_with_transform(img: Image.Image, tf) -> np.ndarray:
        tensor = tf(img.convert("RGB")).unsqueeze(0)
        with torch.no_grad():
            feat = base_clf._extractor(tensor).squeeze(0).cpu().numpy()
        return feat.astype(np.float32)

    fold_accs, fold_f1s = [], []
    fold_carp_f1s, fold_carp_recalls = [], []
    oof_preds = np.empty(len(y), dtype=object)
    t0 = time.perf_counter()

    # Pre-extract clean features for validation
    clean_feats = np.array([extract_with_transform(Image.open(p), val_tf) for p in paths])

    for fold, (tr_idx, val_idx) in enumerate(cv_splits):
        # In training fold, generate clean + augmented features (2x training data)
        tr_feats_clean = clean_feats[tr_idx]
        tr_feats_aug = np.array([extract_with_transform(Image.open(paths[i]), train_aug) for i in tr_idx])
        X_tr = np.vstack([tr_feats_clean, tr_feats_aug])
        y_tr = np.concatenate([y[tr_idx], y[tr_idx]])

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(X_tr_s, y_tr)

        X_val_s = scaler.transform(clean_feats[val_idx])
        preds = clf.predict(X_val_s)
        oof_preds[val_idx] = preds

        acc = float(accuracy_score(y[val_idx], preds))
        f1 = float(f1_score(y[val_idx], preds, average="macro", zero_division=0))
        rep = classification_report(y[val_idx], preds, output_dict=True, zero_division=0)
        fold_accs.append(acc)
        fold_f1s.append(f1)
        fold_carp_f1s.append(float(rep.get("carp", {}).get("f1-score", 0.0)))
        fold_carp_recalls.append(float(rep.get("carp", {}).get("recall", 0.0)))

    fit_time = time.perf_counter() - t0
    report = classification_report(y, oof_preds, target_names=classes, output_dict=True, zero_division=0)
    cm = confusion_matrix(y, oof_preds, labels=classes).tolist()

    return {
        "name": "Exp B: Training-Only Augmentation",
        "mean_accuracy": float(np.mean(fold_accs)),
        "std_accuracy": float(np.std(fold_accs)),
        "mean_macro_f1": float(np.mean(fold_f1s)),
        "std_macro_f1": float(np.std(fold_f1s)),
        "mean_carp_f1": float(np.mean(fold_carp_f1s)),
        "std_carp_f1": float(np.std(fold_carp_f1s)),
        "mean_carp_recall": float(np.mean(fold_carp_recalls)),
        "total_time_sec": fit_time,
        "classification_report": report,
        "confusion_matrix": cm,
    }


# -----------------------------------------------------------------------------
# EXPERIMENT C: Class-Weighted Logistic Regression
# -----------------------------------------------------------------------------
def run_exp_c(
    X_base: np.ndarray,
    y: np.ndarray,
    cv_splits: list[tuple[np.ndarray, np.ndarray]],
    classes: list[str],
    weight_mode: str = "carp_emphasis",
) -> dict:
    print(f"\n--- Running Experiment C: Class-Weighted Logistic Regression ({weight_mode}) ---")
    custom_weights = {"carp": 1.6, "tilapia": 1.2, "catfish": 1.0, "milkfish": 1.0, "pangasius": 1.0} if weight_mode == "carp_emphasis" else "balanced"

    fold_accs, fold_f1s = [], []
    fold_carp_f1s, fold_carp_recalls = [], []
    oof_preds = np.empty(len(y), dtype=object)
    t0 = time.perf_counter()

    for fold, (tr_idx, val_idx) in enumerate(cv_splits):
        scaler = StandardScaler()
        clf = LogisticRegression(max_iter=1000, random_state=42, class_weight=custom_weights)
        X_tr = scaler.fit_transform(X_base[tr_idx])
        clf.fit(X_tr, y[tr_idx])
        X_val = scaler.transform(X_base[val_idx])
        preds = clf.predict(X_val)
        oof_preds[val_idx] = preds

        acc = float(accuracy_score(y[val_idx], preds))
        f1 = float(f1_score(y[val_idx], preds, average="macro", zero_division=0))
        rep = classification_report(y[val_idx], preds, output_dict=True, zero_division=0)
        fold_accs.append(acc)
        fold_f1s.append(f1)
        fold_carp_f1s.append(float(rep.get("carp", {}).get("f1-score", 0.0)))
        fold_carp_recalls.append(float(rep.get("carp", {}).get("recall", 0.0)))

    fit_time = time.perf_counter() - t0
    report = classification_report(y, oof_preds, target_names=classes, output_dict=True, zero_division=0)
    cm = confusion_matrix(y, oof_preds, labels=classes).tolist()

    return {
        "name": f"Exp C: Class-Weighted ({weight_mode})",
        "mean_accuracy": float(np.mean(fold_accs)),
        "std_accuracy": float(np.std(fold_accs)),
        "mean_macro_f1": float(np.mean(fold_f1s)),
        "std_macro_f1": float(np.std(fold_f1s)),
        "mean_carp_f1": float(np.mean(fold_carp_f1s)),
        "std_carp_f1": float(np.std(fold_carp_f1s)),
        "mean_carp_recall": float(np.mean(fold_carp_recalls)),
        "total_time_sec": fit_time,
        "classification_report": report,
        "confusion_matrix": cm,
    }


# -----------------------------------------------------------------------------
# EXPERIMENT D: Fine-Tuning Upper EfficientNet-B0 Layers
# -----------------------------------------------------------------------------
def run_exp_d(
    paths: list[Path],
    y: np.ndarray,
    cv_splits: list[tuple[np.ndarray, np.ndarray]],
    classes: list[str],
) -> dict:
    print("\n--- Running Experiment D: Fine-Tuning Upper EfficientNet-B0 Layers (features.7, features.8) ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    val_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    class CustomDataset(torch.utils.data.Dataset):
        def __init__(self, img_paths: list[Path], labels_arr: np.ndarray, tf):
            self.img_paths = img_paths
            self.labels_arr = labels_arr
            self.tf = tf

        def __len__(self):
            return len(self.img_paths)

        def __getitem__(self, idx):
            img = Image.open(self.img_paths[idx]).convert("RGB")
            tensor = self.tf(img)
            label_idx = classes.index(self.labels_arr[idx])
            return tensor, label_idx

    fold_accs, fold_f1s = [], []
    fold_carp_f1s, fold_carp_recalls = [], []
    oof_preds = np.empty(len(y), dtype=object)
    t0 = time.perf_counter()

    for fold, (tr_idx, val_idx) in enumerate(cv_splits):
        # Build model with frozen bottom and unchilled top
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        # Freeze stages 0-6
        for param in model.features[:7].parameters():
            param.requires_grad = False
        # Unfreeze stages 7 and 8
        for param in model.features[7:].parameters():
            param.requires_grad = True

        num_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_features, len(classes))
        model.to(device)

        tr_ds = CustomDataset([paths[i] for i in tr_idx], y[tr_idx], val_tf)
        val_ds = CustomDataset([paths[i] for i in val_idx], y[val_idx], val_tf)
        tr_loader = torch.utils.data.DataLoader(tr_ds, batch_size=16, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_ds, batch_size=16, shuffle=False)

        # Differential learning rate: small on backbone, slightly higher on linear head
        optimizer = torch.optim.AdamW([
            {"params": model.features[7:].parameters(), "lr": 2e-5},
            {"params": model.classifier.parameters(), "lr": 2e-4},
        ], weight_decay=1e-2)
        criterion = nn.CrossEntropyLoss()

        model.train()
        for epoch in range(5):  # 5 epochs to prevent severe overfitting on 300 samples
            for images_b, targets_b in tr_loader:
                images_b, targets_b = images_b.to(device), targets_b.to(device)
                optimizer.zero_grad()
                outputs = model(images_b)
                loss = criterion(outputs, targets_b)
                loss.backward()
                optimizer.step()

        # Validation
        model.eval()
        val_preds_list = []
        with torch.no_grad():
            for images_b, _ in val_loader:
                images_b = images_b.to(device)
                out = model(images_b)
                preds_b = out.argmax(dim=1).cpu().numpy()
                val_preds_list.extend([classes[p] for p in preds_b])

        preds_arr = np.array(val_preds_list)
        oof_preds[val_idx] = preds_arr

        acc = float(accuracy_score(y[val_idx], preds_arr))
        f1 = float(f1_score(y[val_idx], preds_arr, average="macro", zero_division=0))
        rep = classification_report(y[val_idx], preds_arr, output_dict=True, zero_division=0)
        fold_accs.append(acc)
        fold_f1s.append(f1)
        fold_carp_f1s.append(float(rep.get("carp", {}).get("f1-score", 0.0)))
        fold_carp_recalls.append(float(rep.get("carp", {}).get("recall", 0.0)))
        print(f"  Fold {fold+1}/5 -> Acc: {acc:.3f} | Macro F1: {f1:.3f} | Carp F1: {rep.get('carp', {}).get('f1-score', 0.0):.3f}")

    fit_time = time.perf_counter() - t0
    report = classification_report(y, oof_preds, target_names=classes, output_dict=True, zero_division=0)
    cm = confusion_matrix(y, oof_preds, labels=classes).tolist()

    return {
        "name": "Exp D: Fine-Tuning Upper Layers (features.7, features.8)",
        "mean_accuracy": float(np.mean(fold_accs)),
        "std_accuracy": float(np.std(fold_accs)),
        "mean_macro_f1": float(np.mean(fold_f1s)),
        "std_macro_f1": float(np.std(fold_f1s)),
        "mean_carp_f1": float(np.mean(fold_carp_f1s)),
        "std_carp_f1": float(np.std(fold_carp_f1s)),
        "mean_carp_recall": float(np.mean(fold_carp_recalls)),
        "total_time_sec": fit_time,
        "classification_report": report,
        "confusion_matrix": cm,
    }


def main() -> None:
    paths, y, g, classes = get_training_data()
    print(f"Data: {len(paths)} images across {len(set(g))} observers.")

    # 1. Base frozen feature extraction
    clf_base = TransferLearningClassifier(backbone_name="efficientnet_b0", random_state=42)
    X_base = np.array([clf_base.extract_features(Image.open(p)) for p in paths])

    # 2. Same exact 5-fold StratifiedGroupKFold
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    cv_splits = list(sgkf.split(X_base, y, g))

    # Run experiments
    results = {}
    results["exp_a_baseline"] = run_exp_a(X_base, y, g, cv_splits, classes)
    results["exp_b_augmentation"] = run_exp_b(paths, y, cv_splits, classes)
    results["exp_c_class_weighted"] = run_exp_c(X_base, y, cv_splits, classes, weight_mode="carp_emphasis")
    results["exp_d_finetuning"] = run_exp_d(paths, y, cv_splits, classes)

    # Summary table
    print("\n" + "=" * 105)
    print(f"{'Experiment':<40} | {'CV Accuracy':<15} | {'CV Macro F1':<15} | {'Carp F1':<10} | {'Carp Recall':<12} | {'Time (s)'}")
    print("=" * 105)
    for k, res in results.items():
        name = res["name"]
        acc_str = f"{res['mean_accuracy']:.3f} ± {res['std_accuracy']:.3f}"
        f1_str = f"{res['mean_macro_f1']:.3f} ± {res['std_macro_f1']:.3f}"
        carp_f1_str = f"{res['mean_carp_f1']:.3f} ± {res['std_carp_f1']:.3f}"
        carp_rec_str = f"{res['mean_carp_recall']:.3f}"
        time_str = f"{res['total_time_sec']:.1f}s"
        print(f"{name:<40} | {acc_str:<15} | {f1_str:<15} | {carp_f1_str:<10} | {carp_rec_str:<12} | {time_str}")
    print("=" * 105)

    # Save to JSON
    out_file = PROJECT_ROOT / "experiments" / "phase_7c" / "experiment_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[OK] Experiment results saved to {out_file}")


if __name__ == "__main__":
    main()
