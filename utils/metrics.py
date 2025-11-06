from typing import List, Dict, Tuple
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve, precision_recall_curve, auc
from .paths import ensure_dir, ID_TO_PRETTY


def save_confusion_matrix(y_true, y_pred, save_path: str,
                          class_names=None, order=None) -> None:
    ensure_dir(os.path.dirname(save_path))

    # บังคับลำดับคลาสตาม order ถ้ามี (order คือ index ของคลาสใน y_true/y_pred)
    if order is not None:
        cm = confusion_matrix(y_true, y_pred, labels=order)
    else:
        cm = confusion_matrix(y_true, y_pred)

    if class_names is None:
        labels = [ID_TO_PRETTY[i] for i in range(len(ID_TO_PRETTY))]
    else:
        labels = class_names

    plt.figure(figsize=(6, 5))
    ax = sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                     xticklabels=labels, yticklabels=labels)
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.title("Confusion Matrix")
    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()



def save_classification_report(y_true: List[int], y_pred: List[int], save_txt: str, save_csv: str) -> None:
    ensure_dir(os.path.dirname(save_txt))
    report_dict = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    with open(save_txt, "w", encoding="utf-8") as f:
        f.write(classification_report(y_true, y_pred, zero_division=0))
    df = pd.DataFrame(report_dict).T
    df.to_csv(save_csv, index=True, encoding="utf-8-sig")


def save_roc_pr_curves(y_true: np.ndarray, y_prob: np.ndarray, save_dir: str, class_names: List[str]) -> Dict[str, float]:
    ensure_dir(save_dir)
    # y_true: shape (N,), y_prob: shape (N, C)
    num_classes = y_prob.shape[1]
    y_true_onehot = np.eye(num_classes)[y_true]

    roc_aucs: Dict[str, float] = {}
    pr_aucs: Dict[str, float] = {}

    # ROC curves
    plt.figure(figsize=(6, 5))
    for c in range(num_classes):
        fpr, tpr, _ = roc_curve(y_true_onehot[:, c], y_prob[:, c])
        roc_auc = auc(fpr, tpr)
        roc_aucs[class_names[c]] = roc_auc
        plt.plot(fpr, tpr, label=f"{class_names[c]} (AUC={roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "roc_curves.png"))
    plt.close()

    # PR curves
    plt.figure(figsize=(6, 5))
    for c in range(num_classes):
        precision, recall, _ = precision_recall_curve(y_true_onehot[:, c], y_prob[:, c])
        pr_auc = auc(recall, precision)
        pr_aucs[class_names[c]] = pr_auc
        plt.plot(recall, precision, label=f"{class_names[c]} (AUC={pr_auc:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "pr_curves.png"))
    plt.close()

    # Macro scores
    macro_roc = np.mean(list(roc_aucs.values())) if roc_aucs else float("nan")
    macro_pr = np.mean(list(pr_aucs.values())) if pr_aucs else float("nan")
    with open(os.path.join(save_dir, "auc_summary.txt"), "w", encoding="utf-8") as f:
        f.write(f"Macro ROC-AUC: {macro_roc:.4f}\n")
        f.write(f"Macro PR-AUC: {macro_pr:.4f}\n")

    return {"macro_roc_auc": macro_roc, "macro_pr_auc": macro_pr}


def save_training_curves(history: Dict[str, List[float]], save_dir: str) -> None:
    ensure_dir(save_dir)
    # history with keys: train_loss, val_loss, train_acc, val_acc
    plt.figure(figsize=(6, 5))
    plt.plot(history.get("train_loss", []), label="Train Loss")
    plt.plot(history.get("val_loss", []), label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training/Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "loss_curve.png"))
    plt.close()

    plt.figure(figsize=(6, 5))
    plt.plot(history.get("train_acc", []), label="Train Acc")
    plt.plot(history.get("val_acc", []), label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training/Validation Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "accuracy_curve.png"))
    plt.close()
