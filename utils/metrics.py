# utils/metrics.py
from sklearn.metrics import (
    accuracy_score, f1_score, recall_score, precision_score,
    roc_auc_score, confusion_matrix, classification_report
)
import numpy as np

def compute_global_score(y_true, y_pred, y_proba) -> float:
    """
    Score pondéré utilisé dans ton notebook original (très bon pour la détection de bugs)
    """
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_proba)

    return 0.45 * f1 + 0.30 * rec + 0.20 * auc + 0.05 * acc

def print_full_evaluation(y_true, y_pred, y_proba, model_name="Modèle"):
    print(f"\n=== Évaluation complète : {model_name} ===")
    print(f"Accuracy   : {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision  : {precision_score(y_true, y_pred):.4f}")
    print(f"Recall     : {recall_score(y_true, y_pred):.4f}")
    print(f"F1-Score   : {f1_score(y_true, y_pred):.4f}")
    print(f"ROC-AUC    : {roc_auc_score(y_true, y_proba):.4f}")
    print(f"Global Score : {compute_global_score(y_true, y_pred, y_proba):.4f}")
    print("\nMatrice de confusion :")
    print(confusion_matrix(y_true, y_pred))
    print("\nClassification Report :")
    print(classification_report(y_true, y_pred))