"""Métricas binarias para clase positiva 1 y regla score >= umbral."""
import numpy as np
from sklearn.metrics import confusion_matrix


def threshold_metrics(y_true, y_score, threshold):
    """Devuelve matriz [[TN, FP], [FN, TP]], FPR y TPR.

    Las tasas requieren las dos clases en la muestra de evaluación.
    """
    truth = np.asarray(y_true)
    scores = np.asarray(y_score)
    if truth.ndim != 1 or scores.shape != truth.shape or not truth.size:
        raise ValueError("Se requieren vectores no vacíos de igual tamaño")
    if set(np.unique(truth)) != {0, 1}:
        raise ValueError("Se requieren ambas clases: 0 y 1")
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("El umbral debe estar entre 0 y 1")
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("Las probabilidades deben estar entre 0 y 1")
    cm = confusion_matrix(truth, (scores >= threshold).astype(int), labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return cm, fp / (fp + tn), tp / (tp + fn)
