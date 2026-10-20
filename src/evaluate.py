"""
Evaluation helpers. Macro-F1 is the primary metric fixed in Milestone 2 (section 10);
accuracy, per-class recall and the confusion matrix are secondary evidence.
"""
import matplotlib.pyplot as plt
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, f1_score,
                             precision_recall_fscore_support)

from src.data import CLASS_ORDER


def evaluate(t, y):
    """
    Parameters:
        `t` - true labels (0 = Dropout, 1 = Enrolled, 2 = Graduate)
        `y` - predicted labels
    Returns: a dict with macro_f1, accuracy, and per-class precision/recall/F1,
             e.g. out["recall_Dropout"]
    """
    out = {
        "macro_f1": f1_score(t, y, average="macro", labels=[0, 1, 2], zero_division=0),
        "accuracy": accuracy_score(t, y),
    }
    precision, recall, f1, _ = precision_recall_fscore_support(
        t, y, labels=[0, 1, 2], zero_division=0)
    for i, name in enumerate(CLASS_ORDER):
        out[f"precision_{name}"] = precision[i]
        out[f"recall_{name}"] = recall[i]
        out[f"f1_{name}"] = f1[i]
    return out


def plot_confusion(t, y, title, ax=None):
    """
    Parameters:
        `t` - true labels
        `y` - predicted labels
        `title` - plot title
        `ax` - optional matplotlib axis
    Returns: the matplotlib axis (rows = true class, columns = predicted class)
    """
    disp = ConfusionMatrixDisplay.from_predictions(
        t, y, labels=[0, 1, 2], display_labels=CLASS_ORDER, cmap="Blues", colorbar=False, ax=ax)
    disp.ax_.set_title(title)
    return disp.ax_
