import os
import json
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

from utils.history_logger import load_history

CLASS_NAMES = ['melanoma', 'psoriasis']

def evaluate_user_predictions():
    history = load_history()

    if not history or len(history) == 0:
        print("[INFO] Tidak ada histori prediksi yang ditemukan.")
        return None, None, None, None

    # Filter hanya data yang punya predicted_label dan true_label (bukan kosong)
    filtered = [
        d for d in history
        if d.get("true_label", "").strip().lower() in CLASS_NAMES and
           d.get("predicted_label", "").strip().lower() in CLASS_NAMES
    ]

    if not filtered:
        print("[INFO] Tidak ada data valid yang memiliki true_label dan predicted_label.")
        return None, None, None, None

    # Ekstrak label ground truth dan prediksi
    y_true = [d["true_label"].strip().lower() for d in filtered]
    y_pred = [d["predicted_label"].strip().lower() for d in filtered]

    # Hitung akurasi dan classification report
    accuracy = accuracy_score(y_true, y_pred)
    report_dict = classification_report(
        y_true, y_pred,
        labels=CLASS_NAMES,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0  
    )
    report_df = pd.DataFrame(report_dict).transpose()

    # Buat Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=CLASS_NAMES)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax
    )
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Confusion Matrix")

    return accuracy, report_df, cm, fig