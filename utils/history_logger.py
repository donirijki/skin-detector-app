import os
import json

HISTORY_FILE = "logs/prediction_history.json"

def load_history():
    """Memuat histori prediksi dari file JSON."""
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_history(history):
    """Menyimpan histori prediksi ke file JSON."""
    os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

def log_to_history(image_path, predicted_label, confidence):
    """Menambahkan satu entri baru ke histori prediksi (tanpa actual label)."""
    history = load_history()
    history.append({
        "image_path": image_path,
        "image_name": os.path.basename(image_path),
        "predicted_label": predicted_label,
        "confidence": confidence
    })
    save_history(history)