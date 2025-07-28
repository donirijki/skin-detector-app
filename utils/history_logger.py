import os
import json
import csv

# Lokasi penyimpanan
HISTORY_JSON_FILE = "logs/prediction_history.json"
HISTORY_CSV_FILE = "logs/prediction_log.csv"

# Header untuk file CSV
HISTORY_FIELDNAMES = ["image_name", "predicted_label", "confidence", "true_label"]

def load_history():
    """
    Memuat histori prediksi dari file JSON jika tersedia.
    """
    if os.path.exists(HISTORY_JSON_FILE):
        with open(HISTORY_JSON_FILE, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return []
    return []

def save_history_to_json_and_csv(history):
    """
    Menyimpan histori ke dalam file JSON dan CSV.
    """
    os.makedirs(os.path.dirname(HISTORY_JSON_FILE), exist_ok=True)

    # Simpan ke file JSON
    with open(HISTORY_JSON_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

    # Simpan ke file CSV
    with open(HISTORY_CSV_FILE, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=HISTORY_FIELDNAMES)
        writer.writeheader()
        for row in history:
            writer.writerow({
                "image_name": row.get("image_name", ""),
                "predicted_label": row.get("predicted_label", ""),
                "confidence": row.get("confidence", ""),
                "true_label": row.get("true_label", "")
            })

def log_to_history(image_path, predicted_label, confidence, true_label=None):
    """
    Menambahkan entri baru ke histori jika belum ada.
    """
    history = load_history()
    image_name = os.path.basename(image_path)

    # Cegah duplikat berdasarkan nama file
    if any(item["image_name"] == image_name for item in history):
        return

    entry = {
        "image_name": image_name,
        "predicted_label": predicted_label.strip().lower(),
        "confidence": float(confidence),
        "true_label": true_label.strip().lower() if true_label else ""
    }

    history.append(entry)
    save_history_to_json_and_csv(history)