import os
import json
import csv
import pandas as pd
from datetime import datetime

# Path log file
HISTORY_JSON_FILE = "logs/prediction_history.json"
HISTORY_CSV_FILE = "logs/prediction_log.csv"
USER_LOG_PATH = "outputs/predictions_log.csv"

# Log Prediksi Umum 

def load_history():
    """Memuat histori prediksi dari file JSON."""
    if os.path.exists(HISTORY_JSON_FILE):
        with open(HISTORY_JSON_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_history(history):
    """Menyimpan histori prediksi ke file JSON & CSV."""
    os.makedirs(os.path.dirname(HISTORY_JSON_FILE), exist_ok=True)

    # Simpan JSON
    with open(HISTORY_JSON_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

    # Simpan CSV
    with open(HISTORY_CSV_FILE, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "image_name", "predicted_label", "confidence"])
        writer.writeheader()
        for row in history:
            writer.writerow(row)

def log_to_history(image_path, predicted_label, confidence):
    """Menambahkan entri baru ke histori prediksi umum."""
    try:
        confidence = float(confidence)
        history = load_history()

        # Cek duplikat berdasarkan path gambar
        if any(item["image_path"] == image_path for item in history):
            return

        # Tambah entri baru
        history.append({
            "image_path": image_path,
            "image_name": os.path.basename(image_path),
            "predicted_label": predicted_label.strip().lower(),
            "confidence": confidence
        })

        save_history(history)
    except Exception as e:
        print(f"[ERROR] Gagal menyimpan ke history: {e}")

# Log Prediksi User 

def log_prediction(user_info, image_name, prediction, confidence):
    """Menyimpan log prediksi pengguna ke file CSV."""
    os.makedirs(os.path.dirname(USER_LOG_PATH), exist_ok=True)

    new_row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "nama": user_info.get("nama", ""),
        "usia": user_info.get("usia", ""),
        "jenis_kelamin": user_info.get("jenis_kelamin", ""),
        "nama_file": image_name,
        "prediksi": prediction,
        "confidence": confidence
    }

    try:
        df_new = pd.DataFrame([new_row])
        if not os.path.exists(USER_LOG_PATH):
            df_new.to_csv(USER_LOG_PATH, index=False)
        else:
            df_new.to_csv(USER_LOG_PATH, mode='a', header=False, index=False)
    except Exception as e:
        print(f"[ERROR] Gagal menyimpan log prediksi pengguna: {e}")