import os
import pandas as pd
from datetime import datetime

USER_LOG_PATH = "outputs/predictions_log.csv"
USER_FIELDNAMES = ["timestamp", "nama", "usia", "jenis_kelamin", "nama_file", "prediksi", "confidence", "true_label"]

def log_prediction(user_info, image_name, prediction, confidence, true_label=None):
    os.makedirs(os.path.dirname(USER_LOG_PATH), exist_ok=True)

    # Normalisasi label prediksi dan label asli
    prediction = prediction.strip().lower()
    true_label = true_label.strip().lower() if true_label else ""

    new_row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "nama": user_info.get("nama", "").strip(),
        "usia": user_info.get("usia", ""),
        "jenis_kelamin": user_info.get("jenis_kelamin", "").strip(),
        "nama_file": image_name,
        "prediksi": prediction,
        "confidence": float(confidence),
        "true_label": true_label
    }

    try:
        if os.path.exists(USER_LOG_PATH):
            df = pd.read_csv(USER_LOG_PATH)

            # Hindari duplikasi: hanya log 1x per user + gambar
            is_duplicate = (
                (df["nama"] == new_row["nama"]) &
                (df["nama_file"] == new_row["nama_file"])
            ).any()

            if is_duplicate:
                print(f"[INFO] Duplikat ditemukan. Prediksi untuk {new_row['nama_file']} oleh {new_row['nama']} sudah ada.")
                return

            # Tambahkan baris baru
            df_new = pd.DataFrame([new_row])
            df_new.to_csv(USER_LOG_PATH, mode="a", header=False, index=False, columns=USER_FIELDNAMES)
        else:
            # File belum ada, tulis dengan header
            df_new = pd.DataFrame([new_row])
            df_new.to_csv(USER_LOG_PATH, index=False, columns=USER_FIELDNAMES)

    except Exception as e:
        print(f"[ERROR] Gagal menyimpan log prediksi pengguna: {e}")