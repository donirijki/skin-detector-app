import os
from datetime import datetime

def write_prediction_report(user_info, image_name, predicted_label, confidence, output_dir="reports"):
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Nama file pakai nama pengguna
    file_name = f"laporan_prediksi_{user_info.get('nama', 'user').replace(' ', '_').lower()}.txt"
    output_path = os.path.join(output_dir, file_name)

    report = f"""LAPORAN PREDIKSI KULIT 

Tanggal/Waktu   : {timestamp}
Nama Pengguna   : {user_info.get('nama', '')}
Usia            : {user_info.get('usia', '')}
Jenis Kelamin   : {user_info.get('jenis_kelamin', '')}

Nama Gambar     : {image_name}
Prediksi Model  : {predicted_label.upper()}
Confidence      : {confidence * 100:.2f}%

Penjelasan Model:
Model mendeteksi gambar ini sebagai {predicted_label.upper()} dengan keyakinan {confidence * 100:.2f}%.
Silakan konsultasikan ke dokter kulit untuk konfirmasi lebih lanjut.

"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)

    return output_path