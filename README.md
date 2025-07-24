# Aplikasi Deteksi Kulit: Melanoma vs Psoriasis

Web app ini menggunakan model deep learning **VGG16** untuk mengklasifikasikan gambar kulit menjadi **Melanoma** atau **Psoriasis**.  
Dilengkapi visualisasi **Grad-CAM**, evaluasi performa model, dan penjelasan medis sebagai edukasi pengguna.

---

## Coba Demo Langsung
[Akses Aplikasi via Streamlit Cloud](https://melanosis-app-ikyy.streamlit.app/)

---

## Fitur Utama

- Upload gambar kulit
- Prediksi otomatis: Melanoma atau Psoriasis
- Visualisasi Grad-CAM: Area penting yang dipertimbangkan model
- Logging histori prediksi
- Unduh laporan hasil prediksi (CSV)
- Halaman evaluasi model: akurasi, confusion matrix, dan classification report
- Penjelasan penyakit secara medis

---

## Struktur Proyek
```
project_root/
├── app.py # File utama Streamlit
├── requirements.txt # Daftar dependensi Python
├── evaluation_model_summary.py # Halaman evaluasi model
├── models/ # Model terlatih (.h5)
├── component/ # Komponen UI (modular)
├── utils/ # Fungsi pendukung & preprocessing
├── static/logs/ # Visualisasi hasil pelatihan
├── logs/ # Histori prediksi JSON + CSV
├── reports/ # File laporan hasil prediksi
├── data_testing/ # Dataset pengujian lokal
├── external_test/ # Gambar prediksi user
├── outputs/ # Log tambahan prediksi
└── venv/ # Virtual environment (abaikan saat push)
```

---

## Instalasi Lokal

### Clone Repo

```bash
git clone https://github.com/donirjki/skin-detector-app.git
<<<<<<< HEAD
cd skin-detector-app
=======
cd skin-detector-app
>>>>>>> 8f26518 (Fix: rename requirements file + update for Streamlit Cloud)
