# Aplikasi Deteksi Kulit: Melanoma vs Psoriasis

Web app ini menggunakan model deep learning **VGG16** untuk mengklasifikasikan gambar kulit menjadi **Melanoma** atau **Psoriasis**.  
Dilengkapi dengan visualisasi **Grad-CAM**, evaluasi performa model, serta halaman edukasi medis interaktif untuk pengguna.

---

## Coba Demo Langsung

[Buka Aplikasi di Streamlit Cloud](https://melanosis-app-ikyy.streamlit.app/)

---

## Fitur Utama

- Upload gambar kulit langsung dari antarmuka web
- Prediksi otomatis dengan output Melanoma / Psoriasis
- Visualisasi **Grad-CAM** untuk area penting yang diperhatikan model
- Logging histori prediksi (tersimpan otomatis)
- Unduh laporan hasil prediksi (.txt)
- Evaluasi model: akurasi, confusion matrix, classification report
- Edukasi visual interaktif tentang jenis penyakit kulit

---

## Struktur Proyek

```text
skin-detector-app/
├── app.py                       # File utama Streamlit
├── requirements.txt            # Daftar dependensi Python
├── evaluation_model_summary.py # Halaman evaluasi model (opsional)
├── models/                     # Model CNN terlatih (.h5)
├── component/                  # Komponen UI modular (prediksi, advice, evaluasi)
│   ├── prediction_section.py
│   ├── advice_section.py
│   └── evaluation_section.py
├── utils/                      # Preprocessing, Grad-CAM, logging
├── static/logs/                # Visualisasi hasil training (loss/accuracy PNG)
├── logs/                       # Histori prediksi pengguna (JSON, CSV)
├── reports/                    # Laporan hasil prediksi (.txt)
├── data_testing/               # Dataset pengujian (gambar Melanoma & Psoriasis)
├── external_test/              # Gambar hasil prediksi user disimpan otomatis
├── outputs/                    # Log tambahan & file prediksi
├── .streamlit/config.toml      # Tema UI kustom (warna latar dan font)
└── venv/                       # Virtual environment (tidak di-push ke GitHub)
```

---

## Instalasi Lokal

Ikuti langkah-langkah berikut untuk menjalankan aplikasi secara lokal:

```bash
```
# 1. Clone Repository
git clone https://github.com/donirijki/skin-detector-app.git
cd skin-detector-app
```

```
# 2. Buat Virtual Environment
python -m venv venv
```

```
# 3. Aktifkan Virtual Environment
# (Windows)
.\venv\Scripts\activate
```
```
# (macOS/Linux)
source venv/bin/activate
```

```
# 4. Install Dependencies
pip install -r requirements.txt
```

```
# 5. Jalankan Aplikasi
streamlit run app.py
```

```
# 6. Akses Aplikasi
# Buka browser dan kunjungi:
http://localhost:8501