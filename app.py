import streamlit as st
import os

# Import modul dan komponen
from utils.model_loader import load_model
from component.prediction_section import render_prediction_section
from component.evaluation_section import render_evaluation_section
from component.advice_section import render_advice_section

# Konfigurasi Halaman
st.set_page_config(
    page_title="Deteksi Kulit Melanoma dan Psoriasis",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Judul Aplikasi
st.title("Aplikasi Deteksi Gambar Kulit untuk Melanoma dan Psoriasis")

# Inisialisasi Session State
if "user_info" not in st.session_state:
    st.session_state.user_info = {}

if "latest_image" not in st.session_state:
    st.session_state.latest_image = None
    st.session_state.latest_y_pred = None
    st.session_state.latest_y_true = None
    st.session_state.latest_confidence = None

if "history" not in st.session_state:
    st.session_state.history = []

if "navigate_to_evaluation" not in st.session_state:
    st.session_state.navigate_to_evaluation = False

# Load Model Sekali (Cache)
try:
    base_dir = os.path.dirname(__file__)
except NameError:
    base_dir = os.getcwd()

MODEL_PATH = os.path.join(base_dir, "models", "best_vgg16_model.h5")

if not os.path.exists(MODEL_PATH):
    st.error(f"File model tidak ditemukan di path: `{MODEL_PATH}`.\n\n"
             f"Pastikan file `best_vgg16_model.h5` ada di folder `models/`.")
    st.stop()

try:
    with st.spinner("Memuat model..."):
        model = load_model(MODEL_PATH)
except Exception as e:
    st.error("Gagal memuat model.")
    st.exception(e)
    st.stop()

# Sidebar Navigasi
menu = st.sidebar.radio(
    "Navigasi",
    [
        "Panduan & Identitas",
        "Prediksi Gambar",
        "Evaluasi Model",
        "Penjelasan Medis"
    ]
)

# Navigasi otomatis dari tombol di halaman prediksi
if st.session_state.navigate_to_evaluation:
    menu = "Evaluasi Model"
    st.session_state.navigate_to_evaluation = False

# Routing Berdasarkan Menu
if menu == "Panduan & Identitas":
    st.header("Panduan Penggunaan Aplikasi")
    st.markdown("""
Aplikasi ini dirancang untuk membantu klasifikasi gambar kulit menjadi **Melanoma** atau **Psoriasis** menggunakan model CNN (**Convolutional Neural Network**).

---

### Cara Menggunakan Aplikasi:

1. **Isi Identitas Pengguna**  
   Masukkan nama, usia, dan jenis kelamin terlebih dahulu agar dapat melanjutkan ke menu prediksi.

2. **Prediksi Gambar Kulit**  
   - Masuk ke menu **Prediksi Gambar**
   - Unggah gambar kulit dengan format `.jpg`, `.jpeg`, atau `.png`

   **Ketentuan Gambar yang Disarankan:**
   - Ukuran maksimal file: 5 MB
   - Resolusi ideal: 224 x 224 piksel  
     (Jika gambar beresolusi lebih besar, akan diubah otomatis oleh sistem)
   - Panduan pengambilan gambar:
     - Ambil gambar dari jarak dekat dengan fokus pada area kulit yang mengalami kelainan atau lesi
     - Pastikan gambar tidak buram (tidak blur)
     - Gunakan pencahayaan yang cukup dan merata (hindari pencahayaan yang terlalu gelap atau terlalu terang)
     - Hindari latar belakang yang mengganggu atau objek lain seperti jari tangan lain, perban, kain, atau benda asing lainnya

   - Aplikasi akan menampilkan:
     - Hasil prediksi kelas
     - Persentase keyakinan model
     - Visualisasi Grad-CAM
     - Distribusi probabilitas antar kelas
   - Setelah prediksi, silakan isi label sebenarnya untuk evaluasi model

3. **Evaluasi Model**  
   - Menampilkan performa model berdasarkan seluruh histori prediksi pengguna
   - Termasuk:
     - Akurasi
     - Confusion Matrix
     - Classification Report
   - Histori tersimpan otomatis setiap kali pengguna melakukan prediksi

4. **Penjelasan Medis**  
   Menyediakan informasi medis terkait Melanoma dan Psoriasis sebagai edukasi tambahan.

---

**Catatan Penting:**
- Aplikasi ini **bukan pengganti diagnosis medis**
- Gunakan sebagai referensi awal dan selalu konsultasikan dengan dokter spesialis kulit
""")

    st.subheader("Formulir Identitas Pengguna")
    with st.form("user_info_form"):
        name = st.text_input("Nama Lengkap")
        age = st.number_input("Usia", min_value=0, max_value=120, value=0)
        gender = st.selectbox("Jenis Kelamin", ["", "Laki-laki", "Perempuan"], index=0)
        submitted = st.form_submit_button("Simpan")

        if submitted:
            if not name.strip():
                st.error("Nama tidak boleh kosong.")
            elif gender not in ["Laki-laki", "Perempuan"]:
                st.error("Silakan pilih jenis kelamin.")
            elif age == 0:
                st.error("Silakan isi usia Anda terlebih dahulu.")
            else:
                st.session_state.user_info = {
                    "nama": name,
                    "usia": age,
                    "jenis_kelamin": gender
                }
                st.success(
                    f"Data tersimpan. Halo, **{name}** ({gender}, {age} tahun). "
                    "Silakan lanjut ke menu *Prediksi Gambar*."
                )

elif menu == "Prediksi Gambar":
    if not st.session_state.user_info:
        st.warning("Harap isi identitas terlebih dahulu di menu 'Panduan & Identitas'.")
    else:
        render_prediction_section(model, st.session_state.user_info)

elif menu == "Evaluasi Model":
    if not st.session_state.history:
        st.warning("Anda belum melakukan prediksi gambar apa pun. Silakan lakukan minimal satu prediksi terlebih dahulu.")
    else:
        render_evaluation_section()

elif menu == "Penjelasan Medis":
    render_advice_section()