import streamlit as st
import os

# Import modul dan komponen
from utils.model_loader import load_model
from component.prediction_section import render_prediction_section
from evaluation_model_summary import render_evaluation_model_page
from component.advice_section import render_advice_section 

# Konfigurasi Halaman
st.set_page_config(
    page_title="Deteksi Kulit - Melanoma vs Psoriasis",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Judul Aplikasi
st.title("Aplikasi Deteksi Gambar Kulit: Melanoma vs Psoriasis")

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

# Routing Berdasarkan Menu
if menu == "Panduan & Identitas":
    st.header("Panduan Penggunaan Aplikasi")
    st.markdown("""
Aplikasi ini dirancang untuk membantu klasifikasi gambar kulit menjadi **Melanoma** atau **Psoriasis** menggunakan model CNN (**Convolutional Neural Network**).

### Cara Menggunakan Aplikasi:

1. **Isi Identitas Pengguna** 
   Harap isi terlebih dahulu! 
   Masukkan nama, usia, dan jenis kelamin pada formulir di bawah untuk keperluan dokumentasi dan log prediksi.

2. **Prediksi Gambar Kulit**  
   Masuk ke menu **Prediksi Gambar** untuk mengunggah foto kulit dalam format `.jpg`, `.jpeg`, atau `.png`.  
   Aplikasi akan memproses gambar menggunakan model CNN (VGG16) dan menampilkan:
   - Hasil prediksi (Melanoma atau Psoriasis)
   - Persentase keyakinan model
   - Visualisasi Grad-CAM

3. **Evaluasi Model**  
   Menu **Evaluasi Model** menampilkan ringkasan akurasi, confusion matrix, classification report, grafik training, serta ringkasan performa model saat pelatihan.

4. **Penjelasan Medis**  
   Buka menu **Penjelasan Medis** untuk mempelajari lebih lanjut mengenai penyakit Melanoma dan Psoriasis.

**Catatan Penting:**
- Aplikasi ini hanya alat bantu, bukan diagnosis medis.
- Tetap konsultasikan hasil dengan dokter spesialis.
""")

    st.subheader("Formulir Identitas Pengguna")
    with st.form("user_info_form"):
        name = st.text_input("Nama Lengkap")
        age = st.number_input("Usia", min_value=0, max_value=120, value=0)
        gender = st.selectbox("Jenis Kelamin", ["", "Laki-laki", "Perempuan", "Lainnya"], index=0)
        submitted = st.form_submit_button("Simpan")

        if submitted:
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
    render_evaluation_model_page()

elif menu == "Penjelasan Medis":
    render_advice_section()