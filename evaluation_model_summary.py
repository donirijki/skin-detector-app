import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from PIL import Image

# Path ke folder logs
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static", "logs")

# Nama file
PLOT_PATH = os.path.join(STATIC_DIR, "vgg16_training_plot.png")
CM_PATH = os.path.join(STATIC_DIR, "vgg16_confusion_matrix.png")
CR_PATH = os.path.join(STATIC_DIR, "vgg16_classification_report.csv")

def render_evaluation_model_page():
    st.header("Evaluasi Model Deteksi Penyakit Kulit")

    st.markdown("""
    Halaman ini menyajikan hasil evaluasi model klasifikasi gambar kulit antara **Melanoma** dan **Psoriasis** 
    menggunakan arsitektur **VGG16** yang telah dilatih dan diuji di Google Colab.
    """)

    # Deskripsi Dataset
    st.subheader("Distribusi Dataset")
    st.markdown("""
    Dataset dibagi menjadi 3 subset:
    
    - **Train:** 60% (melanoma: 393 gambar, psoriasis: 394 gambar)  
    - **Validation:** 20% (melanoma: 131 gambar, psoriasis: 131 gambar)  
    - **Test:** 20% (melanoma: 132 gambar, psoriasis: 131 gambar)
    """)

    # Ringkasan Metrik
    st.subheader("Ringkasan Hasil Evaluasi")
    st.markdown("""
    - **Test Accuracy:** 98.10%  
    - **Test Loss:** 0.0700
    """)

    # Confusion Matrix detail
    st.markdown("""
    **Rincian Confusion Matrix (Biner):**

    - **True Positive  (TP):** 129  
    - **False Positive (FP):** 3   
    - **False Negative (FN):** 2  
    - **True Negative  (TN):** 129
    """)

    # Plot Training
    st.subheader("Training & Validation Accuracy/Loss")
    if os.path.exists(PLOT_PATH):
        st.image(PLOT_PATH, caption="Plot Akurasi dan Loss Selama Training", use_container_width=True)
    else:
        st.warning(f"Gambar training plot tidak ditemukan di: `{PLOT_PATH}`")

    # Confusion Matrix
    st.subheader("Confusion Matrix")
    if os.path.exists(CM_PATH):
        st.image(CM_PATH, caption="Confusion Matrix", use_container_width=True)
    else:
        st.warning(f"Gambar confusion matrix tidak ditemukan di: `{CM_PATH}`")

    # Classification Report
    st.subheader("Classification Report")
    if os.path.exists(CR_PATH):
        try:
            df = pd.read_csv(CR_PATH, index_col=0)
            st.dataframe(df.style.format("{:.4f}"), use_container_width=True)
        except Exception as e:
            st.error(f"Gagal memuat classification report: {e}")
    else:
        st.warning(f"File classification report tidak ditemukan di: `{CR_PATH}`")

    # Alasan Pemilihan Model
    st.subheader("Alasan Pemilihan Model VGG16")
    st.markdown("""
    Model **VGG16** dipilih karena arsitekturnya sederhana namun cukup dalam (deep), cocok untuk klasifikasi dua kelas, 
    dan menunjukkan performa yang stabil dalam eksperimen awal dibandingkan arsitektur lain seperti EfficientNet.
    """)

    # Disclaimer
    st.info("""
    Hasil evaluasi ini berasal dari dataset uji terpisah dan model yang sudah dilatih sebelumnya.  
    Aplikasi ini hanya sebagai **alat bantu**, bukan pengganti diagnosis medis.  
    Silakan konsultasikan ke dokter spesialis kulit untuk kepastian lebih lanjut.
    """)