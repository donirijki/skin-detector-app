import streamlit as st
import os
from PIL import Image

# Deskripsi per Gambar

melanoma_descriptions = {
    "Acral-Lentiginous Melanoma": "Melanoma langka yang muncul di telapak tangan, kaki, atau bawah kuku. Umum pada orang dengan kulit gelap.",
    "Lentigo Maligna Melanoma": "Biasanya tumbuh perlahan di kulit yang sering terpapar matahari seperti wajah dan leher.",
    "Malignant Melanoma": "Melanoma ganas yang sangat agresif. Harus ditangani cepat karena mudah menyebar.",
    "Nodular Melanoma": "Tipe melanoma berbentuk benjolan bulat yang tumbuh cepat dan dapat berdarah.",
    "Superficial Spreading Melanoma": "Jenis melanoma paling umum. Tumbuh menyebar di permukaan kulit sebelum masuk lebih dalam."
}

psoriasis_descriptions = {
    "Flexural Psoriasis": "Psoriasis di area lipatan tubuh seperti ketiak atau selangkangan. Tampak merah dan tidak bersisik.",
    "Palmoplantar Psoriasis": "Muncul di telapak tangan dan kaki. Kulit tampak menebal, kering, dan dapat pecah-pecah.",
    "Plaque Psoriasis": "Jenis paling umum. Ditandai dengan plak merah tebal dan sisik putih keperakan.",
    "Psoriasis Guttata": "Berbentuk bintik kecil seperti tetesan air. Sering muncul setelah infeksi saluran pernapasan atas.",
    "Psoriasis Vulgaris": "Nama lain dari plaque psoriasis. Bentuk paling sering dijumpai, kronis, dan berulang."
}

# Fungsi Utilitas 

def load_images_from_folder(folder_path):
    images = []
    for filename in sorted(os.listdir(folder_path)):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            images.append(os.path.join(folder_path, filename))
    return images

def display_image_grid(image_paths, description_dict, max_per_row=2, image_height=250):
    for i in range(0, len(image_paths), max_per_row):
        cols = st.columns(max_per_row)
        for col, img_path in zip(cols, image_paths[i:i + max_per_row]):
            img = Image.open(img_path)
            img_resized = img.copy()
            img_resized.thumbnail((600, image_height))  # Resize proporsional

            name = os.path.splitext(os.path.basename(img_path))[0].replace("_", " ").title()
            col.image(img_resized, caption=name, use_container_width=True)

            description = description_dict.get(name, "Deskripsi belum tersedia.")
            col.markdown(f"**Penjelasan:** {description}")

# Fungsi Utama 

def render_advice_section():
    st.header("Edukasi Penyakit Kulit")

    tab1, tab2 = st.tabs(["Melanoma", "Psoriasis"])

    # MELANOMA 
    with tab1:
        st.subheader("Tentang Melanoma")
        st.markdown("""
Melanoma adalah bentuk paling berbahaya dari kanker kulit yang berasal dari sel melanosit.

- Dapat berkembang dari tahi lalat atau langsung di kulit normal.
- Tanda-tanda mengikuti pola **ABCDE**:
  - **A**symmetry: Bentuk tidak simetris
  - **B**order: Tepi tidak teratur
  - **C**olor: Warna tidak seragam
  - **D**iameter: Lebih dari 6 mm
  - **E**volving: Terjadi perubahan ukuran, bentuk, atau warna
- Risiko meningkat akibat paparan sinar UV dan riwayat keluarga.
        """)

        st.markdown("### Jenis dan Karakteristik Klinis:")
        melanoma_folder = os.path.join("data_testing", "melanoma")
        melanoma_images = load_images_from_folder(melanoma_folder)
        display_image_grid(melanoma_images, melanoma_descriptions)

        st.markdown("""
### Penanganan Melanoma:
- Periksa segera ke **dokter spesialis kulit** jika terdapat lesi mencurigakan.
- Diagnosis memerlukan **biopsi kulit**.
- Terapi bergantung pada stadium: operasi, imunoterapi, atau kemoterapi.

*Sumber: American Cancer Society, WHO*
        """)

    # PSORIASIS 
    with tab2:
        st.subheader("Tentang Psoriasis")
        st.markdown("""
Psoriasis adalah penyakit autoimun kronis yang mempercepat siklus hidup sel kulit.

- Menyebabkan penumpukan sel yang membentuk plak tebal bersisik.
- Tidak menular, tapi bisa kambuh dalam waktu tertentu.
- Dipicu oleh stres, infeksi, cedera, atau obat tertentu.
        """)

        st.markdown("### Jenis dan Karakteristik Klinis:")
        psoriasis_folder = os.path.join("data_testing", "psoriasis")
        psoriasis_images = load_images_from_folder(psoriasis_folder)
        display_image_grid(psoriasis_images, psoriasis_descriptions)

        st.markdown("""
### Penanganan Psoriasis:
- Pengobatan topikal: **kortikosteroid, vitamin D analog**
- Terapi cahaya: **UVB fototerapi**
- Obat sistemik: **imunosupresan atau biologik**
- Disarankan konsultasi rutin dengan dokter spesialis kulit

*Sumber: National Psoriasis Foundation, Mayo Clinic*
        """)

    st.info("Informasi ini untuk edukasi. Untuk diagnosis dan penanganan yang tepat, konsultasikan dengan tenaga medis profesional.")