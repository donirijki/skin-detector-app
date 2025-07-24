import streamlit as st
from PIL import Image
import io

def render_upload_section():
    st.subheader("Upload Gambar Kulit")

    uploaded_file = st.file_uploader(
        "Pilih gambar kulit (.jpg / .jpeg / .png)",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        try:
            # Baca gambar dari file uploader (stream)
            image = Image.open(uploaded_file).convert("RGB")

            # Tampilkan gambar
            st.image(
                image,
                caption="Gambar yang diunggah",
                use_container_width=True
            )

            # Simpan salinan bytes-nya agar bisa digunakan ulang atau dievaluasi
            image_bytes = uploaded_file.getvalue()
            image_name = uploaded_file.name

            return image, image_bytes, image_name

        except Exception as e:
            st.error(f"Gagal memuat gambar: {e}")
            return None, None, None

    return None, None, None