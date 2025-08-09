import streamlit as st
from PIL import Image

def render_upload_section():
    st.subheader("Upload Gambar Kulit")
    st.markdown("Silakan unggah gambar kulit dalam format **.jpg**, **.jpeg**, atau **.png**. Ukuran maksimal 5 MB.")

    uploaded_file = st.file_uploader(
        label="Pilih gambar dari perangkat Anda:",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        # Validasi ukuran file (maks 5MB)
        max_size_bytes = 5 * 1024 * 1024
        if uploaded_file.size > max_size_bytes:
            st.error("Ukuran file melebihi 5 MB. Silakan unggah gambar yang lebih kecil.")
            return None, None, None

        try:
            # Buka dan konversi gambar ke RGB
            image = Image.open(uploaded_file).convert("RGB")

            # Tampilkan gambar preview
            st.image(
                image,
                caption="Gambar yang diunggah",
                use_container_width=True
            )

            # Ambil nama dan bytes untuk keperluan berikutnya
            image_bytes = uploaded_file.getvalue()
            image_name = uploaded_file.name

            return image, image_bytes, image_name

        except Exception as e:
            st.error(f"Gagal memuat gambar: {e}")
            return None, None, None

    # Jika belum ada file diunggah
    return None, None, None