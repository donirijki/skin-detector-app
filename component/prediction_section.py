import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import hashlib
import io

# Import modul utilitas
from utils.image_preprocessor import preprocess_image
from utils.gradcam_utils import make_gradcam_heatmap, apply_heatmap_on_image
from utils.logging import log_prediction
from utils.history_logger import log_to_history
from utils.report_writer import write_prediction_report

# Daftar label kelas
CLASS_NAMES = ['melanoma', 'psoriasis']

# Fungsi untuk menyimpan gambar hasil unggahan ke folder sesuai hasil prediksi
def save_uploaded_image(image: Image.Image, predicted_label: str, original_name: str) -> str:
    save_dir = os.path.join("external_test", predicted_label.lower())
    os.makedirs(save_dir, exist_ok=True)

    img_bytes = image.tobytes()
    img_hash = hashlib.md5(img_bytes).hexdigest()
    filename = f"{img_hash}_{original_name.replace(' ', '_')}"
    save_path = os.path.join(save_dir, filename)

    if not os.path.exists(save_path):
        image.save(save_path)

    return save_path

# Fungsi utama untuk menampilkan halaman prediksi
def render_prediction_section(model, user_info=None):
    st.header("Prediksi Gambar Kulit")
    st.markdown("Unggah gambar kulit untuk diprediksi apakah termasuk **Melanoma** atau **Psoriasis**.")

    uploaded_file = st.file_uploader("Unggah gambar (.jpg / .jpeg / .png)", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Validasi ukuran file maksimal 5MB
        max_size_bytes = 5 * 1024 * 1024
        if uploaded_file.size > max_size_bytes:
            st.error("Ukuran file melebihi 5 MB. Harap unggah gambar yang lebih kecil.")
            return

        try:
            # Buka gambar dan konversi ke RGB
            image = Image.open(uploaded_file).convert("RGB")

            # Resize ke 224x224
            image_resized = image.resize((224, 224))

            st.subheader("Gambar yang Diupload")
            st.image(image, caption="Gambar Input", width=400)

            # Pra-pemrosesan
            input_data = preprocess_image(image_resized)
            if input_data is None or input_data.shape != (1, 224, 224, 3):
                st.error("Gambar tidak valid atau gagal diproses.")
                return

            # Prediksi menggunakan model
            prediction = model.predict(input_data)[0]
            predicted_index = int(np.argmax(prediction))
            predicted_class = CLASS_NAMES[predicted_index]
            confidence = float(prediction[predicted_index])
            confidence_percent = confidence * 100

            # Tampilkan hasil prediksi
            st.subheader("Hasil Prediksi")
            if confidence >= 0.85:
                st.success(f"Model sangat yakin ini adalah **{predicted_class.upper()}** ({confidence_percent:.2f}%)")
            elif confidence >= 0.6:
                st.warning(f"Model cukup yakin ini adalah **{predicted_class.upper()}** ({confidence_percent:.2f}%)")
            else:
                st.error(f"Model kurang yakin. Prediksi: **{predicted_class.upper()}** ({confidence_percent:.2f}%)")

            # Visualisasi distribusi probabilitas
            st.subheader("Distribusi Probabilitas")
            fig, ax = plt.subplots(figsize=(4, 3))
            bars = ax.bar(CLASS_NAMES, prediction * 100, color=["#1976D2", "#C2185B"])
            ax.set_ylabel("Probabilitas (%)")
            ax.set_ylim([0, 110])
            ax.set_title("Distribusi Probabilitas Model")
            for bar, prob in zip(bars, prediction * 100):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2, f"{prob:.1f}%", ha='center')

            buf = io.BytesIO()
            fig.tight_layout()
            plt.savefig(buf, format="png")
            buf.seek(0)
            plt.close(fig)
            st.image(buf, caption="Distribusi Probabilitas", width=400)

            # Visualisasi Grad-CAM
            st.subheader("Visualisasi Grad-CAM")
            try:
                heatmap = make_gradcam_heatmap(input_data, model, last_conv_layer_name="block5_conv3")
                result_image = apply_heatmap_on_image(image_resized, heatmap)
                st.image(result_image, caption="Grad-CAM Overlay", width=400)
            except Exception as e:
                st.error(f"Gagal membuat Grad-CAM: {e}")

            # Input label sebenarnya
            st.subheader("Label Sebenarnya")
            true_label = st.selectbox(
                "Pilih label sebenarnya dari gambar (diperlukan untuk evaluasi):",
                ["", "melanoma", "psoriasis"]
            )

            if not true_label:
                st.warning("Silakan pilih label sebenarnya untuk menyimpan hasil ke evaluasi.")
                return

            # Simpan gambar dan histori
            saved_image_path = save_uploaded_image(image, predicted_class, uploaded_file.name)
            st.caption(f"Gambar disimpan di: `{os.path.basename(saved_image_path)}`")

            if "history" not in st.session_state:
                st.session_state.history = []

            # Cek jika gambar belum dicatat ke histori
            if not any(item["image_path"] == saved_image_path for item in st.session_state.history):
                st.session_state.history.append({
                    "image_path": saved_image_path,
                    "image_name": os.path.basename(saved_image_path),
                    "predicted_label": predicted_class,
                    "confidence": confidence,
                    "true_label": true_label
                })

                # Log histori
                log_to_history(
                    image_path=saved_image_path,
                    predicted_label=predicted_class,
                    confidence=confidence,
                    true_label=true_label
                )

                # Log prediksi (jika info pengguna tersedia)
                if user_info:
                    log_prediction(
                        user_info=user_info,
                        image_name=os.path.basename(saved_image_path),
                        prediction=predicted_class,
                        confidence=confidence,
                        true_label=true_label
                    )

            # Unduh laporan prediksi (jika info pengguna tersedia)
            if user_info:
                report_path = write_prediction_report(
                    user_info=user_info,
                    image_name=os.path.basename(saved_image_path),
                    predicted_label=predicted_class,
                    confidence=confidence
                )
                with open(report_path, "r", encoding="utf-8") as f:
                    st.download_button(
                        label="Unduh Laporan Prediksi (.txt)",
                        data=f.read(),
                        file_name=os.path.basename(report_path),
                        mime="text/plain"
                    )

        except Exception as e:
            st.error(f"Terjadi kesalahan saat memproses gambar: {e}")