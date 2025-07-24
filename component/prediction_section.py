import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import time

from utils.image_preprocessor import preprocess_image
from utils.gradcam_utils import make_gradcam_heatmap, apply_heatmap_on_image
from utils.logging import log_prediction, log_to_history
from utils.report_writer import write_prediction_report

# Kelas target
CLASS_NAMES = ['melanoma', 'psoriasis']

def save_uploaded_image(image: Image.Image, predicted_label: str, original_name: str) -> str:
    save_dir = os.path.join("external_test", predicted_label.lower())
    os.makedirs(save_dir, exist_ok=True)

    timestamp = int(time.time())
    filename = f"{timestamp}_{original_name}"
    save_path = os.path.join(save_dir, filename)
    image.save(save_path)

    return save_path

def render_prediction_section(model, user_info=None):
    st.header("Prediksi Gambar Kulit")
    st.markdown("Unggah gambar kulit untuk diprediksi apakah termasuk **Melanoma** atau **Psoriasis**.")

    uploaded_file = st.file_uploader("Unggah gambar (.jpg / .jpeg / .png)", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Gambar yang diunggah", use_container_width=True)

            # Preprocessing
            input_data = preprocess_image(image)
            if input_data is None or input_data.shape != (1, 224, 224, 3):
                st.error("Gambar tidak valid atau gagal diproses.")
                return

            # Prediksi model
            prediction = model.predict(input_data)
            if prediction is None or len(prediction[0]) != len(CLASS_NAMES):
                st.error("Model gagal memberikan prediksi yang valid.")
                return

            prediction = prediction[0]
            predicted_index = int(np.argmax(prediction))
            predicted_class = CLASS_NAMES[predicted_index]
            confidence = float(prediction[predicted_index])
            confidence_percent = confidence * 100

            # Hasil Prediksi
            st.subheader("Hasil Prediksi")
            if confidence >= 0.85:
                st.success(f"Model sangat yakin ini adalah **{predicted_class.upper()}** ({confidence_percent:.2f}%)")
            elif confidence >= 0.6:
                st.warning(f"Model cukup yakin ini adalah **{predicted_class.upper()}** ({confidence_percent:.2f}%)")
            else:
                st.error(f"Model kurang yakin. Prediksi: **{predicted_class.upper()}** ({confidence_percent:.2f}%)")

            # Visualisasi Probabilitas
            st.subheader("Distribusi Probabilitas")
            fig, ax = plt.subplots(figsize=(6, 4))
            bars = ax.bar(CLASS_NAMES, prediction * 100, color=["#1976D2", "#C2185B"])
            ax.set_ylabel("Probabilitas (%)")
            ax.set_ylim([0, 110])
            ax.set_title("Distribusi Probabilitas Model")
            for bar, prob in zip(bars, prediction * 100):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2, f"{prob:.1f}%", ha='center')
            st.pyplot(fig)

            # Grad-CAM
            st.subheader("Visualisasi Grad-CAM")
            try:
                heatmap = make_gradcam_heatmap(input_data, model, last_conv_layer_name="block5_conv3")
                result_image = apply_heatmap_on_image(image, heatmap)
                st.image(result_image, caption="Grad-CAM Overlay", use_container_width=True)
            except Exception as e:
                st.error(f"Gagal membuat Grad-CAM: {e}")

            # Simpan gambar
            saved_image_path = save_uploaded_image(image, predicted_class, uploaded_file.name)
            st.caption(f"Gambar disimpan di: `{os.path.basename(saved_image_path)}`")

            # Logging ke session & file
            log_to_history(
                image_path=saved_image_path,
                predicted_label=predicted_class,
                confidence=confidence
            )

            if user_info:
                log_prediction(user_info, os.path.basename(saved_image_path), predicted_class, confidence)

            if "history" not in st.session_state:
                st.session_state.history = []

            if not any(item["image_path"] == saved_image_path for item in st.session_state.history):
                st.session_state.history.append({
                    "image_path": saved_image_path,
                    "image_name": os.path.basename(saved_image_path),
                    "predicted_label": predicted_class,
                    "confidence": confidence
                })

            # Simpan laporan prediksi
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