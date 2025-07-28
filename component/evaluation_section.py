import streamlit as st
from utils.evaluation import evaluate_user_predictions

def render_evaluation_section():
    st.header("Evaluasi Model Deteksi Kulit")

    st.markdown("""
        Evaluasi model dilakukan berdasarkan histori prediksi yang telah diberi label sebenarnya oleh pengguna.
        Klik tombol di bawah untuk menghitung akurasi, classification report, dan confusion matrix.
    """)

    if st.button("Hitung Evaluasi Model (VGG16)"):
        with st.spinner("Menghitung evaluasi model..."):
            try:
                accuracy, report_df, cm, fig = evaluate_user_predictions()

                if report_df is None or report_df.empty:
                    st.warning("Belum ada data prediksi dengan label sebenarnya yang tersedia untuk evaluasi.")
                    return

                st.success("Evaluasi berhasil dihitung.")

                # Tampilkan Akurasi
                st.subheader("Hasil Evaluasi Model (VGG16)")
                st.markdown(f"- **Akurasi Model:** `{accuracy * 100:.2f}%`")

                # Confusion Matrix
                st.markdown("### Confusion Matrix")
                st.pyplot(fig)

                # Classification Report
                st.markdown("### Classification Report")
                st.dataframe(report_df.style.format(precision=2))

            except Exception as e:
                st.error(f"Gagal menghitung evaluasi model: {e}")