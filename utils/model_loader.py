import os
import streamlit as st
import tensorflow as tf

@st.cache_resource(show_spinner="Memuat model deteksi kulit...")
def load_model(model_path='best_vgg16_model.h5'):
    """
    Memuat model Keras dari path yang diberikan.
    """
    abs_path = os.path.abspath(model_path)
    try:
        model = tf.keras.models.load_model(abs_path)
        return model
    except Exception as e:
        st.error(f"Gagal memuat model dari: {abs_path}\n\n**Error:** {str(e)}")
        return None