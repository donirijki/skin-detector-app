import streamlit as st

def render_advice_section():
    st.header("Edukasi Penyakit Kulit")

    with st.expander("Apa itu Melanoma?"):
        st.markdown("""
        **Melanoma** adalah jenis kanker kulit yang berasal dari sel penghasil pigmen (melanosit).  
        - **Berbahaya** jika tidak ditangani cepat  
        - **Deteksi dini** sangat penting untuk meningkatkan angka kesembuhan  
        - Ciri-ciri umum:  
            - Bercak gelap atau tidak simetris  
            - Tepi tidak beraturan  
            - Warna campuran  
            - Diameter > 6mm  
        - Konsultasikan segera ke dokter kulit jika ada perubahan mencurigakan pada kulit
        """)

    with st.expander("Apa itu Psoriasis?"):
        st.markdown("""
        **Psoriasis** adalah penyakit kulit autoimun kronis yang menyebabkan peradangan dan penebalan kulit.  
        - **Bukan penyakit menular**  
        - Berkaitan dengan faktor genetik dan sistem imun  
        - Gejala umum:  
            - Kulit bersisik tebal berwarna perak  
            - Gatal atau perih  
            - Umumnya muncul di siku, lutut, dan kulit kepala  
        - Penanganan: krim topikal, fototerapi, hingga obat sistemik
        """)

    st.info("Informasi ini bukan pengganti diagnosis dokter. Selalu konsultasikan dengan profesional medis.")