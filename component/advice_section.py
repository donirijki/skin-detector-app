import streamlit as st

def render_advice_section():
    st.header("Edukasi Penyakit Kulit")

    with st.expander("Apa itu Melanoma?"):
        st.markdown("""
        **Melanoma** adalah bentuk paling serius dari kanker kulit yang berkembang dari melanosit — sel penghasil melanin (pigmen kulit).  
        
        - Melanoma dapat muncul di kulit normal atau dari tahi lalat yang berubah.  
        - Penyebab utama: paparan sinar UV (matahari atau tanning bed), faktor genetik, dan mutasi sel.  
        - **Gejala umum** mengikuti pola "ABCDE":  
            - **A**symmetry (Asimetri)  
            - **B**order (Tepi tidak teratur)  
            - **C**olor (Warna bervariasi)  
            - **D**iameter (>6 mm)  
            - **E**volving (Perubahan bentuk/ukuran)  
        - **Risiko lebih tinggi** pada individu dengan riwayat keluarga melanoma atau kulit terang.  
        - **Deteksi dini** sangat penting karena jika terdiagnosis awal, peluang kesembuhan >90%.  

        *Sumber: American Cancer Society, WHO, Kemenkes RI*
        """)

    with st.expander("Apa itu Psoriasis?"):
        st.markdown("""
        **Psoriasis** adalah penyakit autoimun kronis yang menyebabkan percepatan regenerasi sel kulit, sehingga menumpuk dan membentuk plak tebal bersisik.  

        - Tidak menular dan bersifat jangka panjang (kronis).  
        - Dipengaruhi oleh **faktor imunologis dan genetik**.  
        - Dapat dipicu oleh stres, infeksi, cedera kulit, atau obat tertentu.  
        - **Gejala utama:**  
            - Plak merah tebal dengan sisik keperakan  
            - Gatal, nyeri, atau kulit pecah-pecah  
            - Umumnya muncul di siku, lutut, kulit kepala, dan punggung bawah  
        - **Jenis umum:** Psoriasis vulgaris (plaque-type), guttata, inversa, pustular  
        - **Terapi:**  
            - Topikal (kortikosteroid, vitamin D analog)  
            - Fototerapi (UVB)  
            - Obat sistemik (imunosupresan, biologik)

        *Sumber: National Psoriasis Foundation, Mayo Clinic, EADV*
        """)

    st.info("Informasi ini hanya untuk tujuan edukatif dan bukan pengganti konsultasi medis. Segera periksakan ke dokter spesialis kulit untuk diagnosis dan perawatan yang tepat.")