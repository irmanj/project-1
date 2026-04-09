import streamlit as st
import joblib
import pandas as pd

# load model
model = joblib.load("model.pkl")

st.title("🎓 Prediksi Kelulusan Siswa")

jam = st.slider("Jam belajar", 0, 10)
hadir = st.slider("Kehadiran", 0, 100)

if st.button("Prediksi"):
    
    input_data = pd.DataFrame({
        'jam_belajar': [jam],
        'kehadiran': [hadir]
    })

    result = model.predict(input_data)

    if result[0] == 1:
        st.success("Siswa kemungkinan LULUS ✅")
    else:
        st.error("Siswa kemungkinan TIDAK LULUS ❌")