import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.title("🎈 Uygulamaya Hoşgeldiniz")

# Veri Seti Yükleyici
st.title('Veri Seti Yükleyici')

# Dosya yükleme widget'ı
uploaded_file = st.file_uploader("Bir CSV dosyası yükleyin", type="csv")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

# Time sütununu sil
if 'Time' in df.columns:
    df = df.drop(columns=['Time'])
    st.write('"Time" sütunu başarıyla silindi.')
else:
    st.write('"Time" sütunu bulunamadı.')
