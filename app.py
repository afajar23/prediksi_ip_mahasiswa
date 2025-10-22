import streamlit as st
import pandas as pd
import joblib

# Load model dan scaler
model = joblib.load('model_random_forest.pkl')
scaler = joblib.load('scaler.pkl')

st.title("🎓 Prediksi Prestasi Akademik Mahasiswa")
st.write("Aplikasi ini memprediksi potensi nilai akhir (IPK) berdasarkan profil mahasiswa.")

# Input fitur dari user
gender = st.selectbox("Jenis Kelamin", ["Male", "Female"])
income = st.selectbox("Pendapatan Keluarga", ["Low (Below 15,000)", "Lower middle (15,000-30,000)", "Upper middle (30,000-50,000)", "High (Above 50,000)"])
hometown = st.selectbox("Asal Daerah", ["City", "Village"])
attendance = st.selectbox("Kehadiran", ["Below 60%", "60%-80%", "80%-100%"])
computer = st.slider("Kemampuan Komputer (1-5)", 1, 5, 3)
hsc = st.number_input("Nilai HSC", 0.0, 5.0, 3.0)
ssc = st.number_input("Nilai SSC", 0.0, 5.0, 3.0)
english = st.slider("Kemampuan Bahasa Inggris (1-5)", 1, 5, 3)
last = st.number_input("Nilai Semester Terakhir", 0.0, 4.0, 3.0)

# Buat dataframe dari input user
input_data = pd.DataFrame({
    'Gender': [gender],
    'Income': [income],
    'Hometown': [hometown],
    'Attendance': [attendance],
    'Computer': [computer],
    'HSC': [hsc],
    'SSC': [ssc],
    'English': [english],
    'Last': [last]
})

# Encoding label sesuai dataset awal
from sklearn.preprocessing import LabelEncoder
label_encoders = {}
categorical_cols = ['Gender', 'Income', 'Hometown', 'Attendance']

for col in categorical_cols:
    le = LabelEncoder()
    le.fit(df[col])  # gunakan label dari dataset asli
    input_data[col] = le.transform(input_data[col])
    label_encoders[col] = le

# Standarisasi
input_scaled = scaler.transform(input_data)

# Prediksi
prediction = model.predict(input_scaled)[0]

st.subheader("📈 Hasil Prediksi IPK:")
st.success(f"Prediksi nilai akhir mahasiswa: {prediction:.3f}")
