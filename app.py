import streamlit as st
import pandas as pd
import joblib

# Load model dan scaler
model = joblib.load('model_random_forest.pkl')
scaler = joblib.load('scaler.pkl')

st.title("🎓 Prediksi Prestasi Akademik Mahasiswa")
st.write("Aplikasi ini memprediksi potensi nilai akhir (IPK) berdasarkan profil mahasiswa.")

# ===============================
# 🔹 Input fitur dari user
# ===============================
department = st.selectbox("Jurusan", ["Business Administration"])
gender = st.selectbox("Jenis Kelamin", ["Male", "Female"])
hsc = st.number_input("Nilai HSC", 0.0, 5.0, 3.0)
ssc = st.number_input("Nilai SSC", 0.0, 5.0, 3.0)
income = st.selectbox("Pendapatan Keluarga", [
    "Low (Below 15,000)", 
    "Lower middle (15,000-30,000)", 
    "Upper middle (30,000-50,000)", 
    "High (Above 50,000)"
])
hometown = st.selectbox("Asal Daerah", ["City", "Village"])
computer = st.slider("Kemampuan Komputer (1-5)", 1, 5, 3)
preparation = st.selectbox("Waktu Belajar", ["0-1 Hour", "2-3 Hours", "More than 3 Hours"])
gaming = st.selectbox("Waktu Bermain Game", ["0-1 Hour", "2-3 Hours", "More than 3 Hours"])
attendance = st.selectbox("Kehadiran", ["Below 60%", "60%-80%", "80%-100%"])
job = st.selectbox("Apakah Bekerja?", ["Yes", "No"])
english = st.slider("Kemampuan Bahasa Inggris (1-5)", 1, 5, 3)
extra = st.selectbox("Ikut Kegiatan Ekstrakurikuler?", ["Yes", "No"])
semester = st.selectbox("Semester Saat Ini", ["1st", "2nd", "3rd", "4th", "5th", "6th", "7th"])
last = st.number_input("Nilai Semester Terakhir", 0.0, 4.0, 3.0)

# ===============================
# 🔹 Buat dataframe input
# ===============================
input_data = pd.DataFrame({
    'Department': [department],
    'Gender': [gender],
    'HSC': [hsc],
    'SSC': [ssc],
    'Income': [income],
    'Hometown': [hometown],
    'Computer': [computer],
    'Preparation': [preparation],
    'Gaming': [gaming],
    'Attendance': [attendance],
    'Job': [job],
    'English': [english],
    'Extra': [extra],
    'Semester': [semester],
    'Last': [last]
})

# ===============================
# 🔹 Encoding manual sesuai dataset
# ===============================
label_maps = {
    'Gender': ['Male', 'Female'],
    'Income': ['Low (Below 15,000)', 'Lower middle (15,000-30,000)', 'Upper middle (30,000-50,000)', 'High (Above 50,000)'],
    'Hometown': ['City', 'Village'],
    'Preparation': ['0-1 Hour', '2-3 Hours', 'More than 3 Hours'],
    'Gaming': ['0-1 Hour', '2-3 Hours', 'More than 3 Hours'],
    'Attendance': ['Below 60%', '60%-80%', '80%-100%'],
    'Job': ['Yes', 'No'],
    'Extra': ['Yes', 'No'],
    'Semester': ['1st', '2nd', '3rd', '4th', '5th', '6th', '7th']
}
categorical_cols = ['Department', 'Gender', 'Income', 'Hometown', 'Attendance']
for col, categories in label_maps.items():
    input_data[col] = input_data[col].apply(lambda x: categories.index(x))

# ===============================
# 🔹 Standarisasi & Prediksi
# ===============================
input_scaled = scaler.transform(input_data)
prediction = model.predict(input_scaled)[0]

st.subheader("📈 Hasil Prediksi IPK:")
st.success(f"Prediksi nilai akhir mahasiswa: {prediction:.3f}")
 
