import streamlit as st
import pandas as pd
import joblib

# ===============================
# 隼 Muat SEMUA artefak
# ===============================
model = joblib.load('model_random_forest.pkl')
scaler = joblib.load('scaler.pkl')
# <-- PERUBAHAN: Muat juga file encoder Anda
encoders = joblib.load('label_encoders.pkl') 

st.title("雌 Prediksi Prestasi Akademik Mahasiswa")
st.write("Aplikasi ini memprediksi potensi nilai akhir (IPK) berdasarkan profil mahasiswa.")

# ===============================
# 隼 Input fitur dari user
# ===============================
# <-- PERUBAHAN: Opsi diambil dari 'encoders' untuk menjamin konsistensi
department = st.selectbox("Jurusan", encoders['Department'].classes_)
gender = st.selectbox("Jenis Kelamin", encoders['Gender'].classes_)
hsc = st.number_input("Nilai HSC", 0.0, 5.0, 3.0)
ssc = st.number_input("Nilai SSC", 0.0, 5.0, 3.0)
income = st.selectbox("Pendapatan Keluarga", encoders['Income'].classes_)
hometown = st.selectbox("Asal Daerah", encoders['Hometown'].classes_)
computer = st.slider("Kemampuan Komputer (1-5)", 1, 5, 3)
preparation = st.selectbox("Waktu Belajar", encoders['Preparation'].classes_)
gaming = st.selectbox("Waktu Bermain Game", encoders['Gaming'].classes_)
attendance = st.selectbox("Kehadiran", encoders['Attendance'].classes_)
job = st.selectbox("Apakah Bekerja?", encoders['Job'].classes_)
english = st.slider("Kemampuan Bahasa Inggris (1-5)", 1, 5, 3)
extra = st.selectbox("Ikut Kegiatan Ekstrakurikuler?", encoders['Extra'].classes_)
semester = st.selectbox("Semester Saat Ini", encoders['Semester'].classes_)
last = st.number_input("Nilai Semester Terakhir", 0.0, 4.0, 3.0)

# ===============================
# 隼 Buat dataframe input
# ===============================
# Tidak ada perubahan di bagian ini
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
# 隼 BAGIAN BARU: Encoding Otomatis
# ===============================
# <-- PERUBAHAN: Hapus 'label_maps' dan ganti dengan loop ini
# Tentukan kolom mana yang kategorikal (sesuai encoders)
categorical_cols = [
    'Department', 'Gender', 'Income', 'Hometown', 'Preparation', 
    'Gaming', 'Attendance', 'Job', 'Extra', 'Semester'
]

# Buat salinan untuk menghindari Peringatan (Warning)
input_data_encoded = input_data.copy()

# Loop dan transform setiap kolom kategorikal
for col in categorical_cols:
    # Ambil nilai string dari input (contoh: "Economics")
    string_value = input_data_encoded[col].iloc[0]
    
    # Ubah string itu menjadi angka (contoh: 2) menggunakan encoder yang sesuai
    # .transform() mengharapkan array, jadi kita bungkus [string_value]
    # lalu ambil hasilnya [0]
    input_data_encoded[col] = encoders[col].transform([string_value])[0]

# ===============================
# 隼 Standarisasi & Prediksi
# ===============================
# <-- PERUBAHAN: Gunakan dataframe yang sudah di-encode
input_scaled = scaler.transform(input_data_encoded)
prediction = model.predict(input_scaled)[0]

st.subheader("嶋 Hasil Prediksi IPK:")
st.success(f"Prediksi nilai akhir mahasiswa: {prediction:.3f}")
