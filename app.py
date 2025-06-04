import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np


TARGET_COLUMN = 'Diagnosis'
FEATURE_COLUMNS = [
    'Age',
    'Gender',
    'Education Level',
    'Disease Duration',
    'Family_History',
    'Substance_Use',
    'Suicide_Attempt',
    'Positive_Symptom_Score',
    'Negative_Symptom_Score',
    'GAF_Score',
    'Social_Support',
    'Hospitalizations',
    'Stress_Factors',
    'Medication_Adherence'
]

@st.cache_data
def load_data(file_path):
    """Memuat dataset dari file CSV."""
    try:
        df = pd.read_csv(file_path)
        st.success(f"Dataset '{file_path}' berhasil dimuat.")
        st.write("Tipe data untuk kolom yang dipilih (sebelum pemrosesan):")
        if all(col in df.columns for col in FEATURE_COLUMNS + [TARGET_COLUMN]):
             st.dataframe(df[FEATURE_COLUMNS + [TARGET_COLUMN]].dtypes.astype(str).to_frame("Tipe Data"))
        else:
            st.warning("Beberapa kolom yang dikonfigurasi tidak ditemukan, tipe data tidak dapat ditampilkan sepenuhnya.")
        return df
    except FileNotFoundError:
        st.error(f"Error: File '{file_path}' tidak ditemukan. Pastikan file tersebut ada di direktori yang sama.")
        return None
    except Exception as e:
        st.error(f"Error saat memuat data: {e}")
        return None

@st.cache_resource
def train_models(df, _feature_cols, _target_col):
    """Melatih model Logistic Regression dan Random Forest."""
    if df is None:
        st.error("Dataset belum dimuat. Model tidak dapat dilatih.")
        return None, None, None, []

    if _target_col not in df.columns:
        st.error(f"Kolom target '{_target_col}' TIDAK DITEMUKAN di CSV. Kolom yang ada: {df.columns.tolist()}")
        return None, None, None, []

    missing_features = [col for col in _feature_cols if col not in df.columns]
    if missing_features:
        st.error(f"Beberapa kolom fitur yang dikonfigurasi TIDAK DITEMUKAN di CSV: {missing_features}.")
        st.error(f"Kolom fitur yang diharapkan (sesuai konfigurasi): {_feature_cols}")
        st.error(f"Kolom yang benar-benar ada di file CSV: {df.columns.tolist()}")
        st.warning("Silakan perbarui daftar 'FEATURE_COLUMNS' di awal kode Python.")
        return None, None, None, []

    df_processed = df.copy()

    if 'Gender' in _feature_cols:
        if df_processed['Gender'].dtype == 'object': 
            gender_map = {'Male': 1, 'Female': 0, 'Laki-laki': 1, 'Perempuan': 0}
            unique_genders = df_processed['Gender'].unique()
            can_map_all = all(g in gender_map for g in unique_genders)
            if can_map_all:
                df_processed['Gender'] = df_processed['Gender'].map(gender_map)
                st.info("Kolom 'Gender' (string) telah diubah menjadi numerik (Male/Laki-laki: 1, Female/Perempuan: 0).")
            else:
                st.warning(f"Kolom 'Gender' mengandung nilai yang tidak dikenal: {unique_genders}. Harap perbarui 'gender_map' dalam kode atau bersihkan data CSV.")

                df_processed['Gender'] = pd.to_numeric(df_processed['Gender'], errors='coerce')

    df_processed = df_processed[_feature_cols + [_target_col]].dropna()
    if df_processed.empty:
        st.error("Setelah menghapus baris dengan nilai kosong (NaN) pada fitur/target terpilih, tidak ada data tersisa untuk melatih model.")
        return None, None, None, []

    X = df_processed[_feature_cols]
    y = df_processed[_target_col]

    if not y.nunique() == 2 or not all(val in [0,1] for val in y.unique()):
        st.warning(f"Kolom target '{_target_col}' tampaknya bukan biner (0/1). Nilai unik: {y.unique()}. Model klasifikasi biner mungkin tidak sesuai.")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    feature_names_actual = X.columns.tolist()

    log_reg_model = LogisticRegression(solver='liblinear', random_state=42)
    log_reg_model.fit(X_scaled, y)

    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_scaled, y)
    
    st.success("Model berhasil dilatih!")
    return log_reg_model, rf_model, scaler, feature_names_actual

data_file = 'schizophrenia_dataset.csv'
df_loaded = load_data(data_file)

log_reg_model, rf_model, scaler, actual_feature_names_from_training = None, None, None, []
if df_loaded is not None:
    log_reg_model, rf_model, scaler, actual_feature_names_from_training = train_models(df_loaded, FEATURE_COLUMNS, TARGET_COLUMN)

st.title("🩺 Aplikasi Deteksi Dini Skizofrenia")
st.markdown("""
Aplikasi ini menggunakan model Machine Learning untuk membantu mendeteksi potensi skizofrenia berdasarkan data yang dimasukkan.
**Disclaimer:** Aplikasi ini adalah alat bantu dan tidak menggantikan diagnosis profesional dari tenaga medis.
""")
st.divider()

st.sidebar.header("📝 Masukkan Data Pasien:")

def get_user_input(feature_names_list_from_training):
    inputs = {}
    if not feature_names_list_from_training:
        st.sidebar.error("Nama fitur tidak tersedia (model mungkin belum dilatih).")
        return None
    
    if 'Age' in feature_names_list_from_training:
        inputs['Age'] = st.sidebar.number_input("Usia (tahun)", min_value=10, max_value=100, value=30, step=1)
    
    if 'Gender' in feature_names_list_from_training:
        inputs['Gender'] = st.sidebar.selectbox("Jenis Kelamin", options=[("Laki-laki", 1), ("Perempuan", 0)], format_func=lambda x: x[0])[1]
    
    if 'Education Level' in feature_names_list_from_training:
        inputs['Education Level'] = st.sidebar.number_input("Tingkat Pendidikan (level numerik)", min_value=0, max_value=20, value=12, step=1, help="Misal: 12 untuk lulus SMA, 16 untuk Sarjana S1")
    
    if 'Disease Duration' in feature_names_list_from_training:
        inputs['Disease Duration'] = st.sidebar.number_input("Durasi Penyakit (misal: bulan)", min_value=0, max_value=600, value=24, step=1)
    
    if 'Family_History' in feature_names_list_from_training:
        inputs['Family_History'] = st.sidebar.selectbox("Riwayat Keluarga dengan Gangguan Jiwa?", options=[("Ya", 1), ("Tidak", 0)], format_func=lambda x: x[0])[1]
    
    if 'Substance_Use' in feature_names_list_from_training:
        inputs['Substance_Use'] = st.sidebar.selectbox("Penggunaan Zat (Narkoba/Alkohol Berlebih)?", options=[("Ya", 1), ("Tidak", 0)], format_func=lambda x: x[0])[1]

    if 'Suicide_Attempt' in feature_names_list_from_training:
        inputs['Suicide_Attempt'] = st.sidebar.selectbox("Riwayat Percobaan Bunuh Diri?", options=[("Ya", 1), ("Tidak", 0)], format_func=lambda x: x[0])[1]

    if 'Positive_Symptom_Score' in feature_names_list_from_training:
        inputs['Positive_Symptom_Score'] = st.sidebar.slider("Skor Gejala Positif", min_value=0, max_value=50, value=10, step=1, help="Contoh: PANSS Positive Subscale (Positive Syndrome Scale)")
    
    if 'Negative_Symptom_Score' in feature_names_list_from_training:
        inputs['Negative_Symptom_Score'] = st.sidebar.slider("Skor Gejala Negatif", min_value=0, max_value=50, value=10, step=1, help="Contoh: PANSS Negative Subscale (Negative Syndrome Scale)")

    if 'GAF_Score' in feature_names_list_from_training:
        inputs['GAF_Score'] = st.sidebar.slider("GAF Score (Global Assessment of Functioning)", min_value=0, max_value=100, value=50, step=1, help="Skor 1-100, semakin tinggi semakin baik fungsinya.")

    if 'Social_Support' in feature_names_list_from_training:
        inputs['Social_Support'] = st.sidebar.slider("Dukungan Sosial (Skor)", min_value=0, max_value=10, value=5, step=1)

    if 'Hospitalizations' in feature_names_list_from_training:
        inputs['Hospitalizations'] = st.sidebar.number_input("Jumlah Rawat Inap Psikiatri", min_value=0, max_value=50, value=1, step=1)
    
    if 'Stress_Factors' in feature_names_list_from_training:
        inputs['Stress_Factors'] = st.sidebar.slider("Faktor Stres (Skor)", min_value=0, max_value=10, value=5, step=1)

    if 'Medication_Adherence' in feature_names_list_from_training:
        inputs['Medication_Adherence'] = st.sidebar.slider("Kepatuhan Minum Obat (Skor/Persentase)", min_value=0, max_value=100, value=75, step=1)

    ordered_inputs = []
    for fn in feature_names_list_from_training:
        if fn in inputs:
            ordered_inputs.append(inputs[fn])
        else:
            st.sidebar.error(f"Fitur '{fn}' yang diharapkan model tidak ditemukan di input pengguna. Periksa 'get_user_input'.")
            return None
    return ordered_inputs

user_input_list = None
if actual_feature_names_from_training:
    user_input_list = get_user_input(actual_feature_names_from_training)
else:
    st.sidebar.warning("Input pengguna tidak dapat ditampilkan karena model belum siap (mungkin karena kesalahan konfigurasi kolom atau data).")


st.sidebar.divider()
model_choice = st.sidebar.selectbox("Pilih Model:", ("Random Forest"))

if st.sidebar.button("🔍 Deteksi Sekarang"):
    if user_input_list and actual_feature_names_from_training and log_reg_model and rf_model and scaler:
        input_array = np.array(user_input_list).reshape(1, -1)
        
        if input_array.shape[1] != len(actual_feature_names_from_training):
            st.error(f"Jumlah fitur input ({input_array.shape[1]}) tidak cocok dengan yang diharapkan model ({len(actual_feature_names_from_training)}).")
        else:
            input_scaled = scaler.transform(input_array)
            prediction, proba = None, None

            st.subheader("Hasil Prediksi:")
            if model_choice == "Logistic Regression":
                prediction = log_reg_model.predict(input_scaled)
                proba = log_reg_model.predict_proba(input_scaled)
                st.write("Model: **Logistic Regression**")
            else:
                prediction = rf_model.predict(input_scaled)
                proba = rf_model.predict_proba(input_scaled)
                st.write("Model: **Random Forest**")

            if prediction[0] == 1:
                st.error("⚠️ **Positif Schizo**")
                st.write(f"Probabilitas (Positif): {proba[0][1]*100:.2f}%")
                st.markdown("Segera konsultasikan dengan profesional kesehatan jiwa.")
            else:
                st.success("✅ **Negatif Schizo**")
                st.write(f"Probabilitas (Negatif): {proba[0][0]*100:.2f}%")
                st.markdown("Tetap jaga kesehatan mental Anda.")
            
            st.divider()
            st.write("Data yang Anda masukkan:")
            input_df = pd.DataFrame([user_input_list], columns=actual_feature_names_from_training)
            st.dataframe(input_df)
    else:
        st.error("Tidak dapat melakukan prediksi. Pastikan data pasien telah dimasukkan, model telah berhasil dilatih, dan tidak ada error konfigurasi.")

else:
    if actual_feature_names_from_training:
         st.info("Masukkan data pasien di sidebar kiri dan klik 'Deteksi Sekarang'.")

st.divider()
st.markdown("<div style='text-align: center; font-size: small;'>Aplikasi demo untuk deteksi dini skizofrenia.</div>", unsafe_allow_html=True)
