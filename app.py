import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample

# Fungsi load data dan preprocessing
@st.cache_data
def load_and_preprocess(data):
    df = data.copy()
    for col in df.columns:
        if df[col].isnull().any():
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col].fillna(df[col].mean(), inplace=True)
            else:
                df[col].fillna(df[col].mode()[0], inplace=True)
    return df

# Fungsi training model
@st.cache_resource
def train_models(df):
    X = df.drop(columns=['Diagnosis'])
    y = df['Diagnosis']

    df_majority = df[df['Diagnosis'] == 0]
    df_minority = df[df['Diagnosis'] == 1]
    df_minority_upsampled = resample(df_minority,
                                     replace=True,
                                     n_samples=len(df_majority),
                                     random_state=42)
    df_balanced = pd.concat([df_majority, df_minority_upsampled]).sample(frac=1, random_state=42).reset_index(drop=True)

    X_bal = df_balanced.drop(columns=['Diagnosis'])
    y_bal = df_balanced['Diagnosis']

    logreg = LogisticRegression(max_iter=1000, random_state=42)
    logreg.fit(X_bal, y_bal)

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_bal, y_bal)

    return logreg, rf

# Fungsi prediksi
def predict(models, input_df):
    logreg, rf = models
    pred_logreg = logreg.predict(input_df)[0]
    pred_rf = rf.predict(input_df)[0]
    return pred_logreg, pred_rf

# Fungsi untuk label diagnosis
def label_diagnosis(pred):
    return "Positif Skizofrenia" if pred == 1 else "Negatif Skizofrenia"

# Main app
def main():
    st.title("Prediksi Skizofrenia dengan Logistic Regression & Random Forest")

    uploaded_file = st.file_uploader("Upload dataset schizophrenia_dataset.csv", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        df_clean = load_and_preprocess(df)

        with st.spinner("Melatih model..."):
            models = train_models(df_clean)
        st.success("Model siap digunakan!")

        st.header("Input Data Pasien untuk Prediksi")

        fitur = df_clean.drop(columns=['Diagnosis']).columns.tolist()

        input_data = {}
        for f in fitur:
            val = st.number_input(f"{f}", value=float(df_clean[f].mean()))
            input_data[f] = val

        input_df = pd.DataFrame([input_data])

        if st.button("Prediksi"):
            pred_logreg, pred_rf = predict(models, input_df)

            st.subheader("Hasil Prediksi")
            st.write(f"Logistic Regression: {label_diagnosis(pred_logreg)}")
            st.write(f"Random Forest: {label_diagnosis(pred_rf)}")

if __name__ == "__main__":
    main()
