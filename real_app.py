import streamlit as st
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import joblib
import neurokit2 as nk
import tempfile
from io import BytesIO

# Load trained model
model = joblib.load("model\heart_Ai.pkl")

st.title("🫀 Heart Disease Risk Prediction App")
st.markdown("Upload your **ECG image or CSV**, and enter your health data to predict heart disease risk.")

# Option to upload image or CSV
upload_mode = st.radio("Choose ECG Input Type:", ["Image (PNG/JPG)", "CSV File"])

ecg_signal = None

if upload_mode == "Image (PNG/JPG)":
    uploaded_img = st.file_uploader("Upload ECG Image", type=["png", "jpg", "jpeg"])
    if uploaded_img:
        try:
            # Load image
            file_bytes = np.asarray(bytearray(uploaded_img.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            # Preprocess image
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = 255 - gray
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            _, binary = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY)

            # Extract waveform
            signal = []
            for x in range(binary.shape[1]):
                column = binary[:, x]
                y_indices = np.where(column > 0)[0]
                if len(y_indices) > 0:
                    avg_y = np.mean(y_indices)
                    signal.append(avg_y)
                else:
                    signal.append(np.nan)

            # Clean signal
            signal = pd.Series(signal).interpolate().fillna(method='bfill').fillna(method='ffill')
            signal = signal.max() - signal
            signal = (signal - signal.min()) / (signal.max() - signal.min())
            signal = signal * 2 - 1

            ecg_signal = signal
            st.success("✅ ECG signal extracted from image")

            # Optional: Preview signal
            st.line_chart(signal[:1000])

        except Exception as e:
            st.error(f"⚠️ Image processing failed: {e}")

else:
    uploaded_csv = st.file_uploader("Upload ECG CSV", type=["csv"])
    if uploaded_csv:
        try:
            df = pd.read_csv(uploaded_csv)
            ecg_signal = df["ECG"]
            st.success("✅ ECG data loaded from CSV")
        except Exception as e:
            st.error(f"⚠️ CSV reading failed: {e}")

# Process ECG signal and extract features
if ecg_signal is not None:
    try:
        signals, info = nk.ecg_process(ecg_signal, sampling_rate=250)
        thalach = np.max(signals["ECG_Rate"])

        rpeaks = info["ECG_R_Peaks"]
        st_slope_values = []
        for r in rpeaks:
            try:
                start = r + int(0.05 * 250)
                end = r + int(0.15 * 250)
                if end < len(ecg_signal):
                    st_slope_values.append(ecg_signal[end] - ecg_signal[start])
            except:
                continue
        slope = np.mean(st_slope_values) if st_slope_values else 0
        slope_category = 1 if slope > 0.05 else (2 if slope < -0.05 else 0)

        baseline = np.mean(ecg_signal[:250])
        oldpeak_values = []
        for r in rpeaks:
            try:
                offset = r + int(0.1 * 250)
                if offset < len(ecg_signal):
                    oldpeak_values.append(baseline - ecg_signal[offset])
            except:
                continue
        oldpeak = np.mean(oldpeak_values) if oldpeak_values else 0

        st.markdown("### Extracted ECG Features")
        st.write(f"**Thalach (Max Heart Rate):** {thalach:.2f}")
        st.write(f"**Slope Category:** {slope_category}")
        st.write(f"**Oldpeak:** {oldpeak:.2f}")

        # User input form
        st.header("📝 Enter Additional Health Information")
        age = st.number_input("Age", 1, 120, 54)
        sex = st.selectbox("Sex", ["Male", "Female"])
        cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3])
        trestbps = st.number_input("Resting BP (mm Hg)", 80, 200, 130)
        chol = st.number_input("Cholesterol (mg/dl)", 100, 400, 250)
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
        restecg = st.selectbox("Resting ECG Result", [0, 1, 2])
        exang = st.selectbox("Exercise-Induced Angina", [0, 1])
        ca = st.selectbox("Number of Major Vessels (0–4)", [0, 1, 2, 3, 4])
        thal = st.selectbox("Thalassemia (3=Normal, 6=Fixed, 7=Reversible)", [3, 6, 7])

        # Prepare for prediction
        sex_binary = 1 if sex == "Male" else 0
        input_data = np.array([[age, sex_binary, cp, trestbps, chol, fbs, restecg,
                                thalach, exang, oldpeak, slope_category, ca, thal]])

        if st.button("🔍 Predict Risk"):
            prediction = model.predict(input_data)
            if prediction[0] == 1:
                st.error("🚨 High Risk of Heart Disease!")
            else:
                st.success("✅ Low Risk of Heart Disease.")

    except Exception as e:
        st.error(f"⚠️ Feature extraction or prediction failed: {e}")
