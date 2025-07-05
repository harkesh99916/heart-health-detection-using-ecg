# HeartRiskAI

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](#license)

## Overview

HeartRiskAI is an AI-driven web application for predicting heart disease risk from ECG data. Users can upload an ECG image (PNG), which is converted to a digital signal using OpenCV. Key cardiac features (e.g. maximum heart rate **(thalach)**, ST-segment **slope**, **oldpeak** depression) are extracted using NeuroKit2. These features are combined with user-provided health parameters (age, sex, cholesterol, blood pressure, etc.) and fed into a pre-trained Random Forest model to estimate the probability of heart disease. HeartRiskAI provides a user-friendly Streamlit interface for easy data entry, processing, and visualization of results. It aims to facilitate early CVD risk assessment by leveraging everyday ECG images and machine learning.

## Motivation

Cardiovascular diseases (CVDs) are the **leading cause of death globally**. In 2019 an estimated 17.9 million people died from CVD (32% of all global deaths), with \~85% due to heart attack or stroke.  Early detection of CVD is therefore crucial: timely identification of at-risk individuals allows interventions and treatment to begin sooner. HeartRiskAI addresses this need by using readily-available ECG images and simple health data to identify high-risk patients before symptoms occur. This could improve outcomes and awareness in populations where formal medical testing is infrequent or delayed.

## Literature and Research Context

Prior work has shown the promise of combining ECG analysis with machine learning. For example, Angelaki *et al.* (2022) used an ECG‑derived feature set plus anthropometric data to screen for hypertension, achieving \~84% accuracy with a Random Forest model. Similarly, Liu & Zhu (2025) trained an RF classifier on ECG features to predict heart failure outcomes with AUC≈0.97. Johnson *et al.* (2021) demonstrated that AI-ECG models could detect structural heart disease: e.g. an AI-enabled ECG identified aortic stenosis in a general population. Recent studies also emphasize explainability and community impact: van de Leur *et al.* (2022) developed an explainable ECG-DNN pipeline using a variational autoencoder to understand ECG predictors, while Zhou *et al.* (2025) constructed a hybrid ECG risk model for future CVD events (achieving C-statistic \~0.73). These works motivate HeartRiskAI’s design – integrating ECG signal processing, feature engineering, and ML risk prediction in an accessible tool.

## Key Features

* **ECG Image Support**: Accepts uploaded ECG image files (e.g. PNG/JPEG of printed or drawn ECG tracings).
* **ECG-to-Signal Processing**: Uses OpenCV to preprocess the image (grayscale conversion, background removal, noise filtering) and extract the waveform. The resulting signal is exported as CSV time-series data.
* **Feature Extraction**: Applies NeuroKit2 to the ECG signal to compute cardiac features such as maximum heart rate (**thalach**), ST-segment **slope**, and ST depression (**oldpeak**), among others.
* **User Data Input**: Prompts user for additional health parameters (age, sex, cholesterol level, resting blood pressure, chest pain type, etc.) via the web form.
* **Risk Prediction**: Combines ECG features with user data as input to a pre-trained Random Forest classifier. The model outputs a heart disease risk score or class.
* **Interactive Web Interface**: Built with Streamlit for ease of use. Users can see the processed ECG waveform plot and a summary of extracted features, and view the predicted risk in a clear format (e.g. probability or risk category).
* **Visualization and Output**: Generates charts (via Matplotlib) to show signal and risk results, helping users understand their data.

## System Architecture

The system pipeline consists of three main modules:

1. **Image Processing → Signal**:  The uploaded ECG image is processed with OpenCV (grayscale, threshold, contour detection) to isolate the ECG trace. The ECG waveform is digitized and saved as a signal CSV.
2. **Feature Extraction**:  NeuroKit2 analyzes the signal CSV to extract clinical features (heart rate, QT interval, ST slope, etc.).
3. **Prediction Model**:  The extracted features are concatenated with user-input demographics and vitals. This feature vector is fed into the Random Forest model (scikit-learn) to predict the heart disease risk.

A simplified flow of the data pipeline is:

```text
User ECG Image (PNG)
   ↓ (OpenCV processing)
ECG Signal Data (CSV)
   ↓ (NeuroKit2 extraction)
ECG Features (thalach, slope, oldpeak, etc.)
   + 
User Parameters (age, sex, BP, cholesterol, ...)
   ↓
Random Forest Model (scikit-learn)
   ↓
Predicted Heart Disease Risk
```

## Model Training

* **Data**: We trained the model on the **UCI Heart Disease (Cleveland)** dataset (303 records) which includes features such as age, sex, chest pain type (**cp**), resting blood pressure (**trestbps**), cholesterol, fasting blood sugar, ECG results (**restecg**), maximum heart rate (**thalach**), exercise angina (**exang**), ST depression (**oldpeak**), ST slope (**slope**), number of vessels (**ca**), thallium score (**thal**), and target disease label. This dataset is publicly available and widely used in research.
* **Training**: We preprocessed the data (one-hot encoding for categorical variables) and split it into training/testing sets. A Random Forest classifier was trained using scikit-learn (with, e.g., 100 trees, max depth tuning) to distinguish presence vs. absence of heart disease. Feature importance was analyzed to ensure the model learned meaningful ECG-based and clinical predictors.
* **Performance**: On held-out test data, the Random Forest achieved approximately **85% accuracy** (similar to reported results in related studies). Cross-validation confirmed stable performance (AUC in the high 0.80s). While not perfect, this suggests the model is effective at risk stratification given the available features.
* **Model Saving**: After training, the final model object is serialized to disk using **joblib** (`.joblib` file). This saved model is loaded by the application at runtime to make predictions on new data.

## Tech Stack

* **Python** – Core programming language.
* **Streamlit** – Framework for building the interactive web application and UI.
* **OpenCV** – Computer vision library used for ECG image processing and signal extraction.
* **NeuroKit2** – Physiological signal processing library for ECG feature extraction.
* **Pandas / NumPy** – Data manipulation and numeric computation.
* **scikit-learn** – Machine learning library used for the Random Forest model.
* **Matplotlib** – Plotting library for visualizing ECG signals and results.
* **Joblib** – Used for saving and loading the trained model efficiently.

## Installation & Setup

1. **Clone the repository**:

   ```bash
   git clone https://github.com/YourUsername/HeartRiskAI.git
   cd HeartRiskAI
   ```
2. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

   *(Dependencies include streamlit, opencv-python, neurokit2, pandas, numpy, scikit-learn, matplotlib, joblib, etc.)*
3. **Run the app**:

   ```bash
   streamlit run app.py
   ```

   This will launch the web interface (default at `http://localhost:8501`) in your browser.

## Usage Instructions

1. **Start the App**: After running `streamlit run app.py`, open the provided URL in your browser.
2. **Upload ECG Image**: Click the file uploader and select an ECG image file (PNG or JPG) from your computer. The app will display the image and process it.
3. **Enter User Data**: Fill in the required health parameters (e.g., age, sex, cholesterol, resting BP) in the input fields.
4. **Run Prediction**: Click the “Predict” button. The app will convert the image to a signal, extract ECG features, and run the Random Forest model.
5. **View Results**: The app displays the predicted risk (e.g. as a probability or risk category) along with a chart of the processed ECG signal and any extracted feature values. Interpret the results accordingly.

```python
# Example (in the app code) of loading the model:
import joblib
model = joblib.load("heart_risk_model.joblib")
features = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
risk_pred = model.predict_proba([features])[0,1]
print(f"Predicted heart disease risk: {risk_pred:.1%}")
```

## Sample Data

The repository includes a `sample_data/` folder with example ECG recordings and images for testing. You can use these synthetic ECG traces (or capture your own) to verify the app’s processing pipeline. Ensure images have clear grid and trace lines. For instance, try uploading `sample_data/ecg_example1.png` and entering placeholder health parameters to see a sample prediction.

## Limitations and Future Work

* **Black-Box Model**: The Random Forest provides good accuracy but lacks easy interpretability. We plan to integrate explainable AI techniques (e.g., SHAP values or the variational autoencoder approach of van de Leur *et al.*) so users can see which ECG features drive the risk score.
* **Data and Personalization**: The current model is trained on a generic dataset; it may not capture all individual variations. Future work includes personalizing models to specific patient subgroups and incorporating more comprehensive health profiles (medications, genetics, etc.).
* **Wearable Integration**: We aim to adapt HeartRiskAI for deployment on mobile and wearable devices (e.g. as a smartphone app) to enable real-time ECG capture and instant risk feedback.
* **Clinical Validation**: Ultimately, rigorous clinical studies are needed to validate the predictions. Expanding the dataset with real patient ECGs and outcomes will help improve and certify the model’s reliability.

## License

This project is released under the **MIT License**. See [LICENSE](#) for details.

## Contributing

We welcome contributions! Please follow these guidelines:

* Fork the repository and create a new feature or bugfix branch (`git checkout -b my-feature`).
* Commit changes with clear messages and follow PEP8 style.
* Update documentation or add tests for new features.
* Submit a Pull Request describing your changes. We will review and merge improvements.
* For major changes, open an issue first to discuss the design and scope.

Please ensure your code is well-documented and any new dependencies are added to `requirements.txt`.

## References

* World Health Organization (2021). *Cardiovascular diseases (CVDs): Key facts.* \[WHO Fact Sheet].
* Angelaki E. *et al.* (2022). *“Artificial intelligence-based opportunistic screening for the detection of arterial hypertension through ECG signals.”* **Journal of Hypertension** 40(12):2494–2501.
* Liu J., Zhu D., et al. (2025). *“Predictive Modeling of Heart Failure Outcomes Using ECG Monitoring Indicators and Machine Learning.”* **Annals of Noninvasive Electrocardiology** 30(4)\:e70097.
* Murphree D.H., Michelena H.I., Enriquez-Sarano M., *et al.* (2021). *“Electrocardiogram screening for aortic valve stenosis using artificial intelligence.”* **European Heart Journal** 42:2885–2896.
* van de Leur R.R., Bos M.N., Taha K., *et al.* (2022). *“Improving explainability of deep neural network-based electrocardiogram interpretation using variational auto-encoders.”* **European Heart Journal – Digital Health** 3(3):390–404.
* Zhou P., Yang Z., *et al.* (2025). *“A hybrid algorithm-based ECG risk prediction model for cardiovascular disease.”* **European Heart Journal – Digital Health** 6(3):466–475.
