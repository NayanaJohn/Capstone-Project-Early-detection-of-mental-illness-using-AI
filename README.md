# Capstone Project 2 - Early Detection Of Mental Illness Using AI
This project is an AI-powered web application designed to *predict mental health diagnosis risk levels* based on user input using a trained machine learning model. It includes data preprocessing, synthetic data generation for imbalanced classes, model training, and a Flask-based REST API to serve predictions.

---

## 🚀 Features

- 🧪 Predicts the likelihood of a mental health disorder based on user input.
- 📊 Classifies risk as *Low, **Medium, or **High*.
- ⚙️ Integrated *machine learning pipeline* for preprocessing and prediction.
- 🔁 Handles *class imbalance* using *CTGAN synthetic data generation*.
- 🌐 Web interface using Flask.

### Data Source
- The dataset contains demographic, professional, and mental health-related survey data.
- Target column: Mental_Health_Diagnosis (Binary: 0 = No, 1 = Yes)

### Preprocessing
- Handled missing values, encoded categorical variables, and selected important features.
- Used a custom class MentalHealthModel with:
  - preprocess_mental_health_data()
  - predict()
  - predict_proba()

### Addressing Imbalanced Data
- Used *CTGANSynthesizer* from SDV (Synthetic Data Vault) to generate:
  - 3,000 synthetic samples of the minority class
  - 7,000 samples of the majority class (real or synthetic)
- Combined into a balanced dataset (imbalanced_3_7_dataset.csv) with a 30:70 class ratio.

## ⚙️ How It Works
1. *User Input* is collected from the frontend or POSTed via API.
2. Input is *preprocessed* and aligned to the model’s feature space.
3. Model returns:
   - prediction (0 or 1)
   - probability (e.g., 0.82)
   - risk_level (Low, Medium, High)

## 🧰 Tech Stack
- *Frontend:* HTML, CSS (Optional JS or Bootstrap)
- *Backend:* Python, Flask
- *ML Model:* Scikit-learn, Pickle
- *Data:* CSV, Pandas, SDV for CTGAN
- *Other:* Flask-CORS

## ⚙️ Installation

### Prerequisites
- Python 3.8+

### Acknowledgements
SDV: Synthetic Data Vault

OSMI Mental Health Survey

Flask, Scikit-learn

### CONTRIBUTING
Contributions are welcome! Feel free to open issues or pull requests for any improvements or new features you would like to see
