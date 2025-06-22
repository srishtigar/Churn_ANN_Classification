# Churn ANN Classification

A project for predicting customer churn using an **Artificial Neural Network (ANN)**, featuring data preprocessing, model training, evaluation, and deployment.

---

## 📂 Repository Structure

- **logs/** – Training logs  
- **regressionlogs/** – Regression experiment logs  
- **Churn_Modelling.csv** – Main dataset  
- **README.md** – Project documentation  
- **app.py** – Streamlit web app for deployment  
- **experiments.ipynb** – Model development and EDA  
- **hyperparametertuningann.ipynb** – Hyperparameter tuning  
- **prediction.ipynb** – Model inference and prediction  
- **salaryregression.ipynb** – Additional regression experiments  
- **model.h5** – Trained ANN model  
- **regression_model.h5** – Trained regression model  
- **label_encoder_gender.pkl** – Saved label encoder for gender  
- **onehot_encoder_geo.pkl** – Saved one-hot encoder for geography  
- **scaler.pkl** – Saved scaler for normalization  
- **requirements.txt** – Python dependencies  

---

## Features

- **ANN-based churn prediction**
- **End-to-end ETL pipeline**: data cleaning, encoding, scaling, splitting
- **Model checkpointing and early stopping**
- **Interactive Streamlit web app for predictions**
- **Comprehensive evaluation:** accuracy, precision, recall, F1-score, confusion matrix

---

## Getting Started

### **Prerequisites**
- Python 3.7+  
- Jupyter Notebook 
- Streamlit (for app deployment)  

### **Installation**

pip install -r requirements.txt


---

## Workflow

1. **Data Loading**
    - Load `Churn_Modelling.csv`
2. **Data Preprocessing**
    - Handle missing values
    - Encode categorical variables (`LabelEncoder`, `OneHotEncoder`)
    - Scale features (`StandardScaler`)
    - Train-test split
3. **Model Building**
    - Build ANN with TensorFlow/Keras (input, hidden, output layers)
4. **Model Training**
    - Compile with Adam optimizer and binary crossentropy loss
    - Train with early stopping and validation split
5. **Model Evaluation**
    - Evaluate on test data
    - Metrics: accuracy, precision, recall, F1-score, confusion matrix
6. **Hyperparameter Tuning**
    - Experiment with layers, activations, optimizers
7. **Deployment**
    - Deploy via `app.py` Streamlit app for real-time predictions

---

## Usage

- **Run Jupyter Notebooks:**
   
  Open and execute `experiments.ipynb`, `hyperparametertuningann.ipynb`, etc.

- **Run the Streamlit App:**
  
  streamlit run app.py
  
  
