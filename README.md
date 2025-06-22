# 🏦 Churn ANN Classification & Regression

A comprehensive machine learning project for **customer churn prediction** and **salary regression** using Artificial Neural Networks (ANN). Features data preprocessing, model training, hyperparameter tuning, TensorBoard monitoring, and Streamlit deployment.

---

## Features

- **Dual ANN Models**: Classification (churn prediction) + Regression (salary prediction)
- **Hyperparameter Tuning**: GridSearchCV optimization with cross-validation
- **TensorBoard Integration**: Real-time training visualization and monitoring
- **Early Stopping**: Prevents overfitting with patience-based callbacks
- **Streamlit Web App**: Interactive prediction interface
- **End-to-End Pipeline**: Data preprocessing, encoding, scaling, training, deployment

---

## 📂 Repository Structure

| File/Folder | Purpose |
|-------------|---------|
| **logs/** | Classification training logs & TensorBoard data |
| **regressionlogs/** | Regression training logs & TensorBoard data |
| **Churn_Modelling.csv** | Main customer dataset |
| **app.py** | Streamlit web application |
| **experiments.ipynb** | Model development & training |
| **hyperparametertuningann.ipynb** | GridSearchCV optimization |
| **prediction.ipynb** | Model inference & evaluation |
| **salaryregression.ipynb** | Regression model experiments |
| **model.h5** | Trained classification ANN |
| **regression_model.h5** | Trained regression ANN |
| **label_encoder_gender.pkl** | Gender label encoder |
| **onehot_encoder_geo.pkl** | Geography one-hot encoder |
| **scaler.pkl** | Feature StandardScaler |

---

## ⚙️ Technical Implementation

### 🏗️ Classification Model Architecture

model = Sequential([
Dense(64, activation='relu', input_shape=(X_train.shape,)),
Dense(32, activation='relu'),
Dense(1, activation='sigmoid') # Binary classification
])

Compilation
model.compile(
optimizer=Adam(learning_rate=0.01),
loss='binary_crossentropy',
metrics=['accuracy']
)


**Model Summary:**
- **Total Parameters**: 2,945 (11.50 KB)
- **Layers**: 3 Dense layers (64→32→1 neurons)
- **Activation**: ReLU (hidden), Sigmoid (output)

### 🏗️ Regression Model Architecture

model = Sequential([
Dense(64, activation='relu', input_shape=(X_train.shape,)),
Dense(32, activation='relu'),
Dense(1) # Regression output
])

Compilation
model.compile(
optimizer='adam',
loss='mean_absolute_error',
metrics=['mae']
)


### 🔧 Training Configuration

Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
tensorboard = TensorBoard(log_dir="logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S"))

Training
history = model.fit(
X_train, y_train,
validation_data=(X_test, y_test),
epochs=100,
callbacks=[early_stopping, tensorboard]
)


---

## 📊 Model Performance

### Classification Results
| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
|-------|------------|-----------|----------|---------|
| 1 | 0.3893 | 83.64% | 0.3501 | 85.70% |
| 8 | 0.3355 | 86.08% | 0.3397 | 86.30% |
| 18 | 0.3133 | 86.76% | 0.3594 | 86.05% |

**Final Performance**: 86.76% training accuracy, 86.05% validation accuracy

### Hyperparameter Tuning Results

Best Score: 85.67%
Best Parameters: {'epochs': 50, 'layers': 1, 'neurons': 128}
Grid Search: 48 combinations tested with 3-fold cross-validation

### 📈 Regression Results
| Metric | Training | Validation |
|--------|----------|------------|
| **MAE** | 49,500 | 50,168 |
| **Loss** | 49,563 | 50,235 |

---

## Getting Started

### 📋 Prerequisites
- Python 3.7+
- TensorFlow 2.x
- Streamlit
- scikit-learn

### Installation

pip install -r requirements.txt

#### Launch Streamlit app
streamlit run app.py

#### View TensorBoard logs
tensorboard --logdir logs/fit
---

User inputs
geography = st.selectbox('Geography', ['France', 'Germany', 'Spain'])
gender = st.selectbox('Gender', ['Male', 'Female'])
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')

Prediction
prediction = model.predict(input_data_scaled)
if prediction > 0.5:
st.write('Customer likely to churn')
else:
st.write('Customer likely to stay')

Example prediction
input_data = [[619, 1, 42, 2, 0.00, 1, 1, 1, 101348.88, 1, 0, 0]]
prediction = model.predict(scaler.transform(input_data))

Output: 0.07 (7% churn probability)


---

## 📈 Data Pipeline

### Preprocessing Steps
1. **Label Encoding**: Gender (Male/Female → 0/1)
2. **One-Hot Encoding**: Geography (France/Germany/Spain → binary vectors)
3. **Feature Scaling**: StandardScaler normalization
4. **Train-Test Split**: 80/20 validation strategy

### Feature Engineering

Gender encoding
label_encoder_gender.fit_transform(data['Gender'])

Geography encoding
onehot_encoder_geo.fit_transform(data[['Geography']])

Output: ['Geography_France', 'Geography_Germany', 'Geography_Spain']
Feature scaling
scaler.fit_transform(features)

## 🎯 Key Insights

- **Best Model**: 128 neurons, 1 hidden layer, 50 epochs
- **Accuracy**: 86.76% on training, 86.05% on validation
- **Balance**: Well-balanced precision/recall performance
- **Deployment**: Real-time predictions via Streamlit interface
- **Monitoring**: TensorBoard integration for training visualization

---



