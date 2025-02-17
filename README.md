# **CPI Prediction with Ridge Regression**

This project implements a **Ridge Regression model** to predict the **Consumer Price Index (CPI)** based on economic indicators. The model has been fine-tuned using **GridSearchCV** and evaluated on real economic data.

---

## **Installation**
### **1. Clone the Repository**
```bash
git clone <repository-url>
cd cpi-prediction
```

### **2. Install Dependencies**
```bash
pip install -r requirements.txt
```

---

## **Usage**
### **1. Data Preprocessing**
Prepare and clean the dataset for model training:
```bash
python scripts/preprocess_data.py
```
- Loads raw CPI and macroeconomic data.
- Handles missing values and feature engineering.
- Saves the cleaned dataset to `./data/processed/feature_engineered_macro_data.csv`.

### **2. Model Training**
Train the Ridge Regression model using fine-tuned hyperparameters:
```bash
python scripts/train_model.py
```
- Uses **GridSearchCV** to find the optimal Ridge regression parameters.
- Saves the trained model to `./models/ridge_model.pkl`.
- Computes **Mean Absolute Error (MAE)** and **R² Score**.
- Generates plots comparing **Actual vs. Predicted CPI**.

### **3. Predict CPI**
Predict CPI based on new economic indicators:
```bash
python scripts/predict_cpi.py
```
- Takes user inputs for key economic factors.
- Outputs the predicted **Consumer Price Index (CPI)**.

---

## **Files and Scripts**
### **1. preprocess_data.py**
- Cleans and prepares macroeconomic data for modeling.
- Saves processed data to the `./data/processed/` folder.

### **2. train_model.py**
- Trains a **Ridge Regression model**.
- Uses **GridSearchCV** for hyperparameter tuning.
- Saves the trained model to `./models/ridge_model.pkl`.
- Evaluates the model’s performance on a test dataset.
- Outputs:
  - **MAE (Mean Absolute Error)**
  - **R² Score**
  - **Actual vs. Predicted CPI Scatter Plot**

### **3. predict_cpi.py**
- Takes user input for key economic indicators.
- Loads the trained model and predicts CPI.

---

## **Results**
### **Final Model Performance**
- **Fine-Tuned Ridge Alpha:** `1.5`
- **Mean Absolute Error (MAE):** `0.5444`
- **R² Score:** `0.9998`

### **Generated Plots**
- **Predicted vs. Actual CPI Scatter Plot**
- **Residual Analysis for Model Performance**

---

## **Dependencies**
- Python 3.12
- `scikit-learn` (Machine Learning)
- `pandas` (Data Handling)
- `numpy` (Numerical Computation)
- `matplotlib` (Visualization)
- `seaborn` (Enhanced Visualization)
- `joblib` (Model Saving and Loading)

To see the full list of dependencies, refer to `requirements.txt`.

---

## **Next Steps**
- Implement **Lasso Regression** for feature selection.
- Experiment with **Deep Learning** models for CPI prediction.
- Automate **data updates** using live economic feeds.
