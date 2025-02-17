import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.ensemble import RandomForestRegressor

file_path = "data/processed/feature_engineered_macro_data.csv"
data = pd.read_csv(file_path)

data["Date"] = pd.to_datetime(data["Year"].astype(str) + "-" + data["Month"], format="%Y-%B")
data = data.sort_values("Date")  

month_mapping = {month: index for index, month in enumerate([
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"
], start=1)}
data["Month"] = data["Month"].map(month_mapping)

data = data.drop(columns=["Date"])

X = data.drop(columns=["CPI_Index"])
y = data["CPI_Index"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Data loaded and preprocessed successfully.")

models = {
    "Linear Regression": Ridge(alpha=1.0),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
}

results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    results.append((name, mae, r2))
    print(f"{name} - MAE: {mae:.4f}, R² Score: {r2:.4f}")

results_df = pd.DataFrame(results, columns=["Model", "MAE", "R² Score"])
print(results_df)

print("Initial model testing completed.")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error, r2_score

file_path = "data/processed/feature_engineered_macro_data.csv"
data = pd.read_csv(file_path)

data["Date"] = pd.to_datetime(data["Year"].astype(str) + "-" + data["Month"], format="%Y-%B")
data = data.sort_values("Date") 

month_mapping = {month: index for index, month in enumerate([
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"
], start=1)}
data["Month"] = data["Month"].map(month_mapping)

X = data.drop(columns=["CPI_Index", "Date"])
y = data["CPI_Index"] 

scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

fine_tuned_alphas = np.linspace(1.5, 3.0, 15)

param_grid_fine = {
    "alpha": fine_tuned_alphas,
    "solver": ["auto", "sparse_cg"] 
}

grid_search_fine = GridSearchCV(Ridge(), param_grid_fine, scoring="neg_mean_absolute_error", cv=5, n_jobs=-1)
grid_search_fine.fit(X_train, y_train)

best_ridge_fine = grid_search_fine.best_estimator_
best_alpha_fine = grid_search_fine.best_params_["alpha"]

print(f" Fine-Tuned Ridge Alpha: {best_alpha_fine}")

best_ridge_fine.fit(X_train, y_train)

y_pred_fine = best_ridge_fine.predict(X_test)

mae_fine = mean_absolute_error(y_test, y_pred_fine)
r2_fine = r2_score(y_test, y_pred_fine)

print(f"Fine-Tuned Ridge MAE: {mae_fine:.4f}")
print(f"Fine-Tuned Ridge R² Score: {r2_fine:.4f}")

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_fine, alpha=0.6, color="blue", label="Predicted vs Actual")
plt.plot(y_test, y_test, color="red", linestyle="dashed", label="Perfect Fit")
plt.xlabel("Actual CPI")
plt.ylabel("Predicted CPI")
plt.title(f"Fine-Tuned Ridge Regression - Actual vs Predicted CPI")
plt.legend()
plt.show()

fine_tuned_ridge = Ridge(alpha=best_alpha_fine)
fine_tuned_ridge.fit(X_train, y_train)

y_pred = fine_tuned_ridge.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f" Fine-Tuned Ridge Alpha: {best_alpha_fine}")
print(f" Fine-Tuned Ridge MAE: {mae:.4f}")
print(f" Fine-Tuned Ridge R² Score: {r2:.4f}")

joblib.dump(fine_tuned_ridge, "ridge_model.pkl")

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.6, color="blue", label="Predicted vs Actual")
plt.plot(y_test, y_test, color="red", linestyle="dashed", label="Perfect Fit")
plt.xlabel("Actual CPI")
plt.ylabel("Predicted CPI")
plt.title("Fine-Tuned Ridge Regression - Actual vs Predicted CPI")
plt.legend()
plt.show()

print("Model training, evaluation, and saving completed.")