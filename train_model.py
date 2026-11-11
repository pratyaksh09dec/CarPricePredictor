import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (r2_score,mean_absolute_error,mean_squared_error)

# ==========================
# Load Cleaned Dataset
# ==========================

df = pd.read_csv("Cleaned_Data.csv")

print("Dataset Shape:", df.shape)

# ==========================
# Feature Encoding
# ==========================

df_encoded = pd.get_dummies(df,columns=['name','company','fuel_type'],drop_first=True)

# Features and Target

X = df_encoded.drop('Price',axis=1)
y = df_encoded['Price']

# Save feature columns

model_columns = X.columns

# ==========================
# Train Test Split
# ==========================

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# ==========================
# Train Model
# ==========================

rf_model = RandomForestRegressor(n_estimators=300,random_state=42)

rf_model.fit(X_train,y_train)

# ==========================
# Model Evaluation
# ==========================

y_pred = rf_model.predict(X_test)

r2 = r2_score(y_test, y_pred)

mae = mean_absolute_error(y_test,y_pred)

mse = mean_squared_error(y_test,y_pred)

rmse = np.sqrt(mse)

print("\nModel Performance")
print("-" * 40)
print("R² Score :", r2)
print("MAE      :", mae)
print("MSE      :", mse)
print("RMSE     :", rmse)

# ==========================
# Save Model Files
# ==========================

with open('car_price_model.pkl', 'wb') as f:
    pickle.dump(rf_model, f)

with open('model_columns.pkl', 'wb') as f:
    pickle.dump(model_columns, f)

print("\nFiles Created Successfully")
print("car_price_model.pkl")
print("model_columns.pkl")