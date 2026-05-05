import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import pickle

print("🚀 Training model using Site 1 data...")

# Load training data
df = pd.read_csv("site_1_train_data.csv")

# Handle missing values
df = df.fillna(df.mean())

# Features (IMPORTANT)
X = df[
    [
        'T_forecast',
        'q_forecast',
        'u_forecast',
        'v_forecast',
        'w_forecast',
        'NO2_satellite',
        'HCHO_satellite'
    ]
]

# Targets
y_no2 = df['NO2_forecast']
y_o3 = df['O3_forecast']

# Train models
model_no2 = RandomForestRegressor(n_estimators=20, max_depth=10)
model_o3 = RandomForestRegressor(n_estimators=20, max_depth=10)
model_no2.fit(X, y_no2)
model_o3.fit(X, y_o3)

# Save models
pickle.dump(model_no2, open("model_no2.pkl", "wb"))
pickle.dump(model_o3, open("model_o3.pkl", "wb"))

print("✅ Models trained successfully!")

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import joblib

df = pd.read_csv("data.csv")

X = df.drop("AQI", axis=1)
y = df["AQI"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lr = LinearRegression()
rf = RandomForestRegressor()

lr.fit(X_train, y_train)
rf.fit(X_train, y_train)

def evaluate(y_test, y_pred):
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    return mae, rmse, r2

lr_metrics = evaluate(y_test, lr.predict(X_test))
rf_metrics = evaluate(y_test, rf.predict(X_test))

print("Linear:", lr_metrics)
print("Random Forest:", rf_metrics)

# Save BEST model (assume RF is better)
joblib.dump(rf, "model.pkl")