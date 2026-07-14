import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

# Load dataset
df = pd.read_csv("diabetes.csv.csv")

# Separate features and target
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save scaler
joblib.dump(scaler, "scaler.pkl")

print("Feature Scaling Completed Successfully!")