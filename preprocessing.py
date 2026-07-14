import pandas as pd

# Load dataset
df = pd.read_csv("diabetes.csv.csv")

# Display first 5 rows
print(df.head())

# Display dataset shape
print(df.shape)

# Display dataset information
print(df.info())
# Display statistical summary
print(df.describe())
# Check missing values
print(df.isnull().sum())
# Check duplicate rows
print(df.duplicated().sum())
# Check zero values in important columns
print("Glucose:", (df["Glucose"] == 0).sum())
print("BloodPressure:", (df["BloodPressure"] == 0).sum())
print("SkinThickness:", (df["SkinThickness"] == 0).sum())
print("Insulin:", (df["Insulin"] == 0).sum())
print("BMI:", (df["BMI"] == 0).sum())
import numpy as np

# Replace 0 values with NaN
columns = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

df[columns] = df[columns].replace(0, np.nan)

# Check missing values after replacement
print(df.isnull().sum())

# Fill missing values using median

df["Glucose"] = df["Glucose"].fillna(df["Glucose"].median())
df["BloodPressure"] = df["BloodPressure"].fillna(df["BloodPressure"].median())
df["SkinThickness"] = df["SkinThickness"].fillna(df["SkinThickness"].median())
df["Insulin"] = df["Insulin"].fillna(df["Insulin"].median())
df["BMI"] = df["BMI"].fillna(df["BMI"].median())

print(df.isnull().sum())

import matplotlib.pyplot as plt

# Boxplot for detecting outliers
df.boxplot(figsize=(12,6))
plt.title("Boxplot of Pima Diabetes Dataset")
plt.show()

