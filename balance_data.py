import pandas as pd
from imblearn.over_sampling import SMOTE

# Load dataset
df = pd.read_csv("diabetes.csv")

# Separate features and target
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Apply SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Combine into new dataframe
balanced_df = pd.DataFrame(X_resampled, columns=X.columns)
balanced_df["Outcome"] = y_resampled

# Save new dataset
balanced_df.to_csv("diabetes_balanced.csv", index=False)

print("Balanced Dataset Created Successfully!\n")
print(balanced_df["Outcome"].value_counts())

