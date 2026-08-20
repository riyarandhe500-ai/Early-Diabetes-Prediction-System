import pandas as pd
from sklearn.utils import resample

# Load dataset
df = pd.read_csv("diabetes.csv")

# Check original class distribution
print("Original Class Distribution:")
print(df["Outcome"].value_counts())

# Separate classes
df_majority = df[df["Outcome"] == 0]
df_minority = df[df["Outcome"] == 1]

# Oversample minority class
df_minority_upsampled = resample(
    df_minority,
    replace=True,
    n_samples=len(df_majority),
    random_state=42
)

# Combine both classes
df_balanced = pd.concat([df_majority, df_minority_upsampled])

# Shuffle dataset
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

# Check new class distribution
print("\nBalanced Class Distribution:")
print(df_balanced["Outcome"].value_counts())

# Save balanced dataset
df_balanced.to_csv("balanced_diabetes.csv", index=False)

print("\nBalanced dataset saved successfully!")