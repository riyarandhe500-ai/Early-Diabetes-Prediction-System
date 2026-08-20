import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("diabetes.csv")

# Display first 5 rows
print(df.head())

# Dataset information
print(df.info())

# Statistical summary
print(df.describe())

# Diabetic and Non-Diabetic count
print(df["Outcome"].value_counts())

# Check whether the dataset is balanced
print("\nClass Distribution:")
print(df["Outcome"].value_counts())

print("\nClass Distribution Percentage:")
print(df["Outcome"].value_counts(normalize=True) * 100)

# ----------------------------
# Graph 1: Outcome Count
# ----------------------------
sns.countplot(x="Outcome", data=df)
plt.title("Diabetic vs Non-Diabetic Patients")
plt.show()

# ----------------------------
# Graph 2: Correlation Heatmap
# ----------------------------
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()
# -----------------------------
# Graph 3: Histograms
# -----------------------------

df.hist(figsize=(12, 10))
plt.suptitle("Distribution of Features")
plt.show()
# ------------------------------
# Graph 4: Boxplots
# ------------------------------

plt.figure(figsize=(15,10))

for i, column in enumerate(df.columns[:-1], 1):
    plt.subplot(3,3,i)
    sns.boxplot(y=df[column])
    plt.title(column)

plt.tight_layout()
plt.show()
# ------------------------------
# Graph 5: Pair Plot
# ------------------------------

sns.pairplot(
    df[["Glucose", "BMI", "Age", "Outcome"]],
    hue="Outcome"
)

plt.show()