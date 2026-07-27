import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Import Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# Load Dataset
df = pd.read_csv("diabetes.csv")

# Features and Target
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Models and Parameters
models = {
    "Logistic Regression": (
        LogisticRegression(max_iter=1000),
        {
            "C": [0.01, 0.1, 1, 10],
            "solver": ["liblinear"]
        }
    ),

    "Decision Tree": (
        DecisionTreeClassifier(random_state=42),
        {
            "criterion": ["gini", "entropy"],
            "max_depth": [3, 5, 7, 10]
        }
    ),

    "Random Forest": (
        RandomForestClassifier(random_state=42),
        {
            "n_estimators": [50, 100, 200],
            "max_depth": [5, 10, None]
        }
    ),

    "Support Vector Machine": (
        SVC(),
        {
            "C": [0.1, 1, 10],
            "kernel": ["linear", "rbf"]
        }
    ),

    "K-Nearest Neighbors": (
        KNeighborsClassifier(),
        {
            "n_neighbors": [3, 5, 7, 9]
        }
    )
}

print("=" * 60)
print("HYPERPARAMETER TUNING RESULTS")
print("=" * 60)

best_model = ""
best_accuracy = 0

for name, (model, params) in models.items():

    print(f"\n{name}")

    grid = GridSearchCV(
        estimator=model,
        param_grid=params,
        cv=5,
        scoring="accuracy",
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    y_pred = grid.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print("Best Parameters :", grid.best_params_)
    print("Accuracy        :", round(accuracy, 4))

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = name

print("\n" + "=" * 60)
print("BEST TUNED MODEL")
print("=" * 60)
print("Model    :", best_model)
print("Accuracy :", round(best_accuracy, 4))