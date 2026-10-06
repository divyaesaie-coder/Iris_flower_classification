# ============================================================
# IRIS FLOWER CLASSIFICATION
# QSkill Internship
# Domain: Artificial Intelligence & Machine Learning
# ============================================================


# 1. IMPORT LIBRARIES
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# 2. LOAD THE IRIS DATASET
# ============================================================

iris = load_iris()

print("Iris dataset loaded successfully!")


# ============================================================
# 3. CREATE A DATAFRAME
# ============================================================

df = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

# Add the target
df["species"] = iris.target


# ============================================================
# 4. DISPLAY BASIC INFORMATION
# ============================================================

print("\nFirst 5 rows of the dataset:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nFeature names:")
print(iris.feature_names)

print("\nTarget names:")
print(iris.target_names)


# ============================================================
# 5. CHECK FOR MISSING VALUES
# ============================================================

print("\nMissing values:")
print(df.isnull().sum())


# ============================================================
# 6. CHECK CLASS DISTRIBUTION
# ============================================================

print("\nClass distribution:")
print(df["species"].value_counts())


# ============================================================
# 7. CONVERT TARGET NUMBERS INTO SPECIES NAMES
# ============================================================

df["species_name"] = df["species"].map({
    0: "Setosa",
    1: "Versicolor",
    2: "Virginica"
})


# ============================================================
# 8. VISUALIZE THE DATA
# ============================================================

sns.pairplot(
    df,
    hue="species_name"
)

plt.suptitle(
    "Iris Flower Feature Relationships",
    y=1.02
)

plt.show()


# ============================================================
# 9. DEFINE FEATURES (X) AND TARGET (y)
# ============================================================

X = df[iris.feature_names]

y = df["species"]


# ============================================================
# 10. SPLIT DATA INTO TRAINING AND TESTING SETS
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining data size:", X_train.shape)
print("Testing data size:", X_test.shape)


# ============================================================
# 11. CREATE THE LOGISTIC REGRESSION MODEL
# ============================================================

model = LogisticRegression(
    max_iter=200
)


# ============================================================
# 12. TRAIN THE MODEL
# ============================================================

model.fit(
    X_train,
    y_train
)

print("\nModel training completed!")


# ============================================================
# 13. MAKE PREDICTIONS
# ============================================================

y_pred = model.predict(X_test)

print("\nPredictions:")
print(y_pred)


# ============================================================
# 14. CALCULATE ACCURACY
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\nAccuracy:", accuracy)
print("Accuracy percentage:", accuracy * 100, "%")


# ============================================================
# 15. CALCULATE PRECISION
# ============================================================

precision = precision_score(
    y_test,
    y_pred,
    average="weighted"
)

print("\nPrecision:", precision)


# ============================================================
# 16. CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=iris.target_names
    )
)


# ============================================================
# 17. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\nConfusion Matrix:")
print(cm)


# ============================================================
# 18. DISPLAY CONFUSION MATRIX
# ============================================================

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=iris.target_names,
    yticklabels=iris.target_names
)

plt.xlabel("Predicted Species")
plt.ylabel("Actual Species")
plt.title("Confusion Matrix - Iris Classification")

plt.show()
