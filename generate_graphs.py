import os

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix


# Create static folder if it doesn't exist
os.makedirs("public", exist_ok=True)


# Load Iris dataset
iris = load_iris()

df = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

df["species"] = iris.target_names[iris.target]


# 1. Species Distribution
plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="species",
    hue="species",
    palette="viridis",
    legend=False
)

plt.title("Distribution of Iris Species")
plt.xlabel("Species")
plt.ylabel("Frequency")

plt.tight_layout()

plt.savefig("public/species_distribution.png")
plt.close()


# 2. Histograms
df.iloc[:, :4].hist(
    figsize=(10, 8),
    bins=15
)

plt.suptitle("Distribution of Iris Numerical Features")
plt.tight_layout()

plt.savefig("public/feature_histograms.png")
plt.close()


# 3. Scatter Plot
plt.figure(figsize=(8, 6))

sns.scatterplot(
    data=df,
    x="petal length (cm)",
    y="petal width (cm)",
    hue="species",
    palette="viridis",
    s=80
)

plt.title("Petal Length vs Petal Width")
plt.xlabel("Petal Length (cm)")
plt.ylabel("Petal Width (cm)")

plt.tight_layout()

plt.savefig("public/petal_scatter.png")
plt.close()


# 4. Correlation Heatmap
correlation_matrix = df.iloc[:, :4].corr()

plt.figure(figsize=(8, 6))

sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Heatmap")

plt.tight_layout()

plt.savefig("public/correlation_heatmap.png")
plt.close()


# 5. Confusion Matrix
X = df.iloc[:, :4]
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=200))
])

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=iris.target_names,
    yticklabels=iris.target_names
)

plt.title("Confusion Matrix - Logistic Regression")
plt.xlabel("Predicted Species")
plt.ylabel("Actual Species")

plt.tight_layout()

plt.savefig("public/confusion_matrix.png")
plt.close()


print("All graphs have been generated successfully!")