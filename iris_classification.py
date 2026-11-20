from sklearn.datasets import load_iris
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Load the Iris dataset
iris = load_iris()

# Display dataset description
print(iris.DESCR)

# Display feature names
print("\nFeature Names:")
print(iris.feature_names)

# Display target names
print("\nTarget Names:")
print(iris.target_names)

# Display the shape of the dataset
print("\nData Shape:")
print(iris.data.shape)

print("\nTarget Shape:")
print(iris.target.shape)

# Display first five observations
print("\nFirst Five Observations:")
print(iris.data[:5])

# Display first five target labels
print("\nFirst Five Target Labels:")
print(iris.target[:5])

# Convert the Iris dataset into a Pandas DataFrame
df = pd.DataFrame(
    data=iris.data,
    columns=iris.feature_names
)

# Add the target variable
df["species"] = iris.target

# Replace numerical labels with species names
df["species"] = df["species"].map({
    0: "setosa",
    1: "versicolor",
    2: "virginica"
})

print(df.head())
print(df)
# Examine the structure of the dataset
df.info()
# Generate descriptive statistics
print(df.describe())
# Check for missing values in each column
print(df.isnull().sum())
# Check the distribution of Iris species
print(df["species"].value_counts())
# Calculate the percentage distribution of species
print(df["species"].value_counts(normalize=True) * 100)

# Visualize the distribution of Iris species
sns.countplot(data=df, x="species")

# Add chart title
plt.title("Distribution of Iris Species")

# Label the x-axis
plt.xlabel("Species")

# Label the y-axis
plt.ylabel("Frequency")

# Display the chart
plt.show()

# Visualize the distribution of numerical features
df.iloc[:, :4].hist(figsize=(10, 8), bins=15)

# Add a main title
plt.suptitle("Distribution of Iris Numerical Features")

# Adjust layout
plt.tight_layout()

# Display the plots
plt.show()

# Boxplots for numerical features
plt.figure(figsize=(12, 8))

for i, column in enumerate(df.columns[:4], 1):
    plt.subplot(2, 2, i)
    sns.boxplot(y=df[column])
    plt.title(f"Boxplot of {column}")

plt.tight_layout()
plt.show()

# Scatter plot of petal length and petal width
plt.figure(figsize=(8, 6))

sns.scatterplot(
    data=df,
    x="petal length (cm)",
    y="petal width (cm)",
    hue="species",
    palette="viridis",
    s=80
)

plt.title("Petal Length vs Petal Width by Iris Species")
plt.xlabel("Petal Length (cm)")
plt.ylabel("Petal Width (cm)")
plt.legend(title="Species")
plt.show()

# Correlation analysis
correlation_matrix = df.iloc[:, :4].corr()

# Visualize correlation using a heatmap
plt.figure(figsize=(8, 6))

sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    linewidths=0.5
)

plt.title("Correlation Heatmap of Iris Features")
plt.show()

# Separate features and target variable

X = df.iloc[:, :4]  # Features
y = df["species"]   # Target

print("Features:")
print(X.head())

print("\nTarget:")
print(y.head())

# Split the dataset into training and testing sets

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Display the sizes of the datasets
print("Training features:", X_train.shape)
print("Testing features:", X_test.shape)
print("Training target:", y_train.shape)
print("Testing target:", y_test.shape)

# Create a Logistic Regression model

model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=200))
])

# Train the Logistic Regression model

model.fit(X_train, y_train)

print("Model training completed successfully!")
# Make predictions using the testing dataset

y_pred = model.predict(X_test)

print("\nPredicted species:")
print(y_pred)

# Evaluate model accuracy

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:")
print(accuracy)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
# Evaluate precision, recall, and F1-score

from sklearn.metrics import classification_report

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Generate confusion matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

# Visualize the confusion matrix

plt.figure(figsize=(8, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["setosa", "versicolor", "virginica"],
    yticklabels=["setosa", "versicolor", "virginica"]
)

plt.title("Confusion Matrix - Logistic Regression")
plt.xlabel("Predicted Species")
plt.ylabel("Actual Species")

plt.show()