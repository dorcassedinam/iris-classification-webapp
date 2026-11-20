from flask import Flask, render_template, request
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# Configure Flask to serve images from the public folder
app = Flask(
    __name__,
    static_folder="public",
    static_url_path=""
)


# Load Iris dataset
iris = load_iris()

# Create DataFrame
df = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

df["species"] = iris.target


# Separate features and target
X = df.iloc[:, :4]
y = df["species"]


# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Create machine learning pipeline
model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=200))
])


# Train model
model.fit(X_train, y_train)


# Make predictions
y_pred = model.predict(X_test)


# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)


# Classification report
report = classification_report(
    y_test,
    y_pred,
    target_names=iris.target_names,
    output_dict=True
)


# Confusion matrix
cm = confusion_matrix(y_test, y_pred)


# Home page
@app.route("/")
def home():

    return render_template(
        "index.html",
        accuracy=round(accuracy * 100, 2),
        report=report,
        species=iris.target_names
    )


# Prediction route
@app.route("/predict", methods=["POST"])
def predict():

    try:

        # Get measurements from form
        sepal_length = float(request.form["sepal_length"])
        sepal_width = float(request.form["sepal_width"])
        petal_length = float(request.form["petal_length"])
        petal_width = float(request.form["petal_width"])


        # Create input DataFrame
        input_data = pd.DataFrame(
            [[
                sepal_length,
                sepal_width,
                petal_length,
                petal_width
            ]],
            columns=iris.feature_names
        )


        # Predict species
        prediction = model.predict(input_data)[0]

        predicted_species = iris.target_names[prediction]


        # Prediction probabilities
        probabilities = model.predict_proba(input_data)[0]


        return render_template(
            "index.html",
            accuracy=round(accuracy * 100, 2),
            report=report,
            species=iris.target_names,
            prediction=predicted_species,
            probabilities=probabilities,
            input_values=request.form
        )


    except ValueError:

        return render_template(
            "index.html",
            accuracy=round(accuracy * 100, 2),
            report=report,
            species=iris.target_names,
            error="Please enter valid numerical measurements."
        )


# Run application locally
if __name__ == "__main__":

    app.run(debug=True)