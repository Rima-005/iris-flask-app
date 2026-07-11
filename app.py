from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load trained model
model = joblib.load("model/model.pkl")
class_names = joblib.load("model/class_names.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    # Get values from form
    sepal_length = float(request.form["sepal_length"])
    sepal_width = float(request.form["sepal_width"])
    petal_length = float(request.form["petal_length"])
    petal_width = float(request.form["petal_width"])

    # Create input array
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

    # Prediction
    prediction = model.predict(features)[0]

    # Confidence
    probability = model.predict_proba(features)
    confidence = round(np.max(probability) * 100, 2)

    species = class_names[prediction]

    return render_template(
        "result.html",
        prediction=species.title(),
        confidence=confidence
    )


if __name__ == "__main__":
    app.run(debug=True)