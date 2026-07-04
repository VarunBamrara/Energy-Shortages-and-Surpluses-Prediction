from flask import Flask, render_template, request
import joblib
import numpy as np

# Load the model
model = joblib.load("random_forest_model.pkl")

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        try:
            features = []
            for i in range(1, 11):
                value = request.form.get(f"feature{i}")
                if value is None or value.strip() == "":
                    raise ValueError(f"Missing value for feature{i}")
                features.append(float(value))
            prediction = model.predict([features])[0]
        except Exception as e:
            prediction = f"Error: {str(e)}"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
