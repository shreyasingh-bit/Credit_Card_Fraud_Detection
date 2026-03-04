from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# load model & scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

@app.route("/")
def home():
    return render_template("fraud-detection (1).html")  # MUST match templates/index.html

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    row = [float(data[col]) for col in FEATURE_COLS]
    arr = np.array(row).reshape(1, -1)
    arr = scaler.transform(arr)

    pred = int(model.predict(arr)[0])

    return jsonify({"prediction": pred})