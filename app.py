from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("model.joblib")
target_names = joblib.load("target_names.joblib")

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy"}), 200

def validate_features(features):
    if not isinstance(features, list):
        return "Features must be a list."
    if len(features) != 4:
        return "Features must contain exactly 4 values."
    for v in features:
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            return "All feature values must be numeric."
    return None

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)
    if not data or "features" not in data:
        return jsonify({"error": "Missing 'features' key"}), 400
    features = data["features"]
    error = validate_features(features)
    if error:
        return jsonify({"error": error}), 400
    X = np.array(features).reshape(1, -1)
    pred = model.predict(X)[0]
    probs = model.predict_proba(X)[0]
    return jsonify({
        "predicted_class": str(target_names[pred]),
        "probabilities": {
            str(name): round(float(p), 4)
            for name, p in zip(target_names, probs)
        }
    })

@app.route("/predict_batch", methods=["POST"])
def predict_batch():
    data = request.get_json(silent=True)
    if not data or "samples" not in data:
        return jsonify({"error": "Missing 'samples' key"}), 400
    samples = data["samples"]
    if not isinstance(samples, list):
        return jsonify({"error": "'samples' must be a list"}), 400
    results = []
    for sample in samples:
        error = validate_features(sample)
        if error:
            return jsonify({"error": error}), 400
        X = np.array(sample).reshape(1, -1)
        pred = model.predict(X)[0]
        probs = model.predict_proba(X)[0]
        results.append({
            "predicted_class": str(target_names[pred]),
            "probabilities": {
                str(name): round(float(p), 4)
                for name, p in zip(target_names, probs)
            }
        })
    return jsonify({"predictions": results})

if __name__ == "__main__":
    app.run(debug=False, port=5000)
