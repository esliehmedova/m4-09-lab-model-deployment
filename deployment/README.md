# Iris Classifier API

## What the Model Does

This service exposes a trained **Random Forest classifier** (100 trees) that predicts the species of an Iris flower from four numeric measurements: sepal length, sepal width, petal length, and petal width (all in centimeters). The model was trained on the classic Iris dataset using an 80/20 train/test split and achieves high accuracy on the held-out test set. It returns one of three species — *setosa*, *versicolor*, or *virginica* — along with the predicted probability for each class. The service is implemented as a Flask HTTP API that loads the serialized model at startup and serves predictions over JSON.

## How to Run

1. Make sure you have the required dependencies installed:

2. Make sure `model.joblib` and `target_names.joblib` exist in the same folder as `app.py`. If not, run the training notebook first to generate them.

3. Start the server from a terminal:

4. The API will be available at `http://127.0.0.1:5000`.

5. To stop the server, press **Ctrl+C** in the terminal.

## API Specification

### `GET /health`
Health check endpoint used by load balancers and monitoring tools.

- **Request body:** none
- **Response (200):**
```json
  {"status": "healthy"}
```

### `POST /predict`
Predict the species for a single Iris sample.

- **Request body:**
```json
  {"features": [5.1, 3.5, 1.4, 0.2]}
```
  `features` must be a list of exactly 4 numeric values.

- **Response (200):**
```json
  {
    "predicted_class": "setosa",
    "probabilities": {
      "setosa": 1.0,
      "versicolor": 0.0,
      "virginica": 0.0
    }
  }
```

- **Response (400):** returned when input is invalid (missing key, wrong length, non-numeric values).
```json
  {"error": "Features must contain exactly 4 values."}
```

### `POST /predict_batch`
Predict species for multiple samples in one request.

- **Request body:**
```json
  {
    "samples": [
      [5.1, 3.5, 1.4, 0.2],
      [6.7, 3.0, 5.2, 2.3]
    ]
  }
```

- **Response (200):**
```json
  {
    "predictions": [
      {"predicted_class": "setosa",    "probabilities": {"setosa": 1.0, "versicolor": 0.0, "virginica": 0.0}},
      {"predicted_class": "virginica", "probabilities": {"setosa": 0.0, "versicolor": 0.05, "virginica": 0.95}}
    ]
  }
```

- **Response (400):** returned when `samples` is missing or any sample fails validation.

## Example Requests

### Using `curl`

```bash
# Health check
curl http://127.0.0.1:5000/health

# Single prediction
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [5.1, 3.5, 1.4, 0.2]}'

# Batch prediction
curl -X POST http://127.0.0.1:5000/predict_batch \
  -H "Content-Type: application/json" \
  -d '{"samples": [[5.1, 3.5, 1.4, 0.2], [6.7, 3.0, 5.2, 2.3]]}'
```

### Using Python `requests`

```python
import requests

# Health check
r = requests.get("http://127.0.0.1:5000/health")
print(r.json())

# Single prediction
r = requests.post(
    "http://127.0.0.1:5000/predict",
    json={"features": [5.1, 3.5, 1.4, 0.2]}
)
print(r.json())

# Batch prediction
r = requests.post(
    "http://127.0.0.1:5000/predict_batch",
    json={"samples": [[5.1, 3.5, 1.4, 0.2], [6.7, 3.0, 5.2, 2.3]]}
)
print(r.json())
``` 