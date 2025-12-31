# API Documentation

## Heart Disease Prediction API

**Base URL (Local)**: `http://localhost:8000`  
**Base URL (Cloud)**: `https://heart-disease-api-sdgp.onrender.com`

---

## Endpoints Overview

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | API information |
| GET | `/health` | Health check |
| GET | `/model-info` | Model metadata |
| POST | `/predict` | Make prediction |
| GET | `/metrics` | Prometheus metrics |
| GET | `/docs` | Swagger UI |

---

## 1. Root Endpoint

### `GET /`

Returns basic API information.

**Response:**
```json
{
  "name": "Heart Disease Prediction API",
  "version": "1.0.0",
  "status": "running",
  "docs": "/docs",
  "health": "/health"
}
```

---

## 2. Health Check

### `GET /health`

Returns API health status and model loading state.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "preprocessor_loaded": true,
  "uptime_seconds": 145.23,
  "timestamp": "2025-12-31T01:52:50.888383"
}
```

**Status Codes:**
- `200 OK` - API is healthy
- `503 Service Unavailable` - Model not loaded

---

## 3. Model Information

### `GET /model-info`

Returns model metadata and performance metrics.

**Response:**
```json
{
  "model_type": "RandomForestClassifier",
  "version": "1.0.0",
  "features": ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"],
  "metrics": {
    "accuracy": 0.8167,
    "roc_auc": 0.9152
  }
}
```

---

## 4. Prediction Endpoint

### `POST /predict`

Make a heart disease prediction based on patient health data.

**Request Headers:**
```
Content-Type: application/json
```

**Request Body:**
```json
{
  "age": 63,
  "sex": 1,
  "cp": 3,
  "trestbps": 145,
  "chol": 233,
  "fbs": 1,
  "restecg": 0,
  "thalach": 150,
  "exang": 0,
  "oldpeak": 2.3,
  "slope": 0,
  "ca": 0,
  "thal": 1
}
```

**Request Parameters:**

| Parameter | Type | Range | Description |
|-----------|------|-------|-------------|
| `age` | int | 0-120 | Patient age in years |
| `sex` | int | 0-1 | Sex (1=male, 0=female) |
| `cp` | int | 0-3 | Chest pain type |
| `trestbps` | int | 0-300 | Resting blood pressure (mm Hg) |
| `chol` | int | 0-600 | Serum cholesterol (mg/dl) |
| `fbs` | int | 0-1 | Fasting blood sugar > 120 mg/dl |
| `restecg` | int | 0-2 | Resting ECG results |
| `thalach` | int | 0-250 | Maximum heart rate achieved |
| `exang` | int | 0-1 | Exercise induced angina |
| `oldpeak` | float | 0-10 | ST depression induced by exercise |
| `slope` | int | 0-2 | Slope of peak exercise ST segment |
| `ca` | int | 0-4 | Number of major vessels colored |
| `thal` | int | 0-3 | Thalassemia type |

**Response:**
```json
{
  "prediction": 0,
  "confidence": 0.2737,
  "risk_level": "Low",
  "probability_no_disease": 0.7263,
  "probability_disease": 0.2737,
  "processing_time_ms": 11.78
}
```

**Response Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `prediction` | int | 0 = No disease, 1 = Disease present |
| `confidence` | float | Probability of disease (0-1) |
| `risk_level` | string | "Low" (<0.3), "Medium" (0.3-0.7), "High" (>0.7) |
| `probability_no_disease` | float | Probability of no heart disease |
| `probability_disease` | float | Probability of heart disease |
| `processing_time_ms` | float | API response time in milliseconds |

**Status Codes:**
- `200 OK` - Prediction successful
- `422 Unprocessable Entity` - Invalid input data
- `503 Service Unavailable` - Model not loaded

---

## 5. Metrics Endpoint

### `GET /metrics`

Returns Prometheus-compatible metrics for monitoring.

**Response (text/plain):**
```
# HELP app_info Application information
# TYPE app_info gauge
app_info{version="1.0.0",model_loaded="True"} 1

# HELP predictions_total Total number of predictions made
# TYPE predictions_total counter
predictions_total 156

# HELP prediction_latency_avg_ms Average prediction latency
# TYPE prediction_latency_avg_ms gauge
prediction_latency_avg_ms 12.45
```

---

## Example Usage

### cURL

```bash
# Health check
curl http://localhost:8000/health

# Make prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age":63,"sex":1,"cp":3,"trestbps":145,"chol":233,"fbs":1,"restecg":0,"thalach":150,"exang":0,"oldpeak":2.3,"slope":0,"ca":0,"thal":1}'
```

### Python

```python
import requests

# Make prediction
response = requests.post(
    "http://localhost:8000/predict",
    json={
        "age": 63, "sex": 1, "cp": 3, "trestbps": 145,
        "chol": 233, "fbs": 1, "restecg": 0, "thalach": 150,
        "exang": 0, "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1
    }
)
print(response.json())
```

### JavaScript

```javascript
fetch('http://localhost:8000/predict', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    age: 63, sex: 1, cp: 3, trestbps: 145,
    chol: 233, fbs: 1, restecg: 0, thalach: 150,
    exang: 0, oldpeak: 2.3, slope: 0, ca: 0, thal: 1
  })
})
.then(res => res.json())
.then(data => console.log(data));
```

---

## Error Responses

### Validation Error (422)
```json
{
  "detail": [
    {
      "loc": ["body", "age"],
      "msg": "ensure this value is less than or equal to 120",
      "type": "value_error.number.not_le"
    }
  ]
}
```

### Model Not Loaded (503)
```json
{
  "detail": "Model not loaded"
}
```

---

## Interactive Documentation

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

---

## Rate Limits

No rate limits are enforced in the current version.

---

## Authentication

No authentication required in the current version.

