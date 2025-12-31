# Heart Disease Prediction - MLOps Assignment

[![CI/CD Pipeline](https://github.com/shahrukhsaba/mlops/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/shahrukhsaba/mlops/actions)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**BITS Pilani - MLOps Assignment (S1-25_AIMLCZG523)**

A production-ready MLOps pipeline for predicting heart disease risk using the UCI Heart Disease dataset. This project demonstrates end-to-end ML model development, CI/CD, containerization, and cloud deployment.

---

## 🚀 Quick Start - Run the Complete Project

```bash
# Step 1: Clone and navigate to the project
git clone https://github.com/shahrukhsaba/mlops.git
cd mlops

# Step 2: Install dependencies
pip install -r requirements.txt

# Step 3: Download and prepare data
python scripts/download_data.py

# Step 4: Run notebooks (generates EDA, training outputs & screenshots)
python scripts/execute_notebooks.py

# Step 5: Train the model for production
python scripts/train_and_save_locally.py

# Step 6: Build Docker image
docker build -t heart-disease-api:latest .

# Step 7: Run Docker container
docker run -d --name heart-disease-api -p 8000:8000 heart-disease-api:latest

# Step 8: Test the API
curl http://localhost:8000/health

# Predict with JSON input
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age":63,"sex":1,"cp":3,"trestbps":145,"chol":233,"fbs":1,"restecg":0,"thalach":150,"exang":0,"oldpeak":2.3,"slope":0,"ca":0,"thal":1}'

# Step 9: Open API documentation (Swagger UI)
open http://localhost:8000/docs
```

---

## 📋 Table of Contents

- [Quick Start](#-quick-start---run-the-complete-project)
- [Overview](#-overview)
- [Features](#-features)
- [Project Structure](#-project-structure)
- [Detailed Execution Steps](#-detailed-execution-steps)
- [Data Acquisition](#-data-acquisition)
- [Model Packaging & Reproducibility](#-model-packaging--reproducibility)
- [Model Training](#-model-training)
- [API Usage](#-api-usage)
- [Model Containerization (Step 6)](#-model-containerization-step-6)
- [Production Deployment (Step 7)](#-production-deployment-step-7)
- [CI/CD Pipeline](#-cicd-pipeline)
- [Monitoring & Logging (Step 8)](#-monitoring--logging-step-8)
- [Testing](#-testing)
- [MLflow Experiment Tracking](#-mlflow-experiment-tracking)

---

## 🎯 Overview

This project builds a machine learning classifier to predict heart disease risk based on patient health data. It includes:

- **Data Pipeline**: Automated data ingestion, validation, and preprocessing
- **Model Training**: Logistic Regression and Random Forest with hyperparameter tuning
- **Experiment Tracking**: MLflow integration for logging parameters, metrics, and artifacts
- **API Serving**: FastAPI-based prediction service
- **Containerization**: Docker image for portable deployment
- **Orchestration**: Kubernetes manifests for production deployment
- **CI/CD**: GitHub Actions workflow for automated testing and deployment
- **Monitoring**: Prometheus metrics and logging

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 📊 EDA | Comprehensive exploratory data analysis with visualizations |
| 🔧 Feature Engineering | Automated feature creation and preprocessing pipeline |
| 🤖 Multiple Models | Logistic Regression & Random Forest with GridSearchCV |
| 📈 Experiment Tracking | MLflow logging of parameters, metrics, and artifacts |
| 🧪 Testing | Unit and integration tests with pytest |
| 🐳 Docker | Containerized API for portable deployment |
| ☸️ Kubernetes | Deployment manifests with HPA and health checks |
| 🔄 CI/CD | GitHub Actions for automated pipeline |
| 📡 Monitoring | Prometheus metrics and structured logging |

---

## 📁 Project Structure

```
heart-disease-mlops/
├── .github/
│   └── workflows/
│       └── ci-cd.yml              # GitHub Actions CI/CD pipeline
├── api/
│   ├── app.py                     # FastAPI application
│   ├── predictor.py               # Prediction logic
│   ├── schemas.py                 # Pydantic schemas
│   ├── middleware/                # API middleware
│   └── monitoring/                # Prometheus metrics
├── data/
│   ├── raw/                       # Raw data files
│   └── processed/                 # Cleaned data
├── k8s/
│   ├── deployment.yaml            # Kubernetes deployment
│   ├── configmap.yaml             # Configuration
│   ├── ingress.yaml               # Ingress rules
│   └── namespace.yaml             # Namespace definition
├── models/
│   └── production/                # Production model artifacts
├── monitoring/
│   ├── prometheus/                # Prometheus configuration
│   └── grafana/                   # Grafana dashboards
├── notebooks/
│   ├── 01_eda.ipynb                          # Exploratory Data Analysis
│   ├── 02_feature_engineering_modeling.ipynb # Feature Engineering & Model Training
│   └── 03_mlflow_experiments.ipynb           # MLflow Experiment Tracking
├── scripts/
│   ├── download_data.py           # Data download script
│   ├── train_and_save_locally.py  # Local training script
│   └── execute_notebooks.py       # Run all notebooks with outputs
├── screenshots/                    # Generated visualizations
│   ├── 01_*.png                   # EDA screenshots
│   ├── 02_*.png                   # Model training screenshots
│   └── 03_*.png                   # MLflow experiment screenshots
├── src/
│   ├── data/                      # Data loading and preprocessing
│   ├── features/                  # Feature engineering
│   ├── models/                    # Model training
│   ├── monitoring/                # Drift detection
│   └── tracking/                  # MLflow utilities
├── tests/
│   ├── unit/                      # Unit tests
│   └── integration/               # Integration tests
├── Dockerfile                     # Docker image definition
├── docker-compose.yml             # Multi-container setup
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

---

## 📝 Detailed Execution Steps

### Prerequisites

- Python 3.9+ or 3.10+
- Docker Desktop (for containerization)
- Git

### Step 1: Clone and Setup

```bash
# Clone the repository
git clone https://github.com/shahrukhsaba/mlops.git
cd mlops

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Download Data

```bash
python scripts/download_data.py
```

**Expected Output:**
```
Dataset download and processing complete!
Final dataset shape: (297, 14)
Target distribution: {0: 160, 1: 137}
```

### Step 3: Run Notebooks (Optional - for EDA & Experiments)

```bash
# Execute all notebooks and generate outputs/screenshots
python scripts/execute_notebooks.py
```

**Expected Output:**
```
======================================================================
          EXECUTING NOTEBOOKS WITH OUTPUT CAPTURE
======================================================================

📓 Executing: notebooks/01_eda.ipynb
   ✅ Executed and saved: notebooks/01_eda.ipynb

📓 Executing: notebooks/02_feature_engineering_modeling.ipynb
   ✅ Executed and saved: notebooks/02_feature_engineering_modeling.ipynb

📓 Executing: notebooks/03_mlflow_experiments.ipynb
   ✅ Executed and saved: notebooks/03_mlflow_experiments.ipynb
```

This will generate 13 screenshots in the `screenshots/` folder.

### Step 4: Train the Model

```bash
python scripts/train_and_save_locally.py
```

**Expected Output:**
```
Training Random Forest model for packaging...
Model saved to models/random_forest/model.pkl
```

### Step 5: Build Docker Image

```bash
docker build -t heart-disease-api:latest .
```

### Step 6: Run Docker Container

```bash
docker run -d --name heart-disease-api -p 8000:8000 heart-disease-api:latest
```

### Step 7: Test the API

```bash
# Health check
curl http://localhost:8000/health

# Make a prediction with JSON input
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age":63,"sex":1,"cp":3,"trestbps":145,"chol":233,"fbs":1,"restecg":0,"thalach":150,"exang":0,"oldpeak":2.3,"slope":0,"ca":0,"thal":1}'
```

**Expected Response:**
```json
{
  "prediction": 0,
  "confidence": 0.2737,
  "risk_level": "Low",
  "probability_no_disease": 0.7263,
  "probability_disease": 0.2737,
  "processing_time_ms": 11.93
}
```

### Step 8: View API Documentation

Open in browser: http://localhost:8000/docs

### Step 9: Stop Container (when done)

```bash
docker stop heart-disease-api
docker rm heart-disease-api
```

---

## 📥 Data Acquisition

### Download Dataset

The UCI Heart Disease dataset can be downloaded automatically:

```bash
python scripts/download_data.py
```

This script will:
- Download data from UCI Machine Learning Repository
- Clean and preprocess the data
- Save to `data/processed/heart_disease_clean.csv`
- Generate metadata in `data/processed/metadata.json`

### Dataset Description

| Feature | Description | Type |
|---------|-------------|------|
| age | Age in years | Numerical |
| sex | Sex (1=male, 0=female) | Binary |
| cp | Chest pain type (0-3) | Categorical |
| trestbps | Resting blood pressure (mm Hg) | Numerical |
| chol | Serum cholesterol (mg/dl) | Numerical |
| fbs | Fasting blood sugar > 120 mg/dl | Binary |
| restecg | Resting ECG results (0-2) | Categorical |
| thalach | Maximum heart rate achieved | Numerical |
| exang | Exercise induced angina | Binary |
| oldpeak | ST depression induced by exercise | Numerical |
| slope | Slope of peak exercise ST segment | Categorical |
| ca | Number of major vessels (0-3) | Numerical |
| thal | Thalassemia type | Categorical |
| target | Heart disease presence (0/1) | Binary |

---

## 📦 Model Packaging & Reproducibility

### Saved Model Artifacts

The following files are included in the repository for full reproducibility:

| File | Description |
|------|-------------|
| `models/production/model.pkl` | Trained RandomForestClassifier (pickle format) |
| `models/production/preprocessor.pkl` | ColumnTransformer for feature preprocessing |
| `models/production/model_metadata.json` | Model version, hyperparameters, performance metrics |
| `models/production/feature_names.json` | List of 19 feature names used by the model |
| `models/production/MODEL_CARD.md` | Model documentation and ethical considerations |
| `mlruns/` | MLflow experiment tracking data (parameters, metrics, artifacts) |
| `requirements.txt` | All Python dependencies with pinned versions |
| `src/features/feature_engineering.py` | Feature creation and preprocessing pipeline |

### Preprocessing Pipeline

The preprocessing pipeline (`src/features/feature_engineering.py`) includes:

```python
# Feature Engineering
create_features(df)  # Creates: age_group, chol_cat, bp_cat, hr_reserve, interaction features

# Preprocessing Pipeline
build_preprocessing_pipeline(numerical_features, categorical_features)
# - Numerical: SimpleImputer(median) + StandardScaler
# - Categorical: SimpleImputer(most_frequent) + OneHotEncoder
```

### Reproducibility

- **Random Seed**: 42 (fixed for all training)
- **Python Version**: 3.10+
- **scikit-learn Version**: 1.3.0
- **All dependencies pinned** in `requirements.txt`

### Load Model for Inference

```python
import joblib

# Load model and preprocessor
model = joblib.load('models/production/model.pkl')
preprocessor = joblib.load('models/production/preprocessor.pkl')

# Make prediction
prediction = model.predict(preprocessed_data)
```

---

## 🎓 Model Training

### Train Models

Train both Logistic Regression and Random Forest models:

```bash
python src/models/train_models.py
```

This will:
1. Load and preprocess the data
2. Train Logistic Regression with hyperparameter tuning
3. Train Random Forest with hyperparameter tuning
4. Log experiments to MLflow
5. Save the best model to `models/production/`

### View MLflow Experiments

```bash
mlflow ui --port 5000
```

Open http://localhost:5000 to view experiment tracking.

---

## 🌐 API Usage

### Run API Locally

```bash
uvicorn api.app:app --reload --host 0.0.0.0 --port 8000
```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API information |
| `/health` | GET | Health check with model status |
| `/model-info` | GET | Model metadata and metrics |
| `/predict` | POST | Make prediction (JSON input) |
| `/metrics` | GET | Prometheus metrics |
| `/docs` | GET | Swagger UI documentation |
| `/redoc` | GET | ReDoc documentation |

### Example Request (JSON Input)

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
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
  }'
```

### Example Response

```json
{
  "prediction": 0,
  "confidence": 0.2737,
  "risk_level": "Low",
  "probability_no_disease": 0.7263,
  "probability_disease": 0.2737,
  "processing_time_ms": 11.93
}
```

### Response Fields

| Field | Description |
|-------|-------------|
| `prediction` | 0 = No heart disease, 1 = Heart disease present |
| `confidence` | Probability of disease (0-1) |
| `risk_level` | "Low" (<0.3), "Medium" (0.3-0.7), "High" (>0.7) |
| `probability_no_disease` | Probability of no heart disease |
| `probability_disease` | Probability of heart disease |
| `processing_time_ms` | API response time in milliseconds |

### Interactive Documentation

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

---

## 🐳 Model Containerization (Step 6)

This section covers Step 6 of the assignment: *"Build a Docker container for the model-serving API (Flask or FastAPI is recommended)"*.

### Why FastAPI?

| Feature | FastAPI | Flask |
|---------|---------|-------|
| **Async Support** | ✅ Native | ❌ Requires extensions |
| **Auto Documentation** | ✅ Built-in (`/docs`, `/redoc`) | ❌ Manual setup |
| **Data Validation** | ✅ Pydantic (automatic) | ❌ Manual validation |
| **JSON Schema** | ✅ Auto-generated | ❌ Manual |
| **Performance** | ✅ Faster (async) | ⚠️ Slower |
| **Type Hints** | ✅ Required & validated | ❌ Optional |

### API Features

- **Framework**: FastAPI with Pydantic validation
- **Endpoint**: `POST /predict` accepts **JSON input**
- **Response**: Returns prediction, confidence, and risk level
- **Documentation**: Auto-generated Swagger UI at `/docs`

### Dockerfile Overview

```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir fastapi uvicorn scikit-learn pandas numpy joblib pydantic
COPY api/ ./api/
COPY models/production/ ./models/production/
EXPOSE 8000
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Build Docker Image

```bash
docker build -t heart-disease-api:latest .
```

### Run Container

```bash
docker run -d --name heart-disease-api -p 8000:8000 heart-disease-api:latest
```

### Using Docker Compose

```bash
docker-compose up -d
```

This starts both the API and MLflow server.

### Test Container with JSON Input

```bash
# Health check
curl http://localhost:8000/health

# Prediction with JSON input (required format)
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age":63,"sex":1,"cp":3,"trestbps":145,"chol":233,"fbs":1,"restecg":0,"thalach":150,"exang":0,"oldpeak":2.3,"slope":0,"ca":0,"thal":1}'
```

### Expected Response

```json
{
  "prediction": 0,
  "confidence": 0.2737,
  "risk_level": "Low",
  "probability_no_disease": 0.7263,
  "probability_disease": 0.2737,
  "processing_time_ms": 11.93
}
```

### Test Different Risk Profiles

```bash
# High-risk patient
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age":70,"sex":1,"cp":2,"trestbps":180,"chol":300,"fbs":1,"restecg":1,"thalach":100,"exang":1,"oldpeak":4.0,"slope":2,"ca":3,"thal":2}'
# Response: {"prediction": 1, "confidence": 0.785, "risk_level": "High", ...}

# Low-risk patient
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age":35,"sex":0,"cp":0,"trestbps":120,"chol":180,"fbs":0,"restecg":0,"thalach":170,"exang":0,"oldpeak":0.0,"slope":1,"ca":0,"thal":1}'
# Response: {"prediction": 0, "confidence": 0.0992, "risk_level": "Low", ...}
```

### Stop Container

```bash
docker stop heart-disease-api
docker rm heart-disease-api
```

---

## ☸️ Production Deployment (Step 7)

This section covers Step 7 of the assignment: *"Deploy the Dockerized API to a public cloud or local Kubernetes"*.

### Deployment Options

| Option | Description | Used in This Project |
|--------|-------------|----------------------|
| **Docker Desktop Kubernetes** | Local K8s cluster | ✅ Primary |
| **Minikube** | Local K8s for development | ✅ Supported |
| **GKE/EKS/AKS** | Cloud Kubernetes | ✅ Manifests ready |

### Kubernetes Manifests (in `k8s/` folder)

The project uses **deployment manifests** (standard Kubernetes YAML) instead of Helm charts:

```
k8s/
├── deployment.yaml    # Deployment + Service (LoadBalancer) + HPA
├── configmap.yaml     # Application configuration
├── ingress.yaml       # Ingress rules (optional, for domain routing)
└── namespace.yaml     # Namespace definition
```

| File | Resources | Description |
|------|-----------|-------------|
| `k8s/deployment.yaml` | Deployment, Service, HPA | Main deployment with LoadBalancer service and auto-scaling |
| `k8s/configmap.yaml` | ConfigMap | Application configuration (paths, ports, settings) |
| `k8s/ingress.yaml` | Ingress | Optional domain-based routing rules |
| `k8s/namespace.yaml` | Namespace | Isolated namespace for the application |

### Resources in `deployment.yaml`

| Resource | API Version | Purpose |
|----------|-------------|---------|
| **Deployment** | apps/v1 | Manages 2 replicas with rolling updates |
| **Service** | v1 (LoadBalancer) | Exposes API on port 80 → container 8000 |
| **HorizontalPodAutoscaler** | autoscaling/v2 | Auto-scales pods based on CPU/memory |

### Deployment Features

| Feature | Configuration |
|---------|---------------|
| **Replicas** | 2 pods (min), 5 pods (max with HPA) |
| **Service Type** | **LoadBalancer** (exposed on `localhost:80`) |
| **Health Checks** | Liveness & Readiness probes on `/health` |
| **Auto-scaling** | HPA based on CPU (70%) and Memory (80%) |
| **Resources** | 256Mi-512Mi memory, 250m-500m CPU |
| **Rolling Updates** | Zero-downtime deployments |
| **Prometheus** | Metrics scraping annotations |
| **Environment** | `ENV=production`, `LOG_LEVEL=INFO` |

### Deploy to Docker Desktop Kubernetes

```bash
# Ensure Docker Desktop Kubernetes is enabled
# (Docker Desktop → Settings → Kubernetes → Enable)

# Build Docker image
docker build -t heart-disease-api:latest .

# Apply Kubernetes manifests
kubectl apply -f k8s/deployment.yaml

# Verify deployment
kubectl get deployments
kubectl get pods
kubectl get svc
kubectl get hpa
```

### Deploy to Minikube

```bash
# Start Minikube
minikube start

# Use Minikube's Docker daemon
eval $(minikube docker-env)

# Build image in Minikube
docker build -t heart-disease-api:latest .

# Apply Kubernetes manifests
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/ingress.yaml

# Get service URL
minikube service heart-disease-api --url
```

### Verify Deployment

```bash
# Check deployment status
kubectl get deployments
# NAME                READY   UP-TO-DATE   AVAILABLE   AGE
# heart-disease-api   2/2     2            2           3h

# Check running pods
kubectl get pods -o wide
# NAME                                 READY   STATUS    RESTARTS   AGE
# heart-disease-api-676668bdb5-877qm   1/1     Running   0          2m
# heart-disease-api-676668bdb5-x6lmb   1/1     Running   0          2m

# Check LoadBalancer service
kubectl get svc
# NAME                TYPE           EXTERNAL-IP   PORT(S)
# heart-disease-api   LoadBalancer   localhost     80:31775/TCP

# Check HorizontalPodAutoscaler
kubectl get hpa
# NAME                    REFERENCE                      MINPODS   MAXPODS   REPLICAS
# heart-disease-api-hpa   Deployment/heart-disease-api   2         5         2
```

### Test Endpoints via LoadBalancer (Port 80)

```bash
# Health check
curl http://localhost:80/health
# {"status": "healthy", "model_loaded": true, ...}

# Model info
curl http://localhost:80/model-info
# {"model_type": "RandomForestClassifier", ...}

# Prediction with JSON
curl -X POST http://localhost:80/predict \
  -H "Content-Type: application/json" \
  -d '{"age":63,"sex":1,"cp":3,"trestbps":145,"chol":233,"fbs":1,"restecg":0,"thalach":150,"exang":0,"oldpeak":2.3,"slope":0,"ca":0,"thal":1}'
# {"prediction": 0, "confidence": 0.2737, "risk_level": "Low", ...}
```

### View Pod Logs

```bash
kubectl logs -l app=heart-disease-api --tail=20
```

### Scale Deployment Manually

```bash
kubectl scale deployment heart-disease-api --replicas=3
```

### Restart Deployment (Update Image)

```bash
kubectl rollout restart deployment heart-disease-api
kubectl rollout status deployment heart-disease-api
```

### Deployment Screenshots

See the `screenshots/` folder for deployment verification:

| Screenshot | Description |
|------------|-------------|
| `07_k8s_deployment_status.txt` | Deployments, pods, services, HPA |
| `07_k8s_api_verification.txt` | API endpoint tests via LoadBalancer |
| `07_k8s_pod_details.txt` | Pod describe and logs |
| `07_docker_container_status.txt` | Docker container status |

---

## 🔄 CI/CD Pipeline

The GitHub Actions workflow (`.github/workflows/ci-cd.yml`) includes:

| Job | Description | Artifacts |
|-----|-------------|-----------|
| **lint** | flake8, black, isort code quality | - |
| **test** | pytest with coverage | Coverage reports (XML, HTML) |
| **train** | Model training & validation | Model artifacts, MLflow runs |
| **docker** | Build & test Docker image | Docker image (.tar.gz) |
| **integration** | API integration tests | Test results |
| **security** | bandit, safety scans | Security reports |

### Trigger Pipeline

- Push to `main`, `master`, or `develop` branch
- Pull request to `main` or `master`
- Manual trigger via `workflow_dispatch`

### View Results

Check the [Actions tab](https://github.com/shahrukhsaba/mlops/actions) in the GitHub repository.

---

## 📊 Monitoring & Logging (Step 8)

This section covers Step 8 of the assignment: *"Integrate logging of API requests. Demonstrate simple monitoring (Prometheus + Grafana or API metrics/logs dashboard)"*.

### 1. API Request Logging

All API requests are logged with structured logging:

```python
# Example log output
2025-12-31 00:30:08 - INFO - Prediction: 0, Confidence: 0.274, Risk: Low, Duration: 0.012s
2025-12-31 00:30:08 - INFO - Prediction: 1, Confidence: 0.785, Risk: High, Duration: 0.011s
```

**Logging features:**
- File logging: `logs/api.log`
- Console logging: stdout (visible in `docker logs`)
- Structured format: timestamp, level, message
- Request details: prediction, confidence, risk level, duration

### 2. Prometheus Metrics

The API exposes comprehensive metrics at `/metrics`:

```bash
curl http://localhost:8000/metrics
```

| Metric | Type | Description |
|--------|------|-------------|
| `app_info` | gauge | Application version and model status |
| `app_uptime_seconds` | gauge | API uptime in seconds |
| `model_loaded` | gauge | Model loading status (0/1) |
| `predictions_total` | counter | Total number of predictions |
| `predictions_success_total` | counter | Successful predictions |
| `predictions_errors_total` | counter | Failed predictions |
| `predictions_by_class{class="0"}` | counter | Predictions: No disease |
| `predictions_by_class{class="1"}` | counter | Predictions: Disease |
| `prediction_latency_avg_ms` | gauge | Average latency (ms) |
| `prediction_latency_total_ms` | counter | Total latency (ms) |

### 3. Monitoring Stack (Prometheus + Grafana)

```
monitoring/
├── docker-compose-monitoring.yml   # Full monitoring stack
├── prometheus/
│   └── prometheus.yml              # Prometheus scrape config
└── grafana/
    ├── dashboards/
    │   ├── dashboards.yml          # Dashboard provisioning
    │   └── heart-disease-api.json  # Pre-built dashboard
    └── datasources/
        └── prometheus.yml          # Prometheus datasource
```

### Start Monitoring Stack

```bash
# Build API image first (if not already built)
docker build -t heart-disease-api:latest .

# Start Prometheus + Grafana + API
cd monitoring
docker-compose -f docker-compose-monitoring.yml up -d
```

### Access Dashboards

| Service | URL | Credentials |
|---------|-----|-------------|
| **API** | http://localhost:8000 | - |
| **Prometheus** | http://localhost:9090 | - |
| **Grafana** | http://localhost:3000 | admin / admin |

### Grafana Dashboard Features

The pre-built dashboard (`heart-disease-api.json`) includes:

- **Total Predictions** - Counter stat panel
- **Average Latency** - Latency gauge with thresholds
- **Model Status** - Loaded/Not loaded indicator
- **Uptime** - Application uptime
- **Predictions by Class** - Pie chart (Disease vs No Disease)
- **Prediction Counts Over Time** - Time series graph

### View Container Logs

```bash
# Docker container logs
docker logs heart-disease-api --tail=50

# Kubernetes pod logs
kubectl logs -l app=heart-disease-api --tail=50
```

### Monitoring Screenshots

See the `screenshots/` folder:

| Screenshot | Description |
|------------|-------------|
| `08_api_metrics.txt` | Prometheus metrics output |
| `08_api_logs.txt` | API request logs |
| `08_monitoring_stack.txt` | Monitoring stack configuration |

---

## 🧪 Testing

### Test Coverage

| Test File | Tests | Description |
|-----------|-------|-------------|
| `tests/unit/test_data_processing.py` | 15 | Data loading, validation, transformations |
| `tests/unit/test_model.py` | 12 | Predictions, metrics, risk classification |
| `tests/unit/test_preprocessing.py` | 11 | Feature engineering, preprocessing |
| `tests/unit/test_api.py` | - | API endpoint tests |
| `tests/integration/test_api_integration.py` | - | Full API integration |

**Total: 52+ tests**

### Run All Tests

```bash
pytest tests/ -v
```

### Run Unit Tests

```bash
pytest tests/unit/ -v
```

### Run Integration Tests

```bash
# Start the API first
uvicorn api.app:app --port 8000 &

# Run integration tests
pytest tests/integration/ -v
```

### Test Coverage

```bash
pytest tests/ --cov=src --cov=api --cov-report=html
```

---

## 📈 MLflow Experiment Tracking

### Features Tracked

- **Parameters**: Hyperparameters, model configuration
- **Metrics**: Accuracy, Precision, Recall, F1, ROC-AUC
- **Artifacts**: Model files, preprocessor, plots

### Start MLflow UI

```bash
mlflow ui --backend-store-uri ./mlruns --port 5000
```

### Compare Experiments

View and compare runs at http://localhost:5000.

---

## 📸 Generated Screenshots

Running `python scripts/execute_notebooks.py` generates notebook screenshots:

### EDA Screenshots (01_eda.ipynb)

| Screenshot | Description |
|------------|-------------|
| `01_class_balance.png` | Target class distribution |
| `01_numerical_histograms.png` | Distribution of numerical features |
| `01_correlation_heatmap.png` | Feature correlation matrix |
| `01_categorical_distributions.png` | Categorical feature distributions |
| `01_boxplots_by_target.png` | Features by target class |

### Model Training Screenshots (02_feature_engineering_modeling.ipynb)

| Screenshot | Description |
|------------|-------------|
| `02_model_comparison.png` | Model metrics comparison |
| `02_roc_curve_comparison.png` | ROC curves for both models |
| `02_confusion_matrices.png` | Confusion matrices side-by-side |
| `02_feature_importance.png` | Random Forest feature importance |

### MLflow Experiment Screenshots (03_mlflow_experiments.ipynb)

| Screenshot | Description |
|------------|-------------|
| `03_lr_confusion_matrix.png` | Logistic Regression confusion matrix |
| `03_lr_roc_curve.png` | Logistic Regression ROC curve |
| `03_rf_confusion_matrix.png` | Random Forest confusion matrix |
| `03_rf_roc_curve.png` | Random Forest ROC curve |

### Production Deployment Screenshots (Step 7)

| Screenshot | Description |
|------------|-------------|
| `07_k8s_deployment_status.txt` | Kubernetes deployments, pods, services, HPA |
| `07_k8s_api_verification.txt` | API endpoint tests via LoadBalancer |
| `07_k8s_pod_details.txt` | Pod describe and container logs |
| `07_docker_container_status.txt` | Docker container and image status |

---

## 📝 Model Performance

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | 80.00% | 83.33% | 71.43% | 76.92% | **92.86%** |
| Random Forest | **81.67%** | 81.48% | **78.57%** | **80.00%** | 91.52% |

*Note: Logistic Regression achieved higher ROC-AUC while Random Forest had better overall accuracy.*

---

## 🔒 Security

- Input validation with Pydantic schemas
- Health checks for container orchestration
- Dependency vulnerability scanning with safety
- Code security analysis with Bandit

---

## 📄 License

This project is licensed under the MIT License.

---

## 👥 Authors

- **Sk Shahrukh Saba** - MLOps Assignment

---

## 🙏 Acknowledgments

- UCI Machine Learning Repository for the Heart Disease dataset
- BITS Pilani for the MLOps course

