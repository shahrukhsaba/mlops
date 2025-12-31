# Deployment Guide

## Heart Disease Prediction API - Deployment Options

This guide covers all deployment options for the Heart Disease Prediction API.

---

## Deployment Options Overview

| Option | Port | Best For | Monitoring |
|--------|------|----------|------------|
| **Docker** | 8000 | Development, Testing | Manual |
| **Docker Compose** | 8000 | Full stack with monitoring | ✅ Grafana, Prometheus, MLflow |
| **Kubernetes** | 80 | Production, Scaling | ✅ With manifests |
| **Render (Cloud)** | 443 | Public demo | ❌ Metrics only |

---

## Option 1: Docker (Simple)

### Prerequisites
- Docker Desktop installed and running

### Deploy

```bash
# Build image
docker build -t heart-disease-api:latest .

# Run container
docker run -d --name heart-disease-api -p 8000:8000 heart-disease-api:latest

# Verify
curl http://localhost:8000/health
```

### Stop

```bash
docker stop heart-disease-api
docker rm heart-disease-api
```

---

## Option 2: Docker Compose (Recommended for Local)

### Prerequisites
- Docker Desktop installed and running

### Deploy

```bash
# Build API image first
docker build -t heart-disease-api:latest .

# Start full stack (API + Prometheus + Grafana + MLflow)
cd monitoring
docker-compose -f docker-compose-monitoring.yml up -d
```

### Services Started

| Service | URL | Credentials |
|---------|-----|-------------|
| API | http://localhost:8000 | - |
| Swagger | http://localhost:8000/docs | - |
| Grafana | http://localhost:3000 | admin / admin |
| Prometheus | http://localhost:9090 | - |
| MLflow | http://localhost:5000 | - |

### Stop

```bash
cd monitoring
docker-compose -f docker-compose-monitoring.yml down
```

---

## Option 3: Kubernetes (Docker Desktop)

### Prerequisites
- Docker Desktop with Kubernetes enabled
- kubectl configured

### Enable Kubernetes
1. Open Docker Desktop
2. Settings → Kubernetes → Enable Kubernetes
3. Apply & Restart
4. Wait for green indicator

### Deploy

```bash
# Build image
docker build -t heart-disease-api:latest .

# Deploy to Kubernetes
kubectl apply -f k8s/deployment.yaml

# Verify
kubectl get deployments
kubectl get pods
kubectl get svc
```

### Access

```bash
# API available at port 80
curl http://localhost:80/health

# Make prediction
curl -X POST http://localhost:80/predict \
  -H "Content-Type: application/json" \
  -d '{"age":63,"sex":1,"cp":3,"trestbps":145,"chol":233,"fbs":1,"restecg":0,"thalach":150,"exang":0,"oldpeak":2.3,"slope":0,"ca":0,"thal":1}'
```

### Kubernetes Resources

| Resource | Description |
|----------|-------------|
| `Deployment` | 2 replicas with rolling updates |
| `Service` | LoadBalancer on port 80 |
| `HPA` | Auto-scale 2-5 pods based on CPU/Memory |
| `ConfigMap` | Application configuration |
| `Ingress` | Optional domain routing |

### Scale

```bash
# Manual scaling
kubectl scale deployment heart-disease-api --replicas=3

# Check HPA
kubectl get hpa
```

### View Logs

```bash
kubectl logs -l app=heart-disease-api --tail=50
```

### Stop

```bash
kubectl delete -f k8s/deployment.yaml
```

---

## Option 4: Kubernetes (Minikube)

### Prerequisites
- Minikube installed
- kubectl configured

### Deploy

```bash
# Start Minikube
minikube start

# Use Minikube's Docker daemon
eval $(minikube docker-env)

# Build image in Minikube
docker build -t heart-disease-api:latest .

# Deploy
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/ingress.yaml

# Get service URL
minikube service heart-disease-api --url
```

---

## Option 5: Render (Cloud)

### Prerequisites
- Render.com account (free)
- GitHub repository connected

### Deploy

1. Go to [render.com](https://render.com)
2. New → Web Service
3. Connect GitHub repo: `shahrukhsaba/mlops`
4. Configure:
   - Name: `heart-disease-api`
   - Environment: Docker
   - Plan: Free
5. Deploy

### Live URL

```
https://heart-disease-api-sdgp.onrender.com
```

### Test

```bash
curl https://heart-disease-api-sdgp.onrender.com/health

curl -X POST https://heart-disease-api-sdgp.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{"age":63,"sex":1,"cp":3,"trestbps":145,"chol":233,"fbs":1,"restecg":0,"thalach":150,"exang":0,"oldpeak":2.3,"slope":0,"ca":0,"thal":1}'
```

### Limitations

- Free tier sleeps after 15 min inactivity
- First request may take ~30s to wake up
- No Grafana/Prometheus (metrics endpoint only)

---

## Kubernetes Manifests Reference

### k8s/deployment.yaml

```yaml
# Deployment - 2 replicas
apiVersion: apps/v1
kind: Deployment
metadata:
  name: heart-disease-api
spec:
  replicas: 2
  selector:
    matchLabels:
      app: heart-disease-api
  template:
    spec:
      containers:
      - name: api
        image: heart-disease-api:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
        readinessProbe:
          httpGet:
            path: /health
            port: 8000

---
# Service - LoadBalancer
apiVersion: v1
kind: Service
metadata:
  name: heart-disease-api
spec:
  type: LoadBalancer
  ports:
  - port: 80
    targetPort: 8000
  selector:
    app: heart-disease-api

---
# HorizontalPodAutoscaler
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: heart-disease-api-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: heart-disease-api
  minReplicas: 2
  maxReplicas: 5
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ENV` | production | Environment name |
| `LOG_LEVEL` | INFO | Logging level |
| `PORT` | 8000 | API port |
| `MODEL_PATH` | models/production/model.pkl | Model file path |

---

## Health Checks

All deployments include health checks:

```bash
# Liveness probe - is the app running?
GET /health

# Readiness probe - is the app ready to serve?
GET /health
```

---

## Monitoring Setup

### Start Monitoring Stack

```bash
cd monitoring
docker-compose -f docker-compose-monitoring.yml up -d
```

### Access Dashboards

| Service | URL |
|---------|-----|
| Grafana | http://localhost:3000/d/heart-disease-api/heart-disease-api-dashboard |
| Prometheus | http://localhost:9090 |
| MLflow | http://localhost:5000 |

### Grafana Login
- Username: `admin`
- Password: `admin`

---

## Troubleshooting

### Container won't start

```bash
# Check logs
docker logs heart-disease-api

# Check if port is in use
lsof -i :8000
```

### Kubernetes pods not ready

```bash
# Describe pod
kubectl describe pod -l app=heart-disease-api

# Check events
kubectl get events --sort-by=.metadata.creationTimestamp
```

### Model not loading

```bash
# Check model files exist
ls -la models/production/

# Verify in container
docker exec heart-disease-api ls -la /app/models/production/
```

---

## Production Checklist

- [ ] Model files in `models/production/`
- [ ] Docker image built successfully
- [ ] Health check passing
- [ ] Prediction endpoint working
- [ ] Monitoring configured (if needed)
- [ ] Logs accessible
- [ ] Resource limits set (K8s)
- [ ] HPA configured (K8s)

