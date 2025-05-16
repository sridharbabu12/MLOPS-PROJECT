# 🧠 MLOPS-PROJECT

An end-to-end Machine Learning Operations (MLOps) pipeline designed to automate the ML lifecycle — from data ingestion and model training to deployment and version control. This project ensures scalability, reproducibility, and maintainability using modern tools.

---

## 🚀 Project Overview

This project demonstrates an MLOps workflow using:

- **FastAPI** - for serving ML models as REST APIs
- **Docker** - for containerizing applications
- **GitHub Actions** - for CI/CD automation
- **DVC (Data Version Control)** - for dataset and model versioning
- **MLflow** - for experiment tracking and model registry

---

## 🧱 Project Structure
MLOPS-PROJECT/
├── .dvc/ # DVC cache and config files
├── .github/workflows/ # GitHub Actions CI/CD workflows
├── artifacts/ # Trained models and data artifacts
├── config/ # Configuration YAMLs
├── pipeline/ # DVC pipeline stages
├── src/ # Core ML code: data, models, training
├── static/ # Static web files (CSS, JS)
├── templates/ # HTML templates for frontend
├── utils/ # Utility functions and helpers
├── application.py # FastAPI app entry point
├── Dockerfile # Docker container definition
├── dvc.yaml # DVC pipeline definition
├── requirements.txt # Python dependencies
└── setup.py # Package metadata
