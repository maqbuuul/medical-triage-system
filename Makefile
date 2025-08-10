# Medical Triage ML Pipeline Makefile

.PHONY: help install train api dashboard docker-build docker-up lint format clean

# Default target
help:
	@echo "Medical Triage ML Pipeline"
	@echo "=========================="
	@echo ""
	@echo "Available commands:"
	@echo "  install      Install dependencies"
	@echo "  train        Run the full ML training pipeline"
	@echo "  api          Start FastAPI server"
	@echo "  dashboard    Start Streamlit dashboard"
	@echo "  mlflow       Start MLflow UI"
	@echo "  docker-build Build Docker images"
	@echo "  docker-up    Start all services with Docker Compose"
	@echo "  docker-down  Stop Docker services"
	@echo "  lint         Run code linting"
	@echo "  format       Format code with black"
	@echo "  clean        Clean up generated files"
	@echo "  test         Run basic tests"

# Install dependencies
install:
	@echo "Installing dependencies..."
	pip install -r requirements.txt
	@echo "✅ Dependencies installed"

# Create necessary directories
setup:
	@echo "Setting up directories..."
	mkdir -p logs artifacts models plots data .streamlit
	@echo "✅ Directories created"

# Run the full training pipeline
train: setup
	@echo "Starting ML training pipeline..."
	python main.py
	@echo "✅ Training pipeline completed"

# Start FastAPI server
api:
	@echo "Starting FastAPI server..."
	uvicorn fastapi_app:app --host 0.0.0.0 --port 8000 --reload

# Start Streamlit dashboard
dashboard:
	@echo "Starting Streamlit dashboard..."
	streamlit run streamlit_dashboard.py

# Start MLflow UI
mlflow:
	@echo "Starting MLflow UI..."
	mlflow ui

# Docker operations
docker-build:
	@echo "Building Docker images..."
	docker-compose build
	@echo "✅ Docker images built"

docker-up:
	@echo "Starting services with Docker Compose..."
	docker-compose up -d
	@echo "✅ Services started"
	@echo "FastAPI: http://localhost:8000"
	@echo "Streamlit: http://localhost:8501"
	@echo "MLflow: http://localhost:5000"

docker-down:
	@echo "Stopping Docker services..."
	docker-compose down
	@echo "✅ Services stopped"

# Code quality
lint:
	@echo "Running code linting..."
	flake8 --max-line-length=88 --extend-ignore=E203,W503 *.py
	pylint --disable=C0114,C0115,C0116 *.py
	@echo "✅ Linting completed"

format:
	@echo "Formatting code..."
	black --line-length=88 *.py
	@echo "✅ Code formatted"

# Testing
test:
	@echo "Running basic tests..."
	python -c "from data_preprocessing import preprocess_pipeline; print('✅ Data preprocessing OK')"
	python -c "from config import *; print('✅ Configuration OK')"
	python -c "from logging_config import setup_logging; print('✅ Logging OK')"
	@echo "✅ Basic tests passed"

# Clean up
clean:
	@echo "Cleaning up generated files..."
	rm -rf __pycache__/
	rm -rf .pytest_cache/
	rm -rf *.pyc
	rm -rf logs/*.log
	rm -rf artifacts/plots/*.png
	rm -rf models/*.pkl
	@echo "✅ Cleanup completed"

# Full pipeline (install, train, test services)
all: install setup train api-test dashboard-test
	@echo "✅ Full pipeline completed successfully!"

# Quick API test
api-test:
	@echo "Testing API endpoint..."
	python -c "import requests; print('API test passed' if requests.get('http://localhost:8000/health').status_code == 200 else 'API test failed')"

# Development setup
dev-setup: install setup
	@echo "Setting up development environment..."
	pip install -e .
	pre-commit install
	@echo "✅ Development environment ready"

# Production deployment
deploy: docker-build
	@echo "Deploying to production..."
	docker-compose -f docker-compose.prod.yml up -d
	@echo "✅ Production deployment completed"

# Backup artifacts
backup:
	@echo "Backing up artifacts..."
	tar -czf backup_$(shell date +%Y%m%d_%H%M%S).tar.gz artifacts/ models/ logs/
	@echo "✅ Backup created"

# Show service status
status:
	@echo "Service Status:"
	@echo "=============="
	docker-compose ps
	@echo ""
	@echo "Logs:"
	@echo "====="
	docker-compose logs --tail=10