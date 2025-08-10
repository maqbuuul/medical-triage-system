# Medical Triage ML System | Nidaamka Qiimeynta Caafimaadka

🏥 **Modern AI-powered medical triage system with Somali language support**

*Nidaam casri ah oo isticmaala AI si loo qiimeeyo xaaladda caafimaad ee bukaanka*

## 🌟 Features

- **🤖 Advanced ML Pipeline**: Multiple algorithms with automated hyperparameter tuning
- **📊 MLflow Integration**: Complete experiment tracking and model versioning  
- **🚀 FastAPI Service**: RESTful API for real-time predictions
- **🎨 Streamlit Dashboard**: Interactive web interface with Somali language support
- **📈 Comprehensive Evaluation**: SHAP analysis, confusion matrices, ROC curves
- **🔧 Production Ready**: Docker deployment, logging, error handling

## 🏗️ Architecture

```
├── 📊 Data Pipeline
│   ├── data_loader.py          # Data loading with sampling
│   └── data_preprocessing.py   # Feature engineering & SMOTE
├── 🤖 ML Pipeline  
│   ├── train_models.py         # Base model training
│   ├── tune_models.py          # Hyperparameter optimization
│   └── evaluate_models.py      # Comprehensive evaluation
├── 📈 Tracking & Logging
│   ├── mlflow_tracking.py      # Experiment tracking
│   └── logging_config.py       # Centralized logging
├── 🌐 Services
│   ├── fastapi_app.py          # REST API service
│   └── streamlit_dashboard.py  # Interactive dashboard
└── 🚀 Deployment
    ├── Dockerfile              # Container configuration
    ├── docker-compose.yml      # Multi-service deployment
    └── requirements.txt        # Dependencies
```

## 🚀 Quick Start

### Option 1: Using Make (Recommended)

```bash
# Install dependencies and setup
make install setup

# Run the complete ML pipeline
make train

# Start all services
make docker-up

# Access the services
# FastAPI: http://localhost:8000
# Streamlit: http://localhost:8501  
# MLflow: http://localhost:5000
```

### Option 2: Manual Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Create directories
mkdir -p logs artifacts models plots data

# 3. Run training pipeline
python main.py

# 4. Start services (in separate terminals)
uvicorn fastapi_app:app --reload
streamlit run streamlit_dashboard.py
mlflow ui
```

## 📊 Data Requirements

Place your data file at `data/triage_data_cleaned.csv` with the following structure:

### Required Columns

**Target Variable:**
- `xaaladda_bukaanka` - Medical condition classification

**Ordinal Features (mild/moderate/high):**
- `heerka_qandhada` - Fever level
- `muddada_qandhada` - Fever duration  
- `madax_xanuun_daran` - Severe headache
- `muddada_madax_xanuunka` - Headache duration
- `muddada_qufaca` - Cough duration
- `muddada_xanuunka` - Pain duration
- `muddada_daalka` - Fatigue duration
- `muddada_mataga` - Nausea duration
- `daal_badan` - Severe fatigue
- `matag_daran` - Severe nausea

**Binary Features (haa/maya, yes/no, 1/0):**
- `qandho` - Fever
- `qufac` - Cough
- `madax_xanuun` - Headache
- `caloosh_xanuun` - Stomach pain
- `daal` - Fatigue
- `matag` - Nausea
- `dhaxan` - Rash
- `qufac_dhiig` - Cough with blood
- `neeftu_dhibto` - Breathing difficulty
- `iftiinka_dhibayo` - Light sensitivity
- `qoortu_adag_tahay` - Stiff neck
- `lalabo` - Dizziness
- `shuban` - Vomiting
- `miisankaga_isdhimay` - Weight loss
- `qandho_daal_leh` - Fever with fatigue
- `matag_dhiig_leh` - Nausea with blood
- `ceshan_karin_qoyaanka` - Cannot feed family

**Nominal Features:**
- `da'da` - Age group
- `nooca_qufaca` - Cough type
- `halka_xanuunku_kaa_hayo` - Pain location

**Optional:**
- `talooyin` - Medical advice/tips (for matching recommendations)

## 🔧 Configuration

Edit `config.py` to customize:

```python
# Data settings
USE_SAMPLE = True
SAMPLE_SIZE = 1000
APPLY_SMOTE = True

# Model settings  
CV_SPLITS = 5
TUNING_CV_SPLITS = 3

# MLflow settings
MLFLOW_EXPERIMENT_NAME = "medical_triage_pipeline"

# Logging
LOG_LEVEL = "INFO"
LOG_FILE = "logs/medical_triage.log"
```

## 🌐 API Usage

### Health Check
```bash
curl http://localhost:8000/health
```

### Make Prediction
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "qandho": "haa",
       "qufac": "maya", 
       "madax_xanuun": "haa",
       "heerka_qandhada": "moderate",
       "da_da": "middle_age"
     }'
```

### Response Format
```json
{
  "predicted_label": "1",
  "predicted_class": "moderate_risk",
  "confidence": 0.85,
  "risk_level": "medium",
  "risk_level_somali": "dhexe",
  "num_symptoms": 2,
  "matching_tips": [
    "Fadlan la tashii dhakhtarka si loo helo baaris dheeri ah.",
    "Raadi caawimaad caafimaad hadii calaamadahu sii daraan."
  ],
  "all_probabilities": {
    "low_risk": 0.15,
    "moderate_risk": 0.85,
    "high_risk": 0.00
  }
}
```

## 🎨 Dashboard Features

The Streamlit dashboard provides:

### 📝 Interactive Input Form
- **Somali/English** bilingual interface
- **Symptom selection** with severity levels
- **Real-time validation** and feedback

### 📊 Results Display  
- **Risk assessment** with confidence scores
- **Somali recommendations** (`talooyin`)
- **Symptom count** and analysis
- **Probability distributions**

### 📈 Visualizations
- **Confusion matrices**
- **ROC curves**
- **Precision-recall curves**
- **SHAP analysis plots**
- **Feature importance charts**
- **Data distribution insights**

### 🔍 Model Analysis
- **Performance comparisons**
- **Classification reports**
- **TP/FP/TN/FN analysis**
- **Historical predictions**

## 🧪 Model Pipeline

### 1. Base Model Training
- **9 algorithms**: LogReg, KNN, Tree, RF, SVM, XGB, Ada, GB, Bag
- **Cross-validation**: 5-fold stratified CV
- **Accuracy-based** ranking
- **MLflow tracking** for all runs

### 2. Hyperparameter Tuning
- **Top 3 models** selected for tuning
- **Dual optimization**: RandomizedSearchCV + Optuna
- **Parallel processing** with fallback
- **Best method** selection per model

### 3. Model Evaluation
- **Comprehensive metrics**: accuracy, precision, recall, F1, AUC
- **Visual analysis**: confusion matrix, ROC, PR curves
- **SHAP explainability**: summary, dependence, force plots
- **Comparison tables** across all models

### 4. Production Model
- **Best performing** model selection
- **Artifact packaging**: model, preprocessor, encoder
- **MLflow registration** for deployment
- **Performance monitoring** ready

## 🔍 SHAP Analysis

The system provides comprehensive model interpretability:

- **🎯 Summary Plots**: Overall feature importance
- **📊 Bar Plots**: Feature contribution ranking  
- **🔗 Dependence Plots**: Feature interaction analysis
- **⚡ Force Plots**: Individual prediction explanations
- **📈 Waterfall Charts**: Step-by-step decision process

## 📊 MLflow Integration

Complete experiment tracking with:

### 🏃 Run Tracking
- **Base models**: All initial training runs
- **Tuning runs**: Hyperparameter optimization
- **Final model**: Production candidate

### 📈 Metrics Logging
- Classification metrics (accuracy, precision, recall, F1)
- Cross-validation scores
- AUC scores (binary/multiclass)
- Custom business metrics

### 📁 Artifact Storage
- Trained models (pickled)
- Preprocessing pipelines
- Evaluation plots (PNG/SVG)
- SHAP analysis results
- Classification reports (CSV)

### 🏷️ Model Registry
- Production model versioning
- Stage management (staging/production)
- Model lineage tracking
- Performance comparison

## 🐳 Docker Deployment

### Single Service Deployment
```bash
# Build image
docker build -t medical-triage .

# Run FastAPI
docker run -p 8000:8000 medical-triage fastapi

# Run Streamlit  
docker run -p 8501:8501 medical-triage streamlit

# Run training
docker run -v $(pwd)/data:/app/data medical-triage train
```

### Full Stack Deployment
```bash
# Start all services
docker-compose up -d

# Scale services
docker-compose up -d --scale api=3

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Services Overview
- **MLflow**: Port 5000 - Experiment tracking UI
- **FastAPI**: Port 8000 - REST API service  
- **Streamlit**: Port 8501 - Interactive dashboard
- **Volumes**: Persistent storage for artifacts

## ☁️ Cloud Deployment

### Streamlit Cloud
1. **Push to GitHub** with all files
2. **Connect repository** to Streamlit Cloud
3. **Set Python version**: 3.10
4. **Main file**: `streamlit_dashboard.py`
5. **Requirements**: `requirements.txt`

### AWS/Azure/GCP
- **Container registry**: Push Docker images
- **Load balancer**: Distribute traffic  
- **Auto-scaling**: Handle demand spikes
- **Monitoring**: CloudWatch/Application Insights
- **CI/CD**: GitHub Actions deployment

## 🔧 Development

### Code Quality
```bash
# Format code
make format
# or
black --line-length=88 *.py

# Lint code  
make lint
# or
flake8 --max-line-length=88 *.py
pylint *.py
```

### Testing
```bash
# Run basic tests
make test

# Test individual components
python -c "from data_preprocessing import preprocess_pipeline; preprocess_pipeline()"
python -c "from fastapi_app import app; print('FastAPI OK')"
```

### Adding New Features

#### New Algorithm
1. Add to `config.py` CLASSIFIERS
2. Update tuning parameters in TUNING_CONFIGS  
3. Test with `train_models.py`

#### New Evaluation Metric
1. Add to `evaluate_models.py` 
2. Update MLflow logging in `mlflow_tracking.py`
3. Include in dashboard visualization

#### New API Endpoint
1. Add route to `fastapi_app.py`
2. Update Pydantic models
3. Add to Streamlit interface

## 🌍 Language Support

### Somali Integration
- **UI Labels**: Bilingual display (Somali/English)
- **Risk Levels**: Somali translations
- **Medical Tips**: Native Somali recommendations
- **Form Validation**: Somali error messages

### Adding Languages
1. Update `SOMALI_MAPPINGS` in `config.py`
2. Modify dashboard labels in `streamlit_dashboard.py`  
3. Add translation files if needed
4. Update API response format

## 📋 Troubleshooting

### Common Issues

#### 🚨 CUDA/GPU Warnings
```python
# XGBoost automatically falls back to CPU
# Check logs for device switching messages
```

#### 📁 Missing Data File
```bash
# Ensure data file exists
ls data/triage_data_cleaned.csv

# Check column names match config
head data/triage_data_cleaned.csv
```

#### 🔌 API Connection Failed
```bash
# Check if FastAPI is running
curl http://localhost:8000/health

# Restart with logs
uvicorn fastapi_app:app --reload --log-level debug
```

#### 🐳 Docker Issues
```bash
# Rebuild images
docker-compose build --no-cache

# Check container logs
docker-compose logs api

# Reset everything
docker-compose down -v && docker-compose up -d
```

### Performance Optimization

#### 🚀 Speed Up Training
- Reduce `SAMPLE_SIZE` in config.py
- Decrease `CV_SPLITS` and `TUNING_CV_SPLITS`
- Use fewer hyperparameter combinations
- Disable SMOTE for faster training

#### 💾 Memory Usage
- Enable data sampling: `USE_SAMPLE = True`
- Reduce SHAP sample size in evaluation
- Use lighter algorithms (LogReg, Tree)
- Implement batch prediction for large datasets

## 📚 Additional Resources

### Documentation
- **Scikit-learn**: [https://scikit-learn.org/](https://scikit-learn.org/)
- **XGBoost**: [https://xgboost.readthedocs.io/](https://xgboost.readthedocs.io/)
- **MLflow**: [https://mlflow.org/docs/](https://mlflow.org/docs/)
- **FastAPI**: [https://fastapi.tiangolo.com/](https://fastapi.tiangolo.com/)
- **Streamlit**: [https://docs.streamlit.io/](https://docs.streamlit.io/)

### Research Papers
- SMOTE: Chawla et al. (2002) - Synthetic Minority Over-sampling Technique
- SHAP: Lundberg & Lee (2017) - A Unified Approach to Interpreting Model Predictions
- XGBoost: Chen & Guestrin (2016) - XGBoost: A Scalable Tree Boosting System

## 🤝 Contributing

1. **Fork** the repository
2. **Create** feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** changes (`git commit -m 'Add amazing feature'`)
4. **Push** to branch (`git push origin feature/amazing-feature`)  
5. **Open** Pull Request

### Development Setup
```bash
# Clone repository
git clone <repo-url>
cd medical-triage-system

# Setup development environment
make dev-setup

# Run tests before committing
make test
make lint
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Somali Medical Community** for domain expertise
- **Open Source Contributors** for excellent tools
- **Research Community** for algorithmic foundations

## 📞 Support

- **📧 Email**: [your-email@domain.com]
- **🐛 Issues**: [GitHub Issues](https://github.com/your-repo/issues)  
- **💬 Discussions**: [GitHub Discussions](https://github.com/your-repo/discussions)
- **📖 Wiki**: [Project Wiki](https://github.com/your-repo/wiki)

---

*Built with ❤️ for the Somali healthcare community*

*La dhisay jacaylkii bulshada caafimaadka Soomaaliyeed*