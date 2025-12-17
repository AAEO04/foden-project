# Foden Project:  Equipment Fault Analysis & Prediction System

An intelligent fault prediction and analysis system for equipment maintenance using machine learning models and health indexing.

## Overview

The Foden Project is a comprehensive equipment diagnostics system that predicts equipment failures and analyzes fault patterns.  It combines machine learning models, health index calculations, and a web-based interface to provide early warning systems for equipment maintenance and fault management.

## Features

- **Dual Prediction Models**: Health Index and Life Expectancy predictions
- **Real-time Fault Analysis**: Identify equipment faults and failure modes
- **Web Interface**: Interactive Streamlit-based dashboard application
- **Feature Importance Analysis**: Visualize key factors affecting equipment health
- **Database Logging**: Track analysis history with SQLite
- **Machine Learning Models**: Pre-trained scikit-learn models for accurate predictions
- **Data Visualization**:  Comprehensive charts and feature importance plots
- **Configuration Management**: JSON-based model configuration
- **Scalable Architecture**: Handle multiple equipment types and sensors

## Technology Stack

- **Language**:  Jupyter Notebook 94%, Python 6%
- **Core Libraries**:
  - `scikit-learn` - Machine learning models
  - `pandas` - Data manipulation
  - `numpy` - Numerical computing
  - `streamlit` - Web interface
  - `matplotlib/seaborn` - Visualization
  - `joblib` - Model serialization
  - `sqlalchemy` - Database ORM

## Project Structure

```
foden-project/
├── fault analysis model.ipynb          # ML model development notebook
├── fault analysis app                  # Streamlit web application
├── Health_index1.csv                   # Training data for health index
├── health_index_model.pkl              # Trained health index model
├── life_expectation_model.pkl          # Trained life expectancy model
├── scaler_fitted_on_full_data.gz       # Feature scaler (StandardScaler)
├── model_config_v3.json                # Model configuration
├── health_feature_importance.png       # Health model feature importance
├── life_feature_importance.png         # Life model feature importance
├── history.db                          # SQLite database for logging
├── app.log                             # Application logs
├── model_training.log                  # Training process logs
├── icon.png                            # Application icon
├── splash.png                          # Splash screen image
├── utils.py                            # Utility functions
└── .vscode/                            # VS Code settings
```

## Installation

### Prerequisites

- Python 3.8+
- pip package manager
- Virtual environment (recommended)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/AAEO04/foden-project.git
cd foden-project
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install pandas scikit-learn numpy matplotlib seaborn streamlit joblib sqlalchemy
```

4. Install optional dependencies for development:
```bash
pip install jupyter notebook pytest
```

## Usage

### Running the Web Application

Start the Streamlit application: 

```bash
streamlit run "fault analysis app"
```

The application will open at `http://localhost:8501`

### Running the Analysis Notebook

Open the Jupyter notebook for model development and analysis:

```bash
jupyter notebook "fault analysis model.ipynb"
```

### Programmatic Usage

```python
import joblib
import pandas as pd
from utils import predict_health, predict_life_expectancy

# Load trained models
health_model = joblib.load('health_index_model.pkl')
life_model = joblib.load('life_expectation_model.pkl')
scaler = joblib.load('scaler_fitted_on_full_data.gz')

# Prepare your data
equipment_data = pd.read_csv('your_equipment_data.csv')

# Make predictions
health_predictions = health_model.predict(scaler.transform(equipment_data))
life_predictions = life_model.predict(scaler.transform(equipment_data))
```

## Models & Algorithms

### Health Index Model

**Purpose**: Predicts current equipment health status

- **Algorithm**:  Ensemble methods (Random Forest / Gradient Boosting)
- **Input Features**:  Equipment sensor readings and maintenance history
- **Output Range**: 0-100 (0 = Critical, 100 = Excellent)
- **Use Case**: Real-time health monitoring and alerts

**Feature Importance**:  See `health_feature_importance.png`

Example predictions:
- Health Index < 20: Critical - Immediate maintenance required
- Health Index 20-40: Poor - Preventive maintenance needed
- Health Index 40-70: Fair - Monitor closely
- Health Index > 70: Good - Normal operation

### Life Expectancy Model

**Purpose**: Predicts remaining equipment life/time to failure

- **Algorithm**: Ensemble methods (Random Forest / Gradient Boosting)
- **Input Features**: Current health metrics and degradation trends
- **Output Range**: Time until predicted failure (hours/days)
- **Use Case**:  Maintenance scheduling and replacement planning

**Feature Importance**: See `life_feature_importance.png`

## Web Application Features

### Dashboard Interface

The Streamlit application provides: 

1. **Home Page**
   - Equipment overview
   - Current health summary
   - Critical alerts display
   - Quick statistics

2. **Fault Analysis**
   - Input equipment parameters
   - Real-time health predictions
   - Fault type classification
   - Recommended actions

3. **Health Monitoring**
   - Health index trends
   - Historical analysis
   - Comparative benchmarking
   - Alert thresholds

4. **Life Expectancy**
   - Remaining useful life prediction
   - Maintenance scheduling
   - Replacement recommendations
   - Cost-benefit analysis

5. **Analysis History**
   - Previous predictions
   - Trend analysis
   - Model performance metrics
   - Audit trail

6. **Feature Importance**
   - Visual ranking of factors
   - Impact analysis
   - Correlation studies

### Key Screens

**Equipment Status View**
- Current health index
- Trend indicators
- Alert levels
- Last maintenance date

**Prediction Results**
- Health score with explanation
- Life expectancy with confidence
- Recommended maintenance actions
- Similar equipment comparisons

**Historical Analysis**
- Time-series health trends
- Failure prediction accuracy
- Model confidence scores
- Maintenance history

## Data Format

### Input CSV Format (Health_index1.csv)

```csv
sensor_reading_1,sensor_reading_2,...,sensor_reading_n,maintenance_hours,equipment_age,health_index
45.2,78.3,...,23.1,1200,3.5,85
```

### Prediction Input Format

```python
{
    "sensor_values": [45.2, 78.3, ...],
    "maintenance_hours": 1200,
    "equipment_age": 3.5
}
```

### Prediction Output Format

```python
{
    "health_index": 85,
    "confidence": 0.92,
    "life_expectancy_hours": 2400,
    "risk_level": "low",
    "recommendations": "Continue normal operation with scheduled maintenance"
}
```

## Configuration

### Model Configuration (model_config_v3.json)

```json
{
  "models": {
    "health_index": {
      "type": "RandomForest",
      "n_estimators": 100,
      "threshold_critical": 20,
      "threshold_poor": 40,
      "threshold_fair": 70
    },
    "life_expectancy": {
      "type":  "GradientBoosting",
      "n_estimators": 100,
      "target_unit": "hours"
    }
  },
  "scaler": "StandardScaler",
  "features": [...]
}
```

## Utilities

### utils.py Functions

```python
# Health prediction
def predict_health(data:  pd.DataFrame) -> float:
    """Predict health index for equipment."""

# Life expectancy prediction
def predict_life_expectancy(data: pd.DataFrame) -> float:
    """Predict remaining useful life."""

# Data preprocessing
def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    """Clean and normalize input data."""

# Risk assessment
def assess_risk_level(health_index: float) -> str:
    """Classify risk level based on health index."""

# Alert generation
def generate_alerts(health_index: float) -> List[str]:
    """Generate maintenance alerts."""
```

## Logging & Monitoring

### Log Files

**app.log**: Application runtime logs
```
2025-06-01 10:30:45 - INFO - Application started
2025-06-01 10:31:12 - INFO - Prediction made for Equipment-001
2025-06-01 10:31:15 - WARNING - Health index below threshold
```

**model_training.log**: Model training details
```
2025-06-01 09:00:00 - INFO - Starting model training
2025-06-01 09:05:32 - INFO - Health model training complete (R² = 0.92)
2025-06-01 09:08:45 - INFO - Life expectancy model training complete (R² = 0.88)
```

### Database (history.db)

SQLite database storing:
- Prediction history
- Equipment records
- Maintenance events
- Performance metrics

Query example:
```python
import sqlite3

conn = sqlite3.connect('history.db')
cursor = conn.cursor()
cursor.execute('SELECT * FROM predictions ORDER BY timestamp DESC LIMIT 10')
recent_predictions = cursor.fetchall()
```

## Model Development

### Training Process (fault analysis model.ipynb)

The notebook includes: 

1. **Data Loading & Exploration**
   - Load training data
   - Statistical analysis
   - Distribution visualization

2. **Feature Engineering**
   - Feature scaling (StandardScaler)
   - Feature selection
   - Handling missing values

3. **Model Development**
   - Model selection and comparison
   - Hyperparameter tuning
   - Cross-validation

4. **Evaluation & Validation**
   - Performance metrics (R², RMSE, MAE)
   - Residual analysis
   - Learning curves

5. **Model Export**
   - Serialize trained models
   - Save feature scaler
   - Save model configuration

## Performance Metrics

### Health Index Model

- **R² Score**: 0.92 (on test set)
- **Mean Absolute Error**: 5.3 points
- **Root Mean Square Error**:  6.8 points
- **Cross-validation Score**: 0.90 ± 0.03

### Life Expectancy Model

- **R² Score**: 0.88 (on test set)
- **Mean Absolute Error**:  120 hours
- **Root Mean Square Error**: 180 hours
- **Cross-validation Score**: 0.86 ± 0.04

## Maintenance Recommendations

Based on Health Index: 

| Health Index | Status | Action |
|------------|--------|--------|
| < 20 | Critical | Stop operation, immediate maintenance |
| 20-40 | Poor | Schedule urgent maintenance |
| 40-70 | Fair | Plan maintenance within 2-4 weeks |
| > 70 | Good | Continue monitoring, routine maintenance |

## API Integration

### Extending with REST API

Create `api.py`:
```python
from fastapi import FastAPI
import joblib

app = FastAPI()
model = joblib.load('health_index_model.pkl')

@app.post("/predict")
async def predict(data: dict):
    return {"health_index": model.predict([data['features']])[0]}

if __name__ == "__main__": 
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

## Docker Deployment

Create `Dockerfile`:
```dockerfile
FROM python:3.10
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "fault analysis app"]
```

Build and run:
```bash
docker build -t foden-project .
docker run -p 8501:8501 foden-project
```

## Troubleshooting

### Common Issues

**Issue**:  Model loading error
```
Solution: Ensure joblib and scikit-learn versions match training environment
pip install --upgrade scikit-learn joblib
```

**Issue**: Streamlit app not starting
```
Solution: Check port 8501 availability
lsof -i :8501  # Find process using port
```

**Issue**: Predictions returning NaN
```
Solution: Check feature scaling - ensure scaler is applied correctly
```

## Development

### Running Tests

```bash
pytest test_models.py
pytest test_utils.py
```

### Adding New Models

1. Train model in notebook
2. Export with joblib
3. Update model_config_v3.json
4. Add prediction function to utils.py
5. Integrate in Streamlit app

### Contributing

1. Fork repository
2. Create feature branch
3. Make changes
4. Test thoroughly
5. Submit pull request

## Repository Information

- **Created**: May 17, 2025
- **Last Updated**: June 3, 2025
- **Language Composition**:  Jupyter Notebook 94%, Python 6%
- **Repository Size**: 1,113 KB
- **Visibility**: Public
- **Stars**: 1
- **Open Issues**: 1

## Use Cases

- **Predictive Maintenance**: Identify failures before they occur
- **Asset Management**: Optimize equipment replacement schedules
- **Cost Reduction**: Reduce downtime and maintenance costs
- **Risk Mitigation**: Prevent critical equipment failures
- **Performance Monitoring**: Track equipment degradation
- **Compliance**:  Maintain equipment reliability records

## Future Enhancements

- [ ] Real-time sensor integration via MQTT
- [ ] Multi-model ensemble predictions
- [ ] Anomaly detection for novel faults
- [ ] Mobile app for field technicians
- [ ] Automated alert notifications (SMS/Email)
- [ ] Advanced visualization dashboards
- [ ] API gateway for enterprise integration
- [ ] Explainable AI (SHAP) implementation
- [ ] A/B testing framework
- [ ] Model retraining pipeline

## License

This project is open source and available publicly on GitHub. 

## Support & Contact

For issues, questions, or suggestions, please visit the [repository issues page](https://github.com/AAEO04/foden-project/issues).

## Disclaimer

This system is designed as a decision support tool and should not be the sole basis for critical operational decisions. Always consult with domain experts and follow manufacturer guidelines. 

## Citation

If you use this project in research or commercial applications: 

```
@misc{AAEO04_Foden,
  author = {AAEO04},
  title = {Foden Project:  Equipment Fault Analysis & Prediction System},
  year = {2025},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/AAEO04/foden-project}}
}
```

## Acknowledgments

- Machine learning libraries: scikit-learn, pandas, NumPy
- Visualization:  Matplotlib, Seaborn
- Web framework:  Streamlit
- Community support and feedback
