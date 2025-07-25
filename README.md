## End-to-End MLProject
# 📊 Student Performance Predictor

A machine learning web application that predicts student math scores based on demographic information and academic performance indicators.

![Student Performance Predictor](C:\Users\DELL\OneDrive\Pictures\Screenshots\Screenshot (44).png)
<img width="1874" height="944" alt="Screenshot (45)" src="https://github.com/user-attachments/assets/b3e54ce6-62a3-476f-a9ac-a3c8e4fea6fb" />
C:\Users\DELL\OneDrive\Pictures\Screenshots\Screenshot (46).png


## 🎯 Project Overview

This project implements an end-to-end machine learning pipeline to predict student math performance using various demographic and academic factors. The application features a modern web interface built with Flask and employs multiple regression algorithms to achieve optimal prediction accuracy.

## ✨ Features

- **🤖 Machine Learning Pipeline**: Complete ML workflow from data ingestion to model deployment
- **📊 Multiple Algorithms**: Comparison of various regression models (Random Forest, XGBoost, CatBoost, etc.)
- **🔧 Hyperparameter Tuning**: GridSearchCV implementation for optimal model performance
- **🌐 Modern Web Interface**: Responsive design with glassmorphism aesthetics
- **📱 Mobile-Friendly**: Fully responsive design for all devices
- **⚡ Real-time Predictions**: Instant math score predictions based on input parameters

## 🛠️ Technologies Used

### Backend
- **Python 3.8+**
- **Flask** - Web framework
- **Scikit-learn** - Machine learning library
- **Pandas** - Data manipulation
- **NumPy** - Numerical computing
- **XGBoost** - Gradient boosting framework
- **CatBoost** - Gradient boosting library

### Frontend
- **HTML5** - Markup language
- **CSS3** - Styling with modern features (glassmorphism, animations)
- **JavaScript** - Client-side interactions

### Data Processing
- **StandardScaler** - Feature scaling
- **LabelEncoder** - Categorical encoding
- **GridSearchCV** - Hyperparameter tuning

## 📁 Project Structure

```
student-performance-predictor/
│
├── src/
│   ├── components/
│   │   ├── data_ingestion.py          # Data loading and splitting
│   │   ├── data_transformation.py     # Feature engineering and preprocessing
│   │   └── model_trainer.py           # Model training and evaluation
│   │
│   ├── pipelines/
│   │   └── predict_pipeline.py        # Prediction pipeline for new data
│   │
│   ├── utils.py                       # Utility functions
│   ├── exception.py                   # Custom exception handling
│   └── logger.py                      # Logging configuration
│
├── templates/
│   ├── index.html                     # Welcome page
│   └── home.html                      # Prediction form
│
├── artifacts/                         # Saved models and preprocessors
│   ├── model.pkl
│   └── preprocessor.pkl
│
├── app.py                            # Flask application
├── requirements.txt                   # Project dependencies
└── README.md                         # Project documentation
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/student-performance-predictor.git
   cd student-performance-predictor
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   python app.py
   ```

5. **Open your browser**
   Navigate to `http://localhost:5000`

## 📊 Model Performance

The application trains and compares multiple regression algorithms:

| Model | Features |
|-------|----------|
| **Random Forest** | Ensemble method with bagging |
| **Gradient Boosting** | Sequential ensemble learning |
| **XGBoost** | Optimized gradient boosting |
| **CatBoost** | Handling categorical features |
| **Decision Tree** | Interpretable tree-based model |
| **Linear Regression** | Baseline linear model |
| **Ridge/Lasso** | Regularized linear models |

The best performing model is automatically selected based on R² score and deployed for predictions.

## 📈 Input Features

The model uses the following features to predict math scores:

- **Gender**: Male/Female
- **Race/Ethnicity**: Groups A-E
- **Parental Education Level**: High school to Master's degree
- **Lunch Type**: Standard/Free or reduced
- **Test Preparation Course**: Completed/None
- **Reading Score**: 0-100
- **Writing Score**: 0-100

## 🎨 User Interface

### Welcome Page
- Modern glassmorphism design
- Animated background particles
- Feature highlights
- Responsive layout

### Prediction Form
- Clean, intuitive form design
- Real-time validation
- Instant results display
- Mobile-optimized interface

## 🔧 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Welcome page |
| `/predictdata` | GET | Prediction form |
| `/predictdata` | POST | Submit prediction request |

## 📝 Usage Example

1. Navigate to the application homepage
2. Click "Start Prediction" 
3. Fill in the student information:
   - Select demographic details
   - Enter reading and writing scores
4. Click "Predict Math Score"
5. View the predicted math score

## 🧪 Model Training Pipeline

```python
# Data Ingestion
raw_data = pd.read_csv('student_data.csv')

# Data Transformation  
preprocessor = create_preprocessor()
processed_data = preprocessor.fit_transform(raw_data)

# Model Training
models = {
    'Random Forest': RandomForestRegressor(),
    'XGBoost': XGBRegressor(),
    # ... other models
}
best_model = train_and_evaluate_models(models, processed_data)

# Model Deployment
save_model(best_model, 'artifacts/model.pkl')
```

## 🔍 Model Evaluation

The application uses the following metrics for model evaluation:

- **R² Score**: Coefficient of determination
- **Training Score**: Performance on training data
- **Test Score**: Performance on validation data
- **Cross-validation**: 3-fold cross-validation for hyperparameter tuning

## 📊 Data Preprocessing

- **Categorical Encoding**: Label encoding for categorical variables
- **Feature Scaling**: StandardScaler for numerical features
- **Pipeline Integration**: Seamless preprocessing pipeline

## 🛡️ Error Handling

- Custom exception classes for better error tracking
- Comprehensive logging system
- User-friendly error messages
- Graceful failure handling

## 🔮 Future Enhancements

- [ ] Add more sophisticated ML models (Neural Networks, Ensemble methods)
- [ ] Implement model explainability (SHAP, LIME)
- [ ] Add data visualization dashboard
- [ ] Include confidence intervals for predictions
- [ ] Implement A/B testing for model versions
- [ ] Add user authentication and prediction history


---

⭐ **If you found this project helpful, please give it a star!** ⭐
