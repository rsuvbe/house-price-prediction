 🏠 House Price Prediction

Predicting house prices in Ames, Iowa using Machine Learning.

📂 **Dataset:** [Ames Housing Data on Kaggle](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data)

## 🎯 Results

| Metric | Value |
|--------|-------|
| **Model** | Linear Regression |
| **R² Score** | 0.93 |
| **RMSE** | $22,902 |
| **Features** | 301 (after encoding) |

**Interpretation:** The model explains 93% of price variance. Average prediction error is ~$23k.

## 🛠 Tech Stack

- **Language:** Python
- **Data:** Pandas, NumPy
- **ML:** Scikit-Learn (Linear Regression, Ridge, Lasso, Pipelines)
- **Viz:** Matplotlib, Seaborn
- **Env:** Jupyter Notebook

## 📁 Project Structure

```text
house-price-project/
├── data/                    # Raw data (not included in Git)
├── notebooks/               # Jupyter notebooks for EDA & training
├── models/                  # Saved .pkl models (gitignored)
├── results/                 # Generated plots & metrics
├── .gitignore               # Git ignore rules
├── requirements.txt         # Dependencies
└── README.md                # This file
```

## 🚀 Quick Start

```bash
# 1. Clone repo
git clone https://github.com/YOUR_USERNAME/house-price-prediction.git  
cd house-price-prediction

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add data
# Download train.csv from Kaggle and place in data/

# 5. Run analysis
jupyter notebook notebooks/01_exploratory_analysis.ipynb
```

## 💻 Usage Example

```python
import joblib
import pandas as pd

# Load artifacts
model = joblib.load('models/linear_model.pkl')
preprocessor = joblib.load('models/preprocessor.pkl')

# Prepare new data
new_house = pd.DataFrame({
    'OverallQual': [8],
    'GrLivArea': [2000],
    'GarageCars': [2],
    # ... other features
})

# Preprocess and predict
processed = preprocessor.transform(new_house)
price = model.predict(processed)

print(f"Estimated price: ${price[0]:,.0f}")
```

## 📊 Key Findings

- **Top 3 Features:** `OverallQual`, `GrLivArea`, `GarageCars` drive price the most.
- **Data Quality:** Log-transformation of target reduced skew from 1.88 → 0.12, improving model stability.
- **Model Choice:** Simple Linear Regression performed best (R²=0.93); Ridge/Lasso didn't improve metrics.


## 🎓 Key Learnings

- ✅ Log-transformation is critical for skewed target variables in regression.
- ✅ Pipelines prevent data leakage during preprocessing.
- ✅ Feature importance analysis helps interpret model decisions.
- ✅ Regularization (Ridge/Lasso) is powerful but requires careful alpha tuning.


## 📈 Visualizations

### Predictions vs Actual
![Model Predictions](results/07_model_predictions.png)

*Figure: Each point represents a house. Points close to the red diagonal line indicate accurate predictions.*

### Feature Importance
![Feature Importance](results/09_feature_importance.png)

*Figure: Top 15 features by coefficient magnitude in the Linear Regression model.*


## 👤 Author

Beksultan — rsuvbe


