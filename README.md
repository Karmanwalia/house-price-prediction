# House Price Prediction App

A machine learning web app that predicts house sale prices based on key property features, built end-to-end from raw data to a deployed, interactive application.

**Live app:** [Add your Streamlit Cloud link here]
**Dataset:** [Kaggle - House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)

---

## Overview

This project predicts a house's sale price given its features (square footage, quality rating, number of bathrooms, etc.) using regression models trained on the Ames, Iowa housing dataset. It's a feature-based price predictor, not a housing market forecaster, estimating what a specific house should sell for, not where the overall market is headed. 

The project covers the full pipeline: data cleaning, exploratory analysis, feature encoding, model training and comparison, and deployment as a live, interactive web app.

## Dataset

- **Source:** Kaggle's "House Prices - Advanced Regression Techniques" competition dataset
- **Size:** 1,460 houses, 80 original features (property type, quality ratings, square footage, neighborhood, etc.)
- **Target:** `SalePrice` (USD)

## Approach

### 1. Exploratory Data Analysis
- Identified that `SalePrice` is right-skewed (mean > median), which motivated a log-transform later on
- Found the strongest correlated features with price: `OverallQual` (0.79), `GrLivArea` (0.71), `GarageCars` (0.64), `TotalBsmtSF` (0.61)
- Spotted two outliers — houses over 4,000 sq ft that sold far below market value — and traced the cause to their `SaleCondition` being "Partial" (sold before construction was finished). These were removed since they don't reflect true market pricing.

### 2. Data Cleaning
Missing values were handled based on *why* they were missing, not with a blanket approach:
- Columns like `PoolQC`, `Fence`, `GarageType` etc. were missing because the house simply doesn't have that feature — filled with `"None"` (categorical) or `0` (numeric)
- `LotFrontage` and `Electrical` had genuinely unknown missing values — filled with median and mode respectively

### 3. Feature Encoding
Categorical variables were one-hot encoded (`pd.get_dummies`), expanding the dataset from 81 to 260 columns.

### 4. Model Training & Comparison

| Model | MAE | R² |
|---|---|---|
| Linear Regression | Unstable (predicted negative prices) | 0.35 |
| Ridge Regression (raw price) | $16,691 | 0.905 |
| **Ridge Regression (log-transformed price)** | **$14,529** | **0.928** |
| Random Forest | $17,076 | 0.891 |

**Plain Linear Regression failed** — with 259 features (many correlated, e.g. `GarageCars`/`GarageArea`), it produced unstable coefficients and even predicted a negative sale price on the test set. Switching to **Ridge Regression**, which penalizes extreme coefficients, fixed this immediately.

**Log-transforming the target** further improved performance by correcting for the right-skewed price distribution, which linear models handle better.

**Random Forest underperformed Ridge** despite being a more complex model. With ~1,166 training rows spread across 259 mostly-sparse one-hot-encoded columns, it likely struggled to find useful splits — a good reminder that a well-understood simpler model, matched to the data's actual problem (skew), can beat a more complex one used off-the-shelf.

**Final model:** Ridge Regression (alpha=10) on log-transformed `SalePrice`.

### 5. Deployment
The final model was wrapped in a Streamlit app and deployed via Streamlit Community Cloud. The app takes six key inputs (overall quality, living area, garage capacity, basement size, bathrooms, year built) and fills in the remaining 250+ features with dataset medians behind the scenes, so users get a simple interface without needing to understand every underlying feature.

**Deployment debugging:** Initial deployment failed with `ModuleNotFoundError` (missing `requirements.txt`), then with a `Segmentation fault` traced to a scikit-learn version mismatch between the training environment (1.6.1) and the deployment environment (1.9.0). Fixed by upgrading and retraining the model under scikit-learn 1.9.0 to match the deployment environment.

## Tech Stack

- **Data & modeling:** Python, pandas, scikit-learn, joblib
- **Development:** Google Colab
- **Deployment:** Streamlit, Streamlit Community Cloud
- **Visualization:** Matplotlib

## Known Limitations / Future Improvements

- The `Id` column was inadvertently left in as a feature; it carries no predictive meaning and should be dropped
- Trained on Ames, Iowa data — patterns don't generalize to other housing markets (different price scales, different value drivers)
- Random Forest was used with default hyperparameters; tuning (`max_depth`, `min_samples_leaf`, etc.) may improve its performance
- No cross-validation was used — results are based on a single 80/20 train/test split

## Running Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

