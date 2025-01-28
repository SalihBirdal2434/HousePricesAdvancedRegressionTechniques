# House Prices Prediction using Advanced Regression Techniques

## Overview
This project aims to develop a **predictive model** for estimating house prices based on various housing attributes. Utilizing **machine learning regression techniques** and extensive **feature engineering**, the model is trained on a dataset containing **79 explanatory variables** that describe nearly every aspect of residential homes.

## Dataset
The dataset used for this project is derived from the **Kaggle - House Prices: Advanced Regression Techniques** competition. It includes **79 features** such as:
- **Property attributes** (square footage, number of rooms, garage type, etc.)
- **Neighborhood and zoning classifications**
- **Structural characteristics** (house style, condition, and materials)
- **Sale information** (sale type, condition, and year of sale)

The dataset is split into:
- **train.csv** – Includes labeled house prices for training the model.
- **test.csv** – Contains test data for predictions without house prices.
- **submission.csv** – Stores the final model predictions in Kaggle’s submission format.

## Methodology
### 1. **Exploratory Data Analysis (EDA)**
   - Data distribution visualization
   - Handling missing values
   - Outlier detection
   - Correlation analysis

### 2. **Feature Engineering**
   - **Categorical encoding**: One-hot encoding for categorical variables.
   - **Handling missing values**: Imputation and feature transformation.
   - **Creating new features**: Generating additional meaningful features.
   - **Feature selection**: Identifying the most influential variables.

### 3. **Model Training**
   - Multiple regression algorithms were tested, including:
     - **Linear Regression**
     - **Random Forest**
     - **Gradient Boosting (XGBoost, LightGBM, CatBoost)**
   - **Hyperparameter tuning** was conducted to optimize performance.

### 4. **Model Evaluation**
   - **Root Mean Squared Error (RMSE)** was used as the evaluation metric.
   - **Feature Importance Analysis** was performed using SHAP and other interpretability techniques.

## Results
- The **best-performing model** was fine-tuned using **XGBoost**, achieving **low RMSE** on validation data.
- Feature importance analysis revealed **key variables** influencing house prices, such as:
  - Overall Quality (`OverallQual`)
  - Living Area (`GrLivArea`)
  - Neighborhood (`Neighborhood`)
  - Year Built (`YearBuilt`)

## Files in the Repository
- **EDA.ipynb** – Notebook for **exploratory data analysis** and visualization.
- **Project.ipynb** – Contains **feature engineering, model training, and evaluation**.
- **importances.png** – Visual representation of **feature importance**.
- **train.csv, test.csv** – Training and test datasets.
- **submission.csv** – Predictions for Kaggle submission.

## How to Run the Project
### 1. Install Dependencies
Ensure you have Python and the required libraries installed:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost lightgbm catboost
