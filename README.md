# House Prices - Advanced Regression Techniques

Kaggle Competition: Predicting house prices using advanced regression techniques.

## Project Overview

This project implements a comprehensive machine learning pipeline to predict house sale prices based on various features such as lot size, number of bedrooms, neighborhood quality, and many other property characteristics.

## Pedagogical Goals

This project serves as a comprehensive learning exercise covering the following key aspects of machine learning and data science:

### 1. **Navigating Incomplete Data and Documentation**
   - Working with real-world datasets containing missing values
   - Interpreting domain-specific documentation to understand feature meanings
   - Distinguishing between "absence" (NA = "None") vs true missing data
   - Making informed decisions based on incomplete information

### 2. **Handling Different Forms of Imputation**
   - **Simple imputation**: Median, mode, constant values
   - **Conditional imputation**: Complex scenarios requiring custom transformers (e.g., veneer features)
   - **Domain knowledge-based imputation**: Understanding when NA means absence vs. missing data

### 3. **Scaling Strategies**
   - **MinMaxScaler**: For discrete features and ordinal encoded features
   - **RobustScaler**: For continuous features with outliers
   - **StandardScaler**: When appropriate
   - Understanding when and why to apply different scaling methods

### 4. **Encoding Techniques**
   - **Ordinal Encoding**: For features with natural order (quality ratings)
   - **One-Hot Encoding**: For low-cardinality nominal features
   - **Target Encoding**: For high-cardinality categorical features
   - **Rare value grouping**: Handling categories with low frequency

### 5. **Feature Engineering**
   - Creating derived features (e.g., age features from year features)
   - Handling zero-inflated distributions
   - Log transformations for skewed features
   - Cyclical feature encoding (e.g., month of sale)
   - Decomposition strategies for complex distributions

### 6. **Feature Selection**
   - Variance-based selection (VarianceThreshold)
   - Permutation importance analysis
   - Feature importance from multiple models
   - Composite scoring for feature selection

### 7. **Model Tuning**
   - Hyperparameter optimization using Optuna
   - Cross-validation strategies
   - Model comparison and selection
   - Ensemble methods (Stacking Regressor)

## Technical Implementation

### Preprocessing Pipeline
- Custom transformers for conditional imputation (`VeneerImputer`)
- Custom transformers for rare value grouping (`RareValuesGrouper`)
- Feature engineering pipeline (age features, transformations)
- Comprehensive encoding and scaling pipeline

### Models Evaluated
- **Linear Models**: Ridge, Lasso, ElasticNet
- **Tree-based**: Random Forest, XGBoost, LightGBM, CatBoost
- **Ensemble**: Stacking Regressor with Ridge meta-learner

### Final Model
The final submission uses an ensemble approach combining:
- CatBoost (optimized with Optuna)
- XGBoost (optimized with Optuna)
- LightGBM (optimized with Optuna)
- Stacked with Ridge meta-learner

## Results

**Final Kaggle Test Submission Score: 0.12175 (RMSLE)**

This score represents a significant improvement from the initial baseline and demonstrates the effectiveness of:
- Comprehensive EDA and feature understanding
- Robust preprocessing pipeline
- Careful feature engineering
- Systematic model selection and hyperparameter tuning
- Ensemble methods

## Repository Structure

```
housing_kaggle/
├── data/
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
├── models/
│   ├── best_catboost.pkl
│   ├── best_xgboost.pkl
│   ├── best_lightgbm.pkl
│   └── best_elasticnet.pkl
├── houses_kaggle_competition.ipynb
└── README.md
```

## Key Learnings

1. **Domain Knowledge Matters**: Understanding what missing values mean in context (absence vs. missing) is crucial for proper imputation.

2. **Pipeline Design**: Creating reusable, testable preprocessing pipelines with sklearn ensures consistency between train and test sets.

3. **Feature Engineering Impact**: Thoughtful feature engineering (age features, transformations) can significantly improve model performance.

4. **Model Diversity**: Combining different types of models (boosting, bagging, linear) in an ensemble often outperforms individual models.

5. **Hyperparameter Tuning**: Systematic hyperparameter optimization (Optuna) is essential for achieving best performance.

## References

- [Kaggle Competition](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)
- Data documentation and feature descriptions from Kaggle
