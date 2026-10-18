# Week 24 Practice Problem 1: CatBoost Classification on Adult Income

A polished end-to-end machine learning project that classifies whether annual income exceeds $50K using the Adult Census dataset from OpenML.

## Project highlights

- Native CatBoost handling of categorical features without one-hot encoding
- Clean preprocessing for mixed numerical and categorical inputs
- Baseline model evaluation with accuracy, precision, recall, F1-score, confusion matrix, and classification report
- Hyperparameter tuning using 3-fold GridSearchCV
- Final model evaluation with feature importance plotting
- Random forest comparison for context
- Reproducible project structure and artifact outputs

## Dataset

The project uses:

- `sklearn.datasets.fetch_openml('adult', version=2)`
- 48,842 samples
- 14 features, including both numeric and categorical inputs
- Binary target: `<=50K` vs `>50K`

## Project structure

- `Week24_practice_1_catboost.py` — complete end-to-end training pipeline
- `outputs/top_10_feature_importance_catboost.png` — top 10 feature importances
- `requirements.txt` — dependency list
- `.gitignore` — standard Python project exclusions

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# or .venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

## Run the project

```bash
python Week24_practice_1_catboost.py
```

## Verified results

This project was executed successfully in the current environment. The final run produced:

- Baseline CatBoost accuracy: 0.8777
- Best tuned parameters: `{'depth': 8, 'l2_leaf_reg': 1, 'learning_rate': 0.2}`
- Best CV accuracy: 0.8549
- Final CatBoost test accuracy: 0.8704
- RandomForest comparision accuracy: 0.8573

CatBoost outperformed the one-hot RandomForest baseline on this data split.

## Notes

- Missing categorical data is filled with a placeholder label before model training.
- CatBoost receives categorical columns via `cat_features`, keeping the workflow faithful to native categorical handling.
