# Week 24 Practice Problem 1: CatBoost Classification on Adult Income

This project is now completed and implemented in `Week24_practice_1_catboost.py`.

## Completed work

- Loads the Adult income dataset from OpenML using `fetch_openml('adult', version=2)`.
- Identifies categorical versus numerical columns.
- Splits the data into train and test sets using `random_state=42`.
- Trains a CatBoost baseline model without one-hot encoding.
- Runs a 3-fold `GridSearchCV` on the required search grid:
  - `depth`: [4, 6, 8]
  - `learning_rate`: [0.03, 0.1, 0.2]
  - `l2_leaf_reg`: [1, 3, 5]
- Fits the final tuned CatBoost model.
- Evaluates accuracy, precision, recall, F1-score, confusion matrix, and classification report.
- Produces a top-10 feature importance chart.
- Compares CatBoost against a one-hot-encoded RandomForest baseline.

## Run it

```bash
python Week24_practice_1_catboost.py
```

## Verified results

The full workflow was executed successfully in this environment.

- Baseline CatBoost accuracy: 0.8777
- Best tuned parameters: `{'depth': 8, 'l2_leaf_reg': 1, 'learning_rate': 0.2}`
- Best CV accuracy: 0.8549
- Final CatBoost test accuracy: 0.8704
- RandomForest test accuracy: 0.8573

Interpretation:

- CatBoost outperformed the one-hot RandomForest baseline on this test split.
- The strongest features were age, relationship, occupation, fnlwgt, capital-gain, and hours-per-week.

## Output artifacts

- Feature importance chart: `outputs/top_10_feature_importance_catboost.png`

## Notes

- Missing categorical values were filled with a placeholder label before model training.
- The script keeps the categorical features as native CatBoost categories rather than one-hot encoding them.
