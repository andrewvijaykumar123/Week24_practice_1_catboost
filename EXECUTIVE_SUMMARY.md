# Executive Summary

This project builds a CatBoost classifier to predict whether annual income exceeds $50K using the Adult Income dataset from OpenML. The implementation follows a complete machine-learning workflow: data preparation, benchmark model training, hyperparameter tuning, final model evaluation, and feature-importance analysis.

## Key result

The tuned CatBoost model achieved a final test accuracy of 0.8704, outperforming a one-hot-encoded RandomForest baseline at 0.8573. This confirms CatBoost's advantage on mixed data containing both numerical and categorical features.

## Why this matters

CatBoost handles categorical variables natively, which reduces preprocessing complexity and often improves model quality on tabular tasks. In this case, the model was able to use features such as age, relationship, occupation, capital gain, and hours worked without manually one-hot encoding the categories.

## Best tuned parameters

- depth: 8
- l2_leaf_reg: 1
- learning_rate: 0.2

## Output artifacts

- Feature importance chart in outputs/top_10_feature_importance_catboost.png
- Full implementation in Week24_practice_1_catboost.py
- Notebook version in Week24_practice_1_catboost.ipynb

## Conclusion

This project is a strong example of a well-structured, production-quality tabular ML workflow using CatBoost. It demonstrates both practical performance and clean reproducibility for class submissions or portfolio work.
