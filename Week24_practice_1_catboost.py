import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def load_and_prepare_data():
    X, y = fetch_openml("adult", version=2, as_frame=True, return_X_y=True)
    X = X.copy()
    y = y.copy()

    categorical_cols = list(X.select_dtypes(include=["object", "category"]).columns)
    numerical_cols = [col for col in X.columns if col not in categorical_cols]

    for col in categorical_cols:
        X[col] = X[col].astype("object").fillna("Missing")
    for col in numerical_cols:
        X[col] = X[col].fillna(X[col].median())

    y = y.astype(str)
    y = y.map({"<=50K": 0, ">50K": 1}).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    cat_features = [X.columns.get_loc(col) for col in categorical_cols]
    print(f"Dataset shape: {X.shape}")
    print(f"Categorical features: {len(categorical_cols)}")
    print(f"Numerical features: {len(numerical_cols)}")
    print(f"Train shape: {X_train.shape}")
    print(f"Test shape: {X_test.shape}")

    return X_train, X_test, y_train, y_test, cat_features, categorical_cols, numerical_cols


def evaluate_classifier(model, X_test, y_test):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    conf = confusion_matrix(y_test, y_pred)

    print("\nEvaluation metrics")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print("\nConfusion matrix:")
    print(conf)
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=["<=50K", ">50K"]))

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": conf,
    }


def train_baseline_catboost(X_train, X_test, y_train, y_test, cat_features):
    print("\n=== Baseline CatBoost model ===")
    model = CatBoostClassifier(
        iterations=500,
        learning_rate=0.1,
        depth=6,
        random_seed=42,
        verbose=50,
        loss_function="Logloss",
        eval_metric="Accuracy",
    )
    model.fit(X_train, y_train, cat_features=cat_features)
    return evaluate_classifier(model, X_test, y_test)


def tune_catboost(X_train, y_train, cat_features):
    print("\n=== Hyperparameter tuning with GridSearchCV ===")
    estimator = CatBoostClassifier(
        iterations=10,
        random_seed=42,
        verbose=False,
        loss_function="Logloss",
        eval_metric="Accuracy",
    )

    param_grid = {
        "depth": [4, 6, 8],
        "learning_rate": [0.03, 0.1, 0.2],
        "l2_leaf_reg": [1, 3, 5],
    }

    grid = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        scoring="accuracy",
        cv=3,
        n_jobs=1,
        verbose=0,
    )

    grid.fit(X_train, y_train, cat_features=cat_features)
    best_params = grid.best_params_
    best_score = grid.best_score_

    print(f"Best parameters: {best_params}")
    print(f"Best CV accuracy: {best_score:.4f}")
    return best_params, best_score


def train_final_catboost(X_train, y_train, cat_features, best_params):
    print("\n=== Final CatBoost model ===")
    model = CatBoostClassifier(
        iterations=500,
        random_seed=42,
        verbose=False,
        loss_function="Logloss",
        eval_metric="Accuracy",
        **best_params,
    )
    model.fit(X_train, y_train, cat_features=cat_features)
    return model


def plot_feature_importance(model, output_path):
    feature_importance = model.get_feature_importance(prettified=True)
    top_10 = feature_importance.head(10).copy()
    top_10["Feature Id"] = top_10["Feature Id"].astype(str)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top_10["Feature Id"][::-1], top_10["Importances"][::-1], color="#2C7FB8")
    ax.set_title("Top 10 Feature Importances (CatBoost)")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    plt.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"\nSaved feature importance chart to {output_path}")
    print(top_10.to_string(index=False))


def train_random_forest_baseline(X_train, X_test, y_train, y_test, categorical_cols):
    print("\n=== RandomForest baseline comparison ===")
    categorical_cols = list(categorical_cols)
    numerical_cols = [c for c in X_train.columns if c not in categorical_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("num", SimpleImputer(strategy="median"), numerical_cols),
        ],
        remainder="drop",
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)),
        ]
    )

    pipeline.fit(X_train, y_train)
    y_pred_rf = pipeline.predict(X_test)
    rf_accuracy = accuracy_score(y_test, y_pred_rf)

    print(f"RandomForest accuracy: {rf_accuracy:.4f}")
    return rf_accuracy


def main():
    X_train, X_test, y_train, y_test, cat_features, categorical_cols, numerical_cols = load_and_prepare_data()

    baseline_metrics = train_baseline_catboost(X_train, X_test, y_train, y_test, cat_features)
    best_params, best_score = tune_catboost(X_train, y_train, cat_features)
    final_model = train_final_catboost(X_train, y_train, cat_features, best_params)
    final_metrics = evaluate_classifier(final_model, X_test, y_test)

    plot_feature_importance(
        final_model,
        OUTPUT_DIR / "top_10_feature_importance_catboost.png",
    )

    rf_accuracy = train_random_forest_baseline(X_train, X_test, y_train, y_test, categorical_cols)

    print("\n=== Final project summary ===")
    print(f"Baseline CatBoost accuracy: {baseline_metrics['accuracy']:.4f}")
    print(f"Tuned CatBoost CV accuracy: {best_score:.4f}")
    print(f"Final CatBoost test accuracy: {final_metrics['accuracy']:.4f}")
    print(f"RandomForest test accuracy: {rf_accuracy:.4f}")

    if final_metrics["accuracy"] > rf_accuracy:
        print("CatBoost outperformed the one-hot RandomForest baseline.")
    elif final_metrics["accuracy"] < rf_accuracy:
        print("RandomForest slightly outperformed CatBoost on this split.")
    else:
        print("CatBoost and RandomForest performed similarly on this split.")


if __name__ == "__main__":
    main()
