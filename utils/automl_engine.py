import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif, f_regression

from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor
)
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score
)


def detect_problem_type(y):
    if y.dtype == "object" or y.dtype.name == "category" or y.nunique() <= 10:
        return "classification"
    return "regression"


def safe_k_value(feature_k, total_features):
    if feature_k == "all":
        return "all"

    try:
        k = int(feature_k)
        return min(k, total_features)
    except:
        return "all"


def run_automl(df, target_column, numeric_strategy="mean", feature_k="all"):
    df = df.copy()

    if target_column not in df.columns:
        raise ValueError("Target column not found.")

    df = df.dropna(subset=[target_column])

    X = df.drop(columns=[target_column])
    y = df[target_column]

    problem_type = detect_problem_type(y)

    target_encoder = None

    if problem_type == "classification":
        target_encoder = LabelEncoder()
        y = target_encoder.fit_transform(y.astype(str))

    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy=numeric_strategy)),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ])

    try:
        processed_sample = preprocessor.fit_transform(X.head(20))
        total_features = processed_sample.shape[1]
    except:
        total_features = X.shape[1]

    k_value = safe_k_value(feature_k, total_features)

    if problem_type == "classification":
        selector = SelectKBest(score_func=f_classif, k=k_value)

        models = {
            "Logistic Regression": LogisticRegression(max_iter=2000),
            "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
            "Extra Trees": ExtraTreesClassifier(n_estimators=200, random_state=42),
            "Gradient Boosting": GradientBoostingClassifier(random_state=42),
            "SVM": SVC(),
            "KNN": KNeighborsClassifier()
        }

    else:
        selector = SelectKBest(score_func=f_regression, k=k_value)

        models = {
            "Linear Regression": LinearRegression(),
            "Ridge Regression": Ridge(),
            "Lasso Regression": Lasso(),
            "Random Forest Regressor": RandomForestRegressor(n_estimators=200, random_state=42),
            "Extra Trees Regressor": ExtraTreesRegressor(n_estimators=200, random_state=42),
            "Gradient Boosting Regressor": GradientBoostingRegressor(random_state=42),
            "SVR": SVR(),
            "KNN Regressor": KNeighborsRegressor()
        }

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    results = {}
    trained_models = {}

    for name, model in models.items():
        try:
            pipeline = Pipeline([
                ("preprocessor", preprocessor),
                ("feature_selection", selector),
                ("model", model)
            ])

            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)

            trained_models[name] = pipeline

            if problem_type == "classification":
                cv_score = cross_val_score(
                    pipeline,
                    X,
                    y,
                    cv=5,
                    scoring="f1_weighted"
                ).mean()

                results[name] = {
                    "Accuracy": accuracy_score(y_test, y_pred),
                    "Precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
                    "Recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
                    "F1 Score": f1_score(y_test, y_pred, average="weighted", zero_division=0),
                    "CV Score": cv_score
                }

            else:
                cv_score = cross_val_score(
                    pipeline,
                    X,
                    y,
                    cv=5,
                    scoring="r2"
                ).mean()

                results[name] = {
                    "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
                    "MAE": mean_absolute_error(y_test, y_pred),
                    "R2 Score": r2_score(y_test, y_pred),
                    "CV Score": cv_score
                }

        except Exception as e:
            results[name] = {"Error": str(e)}

    results_df = pd.DataFrame(results).T

    valid_models = list(trained_models.keys())

    if not valid_models:
        raise ValueError("No model trained successfully.")

    valid_results = results_df.loc[valid_models]

    if valid_results["CV Score"].notna().any():
        best_model_name = valid_results["CV Score"].astype(float).idxmax()
    else:
        if problem_type == "classification":
            best_model_name = valid_results["F1 Score"].astype(float).idxmax()
        else:
            best_model_name = valid_results["R2 Score"].astype(float).idxmax()

    best_model = trained_models[best_model_name]
    predictions = best_model.predict(X_test)

    return {
        "problem_type": problem_type,
        "results": results_df,
        "best_model_name": best_model_name,
        "best_model": best_model,
        "target_encoder": target_encoder,
        "feature_columns": X.columns.tolist(),
        "X_sample": X,
        "actual": y_test[:20],
        "predictions": predictions[:20]
    }


def predict_user_input(model, input_data, feature_columns, target_encoder=None):
    input_df = pd.DataFrame([input_data], columns=feature_columns)
    prediction = model.predict(input_df)[0]

    if target_encoder is not None:
        prediction = target_encoder.inverse_transform([int(prediction)])[0]

    return prediction