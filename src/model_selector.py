import os
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, classification_report, confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from src.evaluator import evaluate_model, get_primary_metric
from src.evaluator import save_classification_outputs, save_feature_importance

from sklearn.preprocessing import LabelEncoder


def get_model_registry(task_type: str):
    if task_type.lower() == "classification":
        return {
            "Logistic Regression": {
                "model": LogisticRegression(max_iter=1000),
                "params": {"C": [0.1, 1, 10]}
            },
            "Random Forest": {
                "model": RandomForestClassifier(),
                "params": {"n_estimators": [100, 200], "max_depth": [None, 10, 20]}
            },
            "XGBoost": {
                "model": XGBClassifier(eval_metric='logloss'),
                "params": {"n_estimators": [100, 200], "max_depth": [3, 6]}
            },
            "LightGBM": {
                "model": LGBMClassifier(),
                "params": {"num_leaves": [31, 50], "learning_rate": [0.05, 0.1]}
            },
            "CatBoost": {
                "model": CatBoostClassifier(verbose=0),
                "params": {"depth": [4, 6], "learning_rate": [0.03, 0.1]}
            },
            "SVM": {
                "model": SVC(probability=True),
                "params": {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"]}
            },
            "KNN": {
                "model": KNeighborsClassifier(),
                "params": {"n_neighbors": [3, 5, 7]}
            },
            "MLP": {
                "model": MLPClassifier(max_iter=300),
                "params": {"hidden_layer_sizes": [(100,), (50, 50)], "activation": ["relu", "tanh"]}
            }
        }

    elif task_type.lower() == "regression":
        return {
            "Linear Regression": {
                "model": LinearRegression(),
                "params": {}
            },
            "Random Forest": {
                "model": RandomForestRegressor(),
                "params": {"n_estimators": [100, 200], "max_depth": [None, 10, 20]}
            },
            "XGBoost": {
                "model": XGBRegressor(),
                "params": {"n_estimators": [100, 200], "max_depth": [3, 6]}
            },
            "LightGBM": {
                "model": LGBMRegressor(),
                "params": {"num_leaves": [31, 50], "learning_rate": [0.05, 0.1]}
            },
            "CatBoost": {
                "model": CatBoostRegressor(verbose=0),
                "params": {"depth": [4, 6], "learning_rate": [0.03, 0.1]}
            },
            "SVR": {
                "model": SVR(),
                "params": {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"]}
            },
            "KNN": {
                "model": KNeighborsRegressor(),
                "params": {"n_neighbors": [3, 5, 7]}
            },
            "MLP": {
                "model": MLPRegressor(max_iter=300),
                "params": {"hidden_layer_sizes": [(100,), (50, 50)], "activation": ["relu", "tanh"]}
            }
        }

    else:
        raise ValueError("Invalid task type. Choose 'classification' or 'regression'.")

def train_and_evaluate_models(X_train, X_test, y_train, y_test, task_type, tune="none", feature_names=None):
    model_registry = get_model_registry(task_type)
    results = []
    best_score = -np.inf
    best_model = None
    best_name = None
    best_preds = None
    best_probs = None

    primary_metric_name = get_primary_metric(task_type)
    scorer = make_scorer(primary_metric_name, greater_is_better=True) if callable(primary_metric_name) else primary_metric_name

    for name, entry in model_registry.items():
        model = entry["model"]
        param_grid = entry["params"]

        from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

        if tune == "grid" and param_grid:
            try:
                search = GridSearchCV(model, param_grid, scoring=scorer, cv=3, n_jobs=-1)
                search.fit(X_train, y_train)
                model = search.best_estimator_
            except Exception as e:
                print(f"[!] Grid search failed for {name}: {e}")
                continue
        elif tune == "random" and param_grid:
            try:
                search = RandomizedSearchCV(model, param_grid, n_iter=10, scoring=scorer, cv=3, n_jobs=-1,
                                            random_state=42)
                search.fit(X_train, y_train)
                model = search.best_estimator_
            except Exception as e:
                print(f"[!] Random search failed for {name}: {e}")
                continue
        else:
            model.fit(X_train, y_train)

        preds = model.predict(X_test)
        score_dict = evaluate_model(y_test, preds, task_type)
        score_dict["model"] = name
        results.append(score_dict)

        primary_metric = "f1" if task_type == "classification" else "r2"
        current_score = score_dict.get(primary_metric, -np.inf)

        if current_score > best_score:
            best_score = current_score
            best_model = model
            best_name = name
            best_preds = preds
            best_probs = model.predict_proba(X_test) if task_type == "classification" and hasattr(model, "predict_proba") else None

    results_dir = f"results/{best_name}"
    os.makedirs(results_dir, exist_ok=True)
    joblib.dump(best_model, f"{results_dir}/{best_name}.pkl")

    if task_type == "classification":
        save_classification_outputs(y_test, best_preds, best_probs, best_name, results_dir)
    if feature_names is not None:
        save_feature_importance(best_model, feature_names, best_name, results_dir)

    return best_model, best_name, results, best_preds
#python main.py "C:\Users\USER\Downloads\data1.csv" --target diagnosis_M --task classification --tune grid
