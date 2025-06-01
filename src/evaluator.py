import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay,
    mean_squared_error,
    r2_score
)
import numpy as np
from io import StringIO

import os
import joblib

def save_classification_outputs(y_true, y_pred, y_prob, model_name, results_dir):
    os.makedirs(results_dir, exist_ok=True)
    report = classification_report(y_true, y_pred, output_dict=False)
    with open(f"{results_dir}/{model_name}_classification_report.txt", "w") as f:
        f.write(report)

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax)
    plt.savefig(f"{results_dir}/{model_name}_confusion_matrix.png")
    plt.close()

    if y_prob is not None:
        RocCurveDisplay.from_predictions(y_true, y_prob[:, 1]).plot()
        plt.savefig(f"{results_dir}/{model_name}_roc_curve.png")
        plt.close()

        PrecisionRecallDisplay.from_predictions(y_true, y_prob[:, 1]).plot()
        plt.savefig(f"{results_dir}/{model_name}_pr_curve.png")
        plt.close()


def save_feature_importance(model, feature_names, model_name, results_dir):
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        sorted_idx = np.argsort(importances)[::-1]
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=importances[sorted_idx], y=np.array(feature_names)[sorted_idx], ax=ax)
        ax.set_title("Feature Importances")
        plt.tight_layout()
        os.makedirs(results_dir, exist_ok=True)
        plt.savefig(f"{results_dir}/{model_name}_feature_importance.png")
        plt.close()

def plot_confusion_matrix_image(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax)
    plt.title("Confusion Matrix")
    return fig


def get_classification_report_text(y_true, y_pred):
    return classification_report(y_true, y_pred)


def plot_roc_curve_image(y_true, model, X_test):
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_score = model.decision_function(X_test)
    else:
        return None
    fig, ax = plt.subplots()
    RocCurveDisplay.from_predictions(y_true, y_score, ax=ax)
    return fig


def plot_pr_curve_image(y_true, model, X_test):
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_score = model.decision_function(X_test)
    else:
        return None
    fig, ax = plt.subplots()
    PrecisionRecallDisplay.from_predictions(y_true, y_score, ax=ax)
    return fig


def plot_prediction_vs_actual(y_true, y_pred):
    fig, ax = plt.subplots()
    ax.scatter(y_true, y_pred)
    ax.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], '--r')
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title("Prediction vs Actual")
    return fig


def plot_residuals(y_true, y_pred):
    residuals = y_true - y_pred
    fig, ax = plt.subplots()
    sns.histplot(residuals, kde=True, ax=ax)
    ax.set_title("Residuals Distribution")
    return fig


def plot_feature_importance_image(model, feature_names):
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        sorted_idx = np.argsort(importances)[::-1]
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=importances[sorted_idx], y=np.array(feature_names)[sorted_idx], ax=ax)
        ax.set_title("Feature Importances")
        plt.tight_layout()
        return fig
    return None


def get_primary_metric(task_type):
    if task_type == "classification":
        return "f1"
    elif task_type == "regression":
        return r2_score
    else:
        raise ValueError("Invalid task type")


def evaluate_model(y_true, y_pred, task_type):
    if task_type == "classification":
        report = classification_report(y_true, y_pred, output_dict=True)
        return {
            "accuracy": report.get("accuracy", 0),
            "precision": report.get("weighted avg", {}).get("precision", 0),
            "recall": report.get("weighted avg", {}).get("recall", 0),
            "f1": report.get("weighted avg", {}).get("f1-score", 0),
        }
    elif task_type == "regression":
        mse = mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        return {
            "mse": mse,
            "r2": r2,
        }
    else:
        raise ValueError("Invalid task type")
