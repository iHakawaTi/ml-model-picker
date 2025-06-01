import streamlit as st
import pandas as pd
import os
import time
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from src.data_loader import load_data, preprocess_data
from src.model_selector import train_and_evaluate_models
from src.evaluator import (
    plot_confusion_matrix_image,
    plot_roc_curve_image,
    plot_pr_curve_image,
    plot_prediction_vs_actual,
    plot_residuals,
    plot_feature_importance_image,
    get_classification_report_text
)
import shutil, os
shutil.rmtree("results", ignore_errors=True)
os.makedirs("results", exist_ok=True)


st.set_page_config(page_title="ML Model Picker", layout="centered")
st.title("🤖 Machine Learning Model Picker")

uploaded_file = st.file_uploader("📂 Upload your CSV dataset", type=["csv"], help="Upload a CSV file with your dataset")

task_type = st.selectbox("🎯 Select the type of ML task", ["classification", "regression"], help="Choose whether you're predicting categories or numbers")

df_for_target = None
target_column = None

if uploaded_file:
    uploaded_file.seek(0)
    df_for_target = pd.read_csv(uploaded_file)
    st.markdown("### 📌 Dataset Preview")
    st.dataframe(df_for_target.head())
    target_column = st.selectbox("🎯 Select the target column", df_for_target.columns, help="This column will be what the model tries to predict")

tune_option = st.selectbox("🧪 Hyperparameter Tuning Method", ["none", "grid", "random"], help="Choose how model parameters are tuned")

if uploaded_file and target_column:
    uploaded_file.seek(0)
    with st.spinner("⚙️ Loading and preprocessing data..."):
        df = load_data(uploaded_file)
        try:
            X_train, X_test, y_train, y_test, feature_names = preprocess_data(df, target_column)
        except Exception as e:
            st.error(f"❌ Preprocessing Error: {e}")
            st.stop()

    if st.button("🚀 Run Model Selection"):
        with st.spinner("🔍 Training models and evaluating..."):
            try:
                start_time = time.time()
                best_model, best_name, results, preds = train_and_evaluate_models(
                    X_train, X_test, y_train, y_test,
                    task_type, tune=tune_option, feature_names=feature_names
                )
                duration = time.time() - start_time
                st.success(f"✅ Best Model: {best_name} (Completed in {duration:.2f} seconds)")

                st.subheader("📊 Model Comparison Results")
                df_results = pd.DataFrame(results)
                df_results.index = df_results['model']
                st.dataframe(df_results.drop(columns=["model"], errors="ignore"))

                if task_type == "classification":
                    st.subheader("🧱 Confusion Matrix")
                    st.pyplot(plot_confusion_matrix_image(y_test, preds))

                    st.subheader("📃 Classification Report")
                    st.text(get_classification_report_text(y_test, preds))

                    st.subheader("📈 ROC Curve")
                    st.pyplot(plot_roc_curve_image(y_test, best_model, X_test))

                    st.subheader("📉 Precision-Recall Curve")
                    st.pyplot(plot_pr_curve_image(y_test, best_model, X_test))

                else:
                    st.subheader("📐 Prediction vs Actual")
                    st.pyplot(plot_prediction_vs_actual(y_test, preds))

                    st.subheader("📊 Residual Plot")
                    st.pyplot(plot_residuals(y_test, preds))

                st.subheader("🏋️ Feature Importances")
                fig = plot_feature_importance_image(best_model, feature_names)
                if fig:
                    st.pyplot(fig)
                else:
                    st.info("ℹ️ Feature importance is not available for this model.")

                st.subheader("💾 Download Best Model")
                model_path = f"results/{best_name}/{best_name}.pkl"
                if os.path.exists(model_path):
                    with open(model_path, "rb") as f:
                        st.download_button(
                            label="📦 Download .pkl",
                            data=f,
                            file_name=f"{best_name}.pkl",
                            mime="application/octet-stream"
                        )

                st.subheader("🔁 Cross-Validation Scores (10-fold)")
                try:
                    cv_scores = cross_val_score(best_model, X_train, y_train, cv=10)
                    st.write(f"Mean CV Score: {cv_scores.mean():.4f}")
                    st.write(f"Std Dev: {cv_scores.std():.4f}")
                except Exception as e:
                    st.warning(f"⚠️ CV failed: {e}")

            except Exception as e:
                st.error(f"❌ Training failed: {e}")
