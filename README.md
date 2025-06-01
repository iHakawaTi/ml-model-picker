# 🧠 ML-Model-Picker

**Effortlessly select the optimal machine learning model for your data with this powerful Streamlit GUI application.**

ML-Model-Picker automates the often complex process of model selection by evaluating multiple algorithms on your dataset. Whether you're tackling a **classification** or **regression** problem, this app provides built-in tools for evaluation, hyperparameter tuning, feature importance analysis, and easy export of the best-performing model.

-----

## 🚀 Key Features

| Feature                     | Description                                                                                                |
|-----------------------------|------------------------------------------------------------------------------------------------------------|
| **🤖 Automated Model Selection** | Evaluates a diverse set of 8+ popular models including SVM, Random Forest, XGBoost, CatBoost, and more. |
| **🎯 Classification & Regression** | Clearly define your task type through an intuitive user interface.                                     |
| **⚙️ Optional Hyperparameter Tuning** | Leverages GridSearchCV for fine-tuning model parameters to maximize performance (user-selectable).     |
| **🧪 Robust Cross-validation** | Employs 10-fold cross-validation to ensure the stability and reliability of model scoring.                |
| **📊 Visual Performance Insights** | Generates insightful visualizations: Confusion Matrix and ROC/PR curves for classification; Residual plots for regression. |
| **🔑 Feature Importance Analysis** | Provides visual representations of feature importance derived from tree-based models.                     |
| **💾 Easy Model Export** | Allows you to download the top-performing model as a ready-to-use `.pkl` file.                            |
| **✨ Smart Categorical Handling** | Automatically performs label encoding on columns with object or string data types.                      |
| **🧹 Intelligent Missing Value Handling** | Offers options to either drop or fill missing values using standard pandas behavior.                 |
| **⏱️ Enhanced User Experience** | Displays training time, provides helpful tooltips, and features a dynamic and responsive user interface. |

-----

## 📂 Project Structure
```
ml-model-picker/
├── app.py             # Streamlit GUI application logic
├── requirements.txt   # List of Python dependencies
├── .gitignore         # Specifies intentionally untracked files that Git should ignore
├── src/
│   ├── data_loader.py   # Handles data loading and preprocessing
│   ├── evaluator.py     # Implements evaluation metrics and visualizations
│   └── model_selector.py# Contains model training, tuning, and selection logic
├──sample_data/
│   ├── sample_classification.csv
│   └── sample_regression.csv
│   
└── results/         # Directory for temporary results and generated visualizations
```

-----

## ▶️ Getting Started

### 🛠 Installation

1.  **Clone the repository:**

    
    git clone [https://github.com/iHakawaTi/ml-model-picker.git](https://github.com/iHakawaTi/ml-model-picker.git)
    ```bash
    cd ml-model-picker
    ```

3.  **Create and activate a virtual environment (recommended):**

    ```bash
    python -m venv .venv
    source .venv/bin/activate   # Linux/macOS
    .venv\Scripts\activate      # Windows
    ```

4.  **Install the required Python packages:**

    ```bash
    pip install -r requirements.txt
    ```

-----

### ▶️ Run the Application
### 🔧 Launch the GUI

```bash
streamlit run app.py
```
🖥️ Key GUI Interactions

- **Upload Dataset:** Easily upload your dataset in `.csv` format.
- **Select Task Type:** Choose between "Classification" and "Regression" based on your prediction goal.
- **Define Target Column:** Select the column in your dataset that you want to predict.
- **(Optional) Enable Tuning:** Check the box to enable hyperparameter optimization using GridSearchCV for each evaluated model.
- **Explore Results:** View performance metrics and insightful visualizations for each model.
- **Download Best Model:** Click the button to download the highest-performing model as a `.pkl` file for future use.

---

## 🌐 Live Demo

🚀 Try the app now: [Open on Streamlit Cloud](https://ml-model-picker.streamlit.app/)

## 📄 License

MIT License — free to use, modify, and distribute.

---

Contributions welcome!  
If you find bugs or have ideas, feel free to open an issue or pull request.
