# 🧠 ML-Model-Picker

A powerful **Streamlit GUI app** for automatically selecting the best machine learning model based on your dataset. Supports both **classification** and **regression** workflows with built-in evaluation, tuning, feature importance, and model export.

---

## 🚀 Features

| Feature                        | Description                                                                  |
|-------------------------------|------------------------------------------------------------------------------|
| ✅ Model Auto-Selection        | Evaluates 8+ models (SVM, RF, XGB, CatBoost, etc.)                           |
| ✅ Classification + Regression | Choose task type via UI                                                     |
| ✅ GridSearchCV (optional)     | Hyperparameter tuning per model                                             |
| ✅ Cross-validation            | 10-fold scoring stability                                                    |
| ✅ Visual Evaluation           | Confusion Matrix, ROC/PR (Classification), Residuals (Regression)            |
| ✅ Feature Importance          | Visualized using tree-based model scores                                    |
| ✅ Best Model Download         | Export top model as `.pkl`                                                  |
| ✅ Categorical Handling        | Automatic label encoding for object/string columns                          |
| ✅ Missing Value Handling      | Drops or fills based on pandas defaults                                     |
| ✅ Timer & UX Enhancements     | Shows training time, tooltips, dynamic UI                                   |

---

## 📁 Project Structure

ml-model-picker/
├── app.py # Streamlit GUI logic
├── requirements.txt # Python dependencies
├── .gitignore # Ignore results, cache, IDE files
├── src/
│ ├── data_loader.py # Preprocessing and file handling
│ ├── evaluator.py # Visual and metric evaluation
│ └── model_selector.py # Training, tuning, selection
└── results/ # Temporary results and visualizations

---

## ▶️ Getting Started

### 🛠 Installation
```bash
git clone https://github.com/iHakawaTi/ml-model-picker.git
cd ml-model-picker
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
---
▶️ Run the App
```bash
streamlit run app.py
```
---
🖥 GUI Highlights
Upload your .csv dataset

Select task type: Classification or Regression

Pick your target column from dropdown

(Optional) Enable tuning

View visualizations + metrics

Download best .pkl model

---

🌐 Streamlit Cloud Deployment
Push to public GitHub repository

Go to Streamlit Cloud

Create new app → Select your repo → Set app.py

Deploy and share the link!

---

📄 License
MIT — Free to use, modify, and share.

