import pandas as pd

def load_data(file) -> pd.DataFrame:
    """Reads uploaded CSV file and returns a pandas DataFrame"""
    try:
        df = pd.read_csv(file)
        return df
    except Exception as e:
        raise ValueError(f"Error reading file: {e}")

def validate_columns(df: pd.DataFrame, target_col: str):
    """Ensure target column exists and has at least 2 classes/unique values"""
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset.")
    if df[target_col].nunique() < 2:
        raise ValueError("Target column must have at least 2 unique values.")

import pandas as pd
from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd

def preprocess_data(df, target_column):
    df = df.copy()

    # Drop completely empty columns
    df = df.dropna(axis=1, how='all')

    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Fill missing categorical columns (mode)
    for col in X.select_dtypes(include=['object', 'category']).columns:
        X[col] = X[col].fillna(X[col].mode()[0])
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))

    # Fill missing numeric columns (mean)
    for col in X.select_dtypes(include=['number']).columns:
        X[col] = X[col].fillna(X[col].mean())

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, X.columns.tolist()
