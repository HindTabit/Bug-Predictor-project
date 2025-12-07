# core/preprocessing.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def clean_label(df: pd.DataFrame, label_col: str) -> pd.Series:
    y = pd.to_numeric(df[label_col], errors='coerce').fillna(0)
    return (y > 0).astype(int)

def preprocess_features(X: pd.DataFrame):
    X_num = X.select_dtypes(include=[np.number])
    X_clean = X_num.replace([np.inf, -np.inf], np.nan).fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_clean)
    return X_scaled, scaler, X_clean.columns.tolist()