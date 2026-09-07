import pandas as pd
import numpy as np

def load_data(path):
    return pd.read_csv(path)

def missing_profile(df):
    missing = df.isnull().sum()
    missing_percent = (missing / len(df)) * 100
    return pd.DataFrame({
        "missing_count": missing,
        "missing_percent": missing_percent
    })

def distribution_profile(df):
    desc = df.describe(include='all').T
    return desc

def correlation_profile(df):
    return df.corr()

def outlier_profile(df):
    outlier_counts = {}
    for col in df.select_dtypes(include=[np.number]).columns:
        z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
        outlier_counts[col] = (z_scores > 3).sum()
    return pd.DataFrame.from_dict(outlier_counts, orient='index', columns=['outlier_count'])

