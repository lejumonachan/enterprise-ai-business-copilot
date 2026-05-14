import pandas as pd


def profile_dataframe(df):
    profile = {
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "missing_values": df.isnull().sum().to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
        "data_types": df.dtypes.astype(str).to_dict(),
        "numeric_summary": df.describe().to_dict() if len(df.select_dtypes(include="number").columns) > 0 else {},
    }

    return profile


def create_text_profile(df):
    profile = profile_dataframe(df)

    text = f"""
Dataset Shape: {profile['shape']}

Columns:
{profile['columns']}

Missing Values:
{profile['missing_values']}

Duplicate Rows:
{profile['duplicate_rows']}

Data Types:
{profile['data_types']}
"""

    return text