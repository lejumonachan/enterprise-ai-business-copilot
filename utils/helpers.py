import os
import pandas as pd


def ensure_directories():
    folders = [
        "uploads",
        "reports",
        "data",
        "models",
        "vector_store"
    ]

    for folder in folders:
        os.makedirs(folder, exist_ok=True)


def dataframe_to_context(df, max_rows=30):
    context = f"""
Dataset Shape: {df.shape}

Columns:
{list(df.columns)}

Missing Values:
{df.isnull().sum().to_string()}

Preview:
{df.head(max_rows).to_string()}

Statistical Summary:
{df.describe(include='all').to_string()}
"""

    return context


def is_dataframe_available():
    import streamlit as st

    return (
        "file_type" in st.session_state
        and st.session_state["file_type"] == "dataframe"
        and "uploaded_data" in st.session_state
    )


def is_pdf_available():
    import streamlit as st

    return (
        "file_type" in st.session_state
        and st.session_state["file_type"] == "pdf"
        and "uploaded_text" in st.session_state
    )