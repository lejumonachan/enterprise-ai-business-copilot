import pandas as pd
import plotly.express as px


def get_column_types(df):
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    all_cols = df.columns.tolist()

    return numeric_cols, categorical_cols, all_cols


def create_bar_chart(df, x_col, y_col=None, color_col=None):
    if y_col:
        return px.bar(df, x=x_col, y=y_col, color=color_col, template="plotly_white")
    else:
        counts = df[x_col].value_counts().reset_index()
        counts.columns = [x_col, "Count"]
        return px.bar(counts, x=x_col, y="Count", template="plotly_white")


def create_line_chart(df, x_col, y_col, color_col=None):
    return px.line(df, x=x_col, y=y_col, color=color_col, template="plotly_white")


def create_pie_chart(df, category_col):
    pie_data = df[category_col].value_counts().reset_index()
    pie_data.columns = [category_col, "Count"]
    return px.pie(pie_data, names=category_col, values="Count", template="plotly_white")


def create_histogram(df, column, bins=30):
    return px.histogram(df, x=column, nbins=bins, template="plotly_white")


def create_box_plot(df, y_col, x_col=None):
    return px.box(df, x=x_col, y=y_col, template="plotly_white")


def create_scatter_plot(df, x_col, y_col, color_col=None):
    return px.scatter(df, x=x_col, y=y_col, color=color_col, template="plotly_white")


def create_heatmap(df, numeric_cols):
    corr = df[numeric_cols].corr()
    return px.imshow(corr, text_auto=True, template="plotly_white")